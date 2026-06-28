#!/usr/bin/env python3
"""
Re-annotate rows using a third model, then accept final_label only when all three
models agree on the same non-need_review label. Otherwise keep need_review.

Results are merged back into the original CSV with tertiary_label, tertiary_reason.

Workflow:
  1. Read conflict or 3rd_annotaion annotated CSV (--file)
  2. Call qwen/qwen3-next-80b-a3b-instruct (tertiary) per pending row
  3. Unanimous vote: primary == secondary == tertiary and not need_review → final_label
  4. Write back in place (add tertiary_label, tertiary_reason; update final_label)

Examples:
  python run_need_review_triple_vote.py --country MX --dry-run
  python run_need_review_triple_vote.py --country all --resume
  python run_need_review_triple_vote.py --country all --parallel-countries --resume
  python run_need_review_triple_vote.py --country BR --all-rows \\
    --file ../3rd_annotaion/BR/BR_train_pool_R5_review_with_id_annotated.csv --resume
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path as _Path

_SCRIPT_DIR = _Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from run_dual_arbitration_annotation import (
    COUNTRY_CONFIGS,
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRIES,
    CountryConfig,
    ModelSpec,
    build_initial_system_prompt,
    build_user_prompt,
    call_nim_label,
    chunks,
    extract_allowed_labels,
    load_env_for_country,
    row_text,
    stable_content_id,
)

DEFAULT_MAX_WORKERS = 3
DEFAULT_TERTIARY_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
WORKER_KEY_COUNTRIES = ("BR", "MX", "SA")
CONFLICT_DIR = PROJECT_ROOT / "2nd_annotation" / "conflict" / "conflict_annotaion"
THIRD_EXPORT_DIR = PROJECT_ROOT / "3rd_annotaion" / "exports"
TERTIARY_LABEL_COL = "tertiary_label"
TERTIARY_REASON_COL = "tertiary_reason"

DEFAULT_FILE_PATTERN = "{country}_conflict_clean_annotation_2nd.csv"


def read_env_var(env_path: _Path, var_name: str) -> str:
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != var_name:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            return value
    raise RuntimeError(f"{var_name} not found in {env_path}")


def load_worker_api_keys(
    max_workers: int,
    api_key_index: int | None = None,
) -> list[tuple[str, str]]:
    if api_key_index is not None:
        from nv_api_keys import get_all_keys

        pool = get_all_keys()
        keys: list[tuple[str, str]] = []
        for worker_index in range(max_workers):
            idx = (api_key_index + worker_index) % len(pool)
            keys.append((pool[idx], f"nv_api_keys[{idx}]"))
        return keys

    keys = []
    for worker_index in range(max_workers):
        country = WORKER_KEY_COUNTRIES[worker_index % len(WORKER_KEY_COUNTRIES)]
        env_path = PROJECT_ROOT / country / ".env"
        api_key = read_env_var(env_path, "NIM_API_KEY")
        label = f"{country}/.env:NIM_API_KEY"
        keys.append((api_key, label))
    return keys


def conflict_csv_path(country: str) -> _Path:
    return CONFLICT_DIR / DEFAULT_FILE_PATTERN.format(country=country.upper())


def with_arbitrator_override(config: CountryConfig, arbitrator_model: str | None) -> CountryConfig:
    arbitrator = config.arbitrator
    if arbitrator_model:
        arbitrator = ModelSpec(
            arbitrator.role,
            arbitrator_model,
            arbitrator.account,
            arbitrator.timeout,
        )
    return CountryConfig(
        country=config.country,
        default_input=config.default_input,
        prompt_path=config.prompt_path,
        primary=config.primary,
        secondary=config.secondary,
        arbitrator=arbitrator,
        high_risk_labels=config.high_risk_labels,
    )


def load_country_prompt(country_config: CountryConfig) -> tuple[str, set[str]]:
    prompt_path = country_config.prompt_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Compiled prompt not found: {prompt_path}")
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    allowed_labels = extract_allowed_labels(system_prompt)
    return system_prompt, allowed_labels


def choose_triple_vote_final_label(labels: list[str]) -> tuple[str, str]:
    normalized = [str(label).strip() for label in labels if str(label).strip()]
    if not normalized:
        return "need_review", "triple_vote_empty"
    unique = set(normalized)
    if len(unique) == 1:
        label = normalized[0]
        if label == "need_review":
            return "need_review", "triple_vote_unanimous_need_review"
        return label, "triple_vote_unanimous"
    return "need_review", "triple_vote_not_unanimous"


def row_content_id(country: str, row_index: int, row: pd.Series) -> str:
    for column in ("_ann_row_id", "content_id", "id"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return stable_content_id(country, row_index, row)


def ensure_tertiary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TERTIARY_LABEL_COL not in out.columns:
        out[TERTIARY_LABEL_COL] = ""
    if TERTIARY_REASON_COL not in out.columns:
        out[TERTIARY_REASON_COL] = ""
    return out


def tertiary_label_is_filled(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def pending_tertiary_rows(
    df: pd.DataFrame,
    limit: int | None,
    resume: bool,
    all_rows: bool,
) -> list[tuple[int, pd.Series]]:
    if not all_rows and "final_label" not in df.columns:
        raise ValueError("Input missing final_label column")

    if all_rows:
        candidates = df.copy()
    else:
        mask = df["final_label"].astype(str).str.strip() == "need_review"
        candidates = df[mask].copy()
    text_mask = candidates.apply(lambda row: bool(row_text(row)), axis=1)
    candidates = candidates[text_mask]

    pending: list[tuple[int, pd.Series]] = []
    for idx, row in candidates.iterrows():
        if resume and tertiary_label_is_filled(row.get(TERTIARY_LABEL_COL)):
            continue
        pending.append((int(idx), row))
        if limit and limit > 0 and len(pending) >= limit:
            break
    return pending


def apply_row_update(full_df: pd.DataFrame, row_index: int, updates: dict) -> None:
    for key, value in updates.items():
        full_df.at[row_index, key] = value


def process_one_row(
    country_config: CountryConfig,
    allowed_labels: set[str],
    system_prompt: str,
    row_index: int,
    row: pd.Series,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    api_key: str,
    api_key_label: str,
) -> tuple[int, dict]:
    """Call tertiary model with the same prompt path as primary/secondary in run_dual_arbitration_annotation."""
    text = row_text(row)
    user_prompt = build_user_prompt(text)
    initial_system_prompt = build_initial_system_prompt(system_prompt)

    tertiary_result = call_nim_label(
        country_config.country,
        country_config.arbitrator,
        initial_system_prompt,
        user_prompt,
        allowed_labels,
        temperature,
        max_tokens,
        retries,
        backoff_seconds,
        api_key=api_key,
        api_key_label=api_key_label,
    )

    primary_label = str(row.get("primary_label") or "").strip() or "need_review"
    secondary_label = str(row.get("secondary_label") or "").strip() or "need_review"
    tertiary_label = tertiary_result["label"]
    vote_labels = [primary_label, secondary_label, tertiary_label]
    new_final_label, new_final_source = choose_triple_vote_final_label(vote_labels)

    return row_index, {
        TERTIARY_LABEL_COL: tertiary_label,
        TERTIARY_REASON_COL: tertiary_result.get("reason", ""),
        "final_label": new_final_label,
        "final_source": new_final_source,
    }


def export_final_need_review(
    country: str,
    source_csv_path: _Path,
    df: pd.DataFrame,
    export_dir: _Path,
) -> _Path | None:
    """Optional: filter need_review rows, keep the input CSV columns as-is."""
    mask = df["final_label"].astype(str).str.strip() == "need_review"
    need_df = df[mask].copy()
    if need_df.empty:
        print(f"[{country}] no need_review rows to export.", flush=True)
        return None

    export_path = export_dir / f"{source_csv_path.stem}_need_review.csv"
    write_csv_atomic(export_path, need_df)
    print(
        f"[{country}] exported {len(need_df):,} need_review rows to {export_path}",
        flush=True,
    )
    return export_path


def write_csv_atomic(path: _Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".csv", dir=str(path.parent))
    try:
        import os

        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            df.to_csv(handle, index=False)
        shutil.move(tmp_name, path)
    except Exception:
        import os

        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def run_country(
    country_config: CountryConfig,
    csv_path: _Path,
    limit: int | None,
    batch_size: int,
    max_workers: int,
    resume: bool,
    dry_run: bool,
    all_rows: bool,
    api_key_index: int | None,
    export_need_review: bool,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
) -> None:
    country = country_config.country
    load_env_for_country(country)
    worker_api_keys = load_worker_api_keys(max_workers, api_key_index)
    system_prompt, allowed_labels = load_country_prompt(country_config)

    if not csv_path.exists():
        raise FileNotFoundError(f"Conflict CSV not found: {csv_path}")

    full_df = ensure_tertiary_columns(pd.read_csv(csv_path))
    pending = pending_tertiary_rows(full_df, limit, resume, all_rows)
    total_batches = (len(pending) + batch_size - 1) // batch_size if pending else 0
    scope = "all rows" if all_rows else "need_review rows"

    print(f"\n[{country}] file: {csv_path}", flush=True)
    print(f"[{country}] prompt: {country_config.prompt_path}", flush=True)
    print(f"[{country}] allowed labels ({len(allowed_labels)}): {', '.join(sorted(allowed_labels))}", flush=True)
    print(
        f"[{country}] {scope} pending: {len(pending):,}  batches: {total_batches}  "
        f"(in-place update, +{TERTIARY_LABEL_COL}, +{TERTIARY_REASON_COL})",
        flush=True,
    )
    print(
        f"[{country}] tertiary={country_config.arbitrator.model}  "
        f"account={country_config.arbitrator.account}",
        flush=True,
    )
    print(f"[{country}] batch_size={batch_size}  max_workers={max_workers}", flush=True)
    print(
        f"[{country}] api keys: "
        + ", ".join(f"worker{i}={label}" for i, (_, label) in enumerate(worker_api_keys)),
        flush=True,
    )

    if dry_run:
        for idx, row in pending[:5]:
            cid = row_content_id(country, idx, row)
            print(
                f"  - {cid}: primary={row.get('primary_label')} "
                f"secondary={row.get('secondary_label')}  text={row_text(row)[:100]}",
                flush=True,
            )
        if export_need_review:
            export_final_need_review(country, csv_path, full_df, THIRD_EXPORT_DIR)
        return

    if not pending:
        print(f"[{country}] nothing to process.", flush=True)
        if export_need_review:
            export_final_need_review(country, csv_path, full_df, THIRD_EXPORT_DIR)
        return

    processed = 0
    resolved = 0
    for batch_number, batch in enumerate(chunks(pending, batch_size), start=1):
        batch_resolved = 0
        print(f"\n[{country}] ── batch {batch_number}/{total_batches} ({len(batch)} rows) ──", flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_one_row,
                    country_config,
                    allowed_labels,
                    system_prompt,
                    idx,
                    row,
                    temperature,
                    max_tokens,
                    retries,
                    backoff_seconds,
                    worker_api_keys[task_idx % max_workers][0],
                    worker_api_keys[task_idx % max_workers][1],
                )
                for task_idx, (idx, row) in enumerate(batch)
            ]
            for future in as_completed(futures):
                row_index, updates = future.result()
                apply_row_update(full_df, row_index, updates)
                write_csv_atomic(csv_path, full_df)
                processed += 1
                if updates.get("final_label") != "need_review":
                    batch_resolved += 1
                    resolved += 1
                row = full_df.loc[row_index]
                cid = row_content_id(country, row_index, row)
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(
                    f"[{country}] {ts}  batch {batch_number}/{total_batches}  "
                    f"{processed}/{len(pending)}  resolved {resolved}  "
                    f"id={cid}  tertiary={updates.get(TERTIARY_LABEL_COL)}  "
                    f"final={updates.get('final_label')}",
                    flush=True,
                )

        print(
            f"[{country}] ── batch {batch_number}/{total_batches} saved to {csv_path.name} ──",
            flush=True,
        )

    print(
        f"[{country}] finished: {processed} rows updated in place, {resolved} resolved from need_review",
        flush=True,
    )
    if export_need_review:
        export_final_need_review(country, csv_path, full_df, THIRD_EXPORT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True, help="BR, MX, TR, SA, or all")
    parser.add_argument("--file", type=_Path, help="Override conflict CSV path (single country only).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="default: 3, one NIM_API_KEY per worker from BR/MX/SA .env")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument(
        "--parallel-countries",
        action="store_true",
        help="When --country all, run BR/MX/TR/SA concurrently (default: one country after another).",
    )
    parser.add_argument(
        "--arbitrator-model",
        help=f"Override tertiary model (default: {DEFAULT_TERTIARY_MODEL}).",
    )
    parser.add_argument(
        "--export-need-review",
        action="store_true",
        help="Export need_review rows to 3rd_annotaion/exports/<input_stem>_need_review.csv (off by default).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export need_review rows (requires --export-need-review). Does not call the API.",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Annotate every row (not only final_label==need_review). Use with --file for 3rd_annotaion CSVs.",
    )
    parser.add_argument(
        "--api-key-index",
        type=int,
        default=None,
        help="Start index in nv_api_keys for workers (worker i uses index+i). Pair with --max-workers.",
    )
    return parser.parse_args()


def _run_one_country(args: argparse.Namespace, country: str) -> None:
    country_config = with_arbitrator_override(
        COUNTRY_CONFIGS[country], args.arbitrator_model or DEFAULT_TERTIARY_MODEL
    )
    csv_path = args.file or conflict_csv_path(country)
    if args.country.upper() != "ALL" and args.file:
        csv_path = args.file

    if args.export_only:
        if not args.export_need_review:
            print("ERROR: --export-only requires --export-need-review", file=sys.stderr)
            raise SystemExit(2)
        full_df = ensure_tertiary_columns(pd.read_csv(csv_path))
        export_final_need_review(country, csv_path, full_df, THIRD_EXPORT_DIR)
        return

    run_country(
        country_config=country_config,
        csv_path=csv_path,
        limit=args.limit,
        batch_size=max(1, args.batch_size),
        max_workers=max(1, args.max_workers),
        resume=args.resume,
        dry_run=args.dry_run,
        all_rows=args.all_rows,
        api_key_index=args.api_key_index,
        export_need_review=args.export_need_review,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=max(0, args.retries),
        backoff_seconds=max(0.0, args.backoff_seconds),
    )


def configure_line_buffered_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


def main() -> int:
    configure_line_buffered_stdio()
    args = parse_args()
    country_arg = args.country.upper()
    target_countries = ["BR", "MX", "TR", "SA"]
    countries = target_countries if country_arg == "ALL" else [country_arg]

    unknown = [c for c in countries if c not in target_countries]
    if unknown:
        print(f"Unsupported country: {', '.join(unknown)}", file=sys.stderr)
        return 2

    if len(countries) > 1 and args.parallel_countries:
        print(
            f"Running {len(countries)} countries in parallel: {', '.join(countries)} "
            f"(each max_workers={max(1, args.max_workers)})"
        )
        with ThreadPoolExecutor(max_workers=len(countries)) as executor:
            futures = [executor.submit(_run_one_country, args, country) for country in countries]
            for future in as_completed(futures):
                future.result()
    else:
        if len(countries) > 1:
            print(f"Running countries sequentially: {' → '.join(countries)}")
        for country in countries:
            _run_one_country(args, country)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
