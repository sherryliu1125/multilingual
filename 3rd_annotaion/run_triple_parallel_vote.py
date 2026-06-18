#!/usr/bin/env python3
"""
Triple parallel vote: primary + secondary + tertiary run concurrently per row.

Models (TR default, from dual + triple scripts):
  - meta/llama-3.3-70b-instruct
  - google/gemma-3n-e4b-it
  - qwen/qwen3-next-80b-a3b-instruct

Vote policy:
  - all 3 agree on same label          -> final_label = that label
  - exactly 2 agree, 1 differs         -> final_label = need_review
  - all 3 different labels             -> final_label = reject

Examples:
  python run_triple_parallel_vote.py --country TR \\
    --input TR/TR_full_triple_annotation_pending.csv \\
    --selection non_unanimous --dry-run

  python run_triple_parallel_vote.py --country TR \\
    --input TR/TR_full_triple_annotation_pending.csv \\
    --output TR/TR_full_triple_annotation_pending_triple_voted.csv \\
    --selection non_unanimous --resume --use-key-pool --max-workers 8 --batch-size 20

  Input is never modified. Results are appended only to --output (or default
  ``{input_stem}_triple_voted.csv``). If that default already exists without
  ``--resume``, a new ``_triple_voted_v2`` (etc.) path is chosen automatically.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_ROOT / "annotaion_prompts"
if str(PROMPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPTS_DIR))

# run_dual_arbitration_annotation imports pandas at module load; stub it because
# this script only needs the NIM call helpers, not dataframe IO.
if "pandas" not in sys.modules:
    import types

    _pandas_stub = types.ModuleType("pandas")

    class _StubDataFrame:
        columns: list[str] = []

        def __init__(self, *args, **kwargs):
            self.columns = list(kwargs.get("columns") or [])

        def copy(self):
            return self

        def iterrows(self):
            return iter([])

        def apply(self, *args, **kwargs):
            return self

        def __getitem__(self, key):
            return self

        def head(self, *args, **kwargs):
            return self

    _pandas_stub.DataFrame = _StubDataFrame
    _pandas_stub.Series = object
    _pandas_stub.read_csv = lambda *args, **kwargs: _StubDataFrame()
    _pandas_stub.read_parquet = lambda *args, **kwargs: _StubDataFrame()
    sys.modules["pandas"] = _pandas_stub

from run_dual_arbitration_annotation import (  # noqa: E402
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
    extract_allowed_labels,
    load_env_for_country,
)

DEFAULT_TERTIARY_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
DEFAULT_MAX_WORKERS = 5
DEFAULT_KEY_POOL_WORKERS = 8
WORKER_KEY_COUNTRIES = ("BR", "MX", "SA")

TERTIARY_SPEC = ModelSpec(
    role="tertiary",
    model=DEFAULT_TERTIARY_MODEL,
    account="ARBITER",
    timeout=120,
)


def chunks(items: list[tuple[int, dict]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_env_var(env_path: Path, var_name: str) -> str:
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


def build_full_key_pool() -> list[tuple[str, str]]:
    """nv_api_keys (5) + BR/MX/SA .env NIM_API_KEY (3) = 8 interchangeable keys."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from nv_api_keys import get_all_keys

    pool: list[tuple[str, str]] = [
        (key, f"nv_api_keys[{index}]") for index, key in enumerate(get_all_keys())
    ]
    for country in WORKER_KEY_COUNTRIES:
        env_path = PROJECT_ROOT / country / ".env"
        api_key = read_env_var(env_path, "NIM_API_KEY")
        pool.append((api_key, f"{country}/.env:NIM_API_KEY"))
    return pool


def load_worker_api_keys(
    max_workers: int,
    api_key_index: int = 0,
    nv_only: bool = False,
) -> list[tuple[str, str]]:
    if nv_only:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from nv_api_keys import get_all_keys

        pool = [(key, f"nv_api_keys[{index}]") for index, key in enumerate(get_all_keys())]
    else:
        pool = build_full_key_pool()

    keys: list[tuple[str, str]] = []
    for worker_index in range(max_workers):
        idx = (api_key_index + worker_index) % len(pool)
        keys.append((pool[idx][0], pool[idx][1]))
    return keys


def read_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    with output_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        done: set[str] = set()
        for row in reader:
            for col in ("_ann_row_id", "content_id"):
                value = row.get(col)
                if value is not None and str(value).strip():
                    done.add(str(value).strip())
                    break
        return done


def append_results(output_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    fieldnames = list(rows[0].keys())
    with output_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def default_output_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_triple_voted{input_path.suffix}")


def resolve_output_path(
    input_path: Path,
    output_arg: Path | None,
    resume: bool,
) -> Path:
    """Pick a dedicated output file; never reuse the input path."""
    input_resolved = input_path.resolve()
    user_specified = output_arg is not None
    candidate = output_arg if user_specified else default_output_for(input_path)

    if candidate.resolve() == input_resolved:
        raise ValueError(
            f"Output path must not be the input file: {input_path}\n"
            "Choose a different --output (e.g. *_triple_voted.csv)."
        )

    if candidate.exists():
        if resume:
            return candidate
        if user_specified:
            raise FileExistsError(
                f"Output already exists: {candidate}\n"
                "Use --resume to append, or pass a different --output path."
            )
        stem = input_path.stem
        suffix = input_path.suffix
        version = 2
        while True:
            versioned = candidate.parent / f"{stem}_triple_voted_v{version}{suffix}"
            if versioned.resolve() == input_resolved:
                version += 1
                continue
            if not versioned.exists():
                print(
                    f"NOTE: default output {candidate.name} already exists; "
                    f"using new file {versioned.name} (pass --resume to continue existing output)."
                )
                return versioned
            version += 1

    return candidate


def choose_triple_parallel_vote(labels: list[str]) -> tuple[str, str]:
    normalized = [str(label).strip() for label in labels if str(label).strip()]
    if len(normalized) != 3:
        return "reject", "triple_vote_incomplete"

    counts = Counter(normalized)
    if len(counts) == 1:
        label = normalized[0]
        if label == "need_review":
            return "need_review", "triple_vote_unanimous_need_review"
        return label, "triple_vote_unanimous"

    if len(counts) == 3:
        return "reject", "triple_vote_all_disagree"

    return "need_review", "triple_vote_partial_agreement"


def call_model(
    country: str,
    spec: ModelSpec,
    system_prompt: str,
    user_prompt: str,
    allowed_labels: set[str],
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    api_key: str | None = None,
    api_key_label: str | None = None,
) -> dict:
    return call_nim_label(
        country,
        spec,
        system_prompt,
        user_prompt,
        allowed_labels,
        temperature,
        max_tokens,
        retries,
        backoff_seconds,
        api_key=api_key,
        api_key_label=api_key_label,
    )


def annotate_one_row(
    country_config: CountryConfig,
    tertiary_spec: ModelSpec,
    allowed_labels: set[str],
    system_prompt: str,
    row_index: int,
    row: dict,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    api_key: str | None = None,
    api_key_label: str | None = None,
) -> dict:
    text = row_text_from_dict(row)
    content_id = stable_content_id_from_dict(country_config.country, row_index, row)
    user_prompt = build_user_prompt(text)
    initial_system_prompt = build_initial_system_prompt(system_prompt)

    specs = [country_config.primary, country_config.secondary, tertiary_spec]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                call_model,
                country_config.country,
                spec,
                initial_system_prompt,
                user_prompt,
                allowed_labels,
                temperature,
                max_tokens,
                retries,
                backoff_seconds,
                api_key,
                api_key_label,
            )
            for spec in specs
        ]
        results = [future.result() for future in futures]

    primary, secondary, tertiary = results
    vote_labels = [primary["label"], secondary["label"], tertiary["label"]]
    final_label, final_source = choose_triple_parallel_vote(vote_labels)

    out = dict(row)
    out.update(
        {
            "primary_label": primary["label"],
            "primary_reason": primary.get("reason", ""),
            "secondary_label": secondary["label"],
            "secondary_reason": secondary.get("reason", ""),
            "tertiary_label": tertiary["label"],
            "tertiary_reason": tertiary.get("reason", ""),
            "final_label": final_label,
            "final_source": final_source,
            "vote_labels": "|".join(vote_labels),
            "vote_agreement": _vote_agreement_summary(vote_labels),
            "primary_model": country_config.primary.model,
            "secondary_model": country_config.secondary.model,
            "tertiary_model": tertiary_spec.model,
            "primary_raw_output": primary.get("raw_output", ""),
            "secondary_raw_output": secondary.get("raw_output", ""),
            "tertiary_raw_output": tertiary.get("raw_output", ""),
            "primary_error": primary.get("error"),
            "secondary_error": secondary.get("error"),
            "tertiary_error": tertiary.get("error"),
            "total_tokens_used": sum(int(r.get("tokens_used") or 0) for r in results),
            "annotated_at": datetime.now(timezone.utc).isoformat(),
            "source_row_index": row_index,
            "_ann_row_id": content_id,
            "worker_api_key": api_key_label or "",
        }
    )
    return out


def _vote_agreement_summary(labels: list[str]) -> str:
    counts = Counter(labels)
    if len(counts) == 1:
        return "unanimous"
    if len(counts) == 3:
        return "all_disagree"
    return "partial_agreement"


def row_text_from_dict(row: dict) -> str:
    for column in ("clean_text_phaseA", "clean_text", "text", "body", "content", "title"):
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def stable_content_id_from_dict(country: str, row_index: int, row: dict) -> str:
    for column in ("_ann_row_id", "content_id", "id", "index"):
        value = row.get(column)
        if value is not None and str(value).strip():
            text = str(value).strip()
            if column == "content_id":
                return text
            return f"{country}_{column}_{text}"
    return f"{country}_row_{row_index:08d}"


def read_input_rows(
    path: Path,
    limit: int | None,
    selection: str,
) -> list[tuple[int, dict]]:
    sys.path.insert(0, str(SCRIPT_DIR))
    from export_tr_full_quality_filtered import needs_triple_annotation

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    pending: list[tuple[int, dict]] = []
    for idx, row in enumerate(rows):
        if not row_text_from_dict(row):
            continue
        if selection == "requires_review_only":
            if str(row.get("requires_review") or "").strip().lower() != "true":
                continue
        elif selection == "non_unanimous":
            if not needs_triple_annotation(row):
                continue
        elif selection == "all":
            pass
        else:
            raise ValueError(f"Unknown selection: {selection}")
        pending.append((idx, row))
        if limit and limit > 0 and len(pending) >= limit:
            break
    return pending


def run_country(
    country: str,
    input_path: Path,
    output_path: Path,
    limit: int | None,
    batch_size: int,
    max_workers: int,
    resume: bool,
    dry_run: bool,
    requires_review_only: bool,
    selection: str,
    tertiary_model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    use_key_pool: bool,
    api_key_index: int,
    nv_keys_only: bool,
) -> None:
    country_config = COUNTRY_CONFIGS[country.upper()]
    tertiary_spec = ModelSpec(
        role="tertiary",
        model=tertiary_model,
        account="ARBITER",
        timeout=120,
    )
    load_env_for_country(country_config.country)
    system_prompt = country_config.prompt_path.read_text(encoding="utf-8").strip()
    allowed_labels = extract_allowed_labels(system_prompt)

    pending_all = read_input_rows(input_path, limit, selection)
    done_ids = read_done_ids(output_path) if resume else set()
    pending = [
        (idx, row)
        for idx, row in pending_all
        if stable_content_id_from_dict(country_config.country, idx, row) not in done_ids
    ]
    total_batches = (len(pending) + batch_size - 1) // batch_size if pending else 0

    print(f"\n[{country}] input:  {input_path}")
    print(f"[{country}] output: {output_path}")
    print(f"[{country}] models:")
    print(f"  primary   = {country_config.primary.model}")
    print(f"  secondary = {country_config.secondary.model}")
    print(f"  tertiary  = {tertiary_spec.model}")
    print(
        f"[{country}] rows eligible: {len(pending_all):,}  pending: {len(pending):,}  "
        f"batches: {total_batches}  selection={selection}"
    )
    print(
        f"[{country}] vote policy: unanimous->final_label | 2-vs-1->need_review | 3-way split->reject"
    )
    if output_path.exists() and resume:
        print(f"[{country}] write mode: append to existing output ({len(done_ids):,} rows done)")
    else:
        print(f"[{country}] write mode: new output file (input is read-only)")

    worker_api_keys: list[tuple[str, str]] | None = None
    if use_key_pool:
        worker_api_keys = load_worker_api_keys(max_workers, api_key_index, nv_keys_only)
        pool_size = len(build_full_key_pool()) if not nv_keys_only else len(worker_api_keys)
        print(
            f"[{country}] key pool: {pool_size} keys, max_workers={max_workers}, "
            f"api_key_index={api_key_index}, nv_only={nv_keys_only}"
        )
        print(
            f"[{country}] api keys: "
            + ", ".join(f"worker{i}={label}" for i, (_, label) in enumerate(worker_api_keys))
        )
    else:
        print(f"[{country}] api keys: NIM_API_KEY_{country}_PRIMARY/SECONDARY/ARBITER from .env")

    if dry_run:
        for idx, row in pending[:5]:
            cid = stable_content_id_from_dict(country_config.country, idx, row)
            print(f"  - {cid}: {row_text_from_dict(row)[:120]}")
        return

    if not pending:
        print(f"[{country}] nothing to process.")
        return

    processed = 0
    stats = Counter()
    for batch_number, batch in enumerate(chunks(pending, batch_size), start=1):
        print(f"\n[{country}] ── batch {batch_number}/{total_batches} ({len(batch)} rows) ──")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task_idx, (idx, row) in enumerate(batch):
                row_api_key: str | None = None
                row_api_key_label: str | None = None
                if worker_api_keys:
                    row_api_key, row_api_key_label = worker_api_keys[task_idx % max_workers]
                futures.append(
                    executor.submit(
                        annotate_one_row,
                        country_config,
                        tertiary_spec,
                        allowed_labels,
                        system_prompt,
                        idx,
                        row,
                        temperature,
                        max_tokens,
                        retries,
                        backoff_seconds,
                        row_api_key,
                        row_api_key_label,
                    )
                )
            for future in as_completed(futures):
                result = future.result()
                append_results(output_path, [result])
                processed += 1
                stats[result.get("final_label", "")] += 1
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(
                    f"[{country}] {ts}  {processed}/{len(pending)}  "
                    f"id={result.get('_ann_row_id')}  votes={result.get('vote_labels')}  "
                    f"final={result.get('final_label')}  source={result.get('final_source')}",
                    flush=True,
                )

    print(f"\n[{country}] finished {processed} rows")
    for label, count in stats.most_common():
        print(f"  {label}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triple parallel vote annotation (dual + qwen tertiary merged)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--country", required=True, choices=tuple(COUNTRY_CONFIGS))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Dedicated output CSV (never overwrites --input). Default: {input_stem}_triple_voted.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--selection",
        choices=("non_unanimous", "requires_review_only", "all"),
        default="non_unanimous",
        help="non_unanimous=未跑过三模型或旧 jury 非三票一致 (default); requires_review_only=仅 requires_review=True",
    )
    parser.add_argument(
        "--requires-review-only",
        action="store_true",
        help="Deprecated alias for --selection requires_review_only",
    )
    parser.add_argument("--tertiary-model", default=DEFAULT_TERTIARY_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument(
        "--use-key-pool",
        action="store_true",
        help="Use 8-key pool (nv_api_keys + BR/MX/SA .env); one key per row worker, shared by all 3 models",
    )
    parser.add_argument(
        "--api-key-index",
        type=int,
        default=0,
        help="Start index in key pool when --use-key-pool (worker i uses pool[index+i])",
    )
    parser.add_argument(
        "--nv-keys-only",
        action="store_true",
        help="With --use-key-pool, use only nv_api_keys.py (5 keys) instead of full 8-key pool",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else SCRIPT_DIR / args.input
    output_arg = args.output
    if output_arg is not None and not output_arg.is_absolute():
        output_arg = SCRIPT_DIR / output_arg

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 2

    try:
        output_path = resolve_output_path(input_path, output_arg, args.resume)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    selection = "requires_review_only" if args.requires_review_only else args.selection
    max_workers = max(1, args.max_workers)
    if args.use_key_pool and args.max_workers == DEFAULT_MAX_WORKERS:
        max_workers = DEFAULT_KEY_POOL_WORKERS

    run_country(
        country=args.country.upper(),
        input_path=input_path,
        output_path=output_path,
        limit=args.limit,
        batch_size=max(1, args.batch_size),
        max_workers=max_workers,
        resume=args.resume,
        dry_run=args.dry_run,
        requires_review_only=args.requires_review_only,
        selection=selection,
        tertiary_model=args.tertiary_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=max(0, args.retries),
        backoff_seconds=max(0.0, args.backoff_seconds),
        use_key_pool=args.use_key_pool,
        api_key_index=args.api_key_index,
        nv_keys_only=args.nv_keys_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
