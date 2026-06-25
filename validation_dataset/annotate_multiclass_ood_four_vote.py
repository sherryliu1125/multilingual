#!/usr/bin/env python3
"""
Annotate multiclass OOD CSVs with primary + secondary + tertiary models, then
vote together with the original ``label`` column (4-way vote).

Vote policy (4 voters: original label + 3 models):
  - any label with >= 3 votes  -> final_label = that label
  - exactly 2 vs 2 tie         -> final_label = reject
  - otherwise (e.g. 2-1-1)     -> final_label = reject

Concurrency (stable defaults):
  - ``--country all``: BR -> MX -> SA -> TR sequentially
  - Default 2 row workers; each worker owns 3 dedicated keys (one per model)
  - Within each row: primary/secondary/tertiary run in parallel on 3 different keys
  - Max 1 in-flight call per key -> 2 workers x 3 models = 6 keys, 6 concurrent max

Checkpoint / resume (built-in):
  - Skip rows with primary_label and no model errors
  - Each finished row flushed to CSV immediately
  - Use --force to re-annotate all rows

Examples:
  python validation_dataset/annotate_multiclass_ood_four_vote.py --country all
  python validation_dataset/annotate_multiclass_ood_four_vote.py --country BR --max-workers 1
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OOD_DIR = SCRIPT_DIR / "ood"
PROMPTS_DIR = PROJECT_ROOT / "annotaion_prompts"
THIRD_DIR = PROJECT_ROOT / "3rd_annotaion"

if str(PROMPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPTS_DIR))
if str(THIRD_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_DIR))

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
from run_triple_parallel_vote import (  # noqa: E402
    DEFAULT_TERTIARY_MODEL,
    row_text_from_dict,
    stable_content_id_from_dict,
)

OOD_FILES: dict[str, Path] = {
    "BR": OOD_DIR / "multiclass_ood_BR.csv",
    "MX": OOD_DIR / "multiclass_ood_MX.csv",
    "SA": OOD_DIR / "multiclass_ood_SA.csv",
    "TR": OOD_DIR / "multiclass_ood_TR.csv",
}

COUNTRY_ORDER = ("BR", "MX", "SA", "TR")
WORKER_KEY_COUNTRIES = ("BR", "MX", "SA")
MODEL_ROLES = ("primary", "secondary", "tertiary")

ANNOTATION_COLUMNS = [
    "primary_label",
    "primary_reason",
    "secondary_label",
    "secondary_reason",
    "tertiary_label",
    "tertiary_reason",
    "final_label",
    "final_source",
    "vote_labels",
    "vote_distribution",
    "primary_model",
    "secondary_model",
    "tertiary_model",
    "primary_key",
    "secondary_key",
    "tertiary_key",
    "primary_error",
    "secondary_error",
    "tertiary_error",
    "total_tokens_used",
    "annotated_at",
    "_ann_row_id",
]

DEFAULT_MAX_WORKERS = 2
MODELS_PER_ROW = 3
DEFAULT_REQUEST_JITTER_SECONDS = 0.5
DEFAULT_CONNECTION_RESET_RETRIES = 4
DEFAULT_TERTIARY_MODEL_LOCAL = DEFAULT_TERTIARY_MODEL


class WorkerKeyPool:
    """Each row worker owns MODELS_PER_ROW keys; one key per parallel model call."""

    def __init__(self, worker_triplets: list[list[tuple[str, str]]]) -> None:
        if not worker_triplets:
            raise RuntimeError("Worker key pool is empty.")
        self._worker_triplets = worker_triplets
        self._key_sems: dict[str, threading.Semaphore] = {}
        for triplet in worker_triplets:
            for _, label in triplet:
                if label not in self._key_sems:
                    self._key_sems[label] = threading.Semaphore(1)

    @property
    def worker_count(self) -> int:
        return len(self._worker_triplets)

    def key_for_model(self, worker_index: int, model_index: int) -> tuple[str, str]:
        return self._worker_triplets[worker_index % len(self._worker_triplets)][model_index]

    def worker_key_labels(self, worker_index: int) -> list[str]:
        return [label for _, label in self._worker_triplets[worker_index]]

    def guarded_call(self, key_label: str, fn):
        with self._key_sems[key_label]:
            return fn()


class CsvCheckpoint:
    def __init__(self, path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.rows = rows
        self._lock = threading.Lock()

    def flush(self) -> None:
        with self._lock:
            write_csv_rows(self.path, self.fieldnames, self.rows)


def read_env_var(env_path: Path, var_name: str) -> str:
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")
    for line in env_path.read_text(encoding="utf-8").strip().splitlines():
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


def build_full_key_pool(nv_only: bool = False) -> list[tuple[str, str]]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from nv_api_keys import get_all_keys

    pool: list[tuple[str, str]] = [
        (key, f"nv_api_keys[{index}]") for index, key in enumerate(get_all_keys())
    ]
    if nv_only:
        return pool
    for country in WORKER_KEY_COUNTRIES:
        env_path = PROJECT_ROOT / country / ".env"
        api_key = read_env_var(env_path, "NIM_API_KEY")
        pool.append((api_key, f"{country}/.env:NIM_API_KEY"))
    return pool


def build_worker_key_pool(
    max_workers: int,
    nv_keys_only: bool,
    api_key_index: int,
) -> WorkerKeyPool:
    pool = build_full_key_pool(nv_only=nv_keys_only)
    if api_key_index:
        pool = pool[api_key_index:] + pool[:api_key_index]
    keys_needed = max_workers * MODELS_PER_ROW
    if len(pool) < keys_needed:
        raise RuntimeError(
            f"Need {keys_needed} API keys for {max_workers} workers x {MODELS_PER_ROW} models, "
            f"got {len(pool)}. Lower --max-workers or use full 8-key pool."
        )
    triplets = [pool[i * MODELS_PER_ROW : (i + 1) * MODELS_PER_ROW] for i in range(max_workers)]
    return WorkerKeyPool(triplets)


def is_connection_reset_error(exc: BaseException) -> bool:
    text = str(exc)
    return "Connection reset by peer" in text or "Errno 54" in text or "ECONNRESET" in text


def norm_label(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return ""
    return text


def choose_four_vote(labels: list[str]) -> tuple[str, str]:
    normalized = [norm_label(label) for label in labels if norm_label(label)]
    if len(normalized) != 4:
        return "reject", "four_vote_incomplete"
    counts = Counter(normalized)
    top_label, top_count = counts.most_common(1)[0]
    if top_count >= 3:
        return top_label, f"four_vote_majority_{top_count}"
    twos = [label for label, count in counts.items() if count == 2]
    if len(twos) == 2 and len(counts) == 2:
        return "reject", "four_vote_2v2"
    return "reject", "four_vote_no_majority"


def vote_distribution(labels: list[str]) -> str:
    counts = Counter(norm_label(label) for label in labels if norm_label(label))
    return "|".join(f"{label}:{count}" for label, count in counts.most_common())


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8-sig",
        newline="",
        dir=path.parent,
        delete=False,
        suffix=path.suffix,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def row_has_model_error(row: dict) -> bool:
    for column in ("primary_error", "secondary_error", "tertiary_error"):
        error = row.get(column)
        if error is not None and str(error).strip():
            return True
    return False


def row_is_done(row: dict, force: bool) -> bool:
    if force:
        return False
    if not norm_label(row.get("primary_label")):
        return False
    return not row_has_model_error(row)


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
    worker_keys: WorkerKeyPool,
    worker_index: int,
    model_index: int,
    request_jitter_seconds: float,
    connection_reset_retries: int,
) -> dict:
    api_key, api_key_label = worker_keys.key_for_model(worker_index, model_index)
    last_error: BaseException | None = None
    for outer_attempt in range(connection_reset_retries + 1):
        if request_jitter_seconds > 0:
            time.sleep(random.uniform(0, request_jitter_seconds))
        try:
            return worker_keys.guarded_call(
                api_key_label,
                lambda: call_nim_label(
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
                ),
            )
        except RuntimeError as exc:
            last_error = exc
            if is_connection_reset_error(exc) and outer_attempt < connection_reset_retries:
                wait = min(60.0, 10.0 * (outer_attempt + 1) + random.uniform(0, 5))
                print(
                    f"[{country}] {spec.role} connection reset on {api_key_label} "
                    f"(outer {outer_attempt + 1}/{connection_reset_retries + 1}); "
                    f"cooldown {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"[{country}] {spec.role} call failed without error detail")


def annotate_one_row(
    country_config: CountryConfig,
    tertiary_spec: ModelSpec,
    allowed_labels: set[str],
    system_prompt: str,
    row_index: int,
    row: dict,
    worker_keys: WorkerKeyPool,
    worker_index: int,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    request_jitter_seconds: float,
    connection_reset_retries: int,
    sequential_models: bool,
) -> dict:
    text = row_text_from_dict(row)
    content_id = stable_content_id_from_dict(country_config.country, row_index, row)
    user_prompt = build_user_prompt(text)
    initial_system_prompt = build_initial_system_prompt(system_prompt)
    original_label = norm_label(row.get("label"))

    specs = [country_config.primary, country_config.secondary, tertiary_spec]
    key_labels = worker_keys.worker_key_labels(worker_index)

    def run_one(model_index: int, spec: ModelSpec) -> dict:
        return call_model(
            country_config.country,
            spec,
            initial_system_prompt,
            user_prompt,
            allowed_labels,
            temperature,
            max_tokens,
            retries,
            backoff_seconds,
            worker_keys,
            worker_index,
            model_index,
            request_jitter_seconds,
            connection_reset_retries,
        )

    if sequential_models:
        results = [run_one(i, spec) for i, spec in enumerate(specs)]
    else:
        with ThreadPoolExecutor(max_workers=MODELS_PER_ROW) as executor:
            futures = [executor.submit(run_one, i, spec) for i, spec in enumerate(specs)]
            results = [future.result() for future in futures]

    primary, secondary, tertiary = results
    vote_labels = [original_label, primary["label"], secondary["label"], tertiary["label"]]
    final_label, final_source = choose_four_vote(vote_labels)

    row.update(
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
            "vote_distribution": vote_distribution(vote_labels),
            "primary_model": country_config.primary.model,
            "secondary_model": country_config.secondary.model,
            "tertiary_model": tertiary_spec.model,
            "primary_key": key_labels[0],
            "secondary_key": key_labels[1],
            "tertiary_key": key_labels[2],
            "primary_error": primary.get("error"),
            "secondary_error": secondary.get("error"),
            "tertiary_error": tertiary.get("error"),
            "total_tokens_used": sum(
                int(r.get("tokens_used") or 0) for r in (primary, secondary, tertiary)
            ),
            "annotated_at": datetime.now(timezone.utc).isoformat(),
            "_ann_row_id": content_id,
        }
    )
    return row


def run_country(
    country: str,
    input_path: Path,
    limit: int | None,
    max_workers: int,
    force: bool,
    dry_run: bool,
    tertiary_model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    api_key_index: int,
    nv_keys_only: bool,
    request_jitter_seconds: float,
    connection_reset_retries: int,
    sequential_models: bool,
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

    fieldnames, rows = read_csv_rows(input_path)
    for column in ANNOTATION_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    pending: list[tuple[int, dict]] = []
    done_count = 0
    for idx, row in enumerate(rows):
        if not row_text_from_dict(row):
            continue
        if row_is_done(row, force=force):
            done_count += 1
            continue
        pending.append((idx, row))
        if limit and limit > 0 and len(pending) >= limit:
            break

    worker_keys = build_worker_key_pool(
        max_workers=max_workers,
        nv_keys_only=nv_keys_only,
        api_key_index=api_key_index,
    )
    worker_count = max(1, min(max_workers, len(pending) if pending else max_workers))
    model_mode = "sequential" if sequential_models else "3 parallel (1 key per model)"

    print(f"\n[{country}] file:   {input_path}")
    print(f"[{country}] models:")
    print(f"  primary   = {country_config.primary.model}")
    print(f"  secondary = {country_config.secondary.model}")
    print(f"  tertiary  = {tertiary_spec.model}")
    print(
        f"[{country}] rows total: {len(rows):,}  already done: {done_count:,}  "
        f"pending: {len(pending):,}"
    )
    print(
        f"[{country}] concurrency: row_workers={worker_count}  "
        f"models_per_row={model_mode}  max_inflight_per_key=1  "
        f"request_jitter<={request_jitter_seconds:.2f}s"
    )
    print(f"[{country}] worker key triplets (primary|secondary|tertiary):")
    for worker_index in range(worker_count):
        labels = worker_keys.worker_key_labels(worker_index)
        print(f"  worker{worker_index} -> {' | '.join(labels)}")
    print(f"[{country}] checkpoint: flush CSV after every row (resume built-in)")

    if dry_run:
        for task_idx, (idx, row) in enumerate(pending[:5]):
            worker_index = task_idx % worker_count
            cid = stable_content_id_from_dict(country_config.country, idx, row)
            keys = " | ".join(worker_keys.worker_key_labels(worker_index))
            print(
                f"  - {cid}: worker{worker_index} keys={keys}  "
                f"label={row.get('label')}  text={row_text_from_dict(row)[:80]}"
            )
        return

    if not pending:
        print(f"[{country}] nothing to process.")
        return

    checkpoint = CsvCheckpoint(input_path, fieldnames, rows)
    processed = 0
    stats = Counter()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                annotate_one_row,
                country_config,
                tertiary_spec,
                allowed_labels,
                system_prompt,
                idx,
                row,
                worker_keys,
                task_idx % worker_count,
                temperature,
                max_tokens,
                retries,
                backoff_seconds,
                request_jitter_seconds,
                connection_reset_retries,
                sequential_models,
            ): idx
            for task_idx, (idx, row) in enumerate(pending)
        }
        for future in as_completed(futures):
            updated_row = future.result()
            processed += 1
            stats[updated_row.get("final_label", "")] += 1
            checkpoint.flush()
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(
                f"[{country}] {ts}  {processed}/{len(pending)}  "
                f"id={updated_row.get('_ann_row_id')}  "
                f"keys={updated_row.get('primary_key')}|"
                f"{updated_row.get('secondary_key')}|{updated_row.get('tertiary_key')}  "
                f"votes={updated_row.get('vote_labels')}  "
                f"final={updated_row.get('final_label')}  source={updated_row.get('final_source')}",
                flush=True,
            )

    print(f"\n[{country}] finished {processed} rows")
    for label, count in stats.most_common():
        print(f"  {label}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate multiclass OOD CSVs with 3 models + original-label 4-way vote",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--country", default="all", choices=("all", *tuple(OOD_FILES)))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Row workers; each uses 3 dedicated keys (default: 2, needs 6 keys)",
    )
    parser.add_argument(
        "--sequential-models",
        action="store_true",
        help="Serialize primary/secondary/tertiary within each row (most stable)",
    )
    parser.add_argument("--request-jitter-seconds", type=float, default=DEFAULT_REQUEST_JITTER_SECONDS)
    parser.add_argument("--connection-reset-retries", type=int, default=DEFAULT_CONNECTION_RESET_RETRIES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tertiary-model", default=DEFAULT_TERTIARY_MODEL_LOCAL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument("--api-key-index", type=int, default=0)
    parser.add_argument("--nv-keys-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    countries = list(COUNTRY_ORDER) if args.country == "all" else [args.country.upper()]
    limit = args.limit if args.limit and args.limit > 0 else None
    max_workers = max(1, args.max_workers)

    for country in countries:
        if not OOD_FILES[country].exists():
            raise FileNotFoundError(f"Missing OOD file for {country}: {OOD_FILES[country]}")

    if len(countries) > 1:
        print(f"Running {len(countries)} countries sequentially: {' -> '.join(countries)}")
        print(
            f"Workers per country: {max_workers}  "
            f"(each worker: 3 keys, 1 in-flight per key, row内 models parallel by default)"
        )

    run_kwargs = dict(
        limit=limit,
        max_workers=max_workers,
        force=args.force,
        dry_run=args.dry_run,
        tertiary_model=args.tertiary_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
        api_key_index=args.api_key_index,
        nv_keys_only=args.nv_keys_only,
        request_jitter_seconds=max(0.0, args.request_jitter_seconds),
        connection_reset_retries=max(0, args.connection_reset_retries),
        sequential_models=args.sequential_models,
    )

    for country in countries:
        run_country(country=country, input_path=OOD_FILES[country], **run_kwargs)


if __name__ == "__main__":
    main()
