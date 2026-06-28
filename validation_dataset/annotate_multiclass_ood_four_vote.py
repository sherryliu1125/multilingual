#!/usr/bin/env python3
"""
Annotate multiclass OOD CSVs with primary + secondary + tertiary models, then
vote together with the original ``label`` column (4-way vote).

Vote policy (4 voters: original label + 3 models):
  - any label with >= 3 votes  -> final_label = that label
  - exactly 2 vs 2 tie         -> final_label = reject
  - otherwise (e.g. 2-1-1)     -> final_label = reject

Triple vote (UAE cross-job): primary + secondary + tertiary only.

Concurrency:
  - Full 8-key pool: nv_api_keys[0-4] + BR/MX/SA .env NIM_API_KEY
  - Global round-robin: each model call takes the next key from all 8
  - Max 1 in-flight request per key (semaphore per key)
  - BR/MX/SA/TR: default 2 row workers, 3 models parallel per row
  - UAE (25 rows): serial rows, serial models, 8-key round-robin

Examples:
  python validation_dataset/annotate_multiclass_ood_four_vote.py --country all
  python validation_dataset/annotate_multiclass_ood_four_vote.py --country UAE
  python validation_dataset/annotate_multiclass_ood_four_vote.py --country UAE --dry-run
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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

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
    NIM_BASE_URL,
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

OOD_CROSSJOBS: dict[str, dict[str, Path | str]] = {
    "UAE": {
        "input": OOD_DIR / "multiclass_ood_SA.csv",
        "output": OOD_DIR / "multiclass_ood_UAE.csv",
        "prompt_country": "UAE",
        "output_country": "UAE",
        "source_country": "SA",
        "source_label_prefix": "SA_",
        "vote_mode": "triple",
        "default_max_workers": 1,
        "force_sequential_models": True,
    },
}

COUNTRY_ORDER = ("BR", "MX", "SA", "TR")
ALL_COUNTRY_CHOICES = tuple(OOD_FILES) + tuple(OOD_CROSSJOBS)
WORKER_KEY_COUNTRIES = ("BR", "MX", "SA")
MODELS_PER_ROW = 3

ANNOTATION_COLUMNS = [
    "source_country",
    "source_sa_row_id",
    "source_sa_final_label",
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
SMALL_JOB_ROW_THRESHOLD = 30
DEFAULT_REQUEST_JITTER_SECONDS = 0.5
DEFAULT_CONNECTION_RESET_RETRIES = 4
DEFAULT_MODEL_TIMEOUT = 180
DEFAULT_TERTIARY_MODEL_LOCAL = DEFAULT_TERTIARY_MODEL

T = TypeVar("T")


class ApiKeyPool:
    """Thread-safe round-robin over all NIM keys; max 1 in-flight call per key."""

    def __init__(self, keys: list[tuple[str, str]]) -> None:
        if not keys:
            raise RuntimeError("API key pool is empty.")
        self._keys = keys
        self._lock = threading.Lock()
        self._index = 0
        self._sems = {label: threading.Semaphore(1) for _, label in keys}

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def labels(self) -> list[str]:
        return [label for _, label in self._keys]

    def next(self) -> tuple[str, str]:
        with self._lock:
            key, label = self._keys[self._index % len(self._keys)]
            self._index += 1
            return key, label

    def preview_next(self, count: int) -> list[str]:
        with self._lock:
            labels: list[str] = []
            for offset in range(count):
                labels.append(self._keys[(self._index + offset) % len(self._keys)][1])
            return labels

    def guarded(self, key_label: str, fn: Callable[[], T]) -> T:
        with self._sems[key_label]:
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


def assert_nim_dns_resolves() -> None:
    import socket

    host = NIM_BASE_URL.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    try:
        socket.getaddrinfo(host, 443)
    except socket.gaierror as exc:
        raise RuntimeError(
            f"Cannot resolve NVIDIA NIM host {host!r} from this environment. "
            "This is a DNS/network problem before any API key is checked. "
            "Check VPN/proxy/DNS/network access, then rerun the command."
        ) from exc


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


def build_key_pool(nv_keys_only: bool, api_key_index: int) -> ApiKeyPool:
    pool = build_full_key_pool(nv_only=nv_keys_only)
    if api_key_index:
        pool = pool[api_key_index:] + pool[:api_key_index]
    return ApiKeyPool(pool)


def with_timeout(spec: ModelSpec, timeout: int) -> ModelSpec:
    return ModelSpec(spec.role, spec.model, spec.account, timeout)


def is_connection_reset_error(exc: BaseException) -> bool:
    text = str(exc)
    return "Connection reset by peer" in text or "Errno 54" in text or "ECONNRESET" in text


def is_timeout_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


def norm_label(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return ""
    return text


def choose_triple_vote(labels: list[str]) -> tuple[str, str]:
    normalized = [norm_label(label) for label in labels if norm_label(label)]
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
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def resolve_job(country: str) -> dict[str, Path | str | int]:
    country = country.upper()
    if country in OOD_CROSSJOBS:
        job = OOD_CROSSJOBS[country]
        return {
            "country": country,
            "input_path": Path(job["input"]),
            "output_path": Path(job["output"]),
            "prompt_country": str(job.get("prompt_country", country)),
            "output_country": str(job.get("output_country", country)),
            "source_country": str(job.get("source_country", "")),
            "source_label_prefix": str(job.get("source_label_prefix", "")),
            "vote_mode": str(job.get("vote_mode", "four")),
            "default_max_workers": int(job.get("default_max_workers", DEFAULT_MAX_WORKERS)),
            "force_sequential_models": bool(job.get("force_sequential_models", False)),
        }
    return {
        "country": country,
        "input_path": OOD_FILES[country],
        "output_path": OOD_FILES[country],
        "prompt_country": country,
        "output_country": country,
        "source_country": country,
        "source_label_prefix": "",
        "vote_mode": "four",
        "default_max_workers": DEFAULT_MAX_WORKERS,
        "force_sequential_models": False,
    }


def build_job_rows(
    input_path: Path,
    output_path: Path,
    output_country: str,
    source_country: str,
    source_label_prefix: str = "",
) -> tuple[list[str], list[dict]]:
    _, input_rows = read_csv_rows(input_path)
    _, output_rows = read_csv_rows(output_path) if output_path.exists() else ([], [])
    saved_by_id = {
        row["source_sa_row_id"]: row
        for row in output_rows
        if row.get("source_sa_row_id")
    }
    saved_rows = [row for row in output_rows if row_text_from_dict(row)]
    saved_fallback = iter(saved_rows)

    base_fieldnames = [
        "text",
        "label",
        "country",
        "source_country",
        "source_sa_row_id",
        "source_sa_final_label",
    ]
    fieldnames = list(base_fieldnames)
    for column in ANNOTATION_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    rows: list[dict] = []
    for src_idx, src in enumerate(input_rows):
        if not row_text_from_dict(src):
            continue
        gold_label = norm_label(src.get("label"))
        if source_label_prefix and not gold_label.startswith(source_label_prefix):
            continue

        source_sa_row_id = f"SA_row_{src_idx:08d}"
        row = {
            "text": src.get("text", ""),
            "label": gold_label,
            "country": output_country,
            "source_country": source_country or src.get("country", ""),
            "source_sa_row_id": source_sa_row_id,
            "source_sa_final_label": norm_label(src.get("final_label")),
        }
        existing = saved_by_id.get(source_sa_row_id)
        if existing is None and not saved_by_id:
            try:
                existing = next(saved_fallback)
            except StopIteration:
                existing = None
        if existing:
            for column in fieldnames:
                if column in existing and existing[column] not in (None, ""):
                    row[column] = existing[column]
        rows.append(row)
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
    key_pool: ApiKeyPool,
    request_jitter_seconds: float,
    connection_reset_retries: int,
) -> dict:
    api_key, api_key_label = key_pool.next()
    last_error: BaseException | None = None
    for outer_attempt in range(connection_reset_retries + 1):
        if request_jitter_seconds > 0:
            time.sleep(random.uniform(0, request_jitter_seconds))
        try:
            result = key_pool.guarded(
                api_key_label,
                lambda k=api_key, kl=api_key_label, s=spec: call_nim_label(
                    country,
                    s,
                    system_prompt,
                    user_prompt,
                    allowed_labels,
                    temperature,
                    max_tokens,
                    retries,
                    backoff_seconds,
                    api_key=k,
                    api_key_label=kl,
                ),
            )
            result["key_env"] = api_key_label
            return result
        except RuntimeError as exc:
            last_error = exc
            retryable = is_connection_reset_error(exc) or is_timeout_error(exc)
            if retryable and outer_attempt < connection_reset_retries:
                wait = min(60.0, 10.0 * (outer_attempt + 1) + random.uniform(0, 5))
                reason = "connection reset" if is_connection_reset_error(exc) else "timeout"
                print(
                    f"[{country}] {spec.role} ({spec.model}) on {api_key_label}: {reason} "
                    f"(outer {outer_attempt + 1}/{connection_reset_retries + 1}, "
                    f"timeout={spec.timeout}s); cooldown {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
                continue
            print(
                f"[{country}] {spec.role} ({spec.model}) FAILED on {api_key_label} "
                f"timeout={spec.timeout}s: {str(exc)[:220]}",
                flush=True,
            )
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
    key_pool: ApiKeyPool,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    request_jitter_seconds: float,
    connection_reset_retries: int,
    sequential_models: bool,
    vote_mode: str,
) -> dict:
    text = row_text_from_dict(row)
    content_id = stable_content_id_from_dict(country_config.country, row_index, row)
    user_prompt = build_user_prompt(text)
    initial_system_prompt = build_initial_system_prompt(system_prompt)

    specs = [country_config.primary, country_config.secondary, tertiary_spec]

    def run_one(spec: ModelSpec) -> dict:
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
            key_pool,
            request_jitter_seconds,
            connection_reset_retries,
        )

    if sequential_models:
        results = [run_one(spec) for spec in specs]
    else:
        with ThreadPoolExecutor(max_workers=MODELS_PER_ROW) as executor:
            futures = [executor.submit(run_one, spec) for spec in specs]
            results = [future.result() for future in futures]

    primary, secondary, tertiary = results
    model_vote_labels = [primary["label"], secondary["label"], tertiary["label"]]
    if vote_mode == "triple":
        final_label, final_source = choose_triple_vote(model_vote_labels)
        vote_labels = model_vote_labels
    else:
        original_label = norm_label(row.get("label"))
        vote_labels = [original_label, *model_vote_labels]
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
            "primary_key": primary.get("key_env", ""),
            "secondary_key": secondary.get("key_env", ""),
            "tertiary_key": tertiary.get("key_env", ""),
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
    output_path: Path,
    prompt_country: str,
    output_country: str,
    source_country: str,
    source_label_prefix: str,
    limit: int | None,
    max_workers: int,
    force: bool,
    dry_run: bool,
    primary_model: str | None,
    secondary_model: str | None,
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
    vote_mode: str,
    model_timeout: int,
    force_sequential_models: bool,
) -> None:
    country_config = COUNTRY_CONFIGS[prompt_country.upper()]
    primary_model = primary_model or country_config.primary.model
    secondary_model = secondary_model or country_config.secondary.model
    tertiary_spec = ModelSpec(
        role="tertiary",
        model=tertiary_model,
        account="ARBITER",
        timeout=model_timeout,
    )
    country_config = CountryConfig(
        country=country_config.country,
        default_input=country_config.default_input,
        prompt_path=country_config.prompt_path,
        primary=with_timeout(
            ModelSpec(
                country_config.primary.role,
                primary_model,
                country_config.primary.account,
                country_config.primary.timeout,
            ),
            model_timeout,
        ),
        secondary=with_timeout(
            ModelSpec(
                country_config.secondary.role,
                secondary_model,
                country_config.secondary.account,
                country_config.secondary.timeout,
            ),
            model_timeout,
        ),
        arbitrator=with_timeout(country_config.arbitrator, model_timeout),
        high_risk_labels=country_config.high_risk_labels,
    )
    load_env_for_country(country_config.country)
    system_prompt = country_config.prompt_path.read_text(encoding="utf-8").strip()
    allowed_labels = extract_allowed_labels(system_prompt)

    if input_path.resolve() != output_path.resolve():
        fieldnames, rows = build_job_rows(
            input_path,
            output_path,
            output_country,
            source_country,
            source_label_prefix=str(source_label_prefix),
        )
    else:
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

    key_pool = build_key_pool(nv_keys_only=nv_keys_only, api_key_index=api_key_index)
    worker_count = max(1, min(max_workers, len(pending) if pending else max_workers))
    auto_sequential_small = (
        country not in OOD_CROSSJOBS
        and 0 < len(pending) <= SMALL_JOB_ROW_THRESHOLD
        and not force_sequential_models
    )
    effective_sequential = sequential_models or force_sequential_models or auto_sequential_small
    model_mode = "sequential" if effective_sequential else "3 parallel (round-robin keys)"
    row_mode = "serial" if worker_count == 1 else f"{worker_count} parallel rows"

    print(f"\n[{country}] input:  {input_path}")
    if input_path.resolve() != output_path.resolve():
        print(f"[{country}] output: {output_path}  (prompt={prompt_country}, country={output_country})")
    print(f"[{country}] models:")
    print(f"  primary   = {country_config.primary.model}")
    print(f"  secondary = {country_config.secondary.model}")
    print(f"  tertiary  = {tertiary_spec.model}")
    vote_desc = "3-model vote" if vote_mode == "triple" else "4-vote (gold+3 models)"
    print(
        f"[{country}] rows total: {len(rows):,}  already done: {done_count:,}  "
        f"pending: {len(pending):,}  vote={vote_desc}"
    )
    print(
        f"[{country}] concurrency: rows={row_mode}  models_per_row={model_mode}  "
        f"keys={key_pool.size} (global round-robin)  max_inflight_per_key=1  "
        f"timeout={model_timeout}s  request_jitter<={request_jitter_seconds:.2f}s"
    )
    if auto_sequential_small and not sequential_models:
        print(f"[{country}] auto sequential-models (pending<={SMALL_JOB_ROW_THRESHOLD})")
    print(f"[{country}] key pool ({key_pool.size} keys, round-robin order):")
    for label in key_pool.labels:
        print(f"  - {label}")
    print(f"[{country}] checkpoint: flush CSV after every row (resume built-in)")

    if dry_run:
        preview_count = min(len(pending) * MODELS_PER_ROW, key_pool.size * 2)
        preview_keys = key_pool.preview_next(preview_count)
        print(f"[{country}] dry-run key rotation preview (next {preview_count} calls):")
        print("  " + " -> ".join(preview_keys))
        for task_idx, (idx, row) in enumerate(pending[:5]):
            start = task_idx * MODELS_PER_ROW
            keys = preview_keys[start : start + MODELS_PER_ROW]
            cid = stable_content_id_from_dict(prompt_country, idx, row)
            print(
                f"  - {cid}: keys={' | '.join(keys)}  "
                f"sa_gold={row.get('label')}  sa_final={row.get('source_sa_final_label')}  "
                f"text={row_text_from_dict(row)[:80]}"
            )
        return

    assert_nim_dns_resolves()

    if not pending:
        print(f"[{country}] nothing to process.")
        return

    checkpoint = CsvCheckpoint(output_path, fieldnames, rows)
    processed = 0
    stats = Counter()

    def process_row(idx: int, row: dict) -> dict:
        return annotate_one_row(
            country_config,
            tertiary_spec,
            allowed_labels,
            system_prompt,
            idx,
            row,
            key_pool,
            temperature,
            max_tokens,
            retries,
            backoff_seconds,
            request_jitter_seconds,
            connection_reset_retries,
            effective_sequential,
            vote_mode,
        )

    def log_row(updated_row: dict) -> None:
        nonlocal processed
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

    if worker_count == 1:
        for idx, row in pending:
            log_row(process_row(idx, row))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(process_row, idx, row): idx for idx, row in pending
            }
            for future in as_completed(futures):
                log_row(future.result())

    print(f"\n[{country}] finished {processed} rows")
    for label, count in stats.most_common():
        print(f"  {label}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate multiclass OOD CSVs with 3 models + optional 4-way vote",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--country", default="all", choices=("all", *ALL_COUNTRY_CHOICES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Concurrent rows (default: 2; UAE=1 serial)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_MODEL_TIMEOUT,
        help=f"NIM read timeout seconds per model call (default: {DEFAULT_MODEL_TIMEOUT})",
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
    parser.add_argument("--primary-model", default=None)
    parser.add_argument("--secondary-model", default=None)
    parser.add_argument("--tertiary-model", default=DEFAULT_TERTIARY_MODEL_LOCAL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument("--api-key-index", type=int, default=0)
    parser.add_argument("--nv-keys-only", action="store_true")
    return parser.parse_args()


def resolve_max_workers(args: argparse.Namespace, country: str) -> int:
    if args.max_workers is not None:
        return max(1, args.max_workers)
    job = resolve_job(country)
    return max(1, int(job.get("default_max_workers", DEFAULT_MAX_WORKERS)))


def main() -> None:
    args = parse_args()
    countries = list(COUNTRY_ORDER) if args.country == "all" else [args.country.upper()]
    limit = args.limit if args.limit and args.limit > 0 else None
    model_timeout = max(30, args.timeout)

    for country in countries:
        job = resolve_job(country)
        input_path = Path(job["input_path"])
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input for {country}: {input_path}")

    if len(countries) > 1:
        print(f"Running {len(countries)} countries sequentially: {' -> '.join(countries)}")

    for country in countries:
        job = resolve_job(country)
        max_workers = resolve_max_workers(args, country)
        run_country(
            country=country,
            input_path=Path(job["input_path"]),
            output_path=Path(job["output_path"]),
            prompt_country=str(job["prompt_country"]),
            output_country=str(job["output_country"]),
            source_country=str(job["source_country"]),
            source_label_prefix=str(job.get("source_label_prefix", "")),
            vote_mode=str(job.get("vote_mode", "four")),
            limit=limit,
            max_workers=max_workers,
            force=args.force,
            dry_run=args.dry_run,
            primary_model=args.primary_model,
            secondary_model=args.secondary_model,
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
            model_timeout=model_timeout,
            force_sequential_models=bool(job.get("force_sequential_models", False)),
        )


if __name__ == "__main__":
    main()
