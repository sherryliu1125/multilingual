#!/usr/bin/env python3
"""
Dual-model annotation with one fixed arbitration model for NVIDIA NIM.

Default workflow:
  1. Two country-configured models independently assign final_label.
  2. If they agree on a non-need_review label, accept the shared label.
  3. If they disagree, or either returns need_review, write need_review for
     later handling. No live arbitration is called in the main annotation run.

The script is intentionally country-aware but centralized. It reads the compiled
policy prompts from annotaion_prompts/compiled/{COUNTRY}_full.txt and can use
different NVIDIA API-key accounts by country/model role.

Examples:
  python run_dual_arbitration_annotation.py --country SA --limit 30 --dry-run
  python run_dual_arbitration_annotation.py --country SA --limit 300 --resume
  python run_dual_arbitration_annotation.py --country all --batch-size 30 --max-workers 3

API key resolution, in order for each model role:
  NIM_API_KEY_{COUNTRY}_{ACCOUNT}
  NIM_API_KEY_{ACCOUNT}
  NIM_API_KEY_{COUNTRY}
  NIM_API_KEY

For the default accounts this means variables such as:
  NIM_API_KEY_SA_PRIMARY, NIM_API_KEY_SECONDARY, NIM_API_KEY_ARBITER, NIM_API_KEY
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = PROJECT_ROOT / "annotaion_prompts" / "compiled"
OUTPUT_DIR = PROJECT_ROOT / "2nd_annotation"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

DEFAULT_BATCH_SIZE = 30
DEFAULT_MAX_WORKERS = 8
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF_SECONDS = 8
DEFAULT_AGREEMENT_AUDIT_RATE = 0.10
DEFAULT_MAX_TOKENS = 128


@dataclass(frozen=True)
class ModelSpec:
    role: str
    model: str
    account: str
    timeout: int = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class CountryConfig:
    country: str
    default_input: Path | None
    prompt_path: Path
    primary: ModelSpec
    secondary: ModelSpec
    arbitrator: ModelSpec
    high_risk_labels: frozenset[str]


LLAMA_70B = ModelSpec(
    role="primary",
    model="meta/llama-3.3-70b-instruct",
    account="PRIMARY",
    timeout=120,
)

GEMMA_3N_E4B = ModelSpec(
    role="secondary",
    model="google/gemma-3n-e4b-it",
    account="SECONDARY",
    timeout=120,
)

QWEN_122B_ARBITRATOR = ModelSpec(
    role="arbitrator",
    model="qwen/qwen3.5-122b-a10b",
    account="ARBITER",
    timeout=120,
)

COUNTRY_CONFIGS: dict[str, CountryConfig] = {
    "BR": CountryConfig(
        country="BR",
        default_input=PROJECT_ROOT / "BR" / "train_BR.csv",
        prompt_path=PROMPT_DIR / "BR_full.txt",
        primary=LLAMA_70B,
        secondary=GEMMA_3N_E4B,
        arbitrator=QWEN_122B_ARBITRATOR,
        high_risk_labels=frozenset(),
    ),
    "MX": CountryConfig(
        country="MX",
        default_input=PROJECT_ROOT / "MX" / "train_MX.csv",
        prompt_path=PROMPT_DIR / "MX_full.txt",
        primary=LLAMA_70B,
        secondary=GEMMA_3N_E4B,
        arbitrator=QWEN_122B_ARBITRATOR,
        high_risk_labels=frozenset(),
    ),
    "SA": CountryConfig(
        country="SA",
        default_input=PROJECT_ROOT / "SA" / "SA_new_Data.csv",
        prompt_path=PROMPT_DIR / "SA_full.txt",
        primary=LLAMA_70B,
        secondary=GEMMA_3N_E4B,
        arbitrator=QWEN_122B_ARBITRATOR,
        high_risk_labels=frozenset(),
    ),
    "TR": CountryConfig(
        country="TR",
        default_input=PROJECT_ROOT / "TR" / "train_TR.csv",
        prompt_path=PROMPT_DIR / "TR_full.txt",
        primary=LLAMA_70B,
        secondary=GEMMA_3N_E4B,
        arbitrator=QWEN_122B_ARBITRATOR,
        high_risk_labels=frozenset(),
    ),
    "UAE": CountryConfig(
        country="UAE",
        default_input=None,
        prompt_path=PROMPT_DIR / "UAE_full.txt",
        primary=LLAMA_70B,
        secondary=GEMMA_3N_E4B,
        arbitrator=QWEN_122B_ARBITRATOR,
        high_risk_labels=frozenset(),
    ),
}


def load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env_for_country(country: str) -> None:
    load_local_env(PROJECT_ROOT / ".env")
    load_local_env(PROJECT_ROOT / country / ".env")


def resolve_api_key(country: str, account: str) -> tuple[str, str]:
    account = account.upper()
    candidates = [
        f"NIM_API_KEY_{country}_{account}",
        f"NIM_API_KEY_{account}",
        f"NIM_API_KEY_{country}",
        "NIM_API_KEY",
    ]
    for name in candidates:
        value = os.getenv(name, "").strip()
        if value:
            return value, name
    raise RuntimeError(
        f"No NVIDIA API key found for {country}/{account}. Tried: {', '.join(candidates)}"
    )


def extract_allowed_labels(prompt: str) -> set[str]:
    match = re.search(r"Allowed final_label:\s*(.*?)\n\s*Output exactly one line:", prompt, re.S)
    if not match:
        raise ValueError("Could not extract Allowed final_label block from prompt.")
    labels = {line.strip() for line in match.group(1).splitlines() if line.strip()}
    if "safe" not in labels or "need_review" not in labels:
        raise ValueError("Compiled prompt labels must include safe and need_review.")
    return labels


def strip_model_output(raw: object) -> str:
    text = "" if raw is None else str(raw).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip("`").strip()


def extract_json_field(text: str, field: str) -> str:
    if text.startswith("{"):
        try:
            data = json.loads(text)
            value = data.get(field)
            if value is not None and str(value).strip():
                return str(value).strip()
        except json.JSONDecodeError:
            pass

    label_match = re.search(
        rf'"(?:final_label|label|category)"\s*:\s*"([A-Za-z][A-Za-z0-9_]*)"',
        text,
    )
    if field in {"final_label", "label", "category"} and label_match:
        return label_match.group(1).strip()

    reason_match = re.search(r'"reason"\s*:\s*"', text)
    if field == "reason" and reason_match:
        start = reason_match.end()
        end = text.rfind('"}')
        if end > start:
            return text[start:end].strip()

    line_match = re.search(rf"{field}\s*:\s*(.+)", text, re.I | re.S)
    if line_match:
        return line_match.group(1).strip().strip('"').strip()
    return ""


def normalize_label(raw: object, allowed_labels: set[str]) -> tuple[str, str | None]:
    text = strip_model_output(raw)
    extracted = (
        extract_json_field(text, "final_label")
        or extract_json_field(text, "label")
        or extract_json_field(text, "category")
    )
    if extracted:
        text = extracted

    line_match = re.search(r"final_label\s*:\s*([A-Za-z0-9_]+)", text)
    if line_match:
        text = line_match.group(1).strip()

    if text not in allowed_labels:
        for candidate in re.findall(r"\b([A-Za-z][A-Za-z0-9_]+)\b", text):
            if candidate in allowed_labels:
                text = candidate
                break

    aliases = {
        "none": "safe",
        "clean": "safe",
        "no_violation": "safe",
        "no-risk": "safe",
        "no_risk": "safe",
        "scams": "Dangerous_Content",
        "scam": "Dangerous_Content",
        "fraud": "Dangerous_Content",
        "phishing": "Dangerous_Content",
    }
    text = aliases.get(text.lower(), text)

    if text in allowed_labels:
        return text, None
    # Label was structurally parsed but not in taxonomy (e.g. model hallucinated "Scams").
    # Accept need_review and continue; retrying the same model usually won't help.
    if extracted or line_match:
        return "need_review", None
    return "need_review", f"Invalid label: {str(raw)[:300]}"


def extract_reason(raw: object) -> str:
    text = strip_model_output(raw)
    for field in ("reason", "rationale", "explanation"):
        reason = extract_json_field(text, field)
        if reason:
            return reason[:500]
    return ""


def build_initial_system_prompt(system_prompt: str) -> str:
    return (
        f"{system_prompt}\n\n"
        "For this dual-model annotation run, override only the output format. "
        "Still apply all taxonomy and country-policy rules above. "
        "final_label must be exactly one value from Allowed final_label above; "
        "do not invent labels such as Scams or Fraud (map scams/fraud/phishing to Dangerous_Content). "
        "Return one compact JSON object only, with exactly these keys: "
        "{\"final_label\":\"<one allowed final_label>\",\"reason\":\"<one concise sentence>\"}. "
        "Do not add markdown or extra text."
    )


def build_user_prompt(text: str) -> str:
    return (
        "Classify the following user-generated text. "
        "Return the required compact JSON object only.\n\n"
        "Input text:\n"
        f"{text}"
    )


def build_arbiter_system_prompt(allowed_labels: set[str]) -> str:
    labels = ", ".join(sorted(allowed_labels))
    return (
        "You are a lightweight arbitration judge. "
        "You do not need the full policy prompt. "
        "Choose exactly one of the two model decisions based on their labels and reasons. "
        f"Allowed final_label values for this arbitration are: {labels}. "
        "Return exactly one line: final_label: <one allowed final_label>. "
        "No explanation."
    )


def build_arbiter_choose_prompt(
    label_a: str,
    reason_a: str,
    label_b: str,
    reason_b: str,
) -> str:
    return (
        "Two stronger annotators disagree, or one returned need_review.\n"
        f"Model A final_label: {label_a}\n"
        f"Model A reason: {reason_a or '(no reason provided)'}\n\n"
        f"Model B final_label: {label_b}\n"
        f"Model B reason: {reason_b or '(no reason provided)'}\n\n"
        f"You must choose exactly one of these two labels. No other label is allowed.\n"
        f"Return exactly one line in the required format and no explanation."
    )


def call_nim_label(
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
    url = f"{NIM_BASE_URL}/chat/completions"
    if api_key:
        key_env = api_key_label or "explicit"
    else:
        api_key, key_env = resolve_api_key(country, spec.account)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=spec.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"].strip()
            label, parse_error = normalize_label(raw, allowed_labels)
            reason = extract_reason(raw)
            if parse_error:
                error = parse_error
                if attempt < retries:
                    wait = backoff_seconds * (attempt + 1)
                    print(
                        f"[{country}] {spec.role} parse error "
                        f"(attempt {attempt + 1}/{retries + 1}): {parse_error[:200]}; "
                        f"retrying in {wait:.0f}s",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"[{country}] {spec.role} invalid label after {retries + 1} attempts: {raw[:300]}"
                )
            return {
                "role": spec.role,
                "model": spec.model,
                "account": spec.account,
                "key_env": key_env,
                "label": label,
                "reason": reason,
                "raw_output": raw[:1000],
                "parse_error": parse_error,
                "error": parse_error,
                "attempts": attempt + 1,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "tokens_used": body.get("usage", {}).get("total_tokens", 0),
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            error = f"HTTP {exc.code}: {detail}"
        except urllib.error.URLError as exc:
            error = f"URL error: {exc}"
            if isinstance(exc.reason, socket.gaierror):
                raise RuntimeError(
                    f"[{country}] {spec.role} cannot resolve NVIDIA NIM host in "
                    f"{NIM_BASE_URL!r}; this is a DNS/network problem before API-key "
                    "authentication. Check VPN/proxy/DNS/network access and rerun."
                ) from exc
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            is_429 = error.startswith("HTTP 429:")
            wait = 60 * (attempt + 1) if is_429 else backoff_seconds * (attempt + 1)
            print(
                f"[{country}] {spec.role} call failed "
                f"(attempt {attempt + 1}/{retries + 1}): {error[:220]}; "
                f"retrying in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"[{country}] {spec.role} API failed after {retries + 1} attempts: {error[:500]}"
    )


def should_audit_agreement(content_id: str, audit_rate: float) -> bool:
    if audit_rate <= 0:
        return False
    if audit_rate >= 1:
        return True
    bucket = int(sha256(content_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < audit_rate


def needs_arbitration(
    labels: list[str],
    high_risk_labels: set[str],
) -> tuple[bool, str]:
    if any(label == "need_review" for label in labels):
        return True, "need_review_from_initial_model"
    if len(set(labels)) != 1:
        return True, "initial_model_disagreement"
    if labels[0] in high_risk_labels:
        return True, "high_risk_label_confirmation"
    return False, "initial_models_agree"


def choose_final_label(
    initial_results: list[dict],
    arbitration_result: dict | None,
    arbitration_reason: str,
) -> tuple[str, str]:
    labels = [result["label"] for result in initial_results]
    if len(set(labels)) == 1 and labels[0] != "need_review":
        return labels[0], "initial_models_agree"
    return "need_review", arbitration_reason


def stable_content_id(country: str, row_index: int, row: pd.Series) -> str:
    for column in ("content_id", "id", "index"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return f"{country}_{column}_{str(value).strip()}"
    return f"{country}_row_{row_index:08d}"


def row_text(row: pd.Series) -> str:
    for column in ("clean_text", "text", "body", "content"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    title = row.get("title")
    if pd.notna(title) and str(title).strip():
        return str(title).strip()
    return ""


def annotate_one(
    country_config: CountryConfig,
    allowed_labels: set[str],
    system_prompt: str,
    row_index: int,
    row: pd.Series,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    parallel_initial_models: bool,
) -> dict:
    text = row_text(row)
    content_id = stable_content_id(country_config.country, row_index, row)
    user_prompt = build_user_prompt(text)
    initial_system_prompt = build_initial_system_prompt(system_prompt)

    initial_specs = [country_config.primary, country_config.secondary]
    if parallel_initial_models:
        with ThreadPoolExecutor(max_workers=len(initial_specs)) as executor:
            futures = {
                executor.submit(
                    call_nim_label,
                    country_config.country,
                    spec,
                    initial_system_prompt,
                    user_prompt,
                    allowed_labels,
                    temperature,
                    max_tokens,
                    retries,
                    backoff_seconds,
                ): spec.role
                for spec in initial_specs
            }
            by_role = {role: future.result() for future, role in futures.items()}
            initial_results = [by_role[spec.role] for spec in initial_specs]
    else:
        initial_results = [
            call_nim_label(
                country_config.country,
                spec,
                initial_system_prompt,
                user_prompt,
                allowed_labels,
                temperature,
                max_tokens,
                retries,
                backoff_seconds,
            )
            for spec in initial_specs
        ]

    labels = [result["label"] for result in initial_results]
    do_arbitrate, arbitration_reason = needs_arbitration(
        labels,
        set(country_config.high_risk_labels),
    )

    arbitration_result = None

    final_label, final_source = choose_final_label(
        initial_results,
        arbitration_result,
        arbitration_reason,
    )

    orig = {str(k): v for k, v in row.items()}
    ann = {
        # ── 核心标注结果 ──────────────────────────────────────────
        "final_label": final_label,
        "final_source": final_source,
        # ── 两个初标模型的输出 ────────────────────────────────────
        "primary_label": initial_results[0]["label"],
        "primary_reason": initial_results[0].get("reason", ""),
        "secondary_label": initial_results[1]["label"],
        "secondary_reason": initial_results[1].get("reason", ""),
        # ── 后期处理信息 ─────────────────────────────────────────
        "review_reason": arbitration_reason,
        "arbitration_triggered": False,
        "arbitration_reason": "",
        "arbitrator_label": (arbitration_result or {}).get("label", ""),
        # ── 原始输出（调试用）────────────────────────────────────
        "primary_raw_output": initial_results[0].get("raw_output", ""),
        "secondary_raw_output": initial_results[1].get("raw_output", ""),
        "arbitrator_raw_output": (arbitration_result or {}).get("raw_output", ""),
        # ── 错误信息 ─────────────────────────────────────────────
        "primary_error": initial_results[0].get("error"),
        "secondary_error": initial_results[1].get("error"),
        "arbitrator_error": (arbitration_result or {}).get("error", ""),
        # ── 元数据 ───────────────────────────────────────────────
        "primary_model": country_config.primary.model,
        "secondary_model": country_config.secondary.model,
        "arbitrator_model": "",
        "total_tokens_used": sum(int(r.get("tokens_used") or 0) for r in initial_results)
            + int((arbitration_result or {}).get("tokens_used") or 0),
        "annotated_at": datetime.now(timezone.utc).isoformat(),
        "source_row_index": row_index,
        "_ann_row_id": content_id,
    }
    return {**orig, **ann}


def read_input(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if not any(column in df.columns for column in ("clean_text", "text", "body", "content", "title")):
        raise ValueError("Input must contain one of: clean_text, text, body, content, title")

    mask = df.apply(lambda row: bool(row_text(row)), axis=1)
    df = df[mask].copy()
    if limit and limit > 0:
        df = df.head(limit).copy()
    return df


def read_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    for col in ("_ann_row_id", "content_id"):
        if col in existing.columns:
            return set(existing[col].dropna().astype(str))
    return set()


def append_results(output_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    fieldnames = list(rows[0].keys())
    with output_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def chunks(items: list[tuple[int, pd.Series]], size: int) -> Iterable[list[tuple[int, pd.Series]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def audit_agreement_rows(
    country_config: CountryConfig,
    annotation_path: Path,
    audit_output_path: Path,
    audit_rate: float,
    max_workers: int,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    resume: bool,
) -> None:
    load_env_for_country(country_config.country)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation output not found: {annotation_path}")

    system_prompt = country_config.prompt_path.read_text(encoding="utf-8").strip()
    allowed_labels = extract_allowed_labels(system_prompt)
    df = pd.read_csv(annotation_path)
    done_ids = read_done_ids(audit_output_path) if resume else set()

    candidates: list[tuple[int, pd.Series]] = []
    for idx, row in df.iterrows():
        content_id = str(row.get("_ann_row_id") or stable_content_id(country_config.country, int(idx), row))
        primary_label = str(row.get("primary_label") or "").strip()
        secondary_label = str(row.get("secondary_label") or "").strip()
        if not primary_label or primary_label != secondary_label or primary_label == "need_review":
            continue
        if not should_audit_agreement(content_id, audit_rate):
            continue
        if content_id in done_ids:
            continue
        candidates.append((int(idx), row))

    print(f"\n[{country_config.country}] agreement audit input: {annotation_path}")
    print(f"[{country_config.country}] agreement audit output: {audit_output_path}")
    print(
        f"[{country_config.country}] agreement rows sampled: {len(candidates):,} "
        f"(rate={audit_rate:.1%}); max_workers={max_workers}"
    )
    if not candidates:
        return

    def audit_one(row_index: int, row: pd.Series) -> dict:
        content_id = str(row.get("_ann_row_id") or stable_content_id(country_config.country, row_index, row))
        text = row_text(row)
        result = call_nim_label(
            country_config.country,
            country_config.arbitrator,
            system_prompt,
            build_user_prompt(text),
            allowed_labels,
            temperature,
            max_tokens,
            retries,
            backoff_seconds,
        )
        original = {str(k): v for k, v in row.items()}
        audit_label = result["label"]
        final_label = str(row.get("final_label") or "").strip()
        original.update({
            "agreement_audit_label": audit_label,
            "agreement_audit_disagrees": bool(final_label and audit_label != final_label),
            "agreement_audit_raw_output": result.get("raw_output", ""),
            "agreement_audit_error": result.get("error"),
            "agreement_audit_model": country_config.arbitrator.model,
            "agreement_audit_account": result.get("key_env", ""),
            "agreement_audit_tokens_used": int(result.get("tokens_used") or 0),
            "agreement_audit_at": datetime.now(timezone.utc).isoformat(),
            "_ann_row_id": content_id,
        })
        return original

    completed = 0
    disagreements = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(audit_one, idx, row)
            for idx, row in candidates
        ]
        for future in as_completed(futures):
            row = future.result()
            append_results(audit_output_path, [row])
            completed += 1
            if row.get("agreement_audit_disagrees"):
                disagreements += 1
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(
                f"[{country_config.country}] {ts}  agreement audit "
                f"{completed}/{len(candidates)}  disagreements {disagreements}",
                flush=True,
            )


def run_country(
    country_config: CountryConfig,
    input_path: Path,
    output_path: Path,
    limit: int | None,
    batch_size: int,
    max_workers: int,
    resume: bool,
    dry_run: bool,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_seconds: float,
    parallel_initial_models: bool,
) -> None:
    load_env_for_country(country_config.country)
    system_prompt = country_config.prompt_path.read_text(encoding="utf-8").strip()
    allowed_labels = extract_allowed_labels(system_prompt)
    df = read_input(input_path, limit)

    indexed_rows = [(int(idx), row) for idx, row in df.iterrows()]
    done_ids = read_done_ids(output_path) if resume else set()
    pending = [
        (idx, row)
        for idx, row in indexed_rows
        if stable_content_id(country_config.country, idx, row) not in done_ids
    ]
    total_batches = (len(pending) + batch_size - 1) // batch_size if pending else 0

    print(f"\n[{country_config.country}] input: {input_path}")
    print(f"[{country_config.country}] output: {output_path}")
    print(f"[{country_config.country}] rows loaded: {len(df):,}  pending: {len(pending):,}  batches: {total_batches}")
    print(
        f"[{country_config.country}] primary={country_config.primary.model}\n"
        f"[{country_config.country}] secondary={country_config.secondary.model}\n"
        f"[{country_config.country}] arbitrator={country_config.arbitrator.model}"
    )
    model_parallel = "on (primary+secondary per row)" if parallel_initial_models else "off (serial per row)"
    max_inflight = max_workers * (2 if parallel_initial_models else 1)
    print(
        f"[{country_config.country}] batch_size={batch_size}  max_workers={max_workers}  "
        f"row_parallel={max_workers}  model_parallel={model_parallel}  "
        f"max_api_inflight≈{max_inflight}"
    )

    if dry_run:
        for idx, row in pending[:5]:
            print(f"  - {stable_content_id(country_config.country, idx, row)}: {row_text(row)[:120]}")
        return

    if not pending:
        return

    processed = 0
    total_review = 0
    for batch_number, batch in enumerate(chunks(pending, batch_size), start=1):
        batch_review = 0
        print(
            f"\n[{country_config.country}] ── batch {batch_number}/{total_batches} starting "
            f"({len(batch)} rows, up to {max_workers} rows in parallel) ──"
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    annotate_one,
                    country_config,
                    allowed_labels,
                    system_prompt,
                    idx,
                    row,
                    temperature,
                    max_tokens,
                    retries,
                    backoff_seconds,
                    parallel_initial_models,
                )
                for idx, row in batch
            ]
            done_in_batch = 0
            for future in as_completed(futures):
                result = future.result()
                append_results(output_path, [result])
                done_in_batch += 1
                processed += 1
                if result.get("final_label") == "need_review":
                    batch_review += 1
                    total_review += 1
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(
                    f"[{country_config.country}] {ts}  "
                    f"batch {batch_number}/{total_batches}  "
                    f"row {done_in_batch}/{len(batch)}  "
                    f"total {processed}/{len(pending)}  "
                    f"review {total_review}",
                    flush=True,
                )
        print(
            f"[{country_config.country}] ── batch {batch_number}/{total_batches} done: "
            f"{len(batch)} rows saved  {batch_review} need later review  "
            f"total {processed}/{len(pending)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True, help="BR, MX, SA, TR, UAE, or all")
    parser.add_argument("--input", type=Path, help="Input CSV/parquet. Required for UAE unless default exists.")
    parser.add_argument("--output", type=Path, help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument(
        "--audit-agreements",
        action="store_true",
        help="Post-process an existing 2nd annotation CSV and audit a deterministic sample of agreement rows.",
    )
    parser.add_argument(
        "--audit-rate",
        type=float,
        default=DEFAULT_AGREEMENT_AUDIT_RATE,
        help="Deterministic sample rate for --audit-agreements.",
    )
    parser.add_argument("--audit-output", type=Path, help="Output CSV path for --audit-agreements.")
    parser.add_argument(
        "--parallel-initial-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call primary and secondary concurrently inside each row worker (default: on).",
    )
    parser.add_argument("--primary-model", help="Override primary model for this run.")
    parser.add_argument("--secondary-model", help="Override secondary model for this run.")
    parser.add_argument("--arbitrator-model", help="Override fixed arbitrator model for this run.")
    return parser.parse_args()


def with_model_overrides(config: CountryConfig, args: argparse.Namespace) -> CountryConfig:
    primary = config.primary
    secondary = config.secondary
    arbitrator = config.arbitrator
    if args.primary_model:
        primary = ModelSpec(primary.role, args.primary_model, primary.account, primary.timeout)
    if args.secondary_model:
        secondary = ModelSpec(secondary.role, args.secondary_model, secondary.account, secondary.timeout)
    if args.arbitrator_model:
        arbitrator = ModelSpec(arbitrator.role, args.arbitrator_model, arbitrator.account, arbitrator.timeout)
    return CountryConfig(
        country=config.country,
        default_input=config.default_input,
        prompt_path=config.prompt_path,
        primary=primary,
        secondary=secondary,
        arbitrator=arbitrator,
        high_risk_labels=config.high_risk_labels,
    )


def main() -> int:
    args = parse_args()
    country_arg = args.country.upper()
    countries = list(COUNTRY_CONFIGS) if country_arg == "ALL" else [country_arg]

    unknown = [country for country in countries if country not in COUNTRY_CONFIGS]
    if unknown:
        print(f"Unknown country: {', '.join(unknown)}", file=sys.stderr)
        return 2

    for country in countries:
        config = with_model_overrides(COUNTRY_CONFIGS[country], args)
        input_path = args.input or config.default_input
        if input_path is None:
            print(f"[{country}] No default input. Pass --input for this country.", file=sys.stderr)
            return 2
        output_path = args.output or (OUTPUT_DIR / country.upper() / f"{country.lower()}_annotation_2nd.csv")
        if country_arg == "ALL" and args.output:
            output_path = args.output.with_name(f"{args.output.stem}_{country.lower()}{args.output.suffix}")

        if args.audit_agreements:
            audit_output_path = args.audit_output or output_path.with_name(
                f"{output_path.stem}_agreement_audit{output_path.suffix}"
            )
            if country_arg == "ALL" and args.audit_output:
                audit_output_path = args.audit_output.with_name(
                    f"{args.audit_output.stem}_{country.lower()}{args.audit_output.suffix}"
                )
            audit_agreement_rows(
                country_config=config,
                annotation_path=output_path,
                audit_output_path=audit_output_path,
                audit_rate=max(0.0, min(1.0, args.audit_rate)),
                max_workers=max(1, args.max_workers),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=max(0, args.retries),
                backoff_seconds=max(0.0, args.backoff_seconds),
                resume=args.resume,
            )
            continue

        run_country(
            country_config=config,
            input_path=input_path,
            output_path=output_path,
            limit=args.limit,
            batch_size=max(1, args.batch_size),
            max_workers=max(1, args.max_workers),
            resume=args.resume,
            dry_run=args.dry_run,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=max(0, args.retries),
            backoff_seconds=max(0.0, args.backoff_seconds),
            parallel_initial_models=args.parallel_initial_models,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
