#!/usr/bin/env python3
"""
Sample N rows from TR_R3/TR_SEX_merged.csv, annotate with Qwen 80B (NIM),
add audit columns in-place on the same file (no separate output).

Usage:
  cd multilingual/TR
  export TR_SEX_AUDIT_SAMPLE_SIZE=100
  export TR_SEX_AUDIT_WORKERS=2
  python3 annotate_tr_sex_merged_qwen_sample.py

Optional env:
  TR_SEX_AUDIT_SAMPLE_SIZE=100   # rows to sample (default 100)
  TR_SEX_AUDIT_SEED=42
  TR_SEX_AUDIT_WORKERS=2
  TR_SEX_AUDIT_TIMEOUT=120
  TR_SEX_AUDIT_INPUT=TR_R3/TR_SEX_merged.csv
  NIM_API_KEY=...                # or TR/.env / BR/.env / nv_api_keys.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MERGED = SCRIPT_DIR / "TR_R3" / "TR_SEX_merged.csv"
PROMPT_PATH = SCRIPT_DIR / "tr_jury_prompt.txt"

QWEN_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
EXPECTED_LABEL = "Sexually_Explicit_Information"

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MAX_RETRIES = 3
NIM_RETRY_BACKOFF_SECONDS = 5

TR_CATEGORY_PRIORITY = {
    "Dangerous_Content": 100,
    "Hate_Speech": 90,
    "TR_State_Security_Ataturk": 85,
    "Harassment": 80,
    "Sexually_Explicit_Information": 70,
    "Politically_Sensitive_Topics": 60,
    "Cybersecurity_Malware": 50,
    "none": 0,
}
ALLOWED_CATEGORIES = set(TR_CATEGORY_PRIORITY)

AUDIT_COLS = [
    "audit_sampled",
    "qwen_violation",
    "qwen_category",
    "qwen_confidence",
    "qwen_reasoning",
    "qwen_sex_match",
    "qwen_error",
    "qwen_model",
    "qwen_audited_at",
]


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


def resolve_nim_api_key() -> str:
    load_local_env(SCRIPT_DIR / ".env")
    load_local_env(REPO_ROOT / "BR" / ".env")
    key = os.getenv("NIM_API_KEY") or os.getenv("NVIDIA_API_KEY") or ""
    if key:
        return key
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from nv_api_keys import get_key  # noqa: WPS433

        return get_key(0)
    except Exception:
        return ""


NIM_API_KEY = resolve_nim_api_key()


def normalize_category(category: object) -> str | None:
    if category is None:
        return "none"
    text = str(category).strip()
    if text in ALLOWED_CATEGORIES:
        return text
    if text.lower() in {"safe", "none", "no_violation", "clean"}:
        return "none"
    return None


def normalize_verdict_fields(violation: object, category: object) -> tuple[bool | None, str, str | None]:
    raw_category = category
    normalized_category = normalize_category(raw_category)
    if normalized_category is None:
        return None, "none", f"Invalid category from model: {raw_category}"

    if normalized_category == "none":
        return False, "none", None

    if violation is False:
        return False, "none", None

    if violation is True:
        return True, normalized_category, None

    return None, normalized_category, "Missing or invalid violation field from model"


def _extract_json_verdict(
    raw: str,
    juror_key: str,
    model_name: str,
    latency_ms: float,
    tokens: int,
) -> dict:
    text = raw.strip()

    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()

    if not text.startswith("{"):
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            brace_end = -1
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        brace_end = i
                        break
            if brace_end > brace_start:
                text = text[brace_start : brace_end + 1]

    for attempt in range(3):
        try:
            data = json.loads(text)
            violation, category, error = normalize_verdict_fields(
                data.get("violation"),
                data.get("category", "none"),
            )
            return {
                "juror": juror_key,
                "model_name": model_name,
                "violation": violation,
                "category": category,
                "confidence": float(data.get("confidence", 0.0)),
                "reasoning": data.get("reasoning", ""),
                "latency_ms": latency_ms,
                "tokens_used": tokens,
                "error": error,
            }
        except json.JSONDecodeError:
            if attempt == 0:
                if text.rstrip().endswith('"') and not text.rstrip().endswith("}"):
                    open_count = text.count("{")
                    close_count = text.count("}")
                    if open_count > close_count:
                        text = text.rstrip() + "\n}" + "}" * (open_count - close_count - 1)
            elif attempt == 1:
                text = re.sub(r",\s*}", "}", text)
                text = re.sub(r",\s*]", "]", text)

    return {
        "juror": juror_key,
        "model_name": model_name,
        "violation": None,
        "category": "none",
        "confidence": 0.0,
        "reasoning": f"Failed to parse JSON response after 3 attempts. Raw: {raw[:500]}",
        "latency_ms": latency_ms,
        "tokens_used": tokens,
        "error": "JSON parse error: could not extract valid JSON from response",
    }


def call_nim_juror(
    juror_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float = 120.0,
    max_retries: int = NIM_MAX_RETRIES,
    retry_backoff: float = NIM_RETRY_BACKOFF_SECONDS,
) -> dict:
    if not NIM_API_KEY:
        return {
            "juror": juror_key,
            "model_name": model_name,
            "violation": None,
            "category": "none",
            "confidence": 0.0,
            "reasoning": "Missing NIM_API_KEY",
            "latency_ms": 0.0,
            "tokens_used": 0,
            "error": "Missing NIM_API_KEY",
        }

    url = f"{NIM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    last_verdict = None
    total_latency_ms = 0.0
    for attempt in range(max_retries + 1):
        t0 = time.monotonic()
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"]
            tokens = body.get("usage", {}).get("total_tokens", 0)
            latency_ms = (time.monotonic() - t0) * 1000
            total_latency_ms += latency_ms
            verdict = _extract_json_verdict(raw, juror_key, model_name, latency_ms, tokens)
        except urllib.error.HTTPError as e:
            latency_ms = (time.monotonic() - t0) * 1000
            total_latency_ms += latency_ms
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:
                detail = str(e)[:300]
            verdict = {
                "juror": juror_key,
                "model_name": model_name,
                "violation": None,
                "category": "none",
                "confidence": 0.0,
                "reasoning": f"NIM API error: HTTPError: {e}; {detail}",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "error": f"{e}: {detail}"[:300],
            }
        except urllib.error.URLError as e:
            latency_ms = (time.monotonic() - t0) * 1000
            total_latency_ms += latency_ms
            verdict = {
                "juror": juror_key,
                "model_name": model_name,
                "violation": None,
                "category": "none",
                "confidence": 0.0,
                "reasoning": f"NIM API error: {type(e).__name__}: {e}",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "error": str(e)[:300],
            }
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            total_latency_ms += latency_ms
            verdict = {
                "juror": juror_key,
                "model_name": model_name,
                "violation": None,
                "category": "none",
                "confidence": 0.0,
                "reasoning": f"Unexpected error: {type(e).__name__}: {e}",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "error": str(e)[:300],
            }

        verdict["attempts"] = attempt + 1
        verdict["total_retry_latency_ms"] = total_latency_ms
        if verdict.get("violation") is not None and not verdict.get("error"):
            return verdict

        last_verdict = verdict
        if attempt < max_retries:
            sleep_s = retry_backoff * (attempt + 1)
            print(f"    retry {attempt + 1}/{max_retries} after error: {verdict.get('error')}")
            time.sleep(sleep_s)

    return last_verdict


def load_annotation_policy(prompt_path: Path = PROMPT_PATH) -> str:
    if not prompt_path.exists():
        print(f"ERROR: prompt file not found at {prompt_path}")
        sys.exit(1)
    return prompt_path.read_text(encoding="utf-8").strip()


def build_qwen_prompts(text: str, language: str, source_style: str) -> tuple[str, str]:
    policy = load_annotation_policy()
    system = f"""You are a Senior Turkey Safety Annotator (Qwen audit).

{policy}
"""
    user = f"""Metadata:
country: TR
language: {language}
source: synthetic_sex_r3
source_style: {source_style}
expected_train_label: {EXPECTED_LABEL}

Task: Judge whether this text is a valid positive training example for
Sexually_Explicit_Information (adult consensual sexual intent in Turkish UGC).
If it is safe, merely romantic, harassment-only, spam garbage, or wrong label,
set violation accordingly and pick the best category.

Text:
{text}"""
    return system, user


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        Path(tmp).replace(path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def main() -> int:
    sample_size = env_int("TR_SEX_AUDIT_SAMPLE_SIZE", 100)
    seed = env_int("TR_SEX_AUDIT_SEED", 42)
    workers = env_int("TR_SEX_AUDIT_WORKERS", 2)
    timeout = env_float("TR_SEX_AUDIT_TIMEOUT", 120.0)
    input_path = Path(os.getenv("TR_SEX_AUDIT_INPUT", str(MERGED)))
    if not input_path.is_absolute():
        input_path = SCRIPT_DIR / input_path

    if not input_path.exists():
        print(f"Missing: {input_path}")
        return 1

    if not NIM_API_KEY:
        print("ERROR: NIM_API_KEY not found in TR/.env, BR/.env, or nv_api_keys.py")
        return 1

    with input_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        base_fields = list(reader.fieldnames or [])
        rows = list(reader)

    for col in AUDIT_COLS:
        if col not in base_fields:
            base_fields.append(col)

    for row in rows:
        for col in AUDIT_COLS:
            row.setdefault(col, "")

    pending = [r for r in rows if (r.get("audit_sampled") or "").lower() != "yes"]
    pool = pending if pending else rows
    k = min(sample_size, len(pool))
    rng = random.Random(seed)
    sampled = rng.sample(pool, k)

    print(f"Annotating {k} rows with {QWEN_MODEL} (workers={workers}, timeout={timeout}s)")

    def annotate_with_timeout(row: dict) -> dict:
        text = (row.get("text") or "").strip()
        lang = row.get("language") or "tr-TR"
        style = row.get("source_style") or ""
        system, user = build_qwen_prompts(text, lang, style)
        verdict = call_nim_juror(
            juror_key="Q",
            model_name=QWEN_MODEL,
            system_prompt=system,
            user_prompt=user,
            timeout=timeout,
        )
        category = verdict.get("category") or "none"
        violation = verdict.get("violation")
        sex_match = ""
        if violation is True:
            sex_match = "yes" if category == EXPECTED_LABEL else "no"
        elif violation is False:
            sex_match = "no_not_violation"
        return {
            "audit_sampled": "yes",
            "qwen_violation": "" if violation is None else str(bool(violation)).lower(),
            "qwen_category": category,
            "qwen_confidence": str(verdict.get("confidence", "")),
            "qwen_reasoning": (verdict.get("reasoning") or "")[:500],
            "qwen_sex_match": sex_match,
            "qwen_error": (verdict.get("error") or "")[:300],
            "qwen_model": QWEN_MODEL,
            "qwen_audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(annotate_with_timeout, r): r["id"] for r in sampled}
        done = 0
        for fut in as_completed(futures):
            rid = futures[fut]
            try:
                results[rid] = fut.result()
            except Exception as e:
                results[rid] = {
                    "audit_sampled": "yes",
                    "qwen_violation": "",
                    "qwen_category": "",
                    "qwen_confidence": "",
                    "qwen_reasoning": "",
                    "qwen_sex_match": "",
                    "qwen_error": str(e)[:300],
                    "qwen_model": QWEN_MODEL,
                    "qwen_audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            done += 1
            print(f"  [{done}/{k}] {rid}")

    for row in rows:
        if row["id"] in results:
            row.update(results[row["id"]])

    write_csv_atomic(input_path, base_fields, rows)

    sampled_rows = [r for r in rows if r.get("audit_sampled") == "yes"]
    match_yes = sum(1 for r in sampled_rows if r.get("qwen_sex_match") == "yes")
    match_no = sum(1 for r in sampled_rows if r.get("qwen_sex_match") == "no")
    safe = sum(1 for r in sampled_rows if r.get("qwen_sex_match") == "no_not_violation")
    errors = sum(1 for r in sampled_rows if r.get("qwen_error"))
    print(f"Written: {input_path}")
    print(f"  sampled: {len(sampled_rows)}")
    print(f"  qwen_sex_match=yes: {match_yes}")
    print(f"  qwen_sex_match=no (wrong category): {match_no}")
    print(f"  qwen_sex_match=no_not_violation (safe): {safe}")
    print(f"  qwen_error: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
