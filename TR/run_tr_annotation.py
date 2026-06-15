#!/usr/bin/env python3
"""
Batch annotation for Turkey data using Nvidia NIM jury.

Architecture:
  - 3 NIM models serve as Juror A (specialist), B (reasoner), C (senior)
  - Each post is judged by all 3 in parallel
  - Majority voting decides the final label
  - Split verdicts (1-1-1) are flagged for human review

Usage:
  python run_tr_annotation.py                    # default: 20 posts, 3 models
  python run_tr_annotation.py --limit 500        # annotate 500 posts
  python run_tr_annotation.py --limit 1000 --resume  # resume from checkpoint
  python run_tr_annotation.py --dry-run           # show what would be annotated
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT = PROJECT_ROOT / "train_TR.csv"
PROMPT_PATH = PROJECT_ROOT / "tr_jury_prompt.txt"
RESULTS_DIR = DATA_DIR / "annotations_tr"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── NIM API Config ───────────────────────────────────────────────
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MAX_RETRIES = 3
NIM_RETRY_BACKOFF_SECONDS = 5


def load_local_env(env_path: Path) -> None:
    """Load simple KEY=VALUE entries from a local .env file."""
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


load_local_env(PROJECT_ROOT / ".env")
load_local_env(PROJECT_ROOT.parent / "BR" / ".env")
NIM_API_KEY = os.getenv("NIM_API_KEY", "")


class IncompleteJuryError(RuntimeError):
    """Raised when strict 3-juror mode cannot get all juror verdicts."""

# ── Jury Model Configuration ─────────────────────────────────────
# Top-performing models for multilingual/low-resource languages from NIM tests.
# Each juror uses a different model for diversity of judgment.

JURY_MODELS = {
    "A": {
        "name": "google/gemma-3n-e4b-it",
        "role": "Turkish/Turkey Language Specialist",
        "description": "Gemma: reliable multilingual safety baseline",
        "timeout": 60,
    },
    "B": {
        "name": "nvidia/nemotron-nano-12b-v2-vl",
        "role": "Cross-Cultural Reasoner",
        "description": "Nemotron: fast cross-checking model",
        "timeout": 60,
    },
    "C": {
        "name": "qwen/qwen3-next-80b-a3b-instruct",
        "role": "Senior Turkey Safety Arbitrator",
        "description": "Qwen: strong multilingual support and nuanced reasoning",
        "timeout": 120,
    },
}

# ── Alternative model configurations ─────────────────────────────
# Can be swapped via --jury-config parameter

ALTERNATIVE_JURY = {
    "fast": {
        "A": {"name": "google/gemma-3n-e4b-it", "role": "Turkish/Turkey Language Specialist", "timeout": 30},
        "B": {"name": "meta/llama-3.1-8b-instruct", "role": "Cross-Cultural Reasoner", "timeout": 30},
        "C": {"name": "google/gemma-3n-e2b-it", "role": "Senior Turkey Safety Arbitrator", "timeout": 30},
    },
    "multilingual": {
        "A": {
            "name": "google/gemma-3n-e4b-it",
            "role": "Turkish/Turkey Language Specialist",
            "description": "Gemma: reliable multilingual safety baseline",
            "timeout": 60,
        },
        "B": {
            "name": "nvidia/nemotron-nano-12b-v2-vl",
            "role": "Cross-Cultural Reasoner",
            "description": "Nemotron: fast cross-checking model",
            "timeout": 60,
        },
        "C": {
            "name": "qwen/qwen3-next-80b-a3b-instruct",
            "role": "Senior Turkey Safety Arbitrator",
            "description": "Qwen: strong multilingual support and nuanced reasoning",
            "timeout": 120,
        },
    },
    "balanced": {
        "A": {"name": "meta/llama-3.1-8b-instruct", "role": "Turkish/Turkey Language Specialist", "timeout": 45},
        "B": {"name": "qwen/qwen3-next-80b-a3b-instruct", "role": "Cross-Cultural Reasoner", "timeout": 45},
        "C": {"name": "mistralai/mistral-nemotron", "role": "Senior Turkey Safety Arbitrator", "timeout": 45},
    },
    "thorough": {
        "A": {"name": "google/gemma-4-31b-it", "role": "Turkish/Turkey Language Specialist", "timeout": 90},
        "B": {"name": "qwen/qwen3-coder-480b-a35b-instruct", "role": "Cross-Cultural Reasoner", "timeout": 90},
        "C": {"name": "mistralai/mistral-large-3-675b-instruct-2512", "role": "Senior Turkey Safety Arbitrator", "timeout": 90},
    },
}

# ── Vote resolution ──────────────────────────────────────────────

# TR-specific violation categories with priority (higher = more severe)
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


def normalize_category(category: object) -> str | None:
    """Normalize common non-violation variants and reject categories outside TR taxonomy."""
    if category is None:
        return "none"
    text = str(category).strip()
    if text in ALLOWED_CATEGORIES:
        return text
    if text.lower() in {"safe", "none", "no_violation", "clean"}:
        return "none"
    return None


def normalize_verdict_fields(violation: object, category: object) -> tuple[bool | None, str, str | None]:
    """Keep juror category output inside the TR taxonomy."""
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

def resolve_majority_category(categories: list[str]) -> str:
    """When multiple categories are found, pick highest priority."""
    best = "none"
    best_prio = -1
    for c in categories:
        prio = TR_CATEGORY_PRIORITY.get(c, 0)
        if prio > best_prio:
            best_prio = prio
            best = c
    return best


def majority_vote(verdicts: list[dict]) -> dict:
    """Simple majority vote over 3 juror verdicts.

    Returns a dict with:
      - final_violation: bool | None
      - final_category: str
      - agreement: str
      - method: "unanimous" | "majority" | "split" | "all_null"
      - requires_review: bool
    """
    violations = [v.get("violation") for v in verdicts]
    yes_count = sum(1 for v in violations if v is True)
    no_count = sum(1 for v in violations if v is False)
    null_count = sum(1 for v in violations if v is None)

    # Collect categories from jurors who found violation
    violation_categories = [
        v.get("category", "none")
        for v in verdicts
        if v.get("violation") is True
    ]

    agreement = " / ".join(
        f"Juror_{v['juror']}:{'violation' if v.get('violation') else 'clean' if v.get('violation') is False else 'null'}"
        for v in verdicts
    )

    # Too many nulls
    if null_count >= 2:
        return {
            "final_violation": None,
            "final_category": "none",
            "agreement": agreement,
            "method": "all_null",
            "requires_review": True,
            "adopted_juror": "none",
            "violation_count": yes_count,
            "clean_count": no_count,
            "null_count": null_count,
        }

    # 3-0 unanimous
    if yes_count == 3:
        final_cat = resolve_majority_category(violation_categories)
        return {
            "final_violation": True,
            "final_category": final_cat,
            "agreement": agreement,
            "method": "unanimous",
            "requires_review": False,
            "adopted_juror": "consensus",
            "violation_count": 3,
            "clean_count": 0,
            "null_count": 0,
        }

    if no_count == 3:
        return {
            "final_violation": False,
            "final_category": "none",
            "agreement": agreement,
            "method": "unanimous",
            "requires_review": False,
            "adopted_juror": "consensus",
            "violation_count": 0,
            "clean_count": 3,
            "null_count": 0,
        }

    # 2-1 majority (violation or clean)
    if yes_count == 2:
        final_cat = resolve_majority_category(violation_categories)
        return {
            "final_violation": True,
            "final_category": final_cat,
            "agreement": agreement,
            "method": "majority",
            "requires_review": False,
            "adopted_juror": "majority",
            "violation_count": 2,
            "clean_count": 1,
            "null_count": 0,
        }

    if no_count == 2:
        return {
            "final_violation": False,
            "final_category": "none",
            "agreement": agreement,
            "method": "majority",
            "requires_review": False,
            "adopted_juror": "majority",
            "violation_count": 1,
            "clean_count": 2,
            "null_count": 0,
        }

    # 1-1-1 split (or other combos)
    return {
        "final_violation": None,
        "final_category": "none",
        "agreement": agreement,
        "method": "split",
        "requires_review": True,
        "adopted_juror": "none",
        "violation_count": yes_count,
        "clean_count": no_count,
        "null_count": null_count,
    }


# ── NIM API Call ─────────────────────────────────────────────────

def call_nim_juror(
    juror_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float = 45.0,
    max_retries: int = NIM_MAX_RETRIES,
    retry_backoff: float = NIM_RETRY_BACKOFF_SECONDS,
) -> dict:
    """Call a single NIM model as a juror.

    Returns a verdict dict with keys:
      juror, model_name, violation, category, confidence, reasoning,
      latency_ms, tokens_used, error (if any)
    """
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
            print(f"    Juror {juror_key} retry {attempt + 1}/{max_retries} after error: {verdict.get('error')}")
            time.sleep(sleep_s)

    return last_verdict


def _extract_json_verdict(
    raw: str,
    juror_key: str,
    model_name: str,
    latency_ms: float,
    tokens: int,
) -> dict:
    """Robust JSON extraction from model response.

    Handles: markdown code fences, extra text before/after JSON,
    missing closing braces, plain-text responses.
    """
    text = raw.strip()

    # Strategy 1: Extract from ```json ... ``` code block
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

    # Strategy 2: Find the outermost { ... } block
    if not text.startswith("{"):
        brace_start = text.find("{")
        if brace_start >= 0:
            # Find matching closing brace
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
                text = text[brace_start:brace_end + 1]

    # Try parsing
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
                # Try adding missing closing brace
                if text.rstrip().endswith('"') and not text.rstrip().endswith("}"):
                    # Count braces
                    open_count = text.count("{")
                    close_count = text.count("}")
                    if open_count > close_count:
                        text = text.rstrip() + "\n}" + "}" * (open_count - close_count - 1)
            elif attempt == 1:
                # Last resort: try to fix common issues (trailing commas, etc.)
                # Remove trailing commas before closing braces
                import re
                text = re.sub(r",\s*}", "}", text)
                text = re.sub(r",\s*]", "]", text)

    # All attempts failed
    return {
        "juror": juror_key,
        "model_name": model_name,
        "violation": None,
        "category": "none",
        "confidence": 0.0,
        "reasoning": f"Failed to parse JSON response after 3 attempts. Raw: {raw[:500]}",
        "latency_ms": latency_ms,
        "tokens_used": tokens,
        "error": f"JSON parse error: could not extract valid JSON from response",
    }


# ── Batch Processing ─────────────────────────────────────────────

def load_tr_data(
    input_path: Path,
    limit: Optional[int] = None,
    language_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Load TR annotation candidates from CSV or parquet."""
    if not input_path.exists():
        print(f"ERROR: input data not found at {input_path}")
        sys.exit(1)

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    print(f"Loaded TR data: {len(df):,} rows from {input_path}")

    if "clean_text" not in df.columns:
        if "text" in df.columns:
            df["clean_text"] = df["text"]
        elif "body" in df.columns:
            df["clean_text"] = df["body"]
        else:
            print("ERROR: input data must contain one of: clean_text, text, body")
            sys.exit(1)

    if "content_id" not in df.columns:
        df["content_id"] = [f"tr_train_{i:08d}" for i in range(len(df))]
    if "language" not in df.columns:
        df["language"] = "tr"
    if "source" not in df.columns:
        df["source"] = "local_csv"
    if "subreddit" not in df.columns:
        df["subreddit"] = ""
    if "title" not in df.columns:
        df["title"] = ""

    if language_filter and "language" in df.columns:
        df = df[df["language"].isin(language_filter)]
        print(f"After language filter ({language_filter}): {len(df):,} rows")

    df = df[df["clean_text"].notna() & (df["clean_text"].astype(str).str.strip() != "")]
    print(f"After removing empty content: {len(df):,} rows")

    if limit and limit > 0:
        df = df.head(limit)
        print(f"Limited to: {len(df):,} rows")

    return df


def load_checkpoint(checkpoint_path: Path) -> set:
    """Load already-annotated content_ids from checkpoint."""
    if not checkpoint_path.exists():
        return set()
    try:
        existing = read_results_file(checkpoint_path)
        return set(existing["content_id"].unique())
    except Exception:
        return set()


def read_results_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def write_results_file(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        try:
            df.to_parquet(path, index=False)
        except ImportError as e:
            fallback_path = path.with_suffix(".csv")
            print(f"  ! Parquet support unavailable ({e.__class__.__name__}); saving CSV instead: {fallback_path}")
            df.to_csv(fallback_path, index=False, encoding="utf-8-sig")


def parquet_engine_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def save_checkpoint(results: list[dict], checkpoint_path: Path):
    """Save intermediate results."""
    df = pd.DataFrame(results)
    write_results_file(df, checkpoint_path)
    print(f"  ✓ Checkpoint saved: {len(df)} records → {checkpoint_path}")


def load_annotation_policy(prompt_path: Path = PROMPT_PATH) -> str:
    if not prompt_path.exists():
        print(f"ERROR: prompt file not found at {prompt_path}")
        sys.exit(1)
    return prompt_path.read_text(encoding="utf-8").strip()


def build_tr_jury_prompts(
    content: str,
    title: str,
    source: str,
    language: str,
    subreddit: str,
    jury_config: dict,
) -> dict:
    """Build three juror prompts from the shared TR policy prompt."""
    policy = load_annotation_policy()
    prompts = {}
    for key in ["A", "B", "C"]:
        cfg = jury_config[key]
        system = f"""You are Juror {key}: {cfg.get("role", "Turkey content safety annotator")}.

{policy}
"""
        user = f"""Metadata:
country: TR
language: {language}
source: {source}
subreddit: {subreddit}
title: {title}

Text:
{content}"""
        prompts[key] = {"system": system, "user": user}
    return prompts


def safe_text_field(value: object, default: str = "") -> str:
    """Return a clean string for CSV fields that may be NaN/None."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value)


def process_one_post(
    row: pd.Series,
    jury_config: dict,
    juror_retries: int = NIM_MAX_RETRIES,
    retry_backoff: float = NIM_RETRY_BACKOFF_SECONDS,
    require_complete_jury: bool = False,
) -> dict:
    """Process a single post through all 3 jurors and voting."""
    content_id = row.get("content_id", str(uuid.uuid4()))
    title = safe_text_field(row.get("title", ""))
    body = safe_text_field(row.get("body", ""))
    clean_text = safe_text_field(row.get("clean_text", ""), body or title)
    language = safe_text_field(row.get("language", "tr"), "tr")
    source = safe_text_field(row.get("source", "reddit"), "reddit")
    subreddit = safe_text_field(row.get("subreddit", ""))

    # Build prompts
    prompts = build_tr_jury_prompts(
        content=clean_text,
        title=title,
        source=source,
        language=language,
        subreddit=subreddit,
        jury_config=jury_config,
    )

    # Call all 3 jurors in parallel
    juror_futures = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for key in ["A", "B", "C"]:
            cfg = jury_config[key]
            future = executor.submit(
                call_nim_juror,
                juror_key=key,
                model_name=cfg["name"],
                system_prompt=prompts[key]["system"],
                user_prompt=prompts[key]["user"],
                timeout=cfg.get("timeout", 45.0),
                max_retries=juror_retries,
                retry_backoff=retry_backoff,
            )
            juror_futures[future] = key

        verdicts = []
        for future in as_completed(juror_futures):
            verdict = future.result()
            verdicts.append(verdict)

    # Sort verdicts by juror key for consistent output
    verdicts.sort(key=lambda v: v["juror"])

    if require_complete_jury:
        incomplete = [v for v in verdicts if v.get("violation") is None or v.get("error")]
        if incomplete:
            details = "; ".join(
                f"Juror {v.get('juror')}: {v.get('error') or 'null verdict'}"
                for v in incomplete
            )
            raise IncompleteJuryError(f"{content_id} incomplete 3-juror verdict: {details}")

    # Majority vote
    vote_result = majority_vote(verdicts)

    # Calculate total latency
    total_latency = sum(v.get("latency_ms", 0) for v in verdicts)
    max_latency = max((v.get("latency_ms", 0) for v in verdicts), default=0)

    # Build output record
    record = {
        "content_id": content_id,
        "country": "TR",
        "language": language,
        "source": source,
        "subreddit": subreddit,
        "title": title[:500] if title else "",
        "clean_text": clean_text[:2000],
        # Final verdict
        "final_violation": vote_result["final_violation"],
        "final_category": vote_result["final_category"],
        "vote_method": vote_result["method"],
        "vote_agreement": vote_result["agreement"],
        "requires_review": vote_result["requires_review"],
        "adopted_juror": vote_result["adopted_juror"],
        "violation_count": vote_result["violation_count"],
        "clean_count": vote_result["clean_count"],
        # Juror A
        "juror_a_model": verdicts[0]["model_name"] if len(verdicts) > 0 else "",
        "juror_a_violation": verdicts[0]["violation"] if len(verdicts) > 0 else None,
        "juror_a_category": verdicts[0]["category"] if len(verdicts) > 0 else "",
        "juror_a_confidence": verdicts[0]["confidence"] if len(verdicts) > 0 else 0.0,
        "juror_a_reasoning": verdicts[0]["reasoning"] if len(verdicts) > 0 else "",
        "juror_a_latency_ms": verdicts[0]["latency_ms"] if len(verdicts) > 0 else 0,
        "juror_a_error": verdicts[0]["error"] if len(verdicts) > 0 else "",
        # Juror B
        "juror_b_model": verdicts[1]["model_name"] if len(verdicts) > 1 else "",
        "juror_b_violation": verdicts[1]["violation"] if len(verdicts) > 1 else None,
        "juror_b_category": verdicts[1]["category"] if len(verdicts) > 1 else "",
        "juror_b_confidence": verdicts[1]["confidence"] if len(verdicts) > 1 else 0.0,
        "juror_b_reasoning": verdicts[1]["reasoning"] if len(verdicts) > 1 else "",
        "juror_b_latency_ms": verdicts[1]["latency_ms"] if len(verdicts) > 1 else 0,
        "juror_b_error": verdicts[1]["error"] if len(verdicts) > 1 else "",
        # Juror C
        "juror_c_model": verdicts[2]["model_name"] if len(verdicts) > 2 else "",
        "juror_c_violation": verdicts[2]["violation"] if len(verdicts) > 2 else None,
        "juror_c_category": verdicts[2]["category"] if len(verdicts) > 2 else "",
        "juror_c_confidence": verdicts[2]["confidence"] if len(verdicts) > 2 else 0.0,
        "juror_c_reasoning": verdicts[2]["reasoning"] if len(verdicts) > 2 else "",
        "juror_c_latency_ms": verdicts[2]["latency_ms"] if len(verdicts) > 2 else 0,
        "juror_c_error": verdicts[2]["error"] if len(verdicts) > 2 else "",
        # Metadata
        "total_latency_ms": total_latency,
        "max_latency_ms": max_latency,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }
    return record


def run_batch_annotation(
    df: pd.DataFrame,
    jury_config: dict,
    checkpoint_path: Path,
    checkpoint_interval: int = 50,
    already_done: set | None = None,
    juror_retries: int = NIM_MAX_RETRIES,
    retry_backoff: float = NIM_RETRY_BACKOFF_SECONDS,
    require_complete_jury: bool = False,
    max_consecutive_all_null_errors: int = 5,
) -> list[dict]:
    """Run batch annotation with incremental checkpointing.

    Processes posts one at a time (jurors in parallel per post),
    saving checkpoint every N posts.
    """
    if already_done is None:
        already_done = set()

    all_results = []
    to_process = df[~df["content_id"].isin(already_done)]

    if len(to_process) == 0:
        print("All posts already annotated. Nothing to do.")
        return all_results

    print(f"\n{'='*70}")
    print(f"Starting annotation batch")
    print(f"  Posts to annotate: {len(to_process):,}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Checkpoint interval: every {checkpoint_interval} posts")
    print(f"  Juror retries: {juror_retries}")
    print(f"  Retry backoff: {retry_backoff:.1f}s")
    print(f"  Require complete 3-juror verdict: {require_complete_jury}")
    print(f"  Jury config:")
    for key in ["A", "B", "C"]:
        cfg = jury_config[key]
        print(f"    Juror {key}: {cfg['name']} ({cfg.get('description', '')})")
    print(f"{'='*70}\n")

    t_batch_start = time.monotonic()
    violation_count = 0
    review_count = 0
    error_count = 0
    consecutive_all_null_errors = 0

    for idx, (_, row) in enumerate(to_process.iterrows()):
        post_num = idx + 1
        content_id = row.get("content_id", "?")
        language = safe_text_field(row.get("language", "?"), "?")
        title_preview = (
            safe_text_field(row.get("title", ""))
            or safe_text_field(row.get("clean_text", ""))
        )[:80]

        print(f"[{post_num}/{len(to_process)}] {content_id} ({language}) | {title_preview}...")

        try:
            record = process_one_post(
                row,
                jury_config,
                juror_retries=juror_retries,
                retry_backoff=retry_backoff,
                require_complete_jury=require_complete_jury,
            )
            all_results.append(record)

            if record["final_violation"] is True:
                violation_count += 1
                print(f"  ⚠ VIOLATION: {record['final_category']} "
                      f"(method={record['vote_method']}, "
                      f"A={record['juror_a_violation']}, "
                      f"B={record['juror_b_violation']}, "
                      f"C={record['juror_c_violation']})")
            elif record["final_violation"] is None:
                review_count += 1
                print(f"  ⚡ NEEDS REVIEW (method={record['vote_method']})")
            else:
                print(f"  ✓ CLEAN (method={record['vote_method']})")

            # Report latencies
            latencies = f"A:{record['juror_a_latency_ms']:.0f}ms B:{record['juror_b_latency_ms']:.0f}ms C:{record['juror_c_latency_ms']:.0f}ms"
            print(f"  ⏱ {latencies}")

            if record["juror_a_error"] or record["juror_b_error"] or record["juror_c_error"]:
                error_count += 1
                errors = []
                for jk in ["a", "b", "c"]:
                    if record[f"juror_{jk}_error"]:
                        errors.append(f"Juror {jk.upper()}: {record[f'juror_{jk}_error'][:100]}")
                print(f"  ❌ ERRORS: {'; '.join(errors)}")

            all_jurors_failed = all(record.get(f"juror_{jk}_error") for jk in ["a", "b", "c"])
            if record.get("vote_method") == "all_null" and all_jurors_failed:
                consecutive_all_null_errors += 1
                print(
                    f"  🛑 Consecutive all-null API failures: "
                    f"{consecutive_all_null_errors}/{max_consecutive_all_null_errors}"
                )
            else:
                consecutive_all_null_errors = 0

        except IncompleteJuryError as e:
            print(f"  ❌ STRICT 3-JUROR STOP: {e}")
            if all_results:
                if checkpoint_path.exists():
                    existing = read_results_file(checkpoint_path)
                    merged = pd.concat([existing, pd.DataFrame(all_results)], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["content_id"], keep="last")
                    write_results_file(merged, checkpoint_path)
                else:
                    save_checkpoint(all_results, checkpoint_path)
            print("  No record was written for the incomplete post. Re-run with --resume to retry it.")
            sys.exit(2)

        except Exception as e:
            error_count += 1
            print(f"  ❌ FATAL ERROR processing post: {e}")
            # Add error record
            all_results.append({
                "content_id": content_id,
                "country": "TR",
                "language": language,
                "final_violation": None,
                "final_category": "none",
                "vote_method": "error",
                "requires_review": True,
                "juror_a_error": str(e)[:500],
                "annotated_at": datetime.now(timezone.utc).isoformat(),
            })

        if consecutive_all_null_errors >= max_consecutive_all_null_errors:
            if consecutive_all_null_errors <= len(all_results):
                del all_results[-consecutive_all_null_errors:]
            print(
                "\nNetwork/API appears unhealthy: multiple consecutive posts had all jurors fail. "
                "Dropping those incomplete rows, saving checkpoint, and stopping."
            )
            break

        # Checkpoint every N posts
        if post_num % checkpoint_interval == 0:
            # Merge with previous checkpoint if resuming
            if checkpoint_path.exists():
                existing = read_results_file(checkpoint_path)
                merged = pd.concat([existing, pd.DataFrame(all_results)], ignore_index=True)
                merged = merged.drop_duplicates(subset=["content_id"], keep="last")
                write_results_file(merged, checkpoint_path)
            else:
                save_checkpoint(all_results, checkpoint_path)

            elapsed = time.monotonic() - t_batch_start
            rate = post_num / elapsed if elapsed > 0 else 0
            print(f"\n  --- Checkpoint @ {post_num} ---")
            print(f"  Violations: {violation_count} | Needs review: {review_count} | Errors: {error_count}")
            print(f"  Rate: {rate:.1f} posts/min | Elapsed: {elapsed:.0f}s\n")

    # Final checkpoint
    if all_results:
        if checkpoint_path.exists():
            existing = read_results_file(checkpoint_path)
            merged = pd.concat([existing, pd.DataFrame(all_results)], ignore_index=True)
            merged = merged.drop_duplicates(subset=["content_id"], keep="last")
            write_results_file(merged, checkpoint_path)
        else:
            save_checkpoint(all_results, checkpoint_path)

    t_total = time.monotonic() - t_batch_start
    print(f"\n{'='*70}")
    print(f"Batch complete!")
    print(f"  Total posts: {len(all_results):,}")
    print(f"  Violations: {violation_count} ({violation_count/max(len(all_results),1)*100:.1f}%)")
    print(f"  Needs review: {review_count} ({review_count/max(len(all_results),1)*100:.1f}%)")
    print(f"  Errors: {error_count}")
    print(f"  Total time: {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"  Avg per post: {t_total/max(len(all_results),1):.1f}s")
    print(f"{'='*70}")

    return all_results


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Annotate Turkey data with Nvidia NIM jury",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tr_annotation.py                          # full train_TR.csv to tr_annotation_full.csv
  python run_tr_annotation.py --limit 500              # 500 posts
  python run_tr_annotation.py --resume                 # resume from tr_annotation_full.csv
  python run_tr_annotation.py --jury-config fast       # use fast model config
  python run_tr_annotation.py --input path/to/file.csv # use a custom CSV/parquet input
  python run_tr_annotation.py --dry-run                # preview data only
  python run_tr_annotation.py --languages tr,en        # Turkish + English
        """,
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=0,
        help="Number of posts to annotate; 0 means all rows (default: 0)"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help=f"Input CSV/parquet path (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--jury-config", choices=["fast", "balanced", "thorough", "multilingual"],
        default="multilingual",
        help="Jury model configuration preset (default: multilingual)"
    )
    parser.add_argument(
        "--languages", type=str, default="tr,en",
        help="Comma-separated language codes to include (default: tr,en)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=50,
        help="Save checkpoint every N posts (default: 50)"
    )
    parser.add_argument(
        "--juror-retries", type=int, default=NIM_MAX_RETRIES,
        help=f"Retry each failed/null juror call N times before accepting failure (default: {NIM_MAX_RETRIES})"
    )
    parser.add_argument(
        "--retry-backoff", type=float, default=float(NIM_RETRY_BACKOFF_SECONDS),
        help=f"Base seconds to wait between juror retries; waits backoff, 2x backoff, ... (default: {NIM_RETRY_BACKOFF_SECONDS})"
    )
    parser.add_argument(
        "--require-complete-jury", action="store_true",
        help="Strict quality mode: require A/B/C all to return valid verdicts; stop without writing the row otherwise"
    )
    parser.add_argument(
        "--max-consecutive-all-null-errors", type=int, default=5,
        help="Stop after N consecutive posts where all jurors fail (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Custom output file name (saved in data/annotations_tr/)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be annotated without making API calls"
    )

    args = parser.parse_args()

    # Resolve jury config
    jury_config = ALTERNATIVE_JURY.get(args.jury_config, JURY_MODELS)

    # Parse languages
    lang_filter = [l.strip() for l in args.languages.split(",")]

    # Output file
    if args.output:
        checkpoint_name = args.output
        if not checkpoint_name.endswith((".parquet", ".csv")):
            checkpoint_name += ".csv"
    else:
        checkpoint_name = "tr_annotation_full.csv"
    checkpoint_path = RESULTS_DIR / checkpoint_name
    if checkpoint_path.suffix.lower() == ".parquet" and not parquet_engine_available():
        checkpoint_path = checkpoint_path.with_suffix(".csv")
        print(f"Parquet engine not available; using CSV output: {checkpoint_path}")

    # Load data
    df = load_tr_data(
        input_path=Path(args.input).expanduser(),
        limit=args.limit,
        language_filter=lang_filter,
    )

    if args.dry_run:
        print(f"\n[Dry Run] Would annotate {len(df):,} posts")
        print(f"[Dry Run] Output: {checkpoint_path}")
        print(f"[Dry Run] Jury config: {args.jury_config}")
        for key in ["A", "B", "C"]:
            cfg = jury_config[key]
            print(f"  Juror {key}: {cfg['name']}")
        print("\nSample posts:")
        for i, (_, row) in enumerate(df.head(5).iterrows()):
            text = (row.get("clean_text") or row.get("body") or "")[:150]
            print(f"  [{i+1}] ({row.get('language')}) {text}...")
        return

    # Load existing checkpoint for resume
    already_done = set()
    if checkpoint_path.exists():
        already_done = load_checkpoint(checkpoint_path)
        if already_done:
            print(f"Existing output found: {checkpoint_path}")
            print(f"Already annotated in output: {len(already_done):,} posts")

    if args.resume:
        # Find latest checkpoint
        if not already_done:
            existing_checkpoints = sorted(
                list(RESULTS_DIR.glob("tr_annotation_*.parquet"))
                + list(RESULTS_DIR.glob("tr_annotation_*.csv")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if existing_checkpoints:
                latest = existing_checkpoints[0]
                print(f"Resuming from: {latest}")
                already_done = load_checkpoint(latest)
                print(f"Already annotated: {len(already_done):,} posts")
                checkpoint_path = latest  # append to same file
            else:
                print("No existing checkpoint found. Starting fresh.")
        elif not checkpoint_path.exists():
            print("No existing checkpoint found. Starting fresh.")

    # Run annotation
    results = run_batch_annotation(
        df=df,
        jury_config=jury_config,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        already_done=already_done,
        juror_retries=args.juror_retries,
        retry_backoff=args.retry_backoff,
        require_complete_jury=args.require_complete_jury,
        max_consecutive_all_null_errors=args.max_consecutive_all_null_errors,
    )

    # Print summary statistics
    if results:
        df_results = pd.DataFrame(results)
        print("\n" + "="*70)
        print("ANNOTATION SUMMARY")
        print("="*70)

        # Category distribution
        if "final_category" in df_results.columns:
            cat_dist = df_results["final_category"].value_counts()
            print("\nCategory distribution:")
            for cat, count in cat_dist.items():
                pct = count / len(df_results) * 100
                bar = "█" * int(pct / 2)
                print(f"  {cat:<40} {count:>6} ({pct:5.1f}%) {bar}")

        # Violation rate
        if "final_violation" in df_results.columns:
            viol = df_results["final_violation"]
            viol_true = (viol == True).sum()
            viol_false = (viol == False).sum()
            viol_null = viol.isna().sum()
            print(f"\nViolations: {viol_true} ({viol_true/max(len(df_results),1)*100:.1f}%)")
            print(f"Clean: {viol_false} ({viol_false/max(len(df_results),1)*100:.1f}%)")
            if viol_null > 0:
                print(f"Undecided (null): {viol_null}")

        # Vote method distribution
        if "vote_method" in df_results.columns:
            print(f"\nVote methods:")
            for method, count in df_results["vote_method"].value_counts().items():
                print(f"  {method}: {count}")

        # Average latency
        if "total_latency_ms" in df_results.columns:
            avg_lat = df_results["total_latency_ms"].mean()
            print(f"\nAverage total latency: {avg_lat:.0f}ms ({avg_lat/1000:.1f}s)")

        # Juror agreement analysis
        if all(c in df_results.columns for c in ["juror_a_violation", "juror_b_violation", "juror_c_violation"]):
            valid = df_results.dropna(subset=["juror_a_violation", "juror_b_violation", "juror_c_violation"])
            if len(valid) > 0:
                unanimous = (
                    (valid["juror_a_violation"] == valid["juror_b_violation"]) &
                    (valid["juror_b_violation"] == valid["juror_c_violation"])
                ).sum()
                print(f"\nJuror agreement:")
                print(f"  Unanimous (all 3 agree): {unanimous}/{len(valid)} ({unanimous/max(len(valid),1)*100:.1f}%)")

        print(f"\nResults saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
