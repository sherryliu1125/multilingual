#!/usr/bin/env python3
"""
TR R3 Synthetic Data Generator — Sexually_Explicit_Information
Model : mistralai/mistral-small-4-119b-2603 (NVIDIA NIM)
Output: multilingual/TR/TR_R3/{SHARD_ID}.csv

Features:
  - All outputs (CSV, log, manifest) under TR/TR_R3/
  - Round-robin API keys from nv_api_keys.py + root/BR/MX/SA .env
  - Each valid row flushed to disk immediately (resume-safe)
  - Ctrl+C safe: progress kept in CSV

Usage:
  cd multilingual/TR
  export TR_SEX_SAMPLES_PER_SHARD=100
  export TR_SEX_BATCH_SIZE=30
  export TR_SEX_MAX_WORKERS=1
  export TR_SEX_SHARD_FROM=008
  python3 generate_tr_r3_sex.py

Edit API_KEYS_LIMIT in this script to change how many keys are used (default 2).

Optional env:
  TR_SEX_API_CONCURRENT=1   # max simultaneous HTTPS calls (default 1, fixes SSL EOF)
  TR_SEX_REQUEST_INTERVAL=2
  TR_SEX_POST_SUCCESS_INTERVAL=2
  TR_SEX_MAX_FILL_ROUNDS=8
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import concurrent.futures
import threading
from datetime import datetime, timezone
from pathlib import Path

# =====================================================================
# Paths
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROMPTS_DIR = SCRIPT_DIR / "prompts" / "SEX"
OUTPUT_DIR = SCRIPT_DIR / "TR_R3"
LOG_PATH = OUTPUT_DIR / "generation.log"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"

# =====================================================================
# USER CONFIG — edit API_KEYS_LIMIT here; rest via env (see Usage)
# =====================================================================

ALL_SHARD_IDS: list[str] = [
    f"TR_SEX_{i:03d}" for i in range(1, 13)
]

# Default run list (edit here, or set TR_SEX_SHARD_FROM=008 in shell)
SHARDS_TO_RUN: list[str] = [
    "TR_SEX_008", "TR_SEX_009", "TR_SEX_010", "TR_SEX_011", "TR_SEX_012",
]

_shard_from = os.getenv("TR_SEX_SHARD_FROM", "").strip()
if _shard_from:
    _start_id = (
        _shard_from if _shard_from.startswith("TR_SEX_")
        else f"TR_SEX_{int(_shard_from):03d}"
    )
    SHARDS_TO_RUN = [s for s in ALL_SHARD_IDS if s >= _start_id]

# How many API keys to use from the pool (edit in script)
API_KEYS_LIMIT: int = 2

SAMPLES_PER_SHARD: int = int(os.getenv("TR_SEX_SAMPLES_PER_SHARD", "200"))
BATCH_SIZE: int = int(os.getenv("TR_SEX_BATCH_SIZE", "30"))
MAX_WORKERS: int = int(os.getenv("TR_SEX_MAX_WORKERS", "3"))
# Max in-flight HTTPS calls (1 = serial; 3 workers won't burst 3 connections)
API_CONCURRENT: int = max(1, int(os.getenv("TR_SEX_API_CONCURRENT", "1")))
MAX_FILL_ROUNDS: int = int(os.getenv("TR_SEX_MAX_FILL_ROUNDS", "8"))
REQUEST_INTERVAL: float = float(os.getenv("TR_SEX_REQUEST_INTERVAL", "2"))
POST_SUCCESS_INTERVAL: float = float(
    os.getenv("TR_SEX_POST_SUCCESS_INTERVAL", os.getenv("TR_SEX_REQUEST_INTERVAL", "2"))
)
MAX_EMOJI_PER_TEXT: int = int(os.getenv("TR_SEX_MAX_EMOJI", "2"))

MODEL_NAME = os.getenv(
    "TR_SEX_MODEL",
    "mistralai/mistral-small-4-119b-2603",
)
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_TOKENS = 12000
MAX_RETRIES = 3
RETRY_BACKOFF = 8

CSV_COLUMNS = [
    "id", "country", "language", "target_label",
    "source_style", "scenario", "explicitness",
    "length_level", "tone", "structure", "obfuscation",
    "pattern_id", "text",
]

REQUIRED_FIELDS = set(CSV_COLUMNS)

# Only N HTTPS calls at once across all shards (prevents SSLEOFError burst)
_api_gate = threading.Semaphore(API_CONCURRENT)

# =====================================================================
# Logging
# =====================================================================

_print_lock = threading.Lock()
_log_lock = threading.Lock()


def _log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    with _print_lock:
        print(msg, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


# =====================================================================
# API key pool
# =====================================================================

_NVAPI_RE = re.compile(r"nvapi-[A-Za-z0-9_-]+")
_KEY_ENV_NAMES = (
    "NVIDIA_API_KEY", "NIM_API_KEY", "NIW_API_KEY",
    "NIM_API_KEY_BR_PRIMARY", "NIM_API_KEY_BR_SECONDARY", "NIM_API_KEY_BR_ARBITER",
    "NIM_API_KEY_MX_PRIMARY", "NIM_API_KEY_MX_SECONDARY", "NIM_API_KEY_MX_ARBITER",
    "NIM_API_KEY_SA_PRIMARY", "NIM_API_KEY_SA_SECONDARY", "NIM_API_KEY_SA_ARBITER",
)


def _parse_dotenv_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    found: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in _KEY_ENV_NAMES or k.startswith("NIM_API_KEY"):
                if v.startswith("nvapi-"):
                    found.append(v)
            for m in _NVAPI_RE.findall(v):
                found.append(m)
    return found


def collect_api_keys() -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []

    def add(key: str) -> None:
        key = key.strip()
        if not key or not key.startswith("nvapi-") or key in seen:
            return
        seen.add(key)
        keys.append(key)

    # nv_api_keys.py
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from nv_api_keys import get_all_keys  # noqa: WPS433

        for k in get_all_keys():
            add(k)
    except Exception as e:
        _log(f"[keys] nv_api_keys.py skip: {e}")

    # root .env + BR/MX/SA/.env (user request)
    env_paths = [
        REPO_ROOT / ".env",
        REPO_ROOT / "BR" / ".env",
        REPO_ROOT / "MX" / ".env",
        REPO_ROOT / "SA" / ".env",
        SCRIPT_DIR / ".env",
    ]
    for p in env_paths:
        for k in _parse_dotenv_keys(p):
            add(k)

    if not keys:
        raise RuntimeError(
            "No API keys found. Check nv_api_keys.py and BR/MX/SA/.env"
        )
    _log(f"[keys] found {len(keys)} unique API key(s) in nv_api_keys + .env files")
    return keys


def select_keys(all_keys: list[str]) -> list[str]:
    limit = max(1, min(API_KEYS_LIMIT, len(all_keys)))
    return all_keys[:limit]


class KeyPool:
    """Thread-safe round-robin across all loaded keys."""

    def __init__(self, keys: list[str]) -> None:
        self._keys = keys
        self._idx = 0
        self._lock = threading.Lock()
        self._usage: dict[str, int] = {k: 0 for k in keys}

    def next_key(self) -> tuple[str, int]:
        with self._lock:
            key = self._keys[self._idx % len(self._keys)]
            slot = self._idx % len(self._keys)
            self._idx += 1
            self._usage[key] += 1
            return key, slot

    def usage_summary(self) -> dict[str, int]:
        with self._lock:
            return dict(self._usage)


# =====================================================================
# English detection (same logic as BR/MX SEX scripts)
# =====================================================================

_TR_EXCEPTIONS = {
    "live", "chat", "hot", "vip", "dm", "link", "story", "stories",
    "feed", "post", "follow", "like", "bio", "nude", "sex", "sexy",
    "pack", "fake", "real", "ok", "babe", "baby", "show", "group",
    "premium", "free", "close", "friends", "my", "top", "fan",
    "cam", "sexting", "roleplay", "preview", "bottom", "bull",
    "cuckold", "swinger", "sugarbaby", "hookup", "nsfw",
}

_EN_FUNCTION_WORDS = {
    "the", "and", "you", "your", "with", "for", "this", "that",
    "are", "have", "can", "will", "want", "just", "what", "when",
    "but", "not", "its", "been", "they", "them", "our", "more",
    "some", "come", "here", "now", "let", "get", "got", "it's",
    "don't", "i'm", "i'll", "i've", "we're", "you're",
}


def _is_english_sentence(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    en_words = [w for w in words if w not in _TR_EXCEPTIONS]
    return sum(1 for w in en_words if w in _EN_FUNCTION_WORDS) >= 3


_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)


def _emoji_count(text: str) -> int:
    return sum(len(m) for m in _EMOJI_RE.findall(text))


# =====================================================================
# Prompt patch
# =====================================================================

def patch_prompt(prompt: str, n: int, shard_id: str, id_start: int) -> str:
    p = prompt
    p = re.sub(r"Number of samples:\s*\d+", f"Number of samples: {n}", p)
    p = re.sub(
        r"Return exactly \d+ JSONL records\.?",
        f"Return exactly {n} JSONL records.",
        p,
    )
    id_end = id_start + n - 1
    p = re.sub(
        r'id = "[^"]*_\d{4}", "[^"]*_\d{4}", \.\.\.',
        (
            f'id = "{shard_id}_{id_start:04d}", '
            f'"{shard_id}_{id_start + 1:04d}", '
            f'... "{shard_id}_{id_end:04d}"'
        ),
        p,
    )
    return p

# =====================================================================
# API
# =====================================================================

def call_api(prompt_text: str, label: str, key_pool: KeyPool) -> str:
    """One HTTPS flight at a time; retries also hold the gate (no retry stampede)."""
    base_url = (
        os.getenv("NIM_BASE_URL")
        or os.getenv("NVIDIA_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    ).rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    with _api_gate:
        for attempt in range(1, MAX_RETRIES + 1):
            if REQUEST_INTERVAL > 0:
                time.sleep(REQUEST_INTERVAL)

            api_key, slot = key_pool.next_key()
            tag = f"key#{slot + 1}"

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a content-safety synthetic data generator. "
                            "Output only valid JSONL, one JSON object per line. "
                            "No markdown, no explanation, no extra text before or after."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ],
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
            }

            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if POST_SUCCESS_INTERVAL > 0:
                        time.sleep(POST_SUCCESS_INTERVAL)
                    _log(f"  [{label}] API ok ({tag})")
                    return data["choices"][0]["message"]["content"]

            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                http_code = e.code
                err_msg = f"HTTP {http_code} ({tag}) attempt {attempt}: {body[:300]}"
            except Exception as e:
                http_code = None
                err_msg = f"Error ({tag}) attempt {attempt}: {repr(e)[:300]}"

            _log(f"  [{label}] {err_msg}")
            if attempt >= MAX_RETRIES:
                break

            if http_code is not None:
                wait = 60 * attempt if http_code == 429 else RETRY_BACKOFF * attempt
            else:
                err = err_msg.lower()
                if "ssl" in err or "eof" in err or "timed out" in err:
                    wait = max(RETRY_BACKOFF * attempt, 15)
                else:
                    wait = RETRY_BACKOFF * attempt
            wait += random.uniform(0, 2)
            _log(f"  [{label}] retry in {wait:.0f}s...")
            time.sleep(wait)

    raise RuntimeError(f"[{label}] API failed after {MAX_RETRIES} attempts.")

# =====================================================================
# JSONL parser & validator (aligned with BR/MX generate_*_r3_sex.py)
# =====================================================================

def _decode_json_line(line: str) -> list[dict]:
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    pos = 0
    while pos < len(line):
        while pos < len(line) and line[pos] in " \t":
            pos += 1
        if pos >= len(line):
            break
        try:
            obj, end_pos = decoder.raw_decode(line, pos)
            if isinstance(obj, dict):
                objects.append(obj)
            pos = end_pos
        except json.JSONDecodeError:
            break
    return objects


def parse_jsonl(raw: str, label: str) -> tuple[list[dict], int, int]:
    records: list[dict] = []
    english_rejected = 0
    emoji_rejected = 0
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue

        objs = _decode_json_line(line)
        if not objs:
            _log(f"  [{label}] Line {i}: JSON parse error — skipped.")
            continue

        for obj in objs:
            missing = REQUIRED_FIELDS - obj.keys()
            if missing:
                _log(f"  [{label}] Line {i}: missing {missing} — skipped.")
                continue

            text = str(obj.get("text", "")).strip()
            if not text:
                _log(f"  [{label}] Line {i}: empty text — skipped.")
                continue

            if _is_english_sentence(text):
                _log(f"  [{label}] Line {i}: English rejected → \"{text[:80]}\"")
                english_rejected += 1
                continue

            n_emoji = _emoji_count(text)
            if n_emoji > MAX_EMOJI_PER_TEXT:
                _log(
                    f"  [{label}] Line {i}: emoji {n_emoji} > {MAX_EMOJI_PER_TEXT} — skipped."
                )
                emoji_rejected += 1
                continue

            records.append(obj)

    return records, english_rejected, emoji_rejected

# =====================================================================
# CSV: per-row immediate flush
# =====================================================================

_csv_lock = threading.Lock()


def shard_csv_path(shard_id: str) -> Path:
    return OUTPUT_DIR / f"{shard_id}.csv"


def read_existing_records(shard_id: str) -> list[dict]:
    path = shard_csv_path(shard_id)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_one_record(shard_id: str, record: dict, need_header: bool) -> None:
    path = shard_csv_path(shard_id)
    with _csv_lock:
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if need_header:
                writer.writeheader()
            writer.writerow(record)
            f.flush()
            os.fsync(f.fileno())


# =====================================================================
# Shard runner
# =====================================================================

def fetch_batch(
    shard_id: str,
    prompt: str,
    batch_idx: int,
    n: int,
    id_start: int,
    key_pool: KeyPool,
) -> list[dict]:
    label = f"{shard_id}/b{batch_idx}"
    _log(f"[{label}] ▶ request {n} (IDs {id_start:04d}–{id_start + n - 1:04d})")
    patched = patch_prompt(prompt, n, shard_id, id_start)
    raw = call_api(patched, label, key_pool)
    records, en_rejected, emoji_rejected = parse_jsonl(raw, label)

    shortfall = n - len(records)
    if shortfall > 0:
        _log(
            f"[{label}] short {shortfall} "
            f"(english={en_rejected}, emoji={emoji_rejected}), refill..."
        )
        refill_prompt = patch_prompt(prompt, shortfall, shard_id, id_start + len(records))
        refill_raw = call_api(refill_prompt, f"{label}/refill", key_pool)
        refill_records, _, _ = parse_jsonl(refill_raw, f"{label}/refill")
        records.extend(refill_records)
    if len(records) > n:
        records = records[:n]
    _log(f"[{label}] got {len(records)}/{n} valid")
    return records


def run_shard(shard_id: str, key_pool: KeyPool) -> None:
    prompt_path = PROMPTS_DIR / f"{shard_id}.txt"
    if not prompt_path.exists():
        _log(f"[{shard_id}] ERROR: missing prompt {prompt_path}")
        return

    prompt = prompt_path.read_text(encoding="utf-8")
    existing = read_existing_records(shard_id)
    seen_texts = {r.get("text", "").strip() for r in existing if r.get("text")}
    need_header = not shard_csv_path(shard_id).exists() or shard_csv_path(shard_id).stat().st_size == 0

    _log(f"[{shard_id}] resume: {len(existing)}/{SAMPLES_PER_SHARD}")
    if len(existing) >= SAMPLES_PER_SHARD:
        _log(f"[{shard_id}] already complete, skip")
        return

    round_idx = 0
    dup_total = 0

    while len(existing) < SAMPLES_PER_SHARD and round_idx < MAX_FILL_ROUNDS:
        needed = SAMPLES_PER_SHARD - len(existing)
        n = min(BATCH_SIZE, needed)
        id_start = len(existing) + 1

        try:
            new_records = fetch_batch(shard_id, prompt, round_idx, n, id_start, key_pool)
        except Exception as e:
            _log(f"[{shard_id}/r{round_idx}] FAILED: {e}")
            round_idx += 1
            continue

        batch_seen: set[str] = set()
        written = 0
        for rec in new_records:
            if len(existing) >= SAMPLES_PER_SHARD:
                break
            text = rec.get("text", "").strip()
            if not text:
                continue
            if text in seen_texts or text in batch_seen:
                dup_total += 1
                _log(f"  [{shard_id}] dup skip: {text[:60]}")
                continue

            seen_texts.add(text)
            batch_seen.add(text)
            row_num = len(existing) + 1
            rec["id"] = f"{shard_id}_{row_num:04d}"
            append_one_record(shard_id, rec, need_header)
            need_header = False
            existing.append(rec)
            written += 1
            _log(f"[{shard_id}] +1 → {len(existing)}/{SAMPLES_PER_SHARD} | {text[:60]}")

        if written == 0:
            _log(f"[{shard_id}/r{round_idx}] no new rows, retry...")
        round_idx += 1

    if len(existing) >= SAMPLES_PER_SHARD:
        _log(f"[{shard_id}] ✓ done {len(existing)} (dups skipped: {dup_total})")
    else:
        _log(f"[{shard_id}] ⚠ incomplete {len(existing)}/{SAMPLES_PER_SHARD}")


def write_manifest(
    key_pool: KeyPool,
    keys_in_use: list[str],
    keys_available: int,
    workers: int,
) -> None:
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "samples_per_shard": SAMPLES_PER_SHARD,
        "batch_size": BATCH_SIZE,
        "shards": SHARDS_TO_RUN,
        "output_dir": str(OUTPUT_DIR),
        "prompts_dir": str(PROMPTS_DIR),
        "api_keys_available": keys_available,
        "api_keys_in_use": len(keys_in_use),
        "api_keys_limit": API_KEYS_LIMIT,
        "api_concurrent": API_CONCURRENT,
        "max_workers": workers,
        "request_interval_sec": REQUEST_INTERVAL,
        "post_success_interval_sec": POST_SUCCESS_INTERVAL,
        "key_usage": {f"key_{i + 1}": key_pool.usage_summary().get(k, 0)
                      for i, k in enumerate(keys_in_use)},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _log("=" * 60)
    _log("TR R3 SEX generator")
    _log(f"Model     : {MODEL_NAME}")
    _log(f"Output    : {OUTPUT_DIR}")
    _log(f"Shards    : {len(SHARDS_TO_RUN)}")
    _log(f"Target    : {SAMPLES_PER_SHARD} per shard")
    _log(f"Batch     : {BATCH_SIZE}")
    _log(f"Workers   : {MAX_WORKERS}")
    _log(f"API gate  : {API_CONCURRENT} concurrent HTTPS call(s)")
    _log("=" * 60)

    all_keys = collect_api_keys()
    keys = select_keys(all_keys)
    _log(f"Keys      : using {len(keys)}/{len(all_keys)} (API_KEYS_LIMIT={API_KEYS_LIMIT})")
    _log(
        f"Interval  : before={REQUEST_INTERVAL}s, "
        f"after_ok={POST_SUCCESS_INTERVAL}s"
    )

    key_pool = KeyPool(keys)
    write_manifest(key_pool, keys, len(all_keys), MAX_WORKERS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_shard, sid, key_pool): sid for sid in SHARDS_TO_RUN}
        for fut in concurrent.futures.as_completed(futures):
            sid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                _log(f"[{sid}] FATAL: {e}")

    write_manifest(key_pool, keys, len(all_keys), MAX_WORKERS)
    _log(f"Key usage: {key_pool.usage_summary()}")
    _log("All shards finished. See TR_R3/ for CSV + generation.log")


if __name__ == "__main__":
    main()
