#!/usr/bin/env python3
"""
MX R3 Synthetic Data Generator
Model : mistralai/mistral-large-3-675b-instruct-2512 (NVIDIA NIM)
Output: multilingual/MX/MX-R3/{SHARD_ID}.csv

使用方式：
  export NVIDIA_API_KEY=nvapi-xxxx
  python generate_br_r3_sex.py
"""

import os
import csv
import json
import time
import re
import math
import urllib.request
import urllib.error
import concurrent.futures
from pathlib import Path

# 自动加载同目录下的 .env
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_dotenv(Path(__file__).resolve().parent / ".env")

# =====================================================================
# ★ USER CONFIG — 改这里，其余不用动
# =====================================================================

# 要跑的 shard 列表（文件名去掉 .txt）
SHARDS_TO_RUN: list[str] = [
    "MX_SEX_002", "MX_SEX_003", "MX_SEX_004", "MX_SEX_005", "MX_SEX_006", "MX_SEX_007", "MX_SEX_008", "MX_SEX_009", "MX_SEX_010", "MX_SEX_011", "MX_SEX_012"
]

# 每个 shard 总共生成多少条
SAMPLES_PER_SHARD: int = 200

# 每次 API call 让模型一次性生成多少条（建议与 SAMPLES_PER_SHARD 一致，或为其因数）
BATCH_SIZE: int = 30

# 最大并发 API call 数（每个 shard 只需 1 次 API call，并发数 = 同时跑几个 shard）
MAX_WORKERS: int = 3

# 每次 API call 前等待的秒数（0 = 不等待）
REQUEST_INTERVAL: float = 0.0

# =====================================================================
# Model & API
# =====================================================================

MODEL_NAME   = "mistralai/mistral-small-4-119b-2603"
TEMPERATURE  = 0.9
TOP_P        = 0.95
MAX_TOKENS   = 12000   # 每条 ~120 tokens，100 条约 12000

MAX_RETRIES  = 3
RETRY_BACKOFF = 8      # 秒，实际等待 = RETRY_BACKOFF × attempt

# =====================================================================
# Paths（自动推导，不用改）
# =====================================================================

SCRIPT_DIR  = Path(__file__).resolve().parent   # .../multilingual/MX
PROMPTS_DIR = SCRIPT_DIR / "prompts" / "SEX"
OUTPUT_DIR  = SCRIPT_DIR / "MX-R3"

# =====================================================================
# Required fields for validation
# =====================================================================

REQUIRED_FIELDS = {
    "id", "country", "language", "target_label",
    "source_style", "scenario", "explicitness",
    "length_level", "tone", "structure", "obfuscation",
    "pattern_id", "text",
}

# =====================================================================
# Prompt patching
# =====================================================================

def patch_prompt(prompt: str, n: int, shard_id: str, id_start: int) -> str:
    """
    Replace N-count placeholders and ID range in the prompt for this batch.
    """
    p = prompt

    # Replace "Number of samples: 100" → "Number of samples: N"
    p = re.sub(r"Number of samples:\s*\d+", f"Number of samples: {n}", p)

    # Replace "Return exactly 100 JSONL records" → "Return exactly N JSONL records"
    p = re.sub(
        r"Return exactly \d+ JSONL records\.?",
        f"Return exactly {n} JSONL records.",
        p,
    )

    # Replace ID example range
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
# API call
# =====================================================================

def call_api(prompt_text: str, label: str) -> str:
    """
    POST to NVIDIA NIM /chat/completions.
    Returns raw text content from the model.
    Retries up to MAX_RETRIES on failure.
    """
    base_url = (
        os.getenv("NIM_BASE_URL")
        or os.getenv("NVIDIA_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    ).rstrip("/")

    api_key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("NIM_API_KEY")
        or os.getenv("NIW_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError(
            "API key not found. Please set NVIDIA_API_KEY in your environment:\n"
            "  export NVIDIA_API_KEY=nvapi-xxxx"
        )

    endpoint = f"{base_url}/chat/completions"

    # 每次 call 前固定等待，避免触发 rate limit
    if REQUEST_INTERVAL > 0:
        time.sleep(REQUEST_INTERVAL)

    for attempt in range(1, MAX_RETRIES + 1):
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
                time.sleep(2)  # 避免触发 RPM 限额
                return data["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"  [{label}] HTTP {e.code} (attempt {attempt}/{MAX_RETRIES}): {body[:500]}")
            if attempt < MAX_RETRIES:
                # 429 等 60s/120s；其他错误等 8s/16s
                wait = 60 * attempt if e.code == 429 else RETRY_BACKOFF * attempt
                print(f"  [{label}] {'Rate-limited' if e.code == 429 else 'Retrying'} — waiting {wait}s...")
                time.sleep(wait)
                continue

        except Exception as e:
            print(f"  [{label}] Error (attempt {attempt}/{MAX_RETRIES}): {repr(e)[:400]}")

            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"  [{label}] Waiting {wait}s before retry...")
                time.sleep(wait)

    raise RuntimeError(f"[{label}] API failed after {MAX_RETRIES} attempts.")

# =====================================================================
# JSON line decoder (handles concatenated objects on one line)
# =====================================================================

def _decode_json_line(line: str) -> list[dict]:
    """
    Decode one line that may contain one or more concatenated JSON objects.
    Handles the "Extra data" case where model outputs {}{} on a single line.
    """
    decoder = json.JSONDecoder()
    objects = []
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

# =====================================================================
# English detection
# =====================================================================

# 常见 es-MX 网络词（允许出现，不算英文）
_PT_EXCEPTIONS = {
    "live", "chat", "hot", "vip", "dm", "link", "story", "stories",
    "feed", "post", "follow", "like", "bio", "nude", "sex", "sexy",
    "pack", "fake", "real", "ok", "babe", "baby", "show", "group",
    "premium", "free", "close", "friends", "my", "top", "fan",
}

# 常见英文功能词（出现多个说明整句是英文）
_EN_FUNCTION_WORDS = {
    "the", "and", "you", "your", "with", "for", "this", "that",
    "are", "have", "can", "will", "want", "just", "what", "when",
    "but", "not", "its", "been", "they", "them", "our", "more",
    "some", "come", "here", "now", "let", "get", "got", "it's",
    "don't", "i'm", "i'll", "i've", "we're", "you're",
}

def _is_english_sentence(text: str) -> bool:
    """
    检测单条 text 是否包含完整英文句子。
    判断标准：去掉 es-MX 常见借词后，英文功能词命中 ≥ 3 个。
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    # 过滤掉 es-MX 允许的借词
    en_words = [w for w in words if w not in _PT_EXCEPTIONS]
    fn_hits = sum(1 for w in en_words if w in _EN_FUNCTION_WORDS)
    return fn_hits >= 3

# =====================================================================
# JSONL parser & validator
# =====================================================================

def parse_jsonl(raw: str, label: str) -> tuple[list[dict], int]:
    """
    Parse JSONL output from model.
    Returns (valid_records, english_rejected_count).
    Skips lines that are invalid JSON, missing fields, empty text, or full English sentences.
    """
    records = []
    english_rejected = 0
    lines = raw.splitlines()
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue

        objs = _decode_json_line(line)
        if not objs:
            print(f"  [{label}] Line {i}: JSON parse error — skipped.")
            continue

        for obj in objs:
            missing = REQUIRED_FIELDS - obj.keys()
            if missing:
                print(f"  [{label}] Line {i}: missing {missing} — skipped.")
                continue

            text = str(obj.get("text", "")).strip()
            if not text:
                print(f"  [{label}] Line {i}: empty text — skipped.")
                continue

            if _is_english_sentence(text):
                print(f"  [{label}] Line {i}: English detected, rejected → \"{text[:80]}\"")
                english_rejected += 1
                continue

            records.append(obj)

    return records, english_rejected

# =====================================================================
# Single batch
# =====================================================================

def run_batch(
    shard_id: str,
    prompt: str,
    batch_idx: int,
    n: int,
    id_start: int,
) -> list[dict]:
    label = f"{shard_id}/b{batch_idx}"
    id_end = id_start + n - 1
    print(f"[{label}] ▶ Requesting {n} samples (ID {id_start:04d}–{id_end:04d})")

    patched_prompt = patch_prompt(prompt, n, shard_id, id_start)
    raw = call_api(patched_prompt, label)
    records, en_rejected = parse_jsonl(raw, label)

    # 自动补充重生成：英文 reject 导致不足时，追加一次补充 call
    shortfall = n - len(records)
    if shortfall > 0:
        print(f"[{label}] ⚠ Short by {shortfall} (english_rejected={en_rejected}), refetching...")
        refill_prompt = patch_prompt(prompt, shortfall, shard_id, id_start + len(records))
        refill_raw = call_api(refill_prompt, f"{label}/refill")
        refill_records, _ = parse_jsonl(refill_raw, f"{label}/refill")
        records.extend(refill_records)
        print(f"[{label}] Refill got {len(refill_records)}/{shortfall}")

    # 截断到精确 n 条，避免 refill 超量返回导致多写
    if len(records) > n:
        records = records[:n]

    print(f"[{label}] ✓ {len(records)}/{n} valid records")
    return records

# =====================================================================
# Shard runner (submits batches to shared executor)
# =====================================================================

def read_existing_count(shard_id: str) -> int:
    """读取已有 CSV，返回已有条数（不含表头）。文件不存在返回 0。"""
    out_path = OUTPUT_DIR / f"{shard_id}.csv"
    if not out_path.exists():
        return 0
    with open(out_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        count = sum(1 for _ in reader)
    return count


def submit_shard(
    shard_id: str,
    executor: concurrent.futures.ThreadPoolExecutor,
) -> list[tuple[str, int, concurrent.futures.Future]]:
    """
    检查已有条数，只补差值，提交到 executor。
    Returns list of (shard_id, batch_idx, future).
    """
    prompt_path = PROMPTS_DIR / f"{shard_id}.txt"
    if not prompt_path.exists():
        print(f"[{shard_id}] ERROR: prompt file not found: {prompt_path}")
        return []

    existing = read_existing_count(shard_id)
    remaining = SAMPLES_PER_SHARD - existing

    if remaining <= 0:
        print(f"[{shard_id}] Already has {existing} records, target {SAMPLES_PER_SHARD} reached. Skipped.")
        return []

    print(f"[{shard_id}] Existing: {existing}, target: {SAMPLES_PER_SHARD}, need: {remaining} more.")

    prompt = prompt_path.read_text(encoding="utf-8")
    n_batches = math.ceil(remaining / BATCH_SIZE)
    futures = []

    for batch_idx in range(n_batches):
        id_start = existing + batch_idx * BATCH_SIZE + 1
        n = min(BATCH_SIZE, remaining - batch_idx * BATCH_SIZE)
        fut = executor.submit(run_batch, shard_id, prompt, batch_idx, n, id_start)
        futures.append((shard_id, batch_idx, fut))

    return futures

# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 62)
    print("  MX R3 Synthetic Data Generator")
    print("=" * 62)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Shards     : {SHARDS_TO_RUN}")
    print(f"  Samples    : {SAMPLES_PER_SHARD} per shard")
    print(f"  Batch size : {BATCH_SIZE} per API call")
    print(f"  Workers    : {MAX_WORKERS} concurrent")
    print(f"  Prompts    : {PROMPTS_DIR}")
    print(f"  Output     : {OUTPUT_DIR}/<shard>.csv")
    print("=" * 62)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Accumulate records per shard
    shard_records: dict[str, list[dict]] = {sid: [] for sid in SHARDS_TO_RUN}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all batches for all shards concurrently
        all_futures: list[tuple[str, int, concurrent.futures.Future]] = []
        for shard_id in SHARDS_TO_RUN:
            all_futures.extend(submit_shard(shard_id, executor))

        # Collect results as they complete
        future_map = {fut: (sid, bidx) for sid, bidx, fut in all_futures}
        for fut in concurrent.futures.as_completed(future_map):
            sid, bidx = future_map[fut]
            try:
                records = fut.result()
                shard_records[sid].extend(records)
            except Exception as e:
                print(f"[{sid}/b{bidx}] ✗ FAILED: {e}")

    # Append new records to existing CSV (resume-safe)
    print()
    csv_columns = [
        "id", "country", "language", "target_label",
        "source_style", "scenario", "explicitness",
        "length_level", "tone", "structure", "obfuscation",
        "pattern_id", "text",
    ]

    for shard_id in SHARDS_TO_RUN:
        records = shard_records[shard_id]
        if not records:
            continue

        out_path = OUTPUT_DIR / f"{shard_id}.csv"
        file_exists = out_path.exists()

        # ID 从已有条数之后接续编号
        existing = read_existing_count(shard_id) if file_exists else 0
        remaining = SAMPLES_PER_SHARD - existing
        # 截断到恰好补满 target，避免多批并发累积超量
        if len(records) > remaining:
            print(f"[{shard_id}] Trimming {len(records)} → {remaining} to hit target exactly")
            records = records[:remaining]
        for i, rec in enumerate(records, existing + 1):
            rec["id"] = f"{shard_id}_{i:04d}"

        with open(out_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(records)

        total = existing + len(records)
        print(f"[{shard_id}] Appended {len(records)} records (total now: {total}) → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
