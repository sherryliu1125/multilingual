#!/usr/bin/env python3
"""
SA LGBTQ Content Synthetic Data Generator
Model : qwen/qwen3.5-397b-a17b (NVIDIA NIM)
Prompts:
  SA/SA-R3/prompts/base.txt          — 公共指令（含 {N} 和 {SHARD_CONTENT} 占位符）
  SA/SA-R3/prompts/{shard_id}.txt    — 各 shard 专属配置（自动按 shard 加载）
Output: SA/SA-R3/SA_LGBTQ_Content.csv

使用方式：
  python SA/SA-R3/generate_sa_lgbtq.py
"""

import os
import csv
import json
import time
import re
import urllib.request
import urllib.error
import concurrent.futures
import threading
from pathlib import Path

# =====================================================================
# .env loader
# =====================================================================

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

_load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")   # SA/.env

# =====================================================================
# ★ USER CONFIG — 改这里，其余不用动
# =====================================================================

SHARD_TARGETS: dict[str, int] = {
    "001": 200,
    "002": 200,
    "003": 200,
    "004": 200,
    "005": 200,
    "006": 200,
    "007": 200,
    "008": 200,
    "009": 200,
    "010": 200,
    "011": 200,
    "012": 200,
}

SHARDS_TO_RUN: list[str] = list(SHARD_TARGETS.keys())

# 每次 API call 请求条数；模型单次输出不超过 MAX_TOKENS，30 条阿语约 3000–5000 tokens
BATCH_SIZE:  int = 30
MAX_WORKERS: int = 3

# =====================================================================
# Model & API
# =====================================================================

MODEL_NAME    = "qwen/qwen3.5-397b-a17b"
TEMPERATURE   = 0.9
TOP_P         = 0.95
MAX_TOKENS    = 8000

MAX_RETRIES   = 3
RETRY_BACKOFF = 8   # 非 429 错误等待秒数 = RETRY_BACKOFF × attempt

# =====================================================================
# Paths
# =====================================================================

SCRIPT_DIR   = Path(__file__).resolve().parent          # SA/SA-R3/
PROMPTS_DIR  = SCRIPT_DIR / "prompts"
BASE_PROMPT  = PROMPTS_DIR / "base.txt"
OUTPUT_DIR   = SCRIPT_DIR
OUTPUT_CSV   = OUTPUT_DIR / "SA_LGBTQ_Content.csv"

# =====================================================================
# CSV schema
# =====================================================================

CSV_COLUMNS    = ["text", "label", "shard_id"]
EXPECTED_LABEL = "SA_LGBTQ_Content"

# =====================================================================
# Prompt assembly & patching
# =====================================================================

def build_prompt(shard_id: str, n: int) -> str:
    """
    拼合 base.txt + {shard_id}.txt，替换 {SHARD_CONTENT} 和 {N}。
    """
    shard_file = PROMPTS_DIR / f"{shard_id}.txt"
    if not shard_file.exists():
        raise FileNotFoundError(f"Shard prompt not found: {shard_file}")

    base         = BASE_PROMPT.read_text(encoding="utf-8")
    shard_content = shard_file.read_text(encoding="utf-8").strip()

    p = base.replace("{SHARD_CONTENT}", shard_content)
    p = p.replace("{N}", str(n))
    return p

# =====================================================================
# API call
# =====================================================================

def call_api(prompt_text: str, label: str) -> str:
    base_url = (
        os.getenv("NIM_BASE_URL")
        or os.getenv("NVIDIA_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    ).rstrip("/")

    api_key = (
        os.getenv("NIM_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError("API key not found. Set NIM_API_KEY in SA/.env")

    endpoint = f"{base_url}/chat/completions"

    for attempt in range(1, MAX_RETRIES + 1):
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "أنت أداة لتوليد بيانات تدريب لسلامة المحتوى. "
                        "أخرج النصوص العربية المطلوبة فقط، كل نص في سطر واحد، "
                        "بدون ترقيم أو تسمية أو شرح أو أسطر فارغة."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            "temperature": TEMPERATURE,
            "top_p":       TOP_P,
            "max_tokens":  MAX_TOKENS,
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
                "Accept":        "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                time.sleep(2)
                return data["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"  [{label}] HTTP {e.code} (attempt {attempt}/{MAX_RETRIES}): {body[:500]}")
            if attempt < MAX_RETRIES:
                wait = 60 * attempt if e.code == 429 else RETRY_BACKOFF * attempt
                print(f"  [{label}] {'Rate-limited' if e.code == 429 else 'Retrying'} — waiting {wait}s...")
                time.sleep(wait)
                continue

        except Exception as exc:
            print(f"  [{label}] Error (attempt {attempt}/{MAX_RETRIES}): {repr(exc)[:400]}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"  [{label}] Waiting {wait}s before retry...")
                time.sleep(wait)

    raise RuntimeError(f"[{label}] API failed after {MAX_RETRIES} attempts.")

# =====================================================================
# Arabic validation
# =====================================================================

_AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

def _is_arabic(text: str) -> bool:
    """至少 40% 字符属于阿拉伯字符块。"""
    non_space = text.replace(" ", "")
    if not non_space:
        return False
    ar_chars = sum(1 for c in non_space if _AR_RE.match(c))
    return ar_chars / len(non_space) >= 0.4

# =====================================================================
# Plain-text parser
# =====================================================================

def parse_lines(raw: str, label: str) -> list[str]:
    """
    解析模型输出的纯文本，每行一条样本。
    - 去除 markdown 代码块标记
    - 去除行首编号（阿拉伯数字或东阿拉伯数字）
    - 过滤过短（<10词）、过长（>80词）、非阿拉伯语行
    """
    raw = re.sub(r"```[^\n]*\n?", "", raw).strip()
    records: list[str] = []

    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        # 去掉行首编号，例如 "1." / "١-" / "2) "
        line = re.sub(r"^[\d٠-٩]+[\.\-\)]\s*", "", line).strip()
        if not line:
            continue

        words = line.split()
        if len(words) < 10:
            print(f"  [{label}] Line {i}: too short ({len(words)} words) — skipped.")
            continue
        if len(words) > 80:
            print(f"  [{label}] Line {i}: too long ({len(words)} words) — truncated to 80.")
            line = " ".join(words[:80])

        if not _is_arabic(line):
            print(f"  [{label}] Line {i}: not Arabic — skipped.")
            continue

        records.append(line)

    return records

# =====================================================================
# Global state: CSV lock + global row counter
# =====================================================================

_csv_lock       = threading.Lock()
_global_counter = 0


def _init_global_counter() -> None:
    global _global_counter
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, encoding="utf-8", newline="") as f:
            _global_counter = sum(1 for _ in csv.DictReader(f))
    else:
        _global_counter = 0


def read_shard_count(shard_id: str) -> int:
    if not OUTPUT_CSV.exists():
        return 0
    with open(OUTPUT_CSV, encoding="utf-8", newline="") as f:
        return sum(
            1 for row in csv.DictReader(f)
            if (row.get("shard_id") or "").strip() == shard_id
        )

# =====================================================================
# Thread-safe flush
# =====================================================================

def flush_records(shard_id: str, texts: list[str], round_idx: int) -> int:
    global _global_counter

    if not texts:
        return 0

    with _csv_lock:
        # 重新统计，防止并发多写超量
        shard_existing = 0
        if OUTPUT_CSV.exists():
            with open(OUTPUT_CSV, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if (row.get("shard_id") or "").strip() == shard_id:
                        shard_existing += 1

        target    = SHARD_TARGETS[shard_id]
        remaining = target - shard_existing
        if remaining <= 0:
            return 0

        to_write = texts[:remaining]
        records: list[dict] = []
        for text in to_write:
            _global_counter += 1
            records.append({
                "text":     text,
                "label":    EXPECTED_LABEL,
                "shard_id": shard_id,
            })

        file_exists = OUTPUT_CSV.exists()
        with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(records)

    return len(to_write)

# =====================================================================
# Per-shard runner
# =====================================================================

def run_shard(shard_id: str) -> None:
    target    = SHARD_TARGETS[shard_id]
    existing  = read_shard_count(shard_id)
    remaining = target - existing

    if remaining <= 0:
        print(f"[{shard_id}] Already at target ({existing}/{target}). Skipped.")
        return

    print(f"[{shard_id}] ▶ Start (target={target}, existing={existing}, need={remaining})")

    shard_total = existing
    round_idx   = 0

    while shard_total < target:
        n     = min(BATCH_SIZE, target - shard_total)
        label = f"{shard_id}/r{round_idx}"
        print(f"[{label}] Requesting {n} lines...")

        patched = build_prompt(shard_id, n)
        raw     = call_api(patched, label)
        texts   = parse_lines(raw, label)

        print(f"[{label}] Parsed {len(texts)} valid lines")

        written     = flush_records(shard_id, texts, round_idx)
        shard_total += written
        print(f"[{label}] +{written} written → 本轮合计 {shard_total}/{target}")

        round_idx += 1

    print(f"[{shard_id}] ✓ Complete")

# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 62)
    print("  SA LGBTQ Content Synthetic Data Generator")
    print("=" * 62)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Shards     : {SHARDS_TO_RUN}")
    print(f"  Batch size : {BATCH_SIZE} per API call")
    print(f"  Workers    : {MAX_WORKERS} concurrent")
    print(f"  Prompts    : {PROMPTS_DIR}/base.txt + {{shard_id}}.txt")
    print(f"  Output     : {OUTPUT_CSV}")
    print("=" * 62)
    print()

    if not BASE_PROMPT.exists():
        raise FileNotFoundError(f"Base prompt not found: {BASE_PROMPT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _init_global_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(run_shard, shard_id): shard_id
            for shard_id in SHARDS_TO_RUN
        }

        for fut in concurrent.futures.as_completed(future_map):
            shard_id = future_map[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"[{shard_id}] ✗ FAILED: {exc}")

    print(f"\n所有 shard 完成。结果写入 {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
