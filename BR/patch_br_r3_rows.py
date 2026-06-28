#!/usr/bin/env python3
"""
BR R3 Row Patcher
针对指定 ID 重新生成并覆盖 CSV 中的对应行。

使用方式：
  python patch_br_r3_rows.py
"""

import os
import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# =====================================================================
# ★ USER CONFIG
# =====================================================================

# 要修复的文件（去掉 .csv）
SHARD_ID = "BR_SEX_003"

# 要替换的 ID 列表
TARGET_IDS = [
    "BR_SEX_003_0028",
    "BR_SEX_003_0043",
    "BR_SEX_003_0047",
]

# =====================================================================
# Paths & API（不用改）
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "BR-R3"
CSV_PATH   = OUTPUT_DIR / f"{SHARD_ID}.csv"

MODEL_NAME    = "mistralai/mistral-large-3-675b-instruct-2512"
TEMPERATURE   = 0.9
TOP_P         = 0.95
MAX_TOKENS    = 4000
MAX_RETRIES   = 3
RETRY_BACKOFF = 8

CSV_COLUMNS = [
    "id", "country", "language", "target_label",
    "source_style", "scenario", "explicitness",
    "length_level", "tone", "structure", "obfuscation",
    "pattern_id", "text",
]

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

_load_dotenv(SCRIPT_DIR / ".env")

# =====================================================================
# Targeted generation prompt
# =====================================================================

PATCH_PROMPT = f"""You are a multilingual content-safety synthetic data generator.

Task:
Generate exactly 3 replacement samples for a Brazilian Portuguese content-safety training dataset.

Country: BR
Language: Brazilian Portuguese / pt-BR
Target label: Sexually_Explicit_Information
Shard context: BR_SEX_003 (adult livestream / video chat)

These 3 samples are replacements for rejected samples with these problems:
- Sample A: text had abnormal character spacing, was too ambiguous to train on
- Sample B: too ambiguous, could be judged as normal flirting rather than sexual content
- Sample C: written in English instead of Brazilian Portuguese

Replacement requirements:
- ALL 3 must be in natural Brazilian Portuguese (pt-BR). Zero English sentences allowed.
- ALL 3 must have clear adult sexual intent that is unambiguous to a content-safety classifier.
- ALL 3 must feel like realistic live_chat or comment messages from Brazilian adult livestreams.
- Vary the sexual situation across the 3 samples (do not repeat the same setup).
- explicitness must be semi_explicit for all 3 (not implicit — the previous implicit samples were too weak).
- No spacing obfuscation. Text must be normally readable.
- Do not use full English sentences. Short English words mixed into Portuguese (e.g. "live", "chat", "hot") are acceptable.
- Do not generate non-consensual, coercive, revenge, hidden-camera, or minor-related content.

Output exactly 3 JSONL records with these fixed field values:
country = "BR"
language = "pt-BR"
target_label = "Sexually_Explicit_Information"
source_style = "live_chat"
explicitness = "semi_explicit"

Vary freely: scenario, length_level, tone, structure, obfuscation, pattern_id, text.

ID values (use exactly):
"{TARGET_IDS[0]}", "{TARGET_IDS[1]}", "{TARGET_IDS[2]}"

Output format:
- Exactly 3 JSONL lines.
- No markdown, no explanation, no extra text.
- Each line is a valid JSON object with all fields: id, country, language, target_label, source_style, scenario, explicitness, length_level, tone, structure, obfuscation, pattern_id, text.
"""

# =====================================================================
# API call
# =====================================================================

def call_api(prompt_text: str) -> str:
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
        raise RuntimeError("API key not found. Set NVIDIA_API_KEY or NIM_API_KEY.")

    endpoint = f"{base_url}/chat/completions"

    for attempt in range(1, MAX_RETRIES + 1):
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a content-safety synthetic data generator. "
                        "Output only valid JSONL, one JSON object per line. "
                        "No markdown, no explanation, no extra text."
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
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"  HTTP {e.code} (attempt {attempt}/{MAX_RETRIES}): {body[:400]}")
        except Exception as e:
            print(f"  Error (attempt {attempt}/{MAX_RETRIES}): {repr(e)[:400]}")

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF * attempt
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"API failed after {MAX_RETRIES} attempts.")

# =====================================================================
# Parse & validate
# =====================================================================

def parse_new_records(raw: str) -> dict[str, dict]:
    """Parse JSONL, return dict keyed by id."""
    result = {}
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  Line {i}: JSON error — {e}")
            continue
        if not isinstance(obj, dict):
            continue
        rec_id = obj.get("id", "")
        if rec_id not in TARGET_IDS:
            print(f"  Line {i}: unexpected id '{rec_id}', skipped.")
            continue
        if not str(obj.get("text", "")).strip():
            print(f"  Line {i}: empty text, skipped.")
            continue
        result[rec_id] = obj
        print(f"  ✓ {rec_id}: {str(obj.get('text',''))[:80]}")
    return result

# =====================================================================
# CSV patch
# =====================================================================

def patch_csv(new_records: dict[str, dict]) -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    rows = []
    patched_ids = set()
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] in new_records:
                rows.append(new_records[row["id"]])
                patched_ids.add(row["id"])
                print(f"  Replaced: {row['id']}")
            else:
                rows.append(row)

    missed = set(new_records.keys()) - patched_ids
    if missed:
        print(f"  WARNING: IDs not found in CSV (not replaced): {missed}")

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV updated: {CSV_PATH}")
    print(f"  Total rows : {len(rows)}")

# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 60)
    print("  BR R3 Row Patcher")
    print("=" * 60)
    print(f"  Shard  : {SHARD_ID}")
    print(f"  Fixing : {TARGET_IDS}")
    print()

    print("Calling API...")
    raw = call_api(PATCH_PROMPT)

    print("\nParsing new records:")
    new_records = parse_new_records(raw)

    if len(new_records) < len(TARGET_IDS):
        missing = set(TARGET_IDS) - set(new_records.keys())
        print(f"\nWARNING: Only got {len(new_records)}/{len(TARGET_IDS)} records.")
        print(f"Missing IDs: {missing}")
        if not new_records:
            print("Nothing to patch. Exiting.")
            return
        ans = input("Patch with partial results? (y/n): ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    print("\nPatching CSV:")
    patch_csv(new_records)
    print("\nDone.")


if __name__ == "__main__":
    main()
