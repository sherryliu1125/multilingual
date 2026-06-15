#!/usr/bin/env python3
"""
Continue BR R2 synthetic generation only for labels still needed to balance R2 to 2000.

Flow:
1. Read current BR R2 full data.
2. Try to fill deficits from dropped synthetic first.
3. Try to fill remaining deficits from existing add_data synthetic files.
4. If still short, call the same NIW model/prompt/filter style as generate_br_synthetic_R2_notebook_cell.py.
5. Save generated supplement, final balanced data, added details, and label distribution report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =========================
# Config
# =========================

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd() / "BR"

BASE_DIR = str(PROJECT_ROOT / "synthetic_R2")

MODEL_NAME = "qwen/qwen3-next-80b-a3b-instruct"

BATCH_SIZE = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 6

TEMPERATURE = 0.9
TOP_P = 0.95
MAX_TOKENS = 3500

TARGET_MIN_COUNT = 2000

GLOBAL_LABELS = [
    "Dangerous_Content",
    "Harassment",
    "Hate_Speech",
    "Sexually_Explicit_Information",
    "Politically_Sensitive_Topics",
    "Cybersecurity_Malware",
    "BR_State_Security_Democratic_Order",
    "safe",
]

GENERATE_LABELS = [
    "Politically_Sensitive_Topics",
    "Sexually_Explicit_Information",
    "BR_State_Security_Democratic_Order",
]

LABEL_NORMALIZE_MAP = {
    "Politically Sensitive Topics": "Politically_Sensitive_Topics",
    "Sexually Explicit Information": "Sexually_Explicit_Information",
    "BR State Security Democratic Order": "BR_State_Security_Democratic_Order",
    "Cybersecurity Malware": "Cybersecurity_Malware",
}

BR_R2_DATA_DIR = PROJECT_ROOT / "BR_R2_data"
BR_ADD_DATA_DIR = PROJECT_ROOT / "add_data"
BR_SYNTHETIC_R2_DIR = PROJECT_ROOT / "synthetic_R2"

MAIN_PATH = BR_R2_DATA_DIR / "br_annotation_R2_full.csv"
DROPPED_SYN_PATH = BR_R2_DATA_DIR / "br_annotation_R2_dropped_synthetic.csv"
ADD_SYN_PATHS = [
    BR_ADD_DATA_DIR / "br_synthetic_R2_valid_dedup.csv",
    BR_ADD_DATA_DIR / "br_cybersecurity_malware_valid_dedup.csv",
]

SUPPLEMENT_RAW_CSV = BR_SYNTHETIC_R2_DIR / "br_r2_balance_supplement_raw.csv"
SUPPLEMENT_ERROR_CSV = BR_SYNTHETIC_R2_DIR / "br_r2_balance_supplement_errors.csv"
SUPPLEMENT_FINAL_CSV = BR_SYNTHETIC_R2_DIR / "br_r2_balance_supplement_valid_dedup.csv"

OUT_BALANCED_PATH = BR_R2_DATA_DIR / "br_annotation_R2_full_balanced2000.csv"
OUT_ADDED_PATH = BR_R2_DATA_DIR / "br_annotation_R2_added_to_balance2000.csv"
OUT_DIST_PATH = BR_R2_DATA_DIR / "br_annotation_R2_balanced2000_label_distribution.csv"

RAW_COLUMNS = [
    "clean_text",
    "final_category",
    "generation_model",
    "batch_id",
    "item_id",
    "raw_output",
    "parse_error",
    "created_at",
]

FINAL_COLUMNS = ["text", "llm_label", "source"]

LABEL_SPECS = {
    "BR_State_Security_Democratic_Order": """
Generate synthetic Brazil-local content that clearly matches the label:
BR_State_Security_Democratic_Order.

Content direction:
- Denying the legitimacy of Brazilian elections while calling for action
- Demanding military intervention, coup, closing Congresso, STF, TSE, or courts
- Encouraging confrontation against Brazilian democratic institutions
- Inciting rejection of election results or anti-democratic mobilization
- Attacking Brazilian democracy while encouraging real-world resistance

Boundaries:
- Must be Brazil-local context.
- May mention Brasil, Brasilia, STF, TSE, Congresso, Exercito, eleicao, urna, golpe, intervencao militar.
- Do not write ordinary policy criticism like taxes, economy, inflation, corruption complaints, or generic anti-president comments.
- Do not write international politics here.
- The text must still clearly match the target label.
- Avoid operational details such as exact meeting locations, schedules, addresses, weapon instructions, evasion tactics, or step-by-step plans.
""".strip(),
    "Sexually_Explicit_Information": """
Generate synthetic adult sexually explicit pt-BR content that clearly matches the label:
Sexually_Explicit_Information.

Content direction:
- Adult sexual invitation, explicit adult sexual implication, sexting, hookup talk
- Brazilian Portuguese adult slang, flirty explicit messages, emoji sexual implication
- Text should clearly show adult sexual intent, arousal, or explicit adult content

Boundaries:
- Only consenting adults.
- Do not mention minors, school age, teenagers, children, or age ambiguity.
- Do not generate rape, drugging, coercion, hidden camera, blackmail, revenge porn, or non-consensual scenarios.
- Do not generate real contact info, real addresses, IDs, bank card numbers, or phone numbers.
- Do not write pure insults using sexual words.
- Do not write medical education or literary prose.
- The text must still clearly match the target label.
""".strip(),
    "Politically_Sensitive_Topics": """
Generate synthetic non-Brazil international politically sensitive content that clearly matches the label:
Politically_Sensitive_Topics.

Content direction:
- Overseas election manipulation, international political rumors, foreign regime instability
- War or conflict narratives, border clashes, foreign political violence
- Malicious political rumors about non-Brazil leaders
- Overseas mass protests, coups, massacres, election fraud rumors, war propaganda

Boundaries:
- Do not write Brazilian domestic politics.
- Do not write neutral international news summaries.
- Must include political sensitivity, instability, rumors, incitement, conflict, or public-safety risk.
- Do not generate terrorist recruitment, weapon-making, concrete violence tutorial, or operational attack instructions.
- The text must still clearly match the target label.
- Avoid operational details such as exact meeting locations, schedules, addresses, weapon instructions, evasion tactics, or step-by-step plans.
""".strip(),
}


# =========================
# Args / Env / NIW API
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate only the remaining BR R2 supplement needed for balanced2000."
    )
    parser.add_argument("--main-path", type=Path, default=MAIN_PATH)
    parser.add_argument("--dropped-synthetic-path", type=Path, default=DROPPED_SYN_PATH)
    parser.add_argument("--add-synthetic-path", type=Path, nargs="*", default=ADD_SYN_PATHS)
    parser.add_argument("--supplement-raw-csv", type=Path, default=SUPPLEMENT_RAW_CSV)
    parser.add_argument("--supplement-error-csv", type=Path, default=SUPPLEMENT_ERROR_CSV)
    parser.add_argument("--supplement-final-csv", type=Path, default=SUPPLEMENT_FINAL_CSV)
    parser.add_argument("--out-balanced-path", type=Path, default=OUT_BALANCED_PATH)
    parser.add_argument("--out-added-path", type=Path, default=OUT_ADDED_PATH)
    parser.add_argument("--out-dist-path", type=Path, default=OUT_DIST_PATH)
    parser.add_argument("--target-min-count", type=int, default=TARGET_MIN_COUNT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-generation-rounds", type=int, default=200)
    return parser.parse_args()


def load_local_env_files():
    candidate_paths = [
        Path.cwd() / ".env",
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
        PROJECT_ROOT.parent / "MX" / ".env",
        PROJECT_ROOT.parent / "BR" / ".env",
        Path(BASE_DIR) / ".env",
    ]

    loaded_paths = []
    for env_path in candidate_paths:
        if not env_path.exists():
            continue
        loaded_paths.append(env_path)
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value

    if loaded_paths:
        print("Loaded env files:")
        for env_path in loaded_paths:
            print(f"  {env_path}")


def call_niw_model(prompt, model_name=MODEL_NAME, timeout=180):
    base_url = (
        os.getenv("NIW_BASE_URL")
        or os.getenv("NIM_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    ).rstrip("/")

    api_key = (
        os.getenv("NIW_API_KEY")
        or os.getenv("NIM_API_KEY")
        or os.getenv("MX_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
        or ""
    )

    if not api_key:
        raise RuntimeError(
            "Missing API key. Set NIW_API_KEY, NIM_API_KEY, MX_API_KEY, or NVIDIA_API_KEY in env/.env."
        )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate synthetic Brazilian Portuguese content-safety training data. "
                    "Return only valid JSON. No reasoning, no explanation, no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NIW HTTPError {e.code}: {body[:1200]}") from e
    except Exception as e:
        raise RuntimeError(f"NIW request failed: {repr(e)}") from e

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected NIW response shape: {json.dumps(data, ensure_ascii=False)[:1200]}")


# =========================
# Standardize / Balance IO
# =========================

def read_csv_or_empty(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])


def read_csv_or_fail(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{source_name} not found: {path}\n"
            "This script is strict: it will not fall back to BR/data/annotations_br/br_annotation_full.csv. "
            "Put the real R2 full CSV under BR/BR_R2_data/br_annotation_R2_full.csv, "
            "or pass --main-path explicitly."
        )
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_candidate_csv_or_empty(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: {source_name} not found, using empty candidates: {path}")
        return pd.DataFrame(columns=["clean_text", "final_category"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace(LABEL_NORMALIZE_MAP)


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


def standardize_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    if "clean_text" not in df.columns and "text" in df.columns:
        df["clean_text"] = df["text"]
    if "final_category" not in df.columns and "llm_label" in df.columns:
        df["final_category"] = df["llm_label"]

    missing = [col for col in ["clean_text", "final_category"] if col not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns after standardization: {missing}")

    df["clean_text"] = df["clean_text"].astype(str)
    df["final_category"] = normalize_label_series(df["final_category"])
    df["_clean_text_key"] = df["clean_text"].map(normalize_text)
    return df


def prepare_candidate_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = standardize_columns(df, source_name)
    df = df[df["_clean_text_key"].ne("")]
    df = df[df["final_category"].isin(GLOBAL_LABELS)]
    return df


def label_distribution(df: pd.DataFrame) -> pd.Series:
    return df["final_category"].value_counts().reindex(GLOBAL_LABELS, fill_value=0)


def compute_deficits(dist: pd.Series, target_min_count: int) -> dict[str, int]:
    return {
        label: max(0, target_min_count - int(dist.get(label, 0)))
        for label in GLOBAL_LABELS
    }


def take_candidates(
    candidate_df: pd.DataFrame,
    current_text_keys: set[str],
    remaining_deficits: dict[str, int],
    source_bucket: str,
    source_file: Path,
) -> pd.DataFrame:
    picked_parts = []
    candidate_df = candidate_df.copy()
    candidate_df = candidate_df[~candidate_df["_clean_text_key"].isin(current_text_keys)]
    candidate_df = candidate_df.drop_duplicates(subset=["_clean_text_key"], keep="first")

    for label in GLOBAL_LABELS:
        need = remaining_deficits.get(label, 0)
        if need <= 0:
            continue

        label_candidates = candidate_df[candidate_df["final_category"].eq(label)]
        picked = label_candidates.head(need).copy()
        if picked.empty:
            continue

        picked["balance_source"] = source_bucket
        picked["balance_source_file"] = str(source_file)
        picked_parts.append(picked)

        picked_keys = set(picked["_clean_text_key"])
        current_text_keys.update(picked_keys)
        remaining_deficits[label] -= len(picked)
        candidate_df = candidate_df[~candidate_df["_clean_text_key"].isin(picked_keys)]

    if not picked_parts:
        return pd.DataFrame(columns=list(candidate_df.columns) + ["balance_source", "balance_source_file"])

    return pd.concat(picked_parts, ignore_index=True)


def simulate_balance(
    main_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
    add_candidate_dfs: list[tuple[Path, pd.DataFrame]],
    supplement_df: pd.DataFrame,
    target_min_count: int,
) -> tuple[pd.DataFrame, dict[str, int], defaultdict[str, int], defaultdict[str, int], defaultdict[str, int]]:
    original_dist = label_distribution(main_df)
    remaining = compute_deficits(original_dist, target_min_count)
    current_text_keys = set(main_df["_clean_text_key"])
    added_parts = []
    from_dropped = defaultdict(int)
    from_add = defaultdict(int)
    from_generated = defaultdict(int)

    picked_dropped = take_candidates(
        dropped_df,
        current_text_keys,
        remaining,
        "dropped_synthetic",
        DROPPED_SYN_PATH,
    )
    if not picked_dropped.empty:
        added_parts.append(picked_dropped)
        for label, count in picked_dropped["final_category"].value_counts().items():
            from_dropped[label] += int(count)

    for add_path, add_df in add_candidate_dfs:
        picked_add = take_candidates(
            add_df,
            current_text_keys,
            remaining,
            "add_data_synthetic",
            add_path,
        )
        if not picked_add.empty:
            added_parts.append(picked_add)
            for label, count in picked_add["final_category"].value_counts().items():
                from_add[label] += int(count)

    picked_generated = take_candidates(
        supplement_df,
        current_text_keys,
        remaining,
        "generated_balance_supplement",
        SUPPLEMENT_FINAL_CSV,
    )
    if not picked_generated.empty:
        added_parts.append(picked_generated)
        for label, count in picked_generated["final_category"].value_counts().items():
            from_generated[label] += int(count)

    if added_parts:
        added_df = pd.concat(added_parts, ignore_index=True)
    else:
        added_df = pd.DataFrame(columns=list(main_df.columns) + ["balance_source", "balance_source_file"])

    return added_df, remaining, from_dropped, from_add, from_generated


# =========================
# Generation parsing/filtering
# =========================

def strip_code_fence(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_array(text):
    text = strip_code_fence(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        raise ValueError("JSON root is not a list")
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        obj = json.loads(text[start:end + 1])
        if isinstance(obj, list):
            return obj

    raise ValueError("Could not parse JSON array")


def is_chinese_heavy(text):
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", str(text or "")))
    return chinese_count >= 2


def is_obviously_english_response(text):
    s = " " + str(text or "").strip().lower() + " "
    if not s.strip():
        return False

    english_phrases = [
        " as an ai ",
        " as a language model ",
        " i cannot ",
        " i can't ",
        " here are ",
        " below are ",
        " the following ",
        " this text ",
        " these examples ",
        " i'm sorry ",
    ]
    if any(p in s for p in english_phrases):
        return True

    letters = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", s)
    if not letters:
        return False

    ascii_letters = re.findall(r"[A-Za-z]", s)
    ascii_ratio = len(ascii_letters) / max(len(letters), 1)

    pt_markers = [
        " que ", " não ", " nao ", " pra ", " você ", " voce ", " vcs ",
        " gente ", " brasil ", " eleição ", " eleicao ", " urna ", " congresso ",
        " hoje ", " aqui ", " isso ", " tá ", " ta ", " pelo ", " pela ",
        " com ", " sem ", " mais ", " muito ", " agora ", " quero ",
    ]
    pt_hits = sum(1 for p in pt_markers if p in s)

    en_markers = [
        " the ", " and ", " you ", " your ", " with ", " government ",
        " election ", " military ", " court ", " president ", " people ",
    ]
    en_hits = sum(1 for p in en_markers if p in s)

    return ascii_ratio > 0.94 and en_hits >= 3 and pt_hits == 0


def is_refusal_text(text):
    s = " " + str(text or "").strip().lower() + " "
    refusal_patterns = [
        " as an ai ",
        " as a language model ",
        " como uma ia ",
        " como inteligência artificial ",
        " como inteligencia artificial ",
        " não posso ajudar a gerar ",
        " nao posso ajudar a gerar ",
        " não posso gerar conteúdo ",
        " nao posso gerar conteudo ",
        " não posso criar esse tipo ",
        " nao posso criar esse tipo ",
        " não posso fornecer ",
        " nao posso fornecer ",
        " não posso atender ",
        " nao posso atender ",
    ]
    return any(p in s for p in refusal_patterns)


def quality_filter_raw(row, target_label):
    text = normalize_text(row.get("clean_text", ""))
    label = str(row.get("final_category", "")).strip()

    if not text:
        return False, "empty_clean_text"
    if label != target_label:
        return False, "wrong_final_category"
    if len(text) < 40 or len(text) > 240:
        return False, f"bad_length_{len(text)}"
    if is_chinese_heavy(text):
        return False, "chinese_detected"
    if is_obviously_english_response(text):
        return False, "obviously_english_response"
    if is_refusal_text(text):
        return False, "refusal_text"

    return True, ""


def read_raw_csv(path: Path) -> pd.DataFrame:
    df = read_csv_or_empty(path, RAW_COLUMNS)
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[RAW_COLUMNS]


def append_raw_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if not file_exists or path.stat().st_size == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in RAW_COLUMNS})


def build_final_dedup(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if raw_df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS), 0

    rows = []
    for _, r in raw_df.iterrows():
        if str(r.get("parse_error", "")).strip():
            continue

        label = str(r.get("final_category", "")).strip()
        if label not in GENERATE_LABELS:
            continue

        raw_row = {
            "clean_text": normalize_text(r.get("clean_text", "")),
            "final_category": label,
        }
        ok, _reason = quality_filter_raw(raw_row, label)
        if not ok:
            continue

        rows.append({
            "text": raw_row["clean_text"],
            "llm_label": label,
            "source": "llm_generated_balance_supplement",
        })

    final_df = pd.DataFrame(rows, columns=FINAL_COLUMNS)
    before = len(final_df)

    if not final_df.empty:
        final_df["_dedup_key"] = final_df["text"].map(normalize_text)
        final_df = final_df.drop_duplicates(subset=["_dedup_key"], keep="first")
        final_df = final_df.drop(columns=["_dedup_key"])

    duplicate_count = before - len(final_df)
    return final_df, duplicate_count


def save_supplement_final(raw_path: Path, final_path: Path) -> pd.DataFrame:
    raw_df = read_raw_csv(raw_path)
    final_df, duplicate_count = build_final_dedup(raw_df)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(final_path, index=False, encoding="utf-8")
    print(f"supplement_valid_rows: {len(final_df)}, supplement_duplicate_removed: {duplicate_count}")
    return final_df


def make_batch_id(label: str, index: int) -> str:
    return f"{label}__balance_supplement_batch_{index:06d}"


def get_used_batch_ids(raw_df: pd.DataFrame, err_df: pd.DataFrame) -> set[str]:
    used = set()
    if not raw_df.empty:
        used.update(raw_df["batch_id"].dropna().astype(str).tolist())
    if not err_df.empty:
        used.update(err_df["batch_id"].dropna().astype(str).tolist())
    return used


def next_batch_index(label: str, used_ids: set[str]) -> int:
    prefix = f"{label}__balance_supplement_batch_"
    max_idx = -1
    for bid in used_ids:
        bid = str(bid)
        if bid.startswith(prefix):
            tail = bid.replace(prefix, "", 1)
            if tail.isdigit():
                max_idx = max(max_idx, int(tail))
    return max_idx + 1


def sample_existing_texts_for_prompt(
    label: str,
    main_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
    add_candidate_dfs: list[tuple[Path, pd.DataFrame]],
    supplement_df: pd.DataFrame,
    k: int = 8,
) -> list[str]:
    values = []
    for df in [main_df, dropped_df, supplement_df] + [x[1] for x in add_candidate_dfs]:
        if df.empty:
            continue
        sub = df[df["final_category"] == label] if "final_category" in df.columns else df[df["llm_label"] == label]
        col = "clean_text" if "clean_text" in sub.columns else "text"
        values.extend(sub[col].dropna().astype(str).map(normalize_text).tolist())

    values = [x for x in values if x]
    if len(values) <= k:
        return values
    return random.sample(values, k)


def build_prompt(label: str, n: int, existing_samples=None) -> str:
    existing_samples = existing_samples or []
    style_hint = random.choice([
        "Misture comentarios de rede social, forum, grupo de mensagem e resposta curta de timeline.",
        "Varie pontuacao, girias brasileiras, abreviacoes, emojis e pequenos erros de digitacao.",
        "Faca parecer amostra real de crawler: natural, curta, com tom de internet, sem cara de template.",
        "Use estruturas diferentes entre os itens; nao comece todos do mesmo jeito.",
        "Inclua variacao de registro: raiva, ironia, deboche, flerte, boato, convocacao vaga ou desabafo, conforme o label.",
    ])

    avoid_block = ""
    if existing_samples:
        avoid_block = (
            "\nAvoid copying or closely paraphrasing these existing samples/styles:\n"
            + "\n".join(f"- {x}" for x in existing_samples[:8])
        )

    return f"""
Generate exactly N={n} synthetic short-text training items.

Current label:
{label}

Label definition and boundaries:
{LABEL_SPECS[label]}

Hard requirements:
- Output must be Brazilian Portuguese pt-BR.
- Do not use Chinese.
- Do not use English, except common internet/platform loanwords naturally used in Brazil.
- Each clean_text should be around 80-180 characters.
- Text should look like realistic social media, forum, comment, or group-chat content.
- The content must clearly match the current target label.
- Do not output explanations, reasoning, classification rationale, notes, or long paragraphs.
- Do not output markdown.
- Do not repeat the same structure across items.
- You may include Brazilian slang, abbreviations, emojis, typos, caps, sarcasm, and internet tone.
- Do not generate real personal privacy, real contact info, real ID numbers, real bank card numbers, or executable malicious code.
- final_category must be exactly: {label}
- {style_hint}

Output format:
Return strictly a JSON array.
Each item must be exactly:
{{"clean_text": "...", "final_category": "{label}"}}

No extra keys.
No markdown code fence.
No prose before or after the JSON array.
{avoid_block}
""".strip()


def parse_batch_to_raw_rows(raw_output: str, label: str, batch_id: str) -> list[dict]:
    arr = extract_json_array(raw_output)
    rows = []
    now = datetime.now(timezone.utc).isoformat()

    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            continue

        row = {
            "clean_text": normalize_text(item.get("clean_text", "")),
            "final_category": str(item.get("final_category", "")).strip(),
            "generation_model": MODEL_NAME,
            "batch_id": batch_id,
            "item_id": f"{batch_id}__item_{i:02d}",
            "raw_output": raw_output,
            "parse_error": "",
            "created_at": now,
        }

        ok, _reason = quality_filter_raw(row, label)
        if ok:
            rows.append(row)

    return rows


def write_error_row(error_path: Path, batch_id: str, label: str, raw_output: str, error_message: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    append_raw_rows(error_path, [{
        "clean_text": "",
        "final_category": label,
        "generation_model": MODEL_NAME,
        "batch_id": batch_id,
        "item_id": f"{batch_id}__error",
        "raw_output": str(raw_output or ""),
        "parse_error": str(error_message),
        "created_at": now,
    }])


def generate_one_batch(
    label: str,
    batch_id: str,
    n: int,
    args: argparse.Namespace,
    main_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
    add_candidate_dfs: list[tuple[Path, pd.DataFrame]],
    supplement_df: pd.DataFrame,
) -> bool:
    prompt = build_prompt(
        label=label,
        n=n,
        existing_samples=sample_existing_texts_for_prompt(
            label, main_df, dropped_df, add_candidate_dfs, supplement_df, k=8
        ),
    )

    last_raw = ""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[{label}] request {batch_id}, n={n}, attempt {attempt}/{MAX_RETRIES}")
            raw_output = call_niw_model(prompt, MODEL_NAME)
            last_raw = raw_output
            rows = parse_batch_to_raw_rows(raw_output, label, batch_id)
            if not rows:
                raise ValueError("Parsed output but zero rows passed quality filter")

            append_raw_rows(args.supplement_raw_csv, rows)
            print(f"[{label}] saved {len(rows)} valid raw rows from {batch_id}")
            return True
        except Exception as e:
            last_error = e
            print(f"[{label}] {batch_id} failed attempt {attempt}: {repr(e)}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    write_error_row(args.supplement_error_csv, batch_id, label, last_raw, repr(last_error))
    print(f"[{label}] wrote error row for {batch_id}")
    return False


# =========================
# Reports / Main
# =========================

def print_label_counts(title: str, counts) -> None:
    print("\n" + "=" * 80)
    print(title)
    for label in GLOBAL_LABELS:
        print(f"{label}: {int(counts.get(label, 0))}")


def save_balanced_outputs(
    main_df: pd.DataFrame,
    added_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.Series:
    main_output_cols = [col for col in main_df.columns if col != "_clean_text_key"]
    balanced_df = pd.concat(
        [
            main_df[main_output_cols],
            added_df.reindex(columns=main_output_cols),
        ],
        ignore_index=True,
    )

    added_detail_cols = main_output_cols + ["balance_source", "balance_source_file"]
    added_detail_df = added_df.reindex(columns=added_detail_cols)

    final_check_df = standardize_columns(balanced_df, "balanced output")
    final_dist = label_distribution(final_check_df)
    dist_report_df = final_dist.rename_axis("final_category").reset_index(name="count")
    dist_report_df["target_min_count"] = args.target_min_count
    dist_report_df["gap_after_balance"] = dist_report_df["count"].apply(
        lambda count: max(0, args.target_min_count - int(count))
    )

    args.out_balanced_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_added_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_dist_path.parent.mkdir(parents=True, exist_ok=True)

    balanced_df.to_csv(args.out_balanced_path, index=False, encoding="utf-8-sig")
    added_detail_df.to_csv(args.out_added_path, index=False, encoding="utf-8-sig")
    dist_report_df.to_csv(args.out_dist_path, index=False, encoding="utf-8-sig")

    return final_dist


def load_all_inputs(args: argparse.Namespace):
    main_df = standardize_columns(read_csv_or_fail(args.main_path, "main R2 data"), "main R2 data")
    dropped_df = prepare_candidate_df(
        read_candidate_csv_or_empty(args.dropped_synthetic_path, "dropped synthetic"),
        "dropped synthetic",
    )

    add_candidate_dfs = []
    for add_path in args.add_synthetic_path:
        if not add_path.exists():
            print(f"WARNING: add_data candidate file not found, skipped: {add_path}")
            continue
        add_df = prepare_candidate_df(
            read_csv_or_fail(add_path, f"add_data synthetic: {add_path.name}"),
            f"add_data synthetic: {add_path.name}",
        )
        add_candidate_dfs.append((add_path, add_df))

    supplement_final_df = save_supplement_final(args.supplement_raw_csv, args.supplement_final_csv)
    supplement_df = prepare_candidate_df(supplement_final_df, "generated supplement")
    return main_df, dropped_df, add_candidate_dfs, supplement_df


def main() -> None:
    args = parse_args()
    load_local_env_files()

    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"SUPPLEMENT_RAW_CSV: {args.supplement_raw_csv}")
    print(f"SUPPLEMENT_FINAL_CSV: {args.supplement_final_csv}")

    args.supplement_raw_csv.parent.mkdir(parents=True, exist_ok=True)
    args.supplement_error_csv.parent.mkdir(parents=True, exist_ok=True)
    args.supplement_final_csv.parent.mkdir(parents=True, exist_ok=True)

    main_df, dropped_df, add_candidate_dfs, supplement_df = load_all_inputs(args)

    original_dist = label_distribution(main_df)
    print_label_counts("Original R2 label distribution", original_dist)

    raw_df = read_raw_csv(args.supplement_raw_csv)
    err_df = read_raw_csv(args.supplement_error_csv)
    used_ids = get_used_batch_ids(raw_df, err_df)

    for generation_round in range(1, args.max_generation_rounds + 1):
        added_df, remaining, from_dropped, from_add, from_generated = simulate_balance(
            main_df,
            dropped_df,
            add_candidate_dfs,
            supplement_df,
            args.target_min_count,
        )

        print_label_counts(f"Remaining deficits before generation round {generation_round}", remaining)

        needed_generate_labels = [
            label for label in GENERATE_LABELS
            if remaining.get(label, 0) > 0
        ]
        if not needed_generate_labels:
            print("No remaining generation deficit. Writing balanced outputs.")
            break

        label = max(needed_generate_labels, key=lambda x: remaining[x])
        need = remaining[label]
        n = min(args.batch_size, max(need, 5))

        idx = next_batch_index(label, used_ids)
        batch_id = make_batch_id(label, idx)
        while batch_id in used_ids:
            idx += 1
            batch_id = make_batch_id(label, idx)
        used_ids.add(batch_id)

        generate_one_batch(
            label,
            batch_id,
            n,
            args,
            main_df,
            dropped_df,
            add_candidate_dfs,
            supplement_df,
        )

        supplement_final_df = save_supplement_final(args.supplement_raw_csv, args.supplement_final_csv)
        supplement_df = prepare_candidate_df(supplement_final_df, "generated supplement")

    added_df, remaining, from_dropped, from_add, from_generated = simulate_balance(
        main_df,
        dropped_df,
        add_candidate_dfs,
        supplement_df,
        args.target_min_count,
    )
    final_dist = save_balanced_outputs(main_df, added_df, args)

    print_label_counts("Added from dropped synthetic", from_dropped)
    print_label_counts("Added from add_data synthetic", from_add)
    print_label_counts("Added from generated balance supplement", from_generated)
    print_label_counts("Final label distribution", final_dist)

    print("\n" + "=" * 80)
    print("Final warnings")
    has_warning = False
    for label in GLOBAL_LABELS:
        final_count = int(final_dist.get(label, 0))
        if final_count < args.target_min_count:
            has_warning = True
            print(
                f"WARNING: {label} still below {args.target_min_count}: "
                f"{final_count}, missing {args.target_min_count - final_count}"
            )
    if not has_warning:
        print(f"All labels reached at least {args.target_min_count}.")

    print("\n" + "=" * 80)
    print("Saved files")
    print(f"Supplement raw:          {args.supplement_raw_csv}")
    print(f"Supplement errors:       {args.supplement_error_csv}")
    print(f"Supplement valid dedup:  {args.supplement_final_csv}")
    print(f"Balanced training data:  {args.out_balanced_path}")
    print(f"Added sample details:    {args.out_added_path}")
    print(f"Distribution report:     {args.out_dist_path}")


main()
