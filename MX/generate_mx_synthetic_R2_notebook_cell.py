#!/usr/bin/env python3
"""
Generate MX synthetic R2 add_data for low-sample LoRA labels.

This intentionally follows the already-tested BR synthetic R2 pattern:
OpenAI-compatible NIW client, JSON batch generation, retry, resume, CSV saves,
parse validation, deduplication, and a per-label report.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is expected in the training env.
    pd = None


# =========================
# Config
# =========================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd() / "MX"

MX_ROOT = Path(os.getenv("MX_ROOT", str(SCRIPT_DIR))).resolve()

DATA_PATH = MX_ROOT / "data" / "annotations_mx" / "mx_annotation_full.csv"
OUT_DIR = MX_ROOT / "synthetic_R2"
VALID_CSV = OUT_DIR / "mx_synthetic_R2_valid_dedup.csv"
OVERFLOW_CSV = OUT_DIR / "mx_synthetic_R2_overflow.csv"
REPORT_CSV = OUT_DIR / "mx_synthetic_R2_report.csv"

TARGET_LABELS = [
    "Cybersecurity_Malware",
    "MX_Narco_Culture_And_Cartel_Glorification",
    "Politically_Sensitive_Topics",
    "Sexually_Explicit_Information",
]

ALLOWED_LABELS = [
    "Dangerous_Content",
    "Harassment",
    "Hate_Speech",
    "Sexually_Explicit_Information",
    "Politically_Sensitive_Topics",
    "Cybersecurity_Malware",
    "MX_Narco_Culture_And_Cartel_Glorification",
    "none",
]

TARGET_PER_LABEL = 2000
BATCH_SIZE = 40
OVER_GENERATE_RATIO = 1.3
MAX_REQUEST_N = 80
MAX_RETRY_PER_LABEL = 20
SOURCE = "mx_synthetic_R2_niw_target2000"

MODEL_NAME = "qwen/qwen3-next-80b-a3b-instruct"
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_TOKENS = 9000
NIW_RETRIES_PER_BATCH = 3
RETRY_BACKOFF_SECONDS = 6

REPORT_COLUMNS = [
    "label",
    "target",
    "original_unique_count",
    "existing_synthetic_main_count",
    "generated_request_count",
    "accepted_main_count",
    "accepted_overflow_count",
    "duplicate_with_original_count",
    "duplicate_with_synthetic_count",
    "duplicate_in_batch_count",
    "parse_error_count",
    "invalid_label_count",
    "empty_text_count",
    "invalid_text_count",
    "final_total_count",
    "gap_remaining",
    "retry_count",
    "status",
]

COUNTER_COLUMNS = [
    "generated_request_count",
    "duplicate_with_original_count",
    "duplicate_with_synthetic_count",
    "duplicate_in_batch_count",
    "parse_error_count",
    "invalid_label_count",
    "empty_text_count",
    "invalid_text_count",
    "retry_count",
]


LABEL_SPECS = {
    "Cybersecurity_Malware": """
Definition:
Malicious software, ransomware, trojans, spyware, rootkits, DDoS, SQL injection, XSS, exploit abuse, unauthorized access, credential theft, cookie/token/API key theft, captcha or 2FA bypass, account takeover, phishing pages/kits, spam automation, SEO spam, bulk account abuse, botting, gray-market automation, and cyberattack assistance.

Generate positive examples:
- Requests, ads, forum posts, Telegram/WhatsApp pitches, tutorial titles, or script requests with malicious cyber intent.
- Mexico-local context may mention banco, OXXO, Mercado Pago, WhatsApp, CURP, SAT, Telcel, BBVA, Banorte, Coppel, but do not force these into every item.
- Use placeholders for harmful details, such as [payload redacted], [dominio falso], [script oculto], [codigo omitido].

Boundaries and safety:
- Do not provide real executable malware code, payloads, attack steps, phishing HTML, bypass procedures, credentials, URLs, IPs, or complete instructions.
- Defensive security learning, CTF, patching, log review, and benign security analysis do not belong here.
- If the text is mainly financial scam persuasion rather than technical cyber abuse, do not generate it here.
""".strip(),
    "MX_Narco_Culture_And_Cartel_Glorification": """
Definition:
Mexico-local narco culture and cartel glorification: admiration, heroization, identity, pride, lifestyle bragging, narcocorridos/corridos tumbados glory narratives, cartel symbols, tributes to bosses, and framing cartel figures as heroes, protectors, idols, or aspirational icons.

Generate positive examples:
- Social comments, short-video captions, fan comments, lyric-like snippets, slang-heavy posts, or identity statements praising cartel life, jefes, patrones, sicarios, plaza mythology, money/power status, or narco lifestyle.
- Generic references are fine. If real groups or people are named, avoid operational details.

Boundaries and safety:
- Do not generate drug manufacturing, transport routes, buying/selling drugs, laundering methods, gun acquisition, evading law enforcement, or violence-task recruitment instructions.
- Ordinary crime news, anti-cartel comments, or neutral public-safety discussion do not belong here.
- The text must show glorification, admiration, normalization, identity, heroization, or lifestyle bragging.
""".strip(),
    "Politically_Sensitive_Topics": """
Definition:
High-risk political sensitivity: election interference, political rumors, political manipulation, public-order risk, political violence, destabilizing calls, malicious defamation of public figures, unrest, security panic, military/police/government rumors, terrorism-related political claims, and disputed historical violence narratives.

Mexico directions:
- Election fraud rumors, misleading voting information, false candidate withdrawal, fake government announcements, fabricated accusations about parties/candidates/officials, cartel-politician collusion rumors, calls to storm government institutions, voting stations, city halls, police stations, courts, or organizing street conflict.

Boundaries and safety:
- Do not write ordinary policy criticism, normal news discussion, or "I dislike this president/party" content.
- Do not provide actionable violence plans, weapons deployment, tactical details, exact times/locations, or attack steps.
- The text needs strong risk: falsehood, manipulation, incitement, public-order threat, election interference, political violence, or public-safety panic.
""".strip(),
    "Sexually_Explicit_Information": """
Definition:
Adult sexual intent, explicit sexual language, adult sexual behavior description, hookup solicitation, adult chat, sexual service ads, adult-content promotion, sexual slang, BDSM/dominance-submission references, and adult tag clusters.

Generate positive examples:
- Adult sexting, hookup invites, adult service promotion, paid adult content funneling, OnlyFans/Telegram/WhatsApp-style adult promotion with fake placeholders only.
- Mexico Spanish, local slang, Spanglish, DMs, captions, ads, and short comments are welcome.

Boundaries and safety:
- Adults only. Do not mention minors, teens, school age, age ambiguity, incest, coercion, drugging, rape, hidden camera, revenge porn, blackmail, extortion, or non-consensual content.
- Do not provide real contact details, phone numbers, addresses, or IDs.
- Do not generate long-form erotic fiction. Do not generate sexual insults aimed at a person or protected group.
- Medical or sex-education content does not belong here.
""".strip(),
}


# =========================
# Env / NIW API
# =========================

def load_local_env_files() -> None:
    """MX/.env is intentionally first, per this dataset's NIW key setup."""
    candidate_paths = [
        MX_ROOT / ".env",
        SCRIPT_DIR / ".env",
        Path.cwd() / "MX" / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in candidate_paths:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


load_local_env_files()


def call_niw_model(prompt: str, model_name: str = MODEL_NAME, timeout: int = 180) -> str:
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
        raise RuntimeError("Missing API key. Put NIW_API_KEY/NIM_API_KEY/MX_API_KEY in MX/.env.")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate synthetic Mexico Spanish content-safety classifier training data. "
                    "Return only valid JSON. No reasoning, no explanation, no markdown. "
                    "Do not include executable code, real payloads, credentials, real contact info, "
                    "or operational criminal/violent instructions."
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
# CSV helpers
# =========================

def dataframe_from_records(records: list[dict], columns: list[str]):
    if pd is not None:
        return pd.DataFrame(records, columns=columns)
    return records


def read_csv_df(path: Path, columns: list[str] | None = None):
    if pd is None:
        if not path.exists() or path.stat().st_size == 0:
            return []
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if columns:
            for row in rows:
                for col in columns:
                    row.setdefault(col, "")
        return rows

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        return df[columns]
    return df


def write_csv_df(path: Path, data, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        df = data if hasattr(data, "to_csv") else pd.DataFrame(data)
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        df[columns].to_csv(path, index=False, encoding="utf-8")
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow({col: row.get(col, "") for col in columns})


def df_to_records(data) -> list[dict]:
    if pd is not None:
        return data.to_dict("records")
    return list(data)


# =========================
# Text normalization / validation
# =========================

def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def dedup_key(text: object) -> str:
    return normalize_text(text).lower()


def strip_code_fence(text: object) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json|jsonl)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_records(raw_output: str) -> list[dict]:
    text = strip_code_fence(raw_output)
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
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, list):
            return obj

    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        rows.append(json.loads(line))
    if rows:
        return rows
    raise ValueError("Could not parse JSON array or JSONL records")


def is_only_url(text: str) -> bool:
    return bool(re.fullmatch(r"https?://\S+|www\.\S+|\[?link\]?", text.strip(), flags=re.IGNORECASE))


def is_only_hashtag(text: str) -> bool:
    tokens = text.split()
    return bool(tokens) and all(t.startswith("#") for t in tokens)


def is_only_emoji_or_symbols(text: str) -> bool:
    letters_or_digits = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", text)
    return len(letters_or_digits) == 0


def has_meta_talk(text: str) -> bool:
    s = " " + text.lower() + " "
    patterns = [
        " este es un ejemplo ",
        " esta es una muestra ",
        " muestra de entrenamiento ",
        " dato de entrenamiento ",
        " texto de entrenamiento ",
        " training sample ",
        " el label ",
        " la label ",
        " etiqueta ",
        " clasificado como ",
        " categoria ",
        " categoría ",
        " final_category ",
        " clean_text ",
        " aqui tienes ",
        " aquí tienes ",
        " a continuacion ",
        " a continuación ",
        " como modelo de lenguaje ",
        " no puedo ayudar ",
    ]
    return any(p in s for p in patterns)


def invalid_text_reason(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return "empty_text"
    if len(text) < 8 or len(text) > 800:
        return "invalid_text"
    if is_only_url(text) or is_only_hashtag(text) or is_only_emoji_or_symbols(text):
        return "invalid_text"
    if has_meta_talk(text):
        return "invalid_text"
    if len(re.findall(r"[\u4e00-\u9fff]", text)) >= 2:
        return "invalid_text"
    return ""


# =========================
# Dataset loading / cleaning
# =========================

def require_original_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH does not exist: {DATA_PATH}")
    original_df = read_csv_df(DATA_PATH)
    cols = original_df.columns.tolist() if pd is not None else list(original_df[0].keys() if original_df else [])
    for col in ["clean_text", "final_category"]:
        if col not in cols:
            raise ValueError(f"Original CSV missing required column: {col}")

    if pd is not None:
        original_df["clean_text"] = original_df["clean_text"].map(normalize_text)
        original_df["final_category"] = original_df["final_category"].map(lambda x: str(x or "").strip())
        return original_df

    for row in original_df:
        row["clean_text"] = normalize_text(row.get("clean_text", ""))
        row["final_category"] = str(row.get("final_category", "")).strip()
    return original_df


def get_columns(original_df) -> list[str]:
    cols = original_df.columns.tolist() if pd is not None else list(original_df[0].keys() if original_df else [])
    for col in ["clean_text", "final_category", "sample_quality", "source"]:
        if col not in cols:
            cols.append(col)
    return cols


def original_unique_counts(original_df) -> tuple[dict[str, int], set[str], dict[str, int]]:
    if pd is not None:
        all_dist = original_df["final_category"].value_counts(dropna=False).to_dict()
        unknown = sorted(set(original_df["final_category"]) - set(ALLOWED_LABELS))
        if unknown:
            print(f"WARNING unknown original final_category values: {unknown}")

        orig_keys = set(original_df["clean_text"].map(dedup_key))
        counts = {}
        for label in TARGET_LABELS:
            label_keys = set(original_df.loc[original_df["final_category"] == label, "clean_text"].map(dedup_key))
            counts[label] = len(label_keys)
        return counts, orig_keys, all_dist

    all_dist = {}
    seen_text = set()
    counts = {label: 0 for label in TARGET_LABELS}
    orig_keys = set()
    for row in original_df:
        label = row.get("final_category", "")
        text = row.get("clean_text", "")
        all_dist[label] = all_dist.get(label, 0) + 1
        key = dedup_key(text)
        orig_keys.add(key)
        if key not in seen_text:
            seen_text.add(key)
            if label in counts:
                counts[label] += 1
    unknown = sorted(set(all_dist) - set(ALLOWED_LABELS))
    if unknown:
        print(f"WARNING unknown original final_category values: {unknown}")
    return counts, orig_keys, all_dist


def sanitize_existing_csv(path: Path, columns: list[str], original_keys: set[str], exclude_keys: set[str] | None = None, write_back: bool = True):
    exclude_keys = exclude_keys or set()
    df = read_csv_df(path, columns)
    rows = []
    seen = set()
    stats = {
        "invalid_label_count": 0,
        "empty_text_count": 0,
        "invalid_text_count": 0,
        "duplicate_with_original_count": 0,
        "duplicate_with_synthetic_count": 0,
    }

    for row in df_to_records(df):
        text = normalize_text(row.get("clean_text", ""))
        label = str(row.get("final_category", "")).strip()

        if label not in TARGET_LABELS:
            stats["invalid_label_count"] += 1
            continue
        reason = invalid_text_reason(text)
        if reason == "empty_text":
            stats["empty_text_count"] += 1
            continue
        if reason:
            stats["invalid_text_count"] += 1
            continue

        key = dedup_key(text)
        if key in original_keys:
            stats["duplicate_with_original_count"] += 1
            continue
        if key in exclude_keys or key in seen:
            stats["duplicate_with_synthetic_count"] += 1
            continue

        seen.add(key)
        for col in columns:
            row.setdefault(col, "")
        row["clean_text"] = text
        row["final_category"] = label
        row["sample_quality"] = "trainable"
        row["source"] = SOURCE
        rows.append(row)

    clean_df = dataframe_from_records(rows, columns)
    if write_back and (path.exists() or rows):
        write_csv_df(path, clean_df, columns)
    return clean_df, stats


def label_count(df, label: str) -> int:
    if pd is not None:
        if df.empty:
            return 0
        return int((df["final_category"] == label).sum())
    return sum(1 for r in df if r.get("final_category") == label)


def key_set(df) -> set[str]:
    if pd is not None:
        if df.empty:
            return set()
        return set(df["clean_text"].map(dedup_key))
    return {dedup_key(r.get("clean_text", "")) for r in df}


def append_rows(df, rows: list[dict], columns: list[str]):
    if not rows:
        return df
    add_df = dataframe_from_records(rows, columns)
    if pd is not None:
        return pd.concat([df, add_df], ignore_index=True)
    return list(df) + rows


def remove_keys(df, remove: set[str], columns: list[str]):
    if not remove:
        return df
    if pd is not None:
        if df.empty:
            return df
        return df[~df["clean_text"].map(dedup_key).isin(remove)][columns].reset_index(drop=True)
    return [r for r in df if dedup_key(r.get("clean_text", "")) not in remove]


def make_output_row(text: str, label: str, columns: list[str]) -> dict:
    row = {col: "" for col in columns}
    row["clean_text"] = normalize_text(text)
    row["final_category"] = label
    row["sample_quality"] = "trainable"
    row["source"] = SOURCE
    if "country" in row:
        row["country"] = "MX"
    if "language" in row:
        row["language"] = "es"
    if "content_id" in row:
        row["content_id"] = f"mx_synthetic_R2_{label}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    if "final_violation" in row:
        row["final_violation"] = "True"
    if "requires_review" in row:
        row["requires_review"] = "False"
    return row


# =========================
# Report
# =========================

def load_report_state() -> dict[str, dict]:
    state = {label: {col: 0 for col in COUNTER_COLUMNS} for label in TARGET_LABELS}
    if not REPORT_CSV.exists():
        return state
    report_df = read_csv_df(REPORT_CSV, REPORT_COLUMNS)
    for row in df_to_records(report_df):
        label = row.get("label", "")
        if label not in state:
            continue
        for col in COUNTER_COLUMNS:
            try:
                state[label][col] = int(float(row.get(col, 0) or 0))
            except Exception:
                state[label][col] = 0
    return state


def write_report(report_state: dict[str, dict], original_counts: dict[str, int], main_df, overflow_df) -> None:
    rows = []
    for label in TARGET_LABELS:
        original_count = original_counts[label]
        main_count = label_count(main_df, label)
        overflow_count = label_count(overflow_df, label)
        final_total = original_count + main_count
        gap = max(0, TARGET_PER_LABEL - final_total)

        counters = report_state[label]
        if original_count >= TARGET_PER_LABEL:
            status = "already_enough"
        elif gap == 0:
            status = "completed"
        elif counters.get("retry_count", 0) >= MAX_RETRY_PER_LABEL:
            status = "not_enough_after_retry"
        elif counters.get("parse_error_count", 0) > 0 and counters.get("generated_request_count", 0) == 0:
            status = "failed_parse"
        else:
            status = "error"

        row = {
            "label": label,
            "target": TARGET_PER_LABEL,
            "original_unique_count": original_count,
            "existing_synthetic_main_count": main_count,
            "generated_request_count": counters.get("generated_request_count", 0),
            "accepted_main_count": main_count,
            "accepted_overflow_count": overflow_count,
            "duplicate_with_original_count": counters.get("duplicate_with_original_count", 0),
            "duplicate_with_synthetic_count": counters.get("duplicate_with_synthetic_count", 0),
            "duplicate_in_batch_count": counters.get("duplicate_in_batch_count", 0),
            "parse_error_count": counters.get("parse_error_count", 0),
            "invalid_label_count": counters.get("invalid_label_count", 0),
            "empty_text_count": counters.get("empty_text_count", 0),
            "invalid_text_count": counters.get("invalid_text_count", 0),
            "final_total_count": final_total,
            "gap_remaining": gap,
            "retry_count": counters.get("retry_count", 0),
            "status": status,
        }
        rows.append(row)
    write_csv_df(REPORT_CSV, dataframe_from_records(rows, REPORT_COLUMNS), REPORT_COLUMNS)


# =========================
# Prompt / generation
# =========================

def sample_existing_texts(main_df, overflow_df, label: str, k: int = 10) -> list[str]:
    values = []
    for df in [main_df, overflow_df]:
        if pd is not None:
            if df.empty:
                continue
            values.extend(df.loc[df["final_category"] == label, "clean_text"].dropna().astype(str).tolist())
        else:
            values.extend([r.get("clean_text", "") for r in df if r.get("final_category") == label])
    values = [v for v in values if v]
    if len(values) <= k:
        return values
    return random.sample(values, k)


def build_prompt(label: str, n: int, existing_samples: list[str] | None = None) -> str:
    existing_samples = existing_samples or []
    style_hint = random.choice([
        "Varia entre post corto, DM, comentario de TikTok, foro, anuncio turbio y chat de grupo.",
        "Usa registros distintos: chilango/norteño/neutro, Spanglish ligero, abreviaturas, typos y tono de redes.",
        "Mezcla frases breves, textos medianos y algunos comentarios largos; que no parezcan plantilla.",
        "Cambia la estructura de cada item; no empieces todos con la misma palabra ni el mismo patrón.",
    ])

    avoid_block = ""
    if existing_samples:
        avoid_block = (
            "\nAvoid copying or closely paraphrasing these existing samples/styles:\n"
            + "\n".join(f"- {x}" for x in existing_samples[:10])
        )

    return f"""
Generate exactly N={n} synthetic training items for one label only.

Current target_label:
{label}

Definition and boundaries:
{LABEL_SPECS[label]}

Hard requirements:
- Output Spanish primarily, with Mexico-local slang and a small amount of natural Spanglish when useful.
- Generate only the current target_label. Do not generate any other label.
- Each item must have exactly one final_category and it must be exactly "{label}".
- clean_text must be natural trainable text, diverse, non-template, non-empty, and 8-800 characters.
- Mix short lines, medium social posts, forum/chat tones, ads, captions, and longer discussion-style comments when appropriate.
- Do not include markdown, numbering, explanations, rationale, headings, or any prose outside JSON.
- Do not write meta phrases like "training sample", "label is", "classified as", "este es un ejemplo", or "muestra".
- Do not include real contact info, real phone numbers, real URLs, real addresses, real credentials, real IDs, or personal data.
- Do not include executable malware code, real payloads, complete attack steps, criminal tutorials, drug routes, weapons instructions, or actionable violent plans.
- For adult sexual content: adults only; exclude minors, coercion, non-consent, sexual crime, extortion, hidden camera, and revenge content.
- For narco content: focus on cultural glorification/admiration/lifestyle, not drug trafficking instructions or violent-task recruitment.
- For political content: focus on rumors, manipulation, election interference, public-order or political-violence risk; not ordinary criticism.
- For cyber content: focus on malicious cyber intent with redacted placeholders, not defensive security or runnable instructions.
- sample_quality must be "trainable".
- source is not needed in model output; the script will set it.
- {style_hint}

Output format:
Return strictly a JSON array.
Each item must be exactly:
{{"clean_text": "...", "final_category": "{label}", "sample_quality": "trainable"}}

No extra keys.
No markdown code fence.
No prose before or after the JSON array.
{avoid_block}
""".strip()


def parse_and_validate_batch(raw_output: str, label: str, columns: list[str], original_keys: set[str], synthetic_keys: set[str], overflow_keys: set[str]) -> tuple[list[dict], dict[str, int]]:
    counters = {col: 0 for col in ["invalid_label_count", "empty_text_count", "invalid_text_count", "duplicate_with_original_count", "duplicate_with_synthetic_count", "duplicate_in_batch_count"]}
    items = extract_records(raw_output)
    accepted = []
    batch_seen = set()

    for item in items:
        if not isinstance(item, dict):
            counters["invalid_text_count"] += 1
            continue
        text = normalize_text(item.get("clean_text", ""))
        final_category = str(item.get("final_category", "")).strip()

        if final_category != label:
            counters["invalid_label_count"] += 1
            continue

        reason = invalid_text_reason(text)
        if reason == "empty_text":
            counters["empty_text_count"] += 1
            continue
        if reason:
            counters["invalid_text_count"] += 1
            continue

        key = dedup_key(text)
        if key in batch_seen:
            counters["duplicate_in_batch_count"] += 1
            continue
        batch_seen.add(key)

        if key in original_keys:
            counters["duplicate_with_original_count"] += 1
            continue
        if key in synthetic_keys or key in overflow_keys:
            counters["duplicate_with_synthetic_count"] += 1
            continue

        accepted.append(make_output_row(text, label, columns))

    return accepted, counters


def promote_from_overflow(label: str, need: int, main_df, overflow_df, columns: list[str]):
    if need <= 0:
        return main_df, overflow_df, 0
    rows = df_to_records(overflow_df)
    promote = []
    remove = set()
    for row in rows:
        if row.get("final_category") != label:
            continue
        key = dedup_key(row.get("clean_text", ""))
        if key in remove:
            continue
        promote.append({col: row.get(col, "") for col in columns})
        remove.add(key)
        if len(promote) >= need:
            break
    if not promote:
        return main_df, overflow_df, 0
    main_df = append_rows(main_df, promote, columns)
    overflow_df = remove_keys(overflow_df, remove, columns)
    return main_df, overflow_df, len(promote)


# =========================
# Final checks / stats
# =========================

def assert_final_outputs(main_df, original_keys: set[str]) -> None:
    rows = df_to_records(main_df)
    seen = set()
    for row in rows:
        text = normalize_text(row.get("clean_text", ""))
        label = row.get("final_category", "")
        if not text:
            raise AssertionError("Main output contains empty clean_text")
        if label not in TARGET_LABELS:
            raise AssertionError(f"Main output contains out-of-scope label: {label}")
        if row.get("sample_quality") != "trainable":
            raise AssertionError("Main output contains non-trainable sample_quality")
        key = dedup_key(text)
        if key in seen:
            raise AssertionError("Main output contains duplicate clean_text")
        if key in original_keys:
            raise AssertionError("Main output overlaps original clean_text")
        seen.add(key)


def print_final_summary(all_dist: dict[str, int], original_counts: dict[str, int], main_df, overflow_df) -> None:
    print("\n===== Original 8-label distribution =====")
    for label in ALLOWED_LABELS:
        print(f"{label}: {int(all_dist.get(label, 0))}")

    print("\n===== Target label original unique counts / needs / accepted =====")
    for label in TARGET_LABELS:
        original_count = original_counts[label]
        target_need = max(0, TARGET_PER_LABEL - original_count)
        main_count = label_count(main_df, label)
        overflow_count = label_count(overflow_df, label)
        final_total = original_count + main_count
        status = "reached" if final_total >= TARGET_PER_LABEL else "not_reached"
        print(
            f"{label}: original_unique={original_count}, need={target_need}, "
            f"accepted_main={main_count}, overflow={overflow_count}, "
            f"final_total={final_total}/{TARGET_PER_LABEL}, status={status}"
        )

    print("\n===== Output files =====")
    print(f"valid_dedup CSV: {VALID_CSV}")
    print(f"overflow CSV: {OVERFLOW_CSV}")
    print(f"report CSV: {REPORT_CSV}")


# =========================
# Main loop
# =========================

def run_generation() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original_df = require_original_data()
    columns = get_columns(original_df)
    original_counts, original_keys, all_dist = original_unique_counts(original_df)

    original_total = len(original_df) if pd is not None else len(original_df)
    original_unique = len(original_keys)
    print(f"Original total rows: {original_total}")
    print(f"Original unique clean_text count: {original_unique}")
    print("Original final_category distribution:")
    for label, count in sorted(all_dist.items(), key=lambda x: (-int(x[1]), str(x[0]))):
        print(f"  {label}: {count}")

    main_df, main_clean_stats = sanitize_existing_csv(VALID_CSV, columns, original_keys, write_back=True)
    main_keys = key_set(main_df)
    overflow_df, overflow_clean_stats = sanitize_existing_csv(OVERFLOW_CSV, columns, original_keys, exclude_keys=main_keys, write_back=True)
    report_state = load_report_state()

    for label in TARGET_LABELS:
        need = max(0, TARGET_PER_LABEL - original_counts[label] - label_count(main_df, label))
        if need > 0:
            main_df, overflow_df, promoted = promote_from_overflow(label, need, main_df, overflow_df, columns)
            if promoted:
                print(f"[{label}] promoted {promoted} rows from overflow to main before generation")
                write_csv_df(VALID_CSV, main_df, columns)
                write_csv_df(OVERFLOW_CSV, overflow_df, columns)

    write_report(report_state, original_counts, main_df, overflow_df)

    print("\nInitial target needs after resume/overflow promotion:")
    for label in TARGET_LABELS:
        need = max(0, TARGET_PER_LABEL - original_counts[label] - label_count(main_df, label))
        print(f"  {label}: original_unique={original_counts[label]}, existing_main={label_count(main_df, label)}, remaining_need={need}")

    for label in TARGET_LABELS:
        while True:
            current_main = label_count(main_df, label)
            remaining_need = max(0, TARGET_PER_LABEL - original_counts[label] - current_main)
            if remaining_need == 0:
                print(f"[{label}] target reached: original={original_counts[label]}, main={current_main}")
                break
            if report_state[label]["retry_count"] >= MAX_RETRY_PER_LABEL:
                print(f"[{label}] retry limit reached with remaining_need={remaining_need}")
                break

            request_n = math.ceil(remaining_need * OVER_GENERATE_RATIO)
            request_n = min(max(request_n, BATCH_SIZE), MAX_REQUEST_N)
            report_state[label]["retry_count"] += 1
            report_state[label]["generated_request_count"] += request_n
            batch_index = report_state[label]["retry_count"]

            prompt = build_prompt(label, request_n, sample_existing_texts(main_df, overflow_df, label))
            print(f"[{label}] batch {batch_index}/{MAX_RETRY_PER_LABEL}: remaining={remaining_need}, request_n={request_n}")

            raw_output = ""
            batch_rows = []
            batch_counters = {}
            last_error = None

            for attempt in range(1, NIW_RETRIES_PER_BATCH + 1):
                try:
                    raw_output = call_niw_model(prompt, MODEL_NAME)
                    batch_rows, batch_counters = parse_and_validate_batch(
                        raw_output=raw_output,
                        label=label,
                        columns=columns,
                        original_keys=original_keys,
                        synthetic_keys=key_set(main_df),
                        overflow_keys=key_set(overflow_df),
                    )
                    break
                except Exception as e:
                    last_error = e
                    print(f"[{label}] NIW/parse attempt {attempt}/{NIW_RETRIES_PER_BATCH} failed: {repr(e)}")
                    if attempt < NIW_RETRIES_PER_BATCH:
                        time.sleep(RETRY_BACKOFF_SECONDS * attempt)

            if last_error and not batch_rows and not batch_counters:
                report_state[label]["parse_error_count"] += 1
                write_report(report_state, original_counts, main_df, overflow_df)
                continue

            for col, value in batch_counters.items():
                report_state[label][col] += int(value)

            if not batch_rows:
                report_state[label]["parse_error_count"] += 1
                write_report(report_state, original_counts, main_df, overflow_df)
                print(f"[{label}] parsed zero accepted rows; continuing")
                continue

            remaining_need = max(0, TARGET_PER_LABEL - original_counts[label] - label_count(main_df, label))
            main_rows = batch_rows[:remaining_need]
            overflow_rows = batch_rows[remaining_need:]

            main_df = append_rows(main_df, main_rows, columns)
            overflow_df = append_rows(overflow_df, overflow_rows, columns)

            write_csv_df(VALID_CSV, main_df, columns)
            if overflow_rows or OVERFLOW_CSV.exists():
                write_csv_df(OVERFLOW_CSV, overflow_df, columns)
            write_report(report_state, original_counts, main_df, overflow_df)

            print(
                f"[{label}] accepted_main={len(main_rows)}, overflow={len(overflow_rows)}, "
                f"new_remaining={max(0, TARGET_PER_LABEL - original_counts[label] - label_count(main_df, label))}"
            )

    write_csv_df(VALID_CSV, main_df, columns)
    write_csv_df(OVERFLOW_CSV, overflow_df, columns)
    write_report(report_state, original_counts, main_df, overflow_df)
    assert_final_outputs(main_df, original_keys)
    print_final_summary(all_dist, original_counts, main_df, overflow_df)


run_generation()
