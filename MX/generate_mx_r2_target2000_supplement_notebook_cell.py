import os
import re
import csv
import json
import time
import random
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# =========================
# Config
# =========================

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd() / "MX"

BASE_DIR = str(PROJECT_ROOT / "synthetic_R2")
REFERENCE_CSV = str(PROJECT_ROOT / "data" / "annotations_mx" / "mx_annotation_full.csv")

OUT_CSV = BASE_DIR + "/mx_r2_target2000_supplement_raw.csv"
ERROR_CSV = BASE_DIR + "/mx_r2_target2000_supplement_errors.csv"
FINAL_CSV = BASE_DIR + "/mx_r2_target2000_supplement_valid_dedup.csv"

MODEL_NAME = "qwen/qwen3-next-80b-a3b-instruct"

BATCH_SIZE = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 6

TEMPERATURE = 0.9
TOP_P = 0.95
MAX_TOKENS = 3500

# Existing MX annotation + this supplement should reach at least 2000.
# 2200 gives room for later global dedup/cleaning.
TARGET_TOTAL_PER_LABEL = 2200

TARGET_LABELS = [
    "Cybersecurity_Malware",
    "MX_Narco_Culture_And_Cartel_Glorification",
    "Politically_Sensitive_Topics",
    "Sexually_Explicit_Information",
]

ALLOWED_LABELS = {
    "Dangerous_Content",
    "Harassment",
    "Hate_Speech",
    "Sexually_Explicit_Information",
    "Politically_Sensitive_Topics",
    "Cybersecurity_Malware",
    "MX_Narco_Culture_And_Cartel_Glorification",
    "none",
}

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
    "Cybersecurity_Malware": """
Definition:
Malicious software, ransomware, trojans, spyware, DDoS, SQL injection, XSS, exploit abuse, unauthorized access, credential theft, cookie/token/API key theft, captcha bypass, 2FA bypass, account takeover, phishing pages/kits, spam automation, SEO spam, bulk account abuse, botting, gray-market automation, and cyberattack assistance.

Positive sample directions:
- Requests or ads for hacking, account takeover, captcha bypass, token/cookie theft, phishing pages, bulk registration, fake-login pages, bot traffic, DDoS, SQL injection, XSS, ransomware, or gray-market scripts.
- Can look like forum posts, Telegram/WhatsApp pitches, short comments, service ads, or script requests.
- Mexico-local context may mention banco, OXXO, Mercado Pago, WhatsApp, CURP, SAT, Telcel, BBVA, Banorte, Coppel, but do not force these into every item.

Safety boundaries:
- Do not generate real executable malicious code, real payloads, complete commands, real URLs, real IPs, real credentials, real keys, or step-by-step attack instructions.
- Use vague intent or placeholders like [payload omitido], [dominio falso], [script oculto] if needed.
- Defensive security learning, CTF, patching, log review, or benign analysis does not belong here.
- If the text is mainly financial scam persuasion rather than technical cyber abuse, avoid it.
""".strip(),

    "MX_Narco_Culture_And_Cartel_Glorification": """
Definition:
Mexico-local narco culture and cartel glorification: admiration, heroization, identity, pride, lifestyle bragging, narcocorridos/corridos tumbados glory narratives, cartel symbols, tributes to bosses, and framing cartel figures as heroes, protectors, idols, or aspirational icons.

Positive sample directions:
- Praise for cartel, jefe, patrón, sicario, narco vida, plaza mythology, corridos tumbados, narcocorridos, money/power/status lifestyle, or fan-like comments.
- Can be Mexican Spanish slang, short video captions, fan comments, lyric-like short lines, social posts, or group-chat messages.
- Generic references are fine; no need to name real organizations. If real groups or people are named, avoid operational detail.

Safety boundaries:
- Do not generate drug manufacturing, transport routes, buying/selling drugs, laundering methods, gun acquisition, evading law enforcement, or violence-task recruitment instructions.
- Ordinary public-safety discussion, anti-cartel comments, or neutral crime news does not belong here.
- Must show glorification, admiration, normalization, identity, heroization, or lifestyle bragging.
""".strip(),

    "Politically_Sensitive_Topics": """
Definition:
High-risk political sensitivity: election interference, political rumors, political manipulation, public-order risk, political violence, destabilizing calls, malicious defamation of public figures, social unrest, security panic, military/police/government rumors, terrorism-related political claims, and disputed historical violence narratives.

Mexico directions:
- Mexico election fraud rumors, misleading voting information, false candidate withdrawal, fake government announcements, fabricated accusations about parties/candidates/officials.
- Incitement or rumors about government, president, governors, candidates, army, police, courts, city halls, voting stations, or public institutions.
- Cartel-politician collusion rumors can be included when framed as political sensitivity, not narco glorification.

Safety boundaries:
- Do not write ordinary policy criticism or normal news discussion.
- Do not provide actionable violence plans, weapons deployment, tactical details, exact times/locations, or attack steps.
- The text must have strong risk: falsehood, manipulation, incitement, public-order threat, election interference, political violence, or public-safety panic.
""".strip(),

    "Sexually_Explicit_Information": """
Definition:
Adult sexual intent, explicit sexual language, adult sexual behavior description, hookup solicitation, adult chat, sexual service ads, adult-content promotion, sexual slang, BDSM/dominance-submission references, and adult tag clusters.

Positive sample directions:
- Adult sexting, hookup invites, adult service promotion, paid adult content funneling, OnlyFans/Telegram/WhatsApp-style adult promotion with fake placeholders only.
- Mexico Spanish, local slang, DMs, captions, ads, short comments, and internet tone are welcome.

Safety boundaries:
- Adults only. Do not mention minors, teens, school age, age ambiguity, incest, coercion, drugging, rape, hidden camera, revenge porn, blackmail, extortion, or non-consensual content.
- Do not provide real contact details, phone numbers, addresses, or IDs.
- Do not generate long-form erotic fiction. Do not generate sexual insults aimed at a person or protected group.
- Medical or sex-education content does not belong here.
""".strip(),
}


# =========================
# Env / NIW API
# =========================

def load_local_env_files():
    # MX/.env first, because this dataset's API key is stored there.
    candidate_paths = [
        PROJECT_ROOT / ".env",
        Path.cwd() / "MX" / ".env",
        Path.cwd() / ".env",
        PROJECT_ROOT.parent / ".env",
        Path(BASE_DIR) / ".env",
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


def call_niw_model(prompt, model_name=MODEL_NAME, timeout=180):
    """
    NIW model call wrapper.

    当前实现按项目里已有的 NVIDIA NIM / OpenAI-compatible 接口写：
      POST {NIM_BASE_URL}/chat/completions
      Authorization: Bearer {NIM_API_KEY}

    如果 MX 的 NIW 网关不同，只改这个函数里的 base_url/api_key/payload/response parsing。
    """
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
            "Missing API key. Set NIW_API_KEY, NIM_API_KEY, MX_API_KEY, or NVIDIA_API_KEY in MX/.env."
        )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate synthetic Mexico Spanish content-safety training data. "
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
# IO
# =========================

def ensure_dirs():
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)


def read_raw_csv(path):
    if not Path(path).exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=RAW_COLUMNS)

    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[RAW_COLUMNS]


def append_raw_rows(path, rows):
    if not rows:
        return
    file_exists = Path(path).exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if not file_exists or Path(path).stat().st_size == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in RAW_COLUMNS})


def save_final_training_csv(final_df):
    final_df = final_df[FINAL_COLUMNS] if not final_df.empty else pd.DataFrame(columns=FINAL_COLUMNS)
    final_df.to_csv(FINAL_CSV, index=False, encoding="utf-8")


def load_reference_df():
    if not Path(REFERENCE_CSV).exists():
        raise FileNotFoundError(f"REFERENCE_CSV not found: {REFERENCE_CSV}")
    df = pd.read_csv(REFERENCE_CSV, dtype=str, keep_default_na=False)
    for col in ["clean_text", "final_category"]:
        if col not in df.columns:
            raise ValueError(f"REFERENCE_CSV missing required column: {col}")
    df["clean_text"] = df["clean_text"].map(normalize_text)
    df["final_category"] = df["final_category"].map(lambda x: str(x).strip())
    unknown = sorted(set(df["final_category"]) - ALLOWED_LABELS)
    if unknown:
        print(f"WARNING unknown MX labels in reference CSV: {unknown}")
    return df


def load_reference_text_keys():
    df = load_reference_df()
    return set(df["clean_text"].map(lambda x: normalize_text(x).lower()))


def reference_counts():
    df = load_reference_df()
    counts = {}
    for label in TARGET_LABELS:
        sub = df[df["final_category"] == label].copy()
        sub["_key"] = sub["clean_text"].map(lambda x: normalize_text(x).lower())
        counts[label] = int(sub.drop_duplicates("_key").shape[0])
    return counts


# =========================
# Parsing / filtering
# =========================

def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


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
    """
    放宽英文过滤：只丢明显英文为主的整句/拒答/解释。
    不丢墨西哥西语里自然混用的 loanwords，比如 bot, spam, phishing, login, OnlyFans。
    """
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

    es_markers = [
        " que ", " para ", " por ", " con ", " sin ", " alguien ", " banda ",
        " raza ", " compa ", " morra ", " morro ", " jefe ", " patrón ",
        " patron ", " cuenta ", " contraseña ", " contrasena ", " gobierno ",
        " candidato ", " elección ", " eleccion ", " casilla ", " quiero ",
        " vendo ", " ocupo ", " necesito ", " jalo ", " está ", " esta ",
        " méxico ", " mexico ", " no ", " sí ", " si ", " ya ", " pero ",
    ]
    es_hits = sum(1 for p in es_markers if p in s)

    en_markers = [
        " the ", " and ", " you ", " your ", " with ", " people ",
        " account ", " password ", " steal ", " attack ", " government ",
    ]
    en_hits = sum(1 for p in en_markers if p in s)

    return ascii_ratio > 0.94 and en_hits >= 3 and es_hits == 0


def is_refusal_text(text):
    s = " " + str(text or "").strip().lower() + " "
    refusal_patterns = [
        " as an ai ",
        " as a language model ",
        " como ia ",
        " como inteligencia artificial ",
        " como modelo de lenguaje ",
        " no puedo ayudar ",
        " no puedo generar ",
        " no puedo crear ",
        " no puedo proporcionar ",
        " no puedo facilitar ",
    ]
    return any(p in s for p in refusal_patterns)


def contains_forbidden_real_artifact(text):
    s = str(text or "")
    patterns = [
        r"https?://\S+",
        r"www\.\S+",
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        r"\b(?:sk|ghp|xoxb|AKIA)[A-Za-z0-9_\-]{12,}\b",
        r"\b\d{10,13}\b",
    ]
    return any(re.search(p, s, flags=re.IGNORECASE) for p in patterns)


def contains_forbidden_operational_cyber(text):
    s = str(text or "")
    patterns = [
        r"\b(?:curl|wget|powershell|cmd\.exe|bash|python|python3|nmap|sqlmap|hydra|metasploit|msfconsole)\b\s+\S+",
        r"<script\b",
        r"select\s+.+\s+from\s+",
        r"union\s+select",
        r"drop\s+table",
    ]
    return any(re.search(p, s, flags=re.IGNORECASE) for p in patterns)


def quality_filter_raw(row, target_label):
    text = normalize_text(row.get("clean_text", ""))
    label = str(row.get("final_category", "")).strip()

    if not text:
        return False, "empty_clean_text"
    if label != target_label:
        return False, "wrong_final_category"
    if label not in ALLOWED_LABELS:
        return False, "label_not_allowed"
    if len(text) < 40 or len(text) > 240:
        return False, f"bad_length_{len(text)}"
    if is_chinese_heavy(text):
        return False, "chinese_detected"
    if is_obviously_english_response(text):
        return False, "obviously_english_response"
    if is_refusal_text(text):
        return False, "refusal_text"
    if contains_forbidden_real_artifact(text):
        return False, "forbidden_real_artifact"
    if target_label == "Cybersecurity_Malware" and contains_forbidden_operational_cyber(text):
        return False, "forbidden_operational_cyber"

    return True, ""


def build_final_dedup(raw_df):
    """
    从 supplement raw 审计表生成最终训练表：
      text,llm_label,source

    去重：
      1. 排除 MX 原始 annotation 已有 clean_text
      2. supplement 内按 clean_text.strip().lower() 去重
    """
    if raw_df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS), 0

    reference_keys = load_reference_text_keys()

    rows = []
    for _, r in raw_df.iterrows():
        if str(r.get("parse_error", "")).strip():
            continue

        label = str(r.get("final_category", "")).strip()
        if label not in TARGET_LABELS:
            continue

        text = normalize_text(r.get("clean_text", ""))
        raw_row = {"clean_text": text, "final_category": label}
        ok, _reason = quality_filter_raw(raw_row, label)
        if not ok:
            continue

        key = text.lower()
        if key in reference_keys:
            continue

        rows.append({
            "text": text,
            "llm_label": label,
            "source": "llm_generated",
        })

    final_df = pd.DataFrame(rows, columns=FINAL_COLUMNS)
    before = len(final_df)

    if not final_df.empty:
        final_df["_dedup_key"] = final_df["text"].map(lambda x: normalize_text(x).lower())
        final_df = final_df.drop_duplicates(subset=["_dedup_key"], keep="first")
        final_df = final_df.drop(columns=["_dedup_key"])

    duplicate_count = before - len(final_df)
    return final_df, duplicate_count


# =========================
# Prompt
# =========================

def build_prompt(label, n, existing_samples=None):
    existing_samples = existing_samples or []
    style_hint = random.choice([
        "Mezcla comentarios de red social, foro, grupo de WhatsApp, DM y respuesta corta de timeline.",
        "Varía puntuación, slang mexicano, abreviaturas, emojis y errores leves de dedo.",
        "Hazlo sonar como muestra real de crawler: natural, breve, con tono de internet, sin plantilla.",
        "Usa estructuras distintas entre items; no empieces todos igual.",
        "Incluye registros distintos: oferta, rumor, alarde, coqueteo, petición, queja intensa o comentario de grupo, según el label.",
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
- Output must be Mexico-local Spanish.
- Do not use Chinese.
- Do not use English, except common internet/platform/cyber/adult loanwords naturally used in Mexico.
- Each clean_text should be around 80-180 characters.
- Text should look like realistic social media, forum, comment, DM, or group-chat content.
- The content must clearly match the current target label.
- Do not output explanations, reasoning, classification rationale, notes, or long paragraphs.
- Do not output markdown.
- Do not repeat the same structure across items.
- You may include Mexican slang, abbreviations, emojis, typos, caps, sarcasm, and internet tone.
- Do not generate real personal privacy, real accounts, real tokens, real keys, real URLs, real IPs, real emails, or real phone numbers.
- Do not generate executable malicious code, payloads, complete commands, criminal tutorials, attack steps, exact violent plans, drug routes, weapon acquisition, or evasion instructions.
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


# =========================
# Resume / batch logic
# =========================

def make_batch_id(label, index):
    return f"{label}__batch_{index:06d}"


def get_used_batch_ids(raw_df, err_df):
    used = set()
    if not raw_df.empty:
        used.update(raw_df["batch_id"].dropna().astype(str).tolist())
    if not err_df.empty:
        used.update(err_df["batch_id"].dropna().astype(str).tolist())
    return used


def next_batch_index(label, used_ids):
    prefix = f"{label}__batch_"
    max_idx = -1
    for bid in used_ids:
        bid = str(bid)
        if bid.startswith(prefix):
            tail = bid.replace(prefix, "", 1)
            if tail.isdigit():
                max_idx = max(max_idx, int(tail))
    return max_idx + 1


def sample_existing_texts(final_df, label, k=8):
    if final_df.empty:
        return []
    sub = final_df[final_df["llm_label"] == label]
    values = sub["text"].dropna().astype(str).tolist()
    if len(values) <= k:
        return values
    return random.sample(values, k)


def parse_batch_to_raw_rows(raw_output, label, batch_id):
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


def write_error_row(batch_id, label, raw_output, error_message):
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "clean_text": "",
        "final_category": label,
        "generation_model": MODEL_NAME,
        "batch_id": batch_id,
        "item_id": f"{batch_id}__error",
        "raw_output": str(raw_output or ""),
        "parse_error": str(error_message),
        "created_at": now,
    }
    append_raw_rows(ERROR_CSV, [row])


def refresh_outputs():
    raw_df = read_raw_csv(OUT_CSV)
    err_df = read_raw_csv(ERROR_CSV)
    final_df, duplicate_count = build_final_dedup(raw_df)
    save_final_training_csv(final_df)
    return raw_df, err_df, final_df, duplicate_count


def total_counts_with_reference(final_df, ref_counts):
    counts = {}
    for label in TARGET_LABELS:
        supplement = 0 if final_df.empty else int((final_df["llm_label"] == label).sum())
        counts[label] = ref_counts.get(label, 0) + supplement
    return counts


def print_stats(raw_df, err_df, final_df, duplicate_count, ref_counts):
    total_counts = total_counts_with_reference(final_df, ref_counts)

    print("\n===== Stats =====")
    for label in TARGET_LABELS:
        ref = ref_counts.get(label, 0)
        supplement = 0 if final_df.empty else int((final_df["llm_label"] == label).sum())
        total = total_counts[label]
        gap = max(TARGET_TOTAL_PER_LABEL - total, 0)
        print(f"{label}: reference={ref}, supplement_valid={supplement}, total={total}/{TARGET_TOTAL_PER_LABEL}, gap={gap}")

    print(f"raw_rows: {len(raw_df)}")
    print(f"parse_or_batch_errors: {len(err_df)}")
    print(f"duplicate_removed: {duplicate_count}")
    print(f"valid_supplement_rows: {len(final_df)}")
    print(f"REFERENCE_CSV: {REFERENCE_CSV}")
    print(f"OUT_CSV: {OUT_CSV}")
    print(f"ERROR_CSV: {ERROR_CSV}")
    print(f"FINAL_CSV: {FINAL_CSV}")
    print("=================\n")


def generate_one_batch(label, batch_id, final_df):
    prompt = build_prompt(
        label=label,
        n=BATCH_SIZE,
        existing_samples=sample_existing_texts(final_df, label, k=8),
    )

    last_raw = ""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[{label}] request {batch_id}, attempt {attempt}/{MAX_RETRIES}")
            raw_output = call_niw_model(prompt, MODEL_NAME)
            last_raw = raw_output

            rows = parse_batch_to_raw_rows(raw_output, label, batch_id)
            if not rows:
                raise ValueError("Parsed output but zero rows passed quality filter")

            append_raw_rows(OUT_CSV, rows)
            print(f"[{label}] saved {len(rows)} valid raw rows from {batch_id}")
            return True

        except Exception as e:
            last_error = e
            print(f"[{label}] {batch_id} failed attempt {attempt}: {repr(e)}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    write_error_row(batch_id, label, last_raw, repr(last_error))
    print(f"[{label}] wrote error row for {batch_id}")
    return False


def run_generation():
    ensure_dirs()
    ref_counts = reference_counts()

    raw_df, err_df, final_df, duplicate_count = refresh_outputs()
    used_ids = get_used_batch_ids(raw_df, err_df)
    print_stats(raw_df, err_df, final_df, duplicate_count, ref_counts)

    for label in TARGET_LABELS:
        while True:
            raw_df, err_df, final_df, duplicate_count = refresh_outputs()
            total_counts = total_counts_with_reference(final_df, ref_counts)
            current_total = total_counts[label]

            if current_total >= TARGET_TOTAL_PER_LABEL:
                print(f"[{label}] target reached: {current_total}/{TARGET_TOTAL_PER_LABEL}")
                break

            gap = TARGET_TOTAL_PER_LABEL - current_total
            print(f"[{label}] current_total={current_total}, target={TARGET_TOTAL_PER_LABEL}, gap={gap}")

            idx = next_batch_index(label, used_ids)
            batch_id = make_batch_id(label, idx)
            while batch_id in used_ids:
                idx += 1
                batch_id = make_batch_id(label, idx)

            used_ids.add(batch_id)
            generate_one_batch(label, batch_id, final_df)

            raw_df, err_df, final_df, duplicate_count = refresh_outputs()
            print_stats(raw_df, err_df, final_df, duplicate_count, ref_counts)

    raw_df, err_df, final_df, duplicate_count = refresh_outputs()
    print("Done. All MX target labels reached.")
    print_stats(raw_df, err_df, final_df, duplicate_count, ref_counts)


run_generation()
