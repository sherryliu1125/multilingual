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
    PROJECT_ROOT = Path.cwd() / "BR"

BASE_DIR = str(PROJECT_ROOT / "synthetic_R2")
OUT_CSV = BASE_DIR + "/br_synthetic_R2_raw.csv"
ERROR_CSV = BASE_DIR + "/br_synthetic_R2_errors.csv"
FINAL_CSV = BASE_DIR + "/br_synthetic_R2_valid_dedup.csv"

MODEL_NAME = "qwen/qwen3-next-80b-a3b-instruct"

BATCH_SIZE = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 6

TEMPERATURE = 0.9
TOP_P = 0.95
MAX_TOKENS = 3500

TARGETS = {
    "BR_State_Security_Democratic_Order": 1450,
    "Sexually_Explicit_Information": 1500,
    "Politically_Sensitive_Topics": 2200,
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
- May mention Brasil, Brasília, STF, TSE, Congresso, Exército, eleição, urna, golpe, intervenção militar.
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
# Env / NIW API
# =========================

def load_local_env_files():
    candidate_paths = [
        Path.cwd() / ".env",
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
        PROJECT_ROOT.parent / "MX" / ".env",
        PROJECT_ROOT.parent / "BR" / ".env",
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

    如果你的 NIW 网关不同，只改这个函数里的 base_url/api_key/payload/response parsing。
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
    不丢 pt-BR 里的 post, link, live, fake news, DM 等网络词。
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


def build_final_dedup(raw_df):
    """
    从 raw 审计表生成最终训练表：
      text,llm_label,source

    去重逻辑：
      clean_text.strip() 后相同，只保留第一条。
    """
    if raw_df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS), 0

    rows = []
    for _, r in raw_df.iterrows():
        if str(r.get("parse_error", "")).strip():
            continue

        label = str(r.get("final_category", "")).strip()
        if label not in TARGETS:
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
            "source": "llm_generated",
        })

    final_df = pd.DataFrame(rows, columns=FINAL_COLUMNS)
    before = len(final_df)

    if not final_df.empty:
        final_df["_dedup_key"] = final_df["text"].map(lambda x: normalize_text(x))
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
        "Misture comentários de rede social, fórum, grupo de mensagem e resposta curta de timeline.",
        "Varie pontuação, gírias brasileiras, abreviações, emojis e pequenos erros de digitação.",
        "Faça parecer amostra real de crawler: natural, curta, com tom de internet, sem cara de template.",
        "Use estruturas diferentes entre os itens; não comece todos do mesmo jeito.",
        "Inclua variação de registro: raiva, ironia, deboche, flerte, boato, convocação vaga ou desabafo, conforme o label.",
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


def print_stats(raw_df, err_df, final_df, duplicate_count):
    print("\n===== Stats =====")
    for label, target in TARGETS.items():
        current = 0 if final_df.empty else int((final_df["llm_label"] == label).sum())
        print(f"{label}: {current} / {target}")

    print(f"raw_rows: {len(raw_df)}")
    print(f"parse_or_batch_errors: {len(err_df)}")
    print(f"duplicate_removed: {duplicate_count}")
    print(f"valid_final_rows: {len(final_df)}")
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

    raw_df, err_df, final_df, duplicate_count = refresh_outputs()
    used_ids = get_used_batch_ids(raw_df, err_df)
    print_stats(raw_df, err_df, final_df, duplicate_count)

    for label, target in TARGETS.items():
        while True:
            raw_df, err_df, final_df, duplicate_count = refresh_outputs()
            current = 0 if final_df.empty else int((final_df["llm_label"] == label).sum())

            if current >= target:
                print(f"[{label}] target reached: {current}/{target}")
                break

            gap = target - current
            print(f"[{label}] current={current}, target={target}, gap={gap}")

            idx = next_batch_index(label, used_ids)
            batch_id = make_batch_id(label, idx)
            while batch_id in used_ids:
                idx += 1
                batch_id = make_batch_id(label, idx)

            used_ids.add(batch_id)
            generate_one_batch(label, batch_id, final_df)

            raw_df, err_df, final_df, duplicate_count = refresh_outputs()
            print_stats(raw_df, err_df, final_df, duplicate_count)

    raw_df, err_df, final_df, duplicate_count = refresh_outputs()
    print("Done. All label targets reached.")
    print_stats(raw_df, err_df, final_df, duplicate_count)


run_generation()
