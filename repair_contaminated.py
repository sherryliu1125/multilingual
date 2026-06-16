#!/usr/bin/env python3
"""
repair_contaminated.py
针对 malware/data/contamination_ids.json 里记录的污染行，
重新调用 NVIDIA NIM API 翻译，并原地替换回对应 CYBER_{PROFILE}.csv 的 text 字段。
其他行不受影响。
"""

import os
import csv
import io
import json
import time
import re
import threading
import concurrent.futures
import urllib.request
import urllib.error
from pathlib import Path

# =====================================================================
# Paths
# =====================================================================

SCRIPT_DIR        = Path(__file__).resolve().parent
INPUT_CSV         = SCRIPT_DIR / "malware" / "data" / "CYBER_EN.csv"
OUTPUT_DIR        = SCRIPT_DIR / "malware" / "data"
CONTAMINATION_JSON = SCRIPT_DIR / "malware" / "data" / "contamination_ids.json"

CSV_COLUMNS       = ["seed_id", "text", "label", "shard_id"]
EXPECTED_LABEL    = "Cybersecurity_Malware"
CHECKPOINT_JSON   = Path(__file__).resolve().parent / "malware" / "data" / "repair_checkpoint.json"

_checkpoint_lock = threading.Lock()

# =====================================================================
# Model & API config（与 localize_cyber.py 保持一致）
# =====================================================================

MODEL_NAME    = "nvidia/llama-3.3-nemotron-super-49b-v1"
TEMPERATURE   = 0.3
TOP_P         = 0.9
MAX_TOKENS    = 6000
BATCH_SIZE    = 20
MAX_RETRIES   = 3
RETRY_BACKOFF = 8   # 非 429 重试等待基数（秒）

# =====================================================================
# .env loader（从 localize_cyber.py 直接复制）
# =====================================================================

_PROFILE_ENV: dict[str, str] = {
    "GULF_AR": "SA/.env",
    "BR_PT":   "BR/.env",
    "MX_ES":   "MX/.env",
    "TR_TR":   "SA/.env",
}


def _read_api_key(profile: str) -> str:
    """从对应 profile 的 .env 文件直接读取 NIM_API_KEY / NVIDIA_API_KEY，
    不写入 os.environ，避免多线程并发时 key 互相覆盖。"""
    env_rel = _PROFILE_ENV.get(profile)
    if not env_rel:
        raise RuntimeError(f"No .env mapping found for profile '{profile}'")
    env_path = SCRIPT_DIR / env_rel
    if not env_path.exists():
        raise RuntimeError(f"[{profile}] .env file not found: {env_path}")

    keys: dict[str, str] = {}
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            keys[k.strip()] = v.strip()

    api_key = keys.get("NIM_API_KEY") or keys.get("NVIDIA_API_KEY") or ""
    if not api_key:
        raise RuntimeError(
            f"[{profile}] NIM_API_KEY / NVIDIA_API_KEY not found in {env_path}"
        )
    return api_key

# =====================================================================
# Prompt（从 localize_cyber.py 直接复制）
# =====================================================================

_COMMON_PROMPT = """\
You are localizing English Cybersecurity_Malware classifier training data into a target language and local user style.

Task:
Translate and localize the input English samples into TARGET_PROFILE.

This is not free generation.
This is meaning-preserving localization.

Target label:
Cybersecurity_Malware

Label definition:
This label covers user text that seeks, offers, assists, promotes, or requests instructions, tools, code, services, or operational help for cyber abuse, malware, unauthorized access, credential theft, exploit abuse, platform abuse, spam, gray-market traffic manipulation, or malicious automation.

Input CSV columns:
seed_id,text,label,shard_id

Output CSV columns:
seed_id,text,label,shard_id

Output rules:

* Keep the same seed_id.
* Keep the same label.
* Keep the same shard_id.
* Only rewrite the text field.
* Output exactly the same number of rows as the input.
* Do not add sample_quality.
* Do not add reasoning.
* Do not add confidence.
* Do not add country.
* Do not add language.
* Do not add explanations.
* Do not wrap the output in markdown.
* Escape CSV correctly.
* Always enclose the text field in double quotes.
* If the text contains double quotes, escape them as two consecutive double quotes ("").
* Do not drop rows.
* Do not merge rows.
* Do not create new rows.

Core translation requirements:

* Preserve the original cybersecurity-abuse intent.
* Preserve the original shard meaning.
* Preserve the original tone and style when possible.
* Do not mistranslate technical cybersecurity terms.
* Do not make the text safer, more educational, more defensive, or more policy-compliant.
* Do not turn malicious requests into benign cybersecurity learning, bug bounty, CTF, compliance, awareness training, responsible disclosure, or defensive security content.
* Do not turn the sample into financial fraud, investment scam, bank-card theft, fake payment, money laundering, refund scam, or crypto investment scam content.
* Do not add sexual, hate, harassment, political, terrorist, weapon, or drug content.

Strict label-preservation rule:
The localized text must remain an unsafe Cybersecurity_Malware sample.

Do NOT neutralize, sanitize, justify, legalize, or soften the abusive intent.

Do NOT add phrases or meanings such as:

* for learning
* for research
* for a lab
* for CTF
* for bug bounty
* for authorized testing
* for defensive security
* for awareness training
* for my own system
* with permission
* legally
* ethically
* to protect users
* to improve security
* as a security professional
* for compliance
* for education only

If the English source implies malicious or abusive use, the localized version must also imply malicious or abusive use.

If the English source is about stealing, bypassing, evading, abusing, flooding, scraping private data, phishing, credential theft, malware, spam, account abuse, leaked data, unauthorized access, or selling cyber-abuse services, the localized text must preserve that same abusive intent.

Do not add moral disclaimers.
Do not add safety disclaimers.
Do not add legal disclaimers.
Do not change the user into a defender, researcher, admin, student, teacher, auditor, or security professional.

Before outputting each row, internally check:

1. Does the localized text still clearly belong to Cybersecurity_Malware?
2. Did I accidentally make it sound legal, educational, defensive, authorized, or harmless?
3. Did I remove or weaken the abusive intent?
4. Did I replace malicious intent with a harmless cybersecurity topic?

If any answer indicates label drift, rewrite the row before output.

Safety and non-operational constraints:

* Do not add code.
* Do not add commands.
* Do not add payloads.
* Do not add exploit strings.
* Do not add phishing HTML.
* Do not add SQL/XSS examples.
* Do not add step-by-step attack instructions.
* Do not add tool installation instructions.
* Do not add real domains, real URLs, real IP addresses, real email addresses, real handles, real credentials, real API keys, real tokens, real private keys, or real victims.
* Do not make the localized text more operational than the English source.
* Keep the text as user-generated intent text, not an attack guide.

Localization requirements:

* Do not translate literally word-for-word.
* Make the text sound like a real local user would write it.
* Use natural local syntax, phrasing, informality, abbreviations, and chat/forum style where appropriate.
* Keep English cybersecurity terms when they are naturally used locally.
* Do not force every technical term into the local language.
* Do not over-localize with stereotypes.
* Do not make every sentence formal.
* Do not make every sentence long.
* Avoid repeating the same opening phrase across rows.
* Avoid making every row start with equivalents of "I need", "How can I", "Can someone", "Help me", or "Looking for".
* Preserve diversity: short requests, marketplace posts, vague inquiries, troubleshooting complaints, semi-technical questions, slangy chat messages, and promotional posts should remain diverse.

Technical term handling:
Keep or partially translate common cyber terms depending on local usage.

Terms that can often remain in English include:
payload, exploit, bypass, login, panel, RAT, shell, token, cookie, session, phishing, stealer, loader, checker, combo, proxy, bot, spam, keylogger, WAF, EDR, AV, DDoS, botnet, admin panel, brute force.

Do not translate these terms in a way that sounds unnatural or obscure in the target language.

Bad transformation examples:

English source:
Looking for a token grabber that avoids detection.

Bad localized version:
I want to learn how token security works for research.

Good localized version:
I need a token grabber that does not get detected easily.

English source:
Need a phishing page that collects login cookies.

Bad localized version:
I need a fake login page for a cybersecurity awareness demo.

Good localized version:
I need a fake login page that can collect login cookies.

English source:
Want a bot to create fake accounts without getting banned.

Bad localized version:
I want to test account registration limits on my own app.

Good localized version:
I want a bot that creates fake accounts without getting banned.

Quality checks before final output:

* Each output text must still clearly belong to Cybersecurity_Malware.
* Each output text must preserve the source meaning.
* Each output text must sound natural in TARGET_PROFILE.
* Each output text must not be mostly English, except for natural technical terms.
* No row should become benign, defensive, academic, authorized, or educational.
* No row should become more operational than the source.
* No duplicate localized text should be produced within the batch.
* The localized text should not look like machine translation.
* The localized text should preserve the same abusive cybersecurity category as the source.
"""

_PROFILE_STYLE: dict[str, str] = {

    "GULF_AR": """\
TARGET_PROFILE = GULF_AR

Target users:
Arabic-speaking users in Saudi Arabia and the UAE.

Target language:
Arabic.

Style:

* Use mostly Modern Standard Arabic that feels natural online.
* Mix in light Gulf Arabic expressions where appropriate.
* The result should be understandable and natural for both Saudi and UAE users.
* Do not make it purely Saudi dialect.
* Do not make it purely Emirati dialect.
* Do not overuse heavy dialect.
* Keep common English cyber terms naturally mixed into Arabic text.
* Arabic + English code-switching is acceptable, but the main sentence should be Arabic.
* Use natural online phrasing such as "أبغى", "أحتاج", "فيه طريقة", "حد يعرف", "ما يبين", "بدون ما ينكشف" only when it fits.
* Avoid overly formal textbook Arabic.
* Avoid translating every cyber term into formal Arabic if local users would normally keep the English term.
""",

    "BR_PT": """\
TARGET_PROFILE = BR_PT

Target users:
Brazilian users.

Target language:
Brazilian Portuguese.

Style:

* Use natural Brazilian Portuguese.
* Use local internet/chat phrasing when appropriate.
* Allow common English cyber terms mixed into Portuguese grammar.
* Do not use European Portuguese.
* Prefer Brazilian forms such as "preciso", "alguém sabe", "quero", "tem como", "conta", "senha", "painel", "sem cair", "não ser detectado", "sem tomar ban" when natural.
* Informal style is acceptable, but avoid making every row slangy.
* Keep technical expressions realistic for Brazilian cyber/forum contexts.
* Avoid overly formal translation that sounds like a manual or textbook.
""",

    "MX_ES": """\
TARGET_PROFILE = MX_ES

Target users:
Mexican users.

Target language:
Mexican Spanish.

Style:

* Use natural Mexican Spanish.
* Use local internet/chat phrasing where appropriate.
* Allow common English cyber terms mixed into Spanish grammar.
* Do not use Spain-specific wording.
* Prefer Mexican/LatAm phrasing such as "ocupo", "necesito", "alguien sabe", "cuentas", "contraseña", "panel", "que no lo detecten", "sin que me tumben la cuenta", "sin que me baneen" when natural.
* Do not overdo slang.
* Keep the tone varied: casual, semi-technical, marketplace-like, vague, frustrated, promotional.
* Avoid overly neutral or formal Spanish that sounds like a translated policy document.
""",

    "TR_TR": """\
TARGET_PROFILE = TR_TR

Target users:
Turkish users.

Target language:
Turkish.

Style:

* Use natural Turkish.
* Use local forum/chat style when appropriate.
* Allow common English cyber terms mixed into Turkish grammar.
* Keep terms like token, cookie, panel, RAT, bypass, exploit, phishing, proxy, bot, checker when they sound natural.
* Prefer Turkish phrasing such as "lazım", "nasıl yaparım", "bilen var mı", "hesap", "şifre", "yakalanmadan", "ban yemeden", "tespit edilmeden" when natural.
* Do not make every row formal.
* Avoid unnatural literal translation from English.
* Keep the tone varied: direct, casual, marketplace-like, vague, frustrated, semi-technical, promotional.
""",
}


def build_prompt(profile: str, csv_rows: str) -> str:
    """组合公共规则 + 语言风格 + 输入行，生成完整 user prompt。"""
    if profile not in _PROFILE_STYLE:
        raise ValueError(
            f"Unknown TARGET_PROFILE '{profile}'. "
            f"Valid options: {list(_PROFILE_STYLE.keys())}"
        )
    style_block = _PROFILE_STYLE[profile]
    return (
        f"{_COMMON_PROMPT}\n"
        f"{style_block}\n"
        f"Generation instruction:\n"
        f"TARGET_PROFILE = {profile}\n\n"
        f"Input rows:\n"
        f"{csv_rows}\n\n"
        f"Output only the localized CSV rows."
    )

# =====================================================================
# API call（从 localize_cyber.py 直接复制）
# =====================================================================

def call_api(user_prompt: str, label: str, profile: str) -> str:
    base_url = (
        os.getenv("NIM_BASE_URL")
        or os.getenv("NVIDIA_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    ).rstrip("/")

    api_key = _read_api_key(profile)

    endpoint = f"{base_url}/chat/completions"

    system_msg = (
        "You are a professional multilingual localization specialist for AI safety datasets. "
        "Output only valid CSV with exactly 4 columns: seed_id,text,label,shard_id. "
        "No header line. No markdown. No explanation. No extra text before or after. "
        "Preserve the original seed_id, label, and shard_id unchanged."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_prompt},
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
                time.sleep(1)
                return data["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"  [{label}] HTTP {e.code} (attempt {attempt}/{MAX_RETRIES}): {body[:400]}")
            if attempt < MAX_RETRIES:
                wait = 60 * attempt if e.code == 429 else RETRY_BACKOFF * attempt
                print(f"  [{label}] Waiting {wait}s...")
                time.sleep(wait)
                continue

        except Exception as exc:
            print(f"  [{label}] Error (attempt {attempt}/{MAX_RETRIES}): {repr(exc)[:300]}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(f"[{label}] API failed after {MAX_RETRIES} attempts.")

# =====================================================================
# CSV parser + 净化检测（从 localize_cyber.py 直接复制）
# =====================================================================

_SANITIZE_PATTERNS = re.compile(
    r"\b(for research|for learning|for education|for CTF|for bug bounty|"
    r"for authorized testing|for defensive|for awareness|for compliance|"
    r"with permission|legally|ethically|to protect users|to improve security|"
    r"security professional|for my own system|para fins educacionais|"
    r"para aprendizado|para pesquisa|para fins legais|"
    r"para propósitos educativos|para aprender|con permiso|legalmente|"
    r"eğitim amaçlı|araştırma amaçlı|yasal olarak|"
    r"للتعلم|للبحث|للدراسة|بإذن|بشكل قانوني)\b",
    re.IGNORECASE,
)


def _repair_csv_line(line: str):
    """
    尝试修复因 text 字段含逗号未转义导致的列错位。
    期望结构: seed_id, ...text..., Cybersecurity_Malware, shard_id
    """
    parts = line.split(",")
    try:
        label_idx = next(
            i for i in range(len(parts) - 1, 0, -1)
            if parts[i].strip() == "Cybersecurity_Malware"
        )
    except StopIteration:
        return None
    seed_id  = parts[0].strip().strip('"')
    text     = ",".join(parts[1:label_idx]).strip().strip('"')
    label    = "Cybersecurity_Malware"
    shard_id = parts[label_idx + 1].strip().strip('"') if label_idx + 1 < len(parts) else ""
    if not seed_id or not text:
        return None
    return [seed_id, text, label, shard_id]


def parse_and_validate(raw: str, source_rows: list[dict], label: str) -> list[dict]:
    """解析 API 返回的 CSV，验证行数和内容完整性。"""
    raw = re.sub(r"```[^\n]*\n?", "", raw).strip()
    raw_with_header = "seed_id,text,label,shard_id\n" + raw

    records: list[dict] = []
    try:
        reader = csv.DictReader(io.StringIO(raw_with_header))
        for i, row in enumerate(reader):
            row = {k.strip(): v.strip() for k, v in row.items() if k and k.strip()}

            text = row.get("text", "").strip()
            if not text or len(text.split()) < 3:
                print(f"  [{label}] Row {i+1}: text too short — skipped.")
                continue

            lbl = row.get("label", "").strip()
            if lbl != EXPECTED_LABEL:
                raw_lines = raw.splitlines()
                repaired = _repair_csv_line(raw_lines[i]) if i < len(raw_lines) else None
                if repaired:
                    seed_id, text, lbl, shard_id = repaired
                    if not text or len(text.split()) < 3:
                        print(f"  [{label}] Row {i+1}: repaired text too short — skipped.")
                        continue
                    if _SANITIZE_PATTERNS.search(text):
                        print(f"  [{label}] Row {i+1}: SANITIZED text detected — skipped: {text[:80]}")
                        continue
                    records.append({
                        "seed_id":  seed_id,
                        "text":     text,
                        "label":    lbl,
                        "shard_id": shard_id,
                    })
                else:
                    print(f"  [{label}] Row {i+1}: wrong label '{lbl}' and repair failed — skipped.")
                continue

            if _SANITIZE_PATTERNS.search(text):
                print(f"  [{label}] Row {i+1}: SANITIZED text detected — skipped: {text[:80]}")
                continue

            records.append({
                "seed_id":  row.get("seed_id", "").strip(),
                "text":     text,
                "label":    lbl,
                "shard_id": row.get("shard_id", "").strip(),
            })

    except Exception as exc:
        print(f"  [{label}] CSV parse error: {exc}")

    expected = len(source_rows)
    if len(records) != expected:
        print(
            f"  [{label}] WARNING: expected {expected} rows, got {len(records)} valid rows."
        )

    return records

# =====================================================================
# Checkpoint：记录已修复的 seed_id，支持断点续跑
# =====================================================================

def _load_checkpoint() -> dict[str, list[str]]:
    """读取 checkpoint 文件，返回 {profile: [已修复的 seed_id, ...]}。"""
    if CHECKPOINT_JSON.exists():
        try:
            with open(CHECKPOINT_JSON, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_checkpoint(profile: str, repaired_ids: list[str]) -> None:
    """线程安全地把 profile 的已修复 ID 追加写入 checkpoint 文件。"""
    with _checkpoint_lock:
        data = _load_checkpoint()
        existing = set(data.get(profile, []))
        existing.update(repaired_ids)
        data[profile] = sorted(existing)
        tmp = CHECKPOINT_JSON.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(CHECKPOINT_JSON)


# =====================================================================
# 原地替换 CSV 文件中指定 seed_id 的 text 字段
# =====================================================================

def patch_csv_inplace(output_csv: Path, updates: dict[str, str]) -> int:
    """
    读取 output_csv，把 updates 里的 seed_id 对应行的 text 字段替换掉，
    用临时文件 + rename 原子写回。
    返回实际替换的行数。
    """
    if not output_csv.exists():
        print(f"  WARNING: {output_csv} not found, skipping patch.")
        return 0

    with open(output_csv, encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    patched = 0
    for row in all_rows:
        sid = row.get("seed_id", "").strip()
        if sid in updates:
            row["text"] = updates[sid]
            patched += 1

    tmp_path = output_csv.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    tmp_path.replace(output_csv)
    return patched

# =====================================================================
# 单 profile 补跑逻辑（每 batch 立即写回 + checkpoint）
# =====================================================================

def repair_profile(
    profile: str,
    contaminated_ids: list[str],
    en_rows_by_id: dict[str, dict],
) -> tuple[int, list[str]]:
    """
    对单个 profile 补跑翻译，每个 batch 成功后立即写回 CSV 并更新 checkpoint。
    返回 (修复行数, 未修复 seed_id 列表)。
    """
    tag = f"[{profile}]"
    output_csv = OUTPUT_DIR / f"CYBER_{profile}.csv"

    # 断点续跑：跳过已在 checkpoint 里的 ID
    checkpoint = _load_checkpoint()
    already_done = set(checkpoint.get(profile, []))
    if already_done:
        print(f"{tag} Checkpoint: {len(already_done)} already repaired — skipping.")

    # 过滤掉在英文源里找不到的 ID，以及已修复的
    todo_ids = [
        sid for sid in contaminated_ids
        if sid in en_rows_by_id and sid not in already_done
    ]
    missing_in_en = [sid for sid in contaminated_ids if sid not in en_rows_by_id]
    if missing_in_en:
        print(f"{tag} WARNING: {len(missing_in_en)} seed_id(s) not in CYBER_EN.csv — skipped.")

    if not todo_ids:
        print(f"{tag} Nothing to repair.")
        return len(already_done), []

    total_needed = len(contaminated_ids) - len(missing_in_en)
    print(f"{tag} Repairing {len(todo_ids)}/{total_needed} row(s) in {output_csv.name} ...")

    total_patched = len(already_done)
    failed_ids: list[str] = []

    for batch_start in range(0, len(todo_ids), BATCH_SIZE):
        batch_ids  = todo_ids[batch_start: batch_start + BATCH_SIZE]
        batch_rows = [en_rows_by_id[sid] for sid in batch_ids]
        batch_no   = batch_start // BATCH_SIZE + 1
        batch_label = f"{profile}/repair-b{batch_no}"

        buf = io.StringIO()
        csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore").writerows(batch_rows)
        csv_input_str = buf.getvalue().strip()

        print(f"{tag} Batch {batch_no}: translating {len(batch_ids)} row(s) "
              f"({batch_start+1}–{batch_start+len(batch_ids)} / {len(todo_ids)} pending)...")

        prompt_text = build_prompt(profile, csv_input_str)

        try:
            raw_output = call_api(prompt_text, batch_label, profile)
        except RuntimeError as e:
            print(f"{tag} Batch {batch_no} API FAILED: {e} — skipping batch.")
            failed_ids.extend(batch_ids)
            continue

        records = parse_and_validate(raw_output, batch_rows, batch_label)
        print(f"{tag} Batch {batch_no}: {len(records)}/{len(batch_ids)} valid row(s) parsed.")

        parsed_by_id = {r["seed_id"]: r["text"] for r in records}
        batch_updates: dict[str, str] = {}
        for sid in batch_ids:
            if sid in parsed_by_id:
                batch_updates[sid] = parsed_by_id[sid]
            else:
                print(f"{tag} WARNING: seed_id {sid} missing from parsed output — unrepaired.")
                failed_ids.append(sid)

        if batch_updates:
            # 每 batch 立即写回 CSV
            patched = patch_csv_inplace(output_csv, batch_updates)
            total_patched += patched
            # 更新 checkpoint
            _save_checkpoint(profile, list(batch_updates.keys()))
            print(f"{tag} Batch {batch_no}: wrote {patched} row(s). "
                  f"Total so far: {total_patched}/{total_needed}")

    return total_patched, failed_ids

# =====================================================================
# Main（并发跑多个 profile）
# =====================================================================

def main() -> None:
    print("=" * 62)
    print("  repair_contaminated.py — Cybersecurity Malware Re-translator")
    print("=" * 62)
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Input    : {INPUT_CSV.name}")
    print(f"  Contamination list: {CONTAMINATION_JSON.name}")
    print(f"  Checkpoint: {CHECKPOINT_JSON.name}")
    print("=" * 62)
    print()

    if not CONTAMINATION_JSON.exists():
        raise FileNotFoundError(f"contamination_ids.json not found: {CONTAMINATION_JSON}")
    with open(CONTAMINATION_JSON, encoding="utf-8") as f:
        contamination_map: dict[str, list[str]] = json.load(f)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    with open(INPUT_CSV, encoding="utf-8", newline="") as f:
        en_rows_by_id: dict[str, dict] = {
            row["seed_id"].strip(): row
            for row in csv.DictReader(f)
            if row.get("seed_id")
        }
    print(f"Loaded {len(en_rows_by_id)} rows from CYBER_EN.csv.\n")

    # 只跑有污染行的 profile
    active_profiles = {
        p: ids for p, ids in contamination_map.items() if ids
    }
    skipped = [p for p, ids in contamination_map.items() if not ids]
    for p in skipped:
        print(f"[{p}] No contaminated IDs — skipping.")

    summary: dict[str, dict] = {}

    # 并发跑所有有污染行的 profile
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_profiles)) as executor:
        future_map = {
            executor.submit(repair_profile, profile, ids, en_rows_by_id): profile
            for profile, ids in active_profiles.items()
        }
        for fut in concurrent.futures.as_completed(future_map):
            profile = future_map[fut]
            try:
                repaired, failed = fut.result()
                summary[profile] = {
                    "total":    len(active_profiles[profile]),
                    "repaired": repaired,
                    "failed":   failed,
                }
            except Exception as exc:
                print(f"[{profile}] CRASHED: {exc}")
                summary[profile] = {
                    "total":    len(active_profiles[profile]),
                    "repaired": 0,
                    "failed":   active_profiles[profile],
                }

    print()
    print("=" * 62)
    print("  Repair Summary")
    print("=" * 62)
    total_repaired = 0
    any_failed = False
    for profile, info in summary.items():
        total_repaired += info["repaired"]
        status = "✓" if not info["failed"] else "!"
        print(f"  {status} {profile}: {info['repaired']}/{info['total']} repaired", end="")
        if info["failed"]:
            any_failed = True
            print(f"  — {len(info['failed'])} UNREPAIRED: {info['failed'][:5]}"
                  f"{'...' if len(info['failed']) > 5 else ''}", end="")
        print()
    print(f"\n  Total repaired rows: {total_repaired}")
    if any_failed:
        print("  WARNING: Some rows were not repaired.")
    else:
        print("  All rows repaired successfully.")
        # 清除 checkpoint
        if CHECKPOINT_JSON.exists():
            CHECKPOINT_JSON.unlink()
            print(f"  Checkpoint file removed.")
    print("=" * 62)


if __name__ == "__main__":
    main()
