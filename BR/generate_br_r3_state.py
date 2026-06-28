#!/usr/bin/env python3
"""
BR R3 STATE Synthetic Data Generator
Label  : BR_State_Security_Democratic_Order
Model  : mistralai/mistral-small-4-119b-2603 (NVIDIA NIM)
Output : multilingual/BR/BR-R3/BR_STATE.csv

运行方式：
  export NVIDIA_API_KEY=nvapi-xxxx
  python generate_br_r3_state.py

生成流程：
  - 顺序运行 8 个 shard（001–008），每个 220 条
  - 每个 shard 完成后立即追加写入同一个 CSV（防止中断丢数据）
  - 每批写入前做精确 text 去重（跨 shard 全局去重）
  - 若 shard 缺口，自动补充，最多 MAX_FILL_ROUNDS 轮
"""

import os
import csv
import io
import json
import time
import re
import urllib.request
import urllib.error
from pathlib import Path

# ── 自动加载 .env ─────────────────────────────────────────────────────
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
# ★ CONFIG
# =====================================================================

SAMPLES_PER_SHARD: int = 220   # 每 shard 目标条数
BATCH_SIZE: int        = 60    # 每次 API call 请求的条数（模型单次上限）
MAX_FILL_ROUNDS: int   = 5     # 缺口补充最大轮数

MODEL_NAME    = "mistralai/mistral-small-4-119b-2603"
TEMPERATURE   = 0.9
TOP_P         = 0.95
MAX_TOKENS    = 16000   # 220 条 × ~60 tokens ≈ 13200，留余量

MAX_RETRIES   = 3
RETRY_BACKOFF = 8       # 秒，实际等待 = RETRY_BACKOFF × attempt

SCRIPT_DIR  = Path(__file__).resolve().parent
OUTPUT_DIR  = SCRIPT_DIR / "BR-R3"
OUTPUT_FILE = OUTPUT_DIR / "BR_STATE.csv"

CSV_COLUMNS = ["text", "final_category", "shard_id"]

# =====================================================================
# ★ PROMPT 模板
# =====================================================================

BASE_PROMPT = """\
你是一个用于内容安全训练数据合成的巴西葡语数据生成器。
任务：生成 BR_State_Security_Democratic_Order 正样本，用于多分类文本安全模型训练。

目标标签：
BR_State_Security_Democratic_Order
语言：
* 只使用巴西葡萄牙语。
* 必须自然，像真实巴西用户在社交平台、评论区、WhatsApp、Telegram、X/Twitter、TikTok、YouTube、论坛、新闻评论区写的内容。
* 不要英文、西班牙语、中文。
* 不要翻译腔。
* 可以包含巴西网络口语、错别字、缩写、emoji、hashtag、大写、无标点、重复字母、阴阳怪气、暗示、半句式表达。

标签定义：
BR_State_Security_Democratic_Order 指在巴西流量中，威胁、否定、破坏或动员攻击巴西国家安全、民主宪政秩序、选举合法性、制度稳定、公共秩序、国家合法暴力垄断的内容。

严格只覆盖以下 4 类：

A. 攻击巴西电子投票和选举合法性
包括声称 urnas eletrônicas、apuração、TSE、TRE、STF 或选举结果被系统性操纵、盗窃、无效；否认选举结果合法性；以"选举被偷"为理由动员抵抗、封路、占领、拒绝承认民主程序。

B. 军事干预、政变、AI-5、反宪法动员
包括支持、恢复、正当化或动员 intervenção militar、golpe militar、regime militar、AI-5、关闭 Congresso、关闭 STF/TSE、要求 Forças Armadas/Exército 接管国家、推翻民选政府或取代民主机构。

C. 攻击巴西三权机构、煽动反民主暴力
包括鼓励、组织、赞美、重复或正当化对 STF、TSE、Congresso Nacional、Palácio do Planalto、Praça dos Três Poderes 等民主机构的入侵、占领、破坏、围堵或暴力威胁；包括"重复 8 de janeiro"或类似表达。

D. 本地犯罪派系或民兵挑战国家权威
包括对 PCC、Comando Vermelho/CV、milícia 等巴西犯罪组织/民兵的美化、支持、招募、恐吓、领土控制宣传、社区规则发布、宵禁、替代国家治理、声称派系/民兵比政府更能维持秩序。
注意：普通毒品、普通武器、普通洗钱、普通诈骗不是本标签。只有当犯罪派系/民兵被描述为替代国家权威、控制领土、执行秩序、挑战国家合法暴力垄断时，才属于本标签。

强制排除：
* 不要生成普通政治批评。
* 不要生成普通辱骂候选人、总统、法官、政党、公职人员。
* 不要生成普通新闻报道、中立学术讨论、历史解释、政策争论。
* 不要生成 Pix 诈骗、非法博彩、普通毒品交易、普通武器买卖、普通洗钱。
* 不要生成 Hate Speech、色情内容、CSAM、具体网络攻击代码、诈骗脚本。
* 不要生成真实可执行的暴力行动教程、武器制作、爆炸物、详细袭击步骤。
* 不要让所有样本都包含同一批关键词，必须做表达变化。

全局多样化要求：

长度多样化：
* 30%：短句，5–20词。
* 45%：中句，20–60词。
* 25%：长句或群聊转发，60–120词。

风格多样化（在本 shard 中均衡分布）：
口号式 / 阴谋论式 / 愤怒评论 / 冷嘲热讽 / 群聊转发 / 街头动员 /
评论区反问 / 模糊暗示 / 激进口号 / 新闻评论区长评 /
Telegram 频道广播 / WhatsApp 家族群转发 / TikTok/Reels 短评 /
X/Twitter hashtag 帖 / YouTube 直播弹幕式短句

表达显隐多样化：
* 50% 明确命中关键词。
* 50% 用同义、转述、暗示、缩写、代称、隐喻，例如：
  "as máquinas já vieram prontas"
  "o sistema não deixa o povo escolher"
  "só de verde e amarelo na porta dos quartéis"
  "tem que zerar Brasília"
  "os togados não mandam no povo"
  "a comunidade tem dono e não é o Estado"
  "quem segura a rua é a firma"
  "ordem vem do morro, não da prefeitura"

实体多样化（适当出现，不过度集中）：
TSE, TRE, STF, Congresso, Senado, Câmara, Planalto, Praça dos Três Poderes /
Forças Armadas, Exército, quartel, generais /
urnas eletrônicas, boletim de urna, apuração, eleição, resultado /
AI-5, intervenção, regime militar / 8 de janeiro, Brasília /
PCC, CV, Comando Vermelho, milícia, comunidade, morro, quebrada

地域多样化（少量加入）：
São Paulo, Rio de Janeiro, Brasília, Minas, Bahia, Nordeste, Sul,
favela, comunidade, Baixada, Zona Oeste, interior, litoral

输出格式：
* 输出纯 CSV，第一行为表头：text,final_category
* final_category 固定为：BR_State_Security_Democratic_Order
* 不要输出解释、编号、markdown、代码块。
* text 字段如含逗号、换行或引号，必须用标准 CSV 转义（双引号包裹，内部引号加倍）。
* 每条 text 必须唯一，不生成重复或近重复样本。
"""

SHARD_INSTRUCTIONS = {
    "001": """\
SHARD 001：电子投票/选举合法性 - 明确型
* 主题集中在 urnas eletrônicas、TSE、apuração、resultado、eleição roubada。
* 多用直接表达，但不要每条都写"urna fraudada"。
* 需要包含短评论、群聊转发、hashtag、反问句。
""",
    "002": """\
SHARD 002：电子投票/选举合法性 - 隐晦/阴谋论型
* 主题仍是选举合法性，但减少直白关键词。
* 多用"sistema", "máquinas", "resultado pronto", "ninguém escolheu nada", "teatro eleitoral"等表达。
* 语气偏阴谋论、质疑民主程序、号召不承认结果。
* 不要写成普通政策批评。
""",
    "003": """\
SHARD 003：军事干预/政变/AI-5 - 明确型
* 主题集中在 intervenção militar、Forças Armadas、Exército、AI-5、fechar Congresso/STF/TSE。
* 必须表达支持、呼吁、正当化或动员反宪法接管。
* 不要只是历史讨论军政府。
* 避免详细暴力操作步骤。
""",
    "004": """\
SHARD 004：军事干预/政变/反宪法动员 - 隐晦/口号型
* 减少直接使用 golpe、intervenção。
* 多用"quartel", "verde e amarelo", "os militares precisam agir", "limpar Brasília", "tirar os togados", "botar ordem na marra"等表达。
* 包含街头动员、群聊号召、转发口号。
* 不要变成普通爱国口号，必须有反宪法接管或取代民主机构的含义。
""",
    "005": """\
SHARD 005：三权机构暴力/8 de janeiro/占领破坏 - 明确型
* 主题集中在 STF、TSE、Congresso、Planalto、Praça dos Três Poderes、8 de janeiro。
* 包含赞美、正当化、鼓励重复、组织占领、围堵、破坏民主机构。
* 不要给具体袭击教程、路线、武器或操作步骤。
* 重点是政治暴力动员或正当化。
""",
    "006": """\
SHARD 006：三权机构暴力/反机构威胁 - 变体型
* 减少"8 de janeiro"重复。
* 多用"invadir", "ocupar", "cercar", "tomar", "derrubar as portas", "tirar ministro de lá", "ninguém sai do plenário"等表达。
* 可以有怒评、直播弹幕、新闻评论、反问句、暗讽。
* 必须针对巴西民主机构，不要只是骂某个法官。
""",
    "007": """\
SHARD 007：PCC/CV/milícia 挑战国家权威 - 明确型
* 主题集中在 PCC、CV、Comando Vermelho、milícia。
* 必须体现派系/民兵替代国家治理、控制社区、发布规则、宵禁、招募、威胁、宣称国家无权进入。
* 不要只是普通毒品交易、普通枪支、普通诈骗。
* 不要给具体犯罪教程。
* 语言可以像社区公告、恐吓标语、匿名评论、群消息。
""",
    "008": """\
SHARD 008：混合难例/边界增强 shard
* 混合 A/B/C/D 四类，每类约 55 条。
* 重点生成容易和 safe、Harassment、Politically_Sensitive_Topics、Dangerous_Content 混淆但应归入 BR_State_Security_Democratic_Order 的样本。
* 包括：
  1. 看似普通政治评论，但实际否认选举合法性。
  2. 看似爱国口号，但实际要求军队取代民主机构。
  3. 看似骂 STF/TSE，但实际鼓励占领、关闭、围堵机构。
  4. 看似犯罪派系评论，但实际宣称派系/民兵替代国家治理。
* 不要生成明显 safe 的普通批评或普通辱骂。
""",
}

def build_prompt(shard_id: str, n: int) -> str:
    return (
        BASE_PROMPT
        + "\n"
        + SHARD_INSTRUCTIONS[shard_id]
        + f"\n现在生成 {n} 条，输出 CSV（含表头 text,final_category）：\n"
    )

# =====================================================================
# API
# =====================================================================

def _get_api_key() -> str:
    key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("NIM_API_KEY")
        or os.getenv("NIW_API_KEY")
        or ""
    )
    if not key:
        raise RuntimeError(
            "API key not found. Set NVIDIA_API_KEY:\n  export NVIDIA_API_KEY=nvapi-xxxx"
        )
    return key


def call_api(prompt_text: str, label: str) -> str:
    base_url = (
        os.getenv("NIM_BASE_URL")
        or os.getenv("NVIDIA_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    ).rstrip("/")

    api_key = _get_api_key()
    endpoint = f"{base_url}/chat/completions"

    for attempt in range(1, MAX_RETRIES + 1):
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a content-safety synthetic data generator. "
                        "Output only valid CSV as instructed. "
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
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                time.sleep(2)
                return data["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"  [{label}] HTTP {e.code} (attempt {attempt}/{MAX_RETRIES}): {body[:400]}")
            if attempt < MAX_RETRIES:
                wait = 60 * attempt if e.code == 429 else RETRY_BACKOFF * attempt
                print(f"  [{label}] Waiting {wait}s...")
                time.sleep(wait)

        except Exception as e:
            print(f"  [{label}] Error (attempt {attempt}/{MAX_RETRIES}): {repr(e)[:300]}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(f"[{label}] API failed after {MAX_RETRIES} attempts.")

# =====================================================================
# CSV 解析
# =====================================================================

_EN_FUNCTION_WORDS = {
    "the", "and", "you", "your", "with", "for", "this", "that",
    "are", "have", "can", "will", "want", "just", "what", "when",
    "but", "not", "its", "been", "they", "them", "our", "more",
    "some", "come", "here", "now", "let", "get", "got",
    "don't", "i'm", "i'll", "i've", "we're", "you're",
}
_PT_OK = {
    "live", "chat", "hot", "vip", "dm", "link", "story", "stories",
    "feed", "post", "follow", "like", "bio", "sex", "sexy", "pack",
    "fake", "real", "ok", "baby", "show", "group", "free", "top",
}

def _is_english(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    en = [w for w in words if w not in _PT_OK]
    return sum(1 for w in en if w in _EN_FUNCTION_WORDS) >= 3


def parse_csv_output(raw: str, shard_id: str, label: str) -> list[dict]:
    """
    解析模型输出的 CSV，返回 {'text': ..., 'final_category': ..., 'shard_id': ...} 列表。
    鲁棒处理：跳过 markdown 代码块标记、空行、英文行、格式错误行。
    """
    # 去掉 markdown 代码块
    raw = re.sub(r"```[^\n]*\n?", "", raw).strip()

    records = []
    reader = csv.reader(io.StringIO(raw))
    header_skipped = False

    for row in reader:
        if not row:
            continue
        # 跳过表头（第一次遇到 text/final_category 的行）
        if not header_skipped and row[0].strip().lower() in ("text", '"text"'):
            header_skipped = True
            continue

        if len(row) < 2:
            print(f"  [{label}] Skip short row: {row}")
            continue

        text = row[0].strip().strip('"')
        category = row[1].strip().strip('"')

        if not text:
            continue
        if _is_english(text):
            print(f"  [{label}] English rejected: \"{text[:70]}\"")
            continue
        if "BR_State_Security_Democratic_Order" not in category and category:
            # 容错：模型可能输出正确 category 但大小写略有变化，直接覆盖
            pass

        records.append({
            "text": text,
            "final_category": "BR_State_Security_Democratic_Order",
            "shard_id": shard_id,
        })

    return records

# =====================================================================
# 写入
# =====================================================================

def append_records(records: list[dict], file_exists: bool) -> None:
    with open(OUTPUT_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)

# =====================================================================
# Shard 运行
# =====================================================================

def run_shard(shard_id: str, seen_texts: set[str], file_exists: bool) -> tuple[int, bool]:
    """
    运行单个 shard，返回 (新增条数, file_exists)。
    seen_texts 跨 shard 全局共享，保证全局去重。
    """
    label = f"STATE/{shard_id}"
    collected = 0
    round_idx = 0

    print(f"\n[{label}] ▶ Start (target={SAMPLES_PER_SHARD})")

    while collected < SAMPLES_PER_SHARD and round_idx < MAX_FILL_ROUNDS:
        needed = SAMPLES_PER_SHARD - collected
        n = min(BATCH_SIZE, needed)

        print(f"[{label}/r{round_idx}] Requesting {n} rows...")
        try:
            raw = call_api(build_prompt(shard_id, n), f"{label}/r{round_idx}")
        except Exception as e:
            print(f"[{label}/r{round_idx}] ✗ {e}")
            round_idx += 1
            continue

        records = parse_csv_output(raw, shard_id, label)
        print(f"[{label}/r{round_idx}] Parsed {len(records)} valid rows")

        # 精确去重
        deduped = []
        for rec in records:
            t = rec["text"]
            if t not in seen_texts:
                seen_texts.add(t)
                deduped.append(rec)

        dup_removed = len(records) - len(deduped)
        if dup_removed:
            print(f"[{label}/r{round_idx}] Removed {dup_removed} duplicates, kept {len(deduped)}")

        if deduped:
            # 若本 shard 超出目标，截断
            remaining = SAMPLES_PER_SHARD - collected
            deduped = deduped[:remaining]
            append_records(deduped, file_exists)
            file_exists = True
            collected += len(deduped)
            print(f"[{label}/r{round_idx}] Written {len(deduped)} → shard total {collected}/{SAMPLES_PER_SHARD}")
        else:
            print(f"[{label}/r{round_idx}] All duplicates, retrying...")

        round_idx += 1

    status = "✓ Complete" if collected >= SAMPLES_PER_SHARD else f"⚠ Incomplete ({collected}/{SAMPLES_PER_SHARD})"
    print(f"[{label}] {status}")
    return collected, file_exists

# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 62)
    print("  BR R3 STATE Synthetic Data Generator")
    print("=" * 62)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Shards     : 001–008")
    print(f"  Per shard  : {SAMPLES_PER_SHARD} rows")
    print(f"  Total goal : {SAMPLES_PER_SHARD * 8} rows")
    print(f"  Batch size : {BATCH_SIZE} rows/call")
    print(f"  Output     : {OUTPUT_FILE}")
    print("=" * 62)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载已有数据，建立全局 seen_texts
    seen_texts: set[str] = set()
    file_exists = OUTPUT_FILE.exists()
    if file_exists:
        with open(OUTPUT_FILE, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                t = row.get("text", "").strip()
                if t:
                    seen_texts.add(t)
        print(f"  Resumed: {len(seen_texts)} existing rows loaded")

    total = 0
    for shard_id in ["001", "002", "003", "004", "005", "006", "007", "008"]:
        # 检查此 shard 是否已完成
        if file_exists:
            with open(OUTPUT_FILE, encoding="utf-8", newline="") as f:
                shard_count = sum(
                    1 for row in csv.DictReader(f)
                    if row.get("shard_id", "") == shard_id
                )
            if shard_count >= SAMPLES_PER_SHARD:
                print(f"\n[STATE/{shard_id}] Already complete ({shard_count} rows), skipped.")
                total += shard_count
                continue

        added, file_exists = run_shard(shard_id, seen_texts, file_exists)
        total += added

    print(f"\n{'=' * 62}")
    print(f"  Done. Total rows in {OUTPUT_FILE.name}: {total}")
    print("=" * 62)


if __name__ == "__main__":
    main()
