#!/usr/bin/env python3
"""
BR R3 STATE Patch Generator（补充轮）
Label  : BR_State_Security_Democratic_Order
Model  : mistralai/mistral-small-4-119b-2603 (NVIDIA NIM)
Output : multilingual/BR/BR-R3/BR_STATE.csv  （追加写入，不覆盖已有数据）

补充目标（每 shard）：
  001: +70   002: +95   003: +60   004: +75
  005: +70   006: +85   007: +75   008: +70
  合计: +600

运行方式：
  python generate_br_r3_state_patch.py
"""

import os, csv, io, json, time, re, urllib.request, urllib.error
from pathlib import Path

# ── .env ─────────────────────────────────────────────────────────────
def _load_dotenv(p: Path):
    if not p.exists(): return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

_load_dotenv(Path(__file__).resolve().parent / ".env")

# =====================================================================
# ★ CONFIG
# =====================================================================

# 每个 shard 本次补充目标（不计已有行数）
SHARD_TARGETS: dict[str, int] = {
    "001": 70,
    "002": 95,
    "003": 60,
    "004": 75,
    "005": 70,
    "006": 85,
    "007": 75,
    "008": 70,
}

BATCH_SIZE      = 55      # 每次 API 请求条数（严格约束下适当减少避免超时）
MAX_FILL_ROUNDS = 8       # 每 shard 最大补充轮数
MODEL_NAME      = "mistralai/mistral-small-4-119b-2603"
TEMPERATURE     = 0.9
TOP_P           = 0.95
MAX_TOKENS      = 10000
MAX_RETRIES     = 3
RETRY_BACKOFF   = 8

SCRIPT_DIR  = Path(__file__).resolve().parent
OUTPUT_DIR  = SCRIPT_DIR / "BR-R3"
OUTPUT_FILE = OUTPUT_DIR / "BR_STATE.csv"
CSV_COLUMNS = ["text", "final_category", "shard_id"]

# =====================================================================
# ★ PROMPT
# =====================================================================

BASE_PROMPT = """\
你是一个用于内容安全训练数据合成的巴西葡语数据生成器。
任务：生成 BR_State_Security_Democratic_Order 正样本，用于多分类文本安全模型训练。

目标标签：BR_State_Security_Democratic_Order

语言规范：
* 只使用巴西葡萄牙语。
* 必须自然，像真实巴西用户在 WhatsApp、Telegram、X/Twitter、TikTok、YouTube、论坛、新闻评论区写的内容。
* 不要英文、西班牙语、中文；不要翻译腔。
* 可以包含网络口语、错别字、缩写、emoji、hashtag、大写、无标点、重复字母、阴阳怪气、暗示、半句式表达。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
标签定义：
BR_State_Security_Democratic_Order 指在巴西流量中，威胁、否定、破坏或动员攻击巴西国家安全、
民主宪政秩序、选举合法性、制度稳定、公共秩序、国家合法暴力垄断的内容。

严格覆盖以下 4 类：
A. 攻击巴西电子投票和选举合法性
B. 军事干预、政变、AI-5、反宪法动员
C. 攻击巴西三权机构、煽动反民主暴力
D. 本地犯罪派系或民兵挑战国家权威

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【硬性长度约束】
* 禁止生成少于 8 个葡语词的样本。
* 每条 text 必须是完整可判断句，不能是实体词、半句话、口号碎片。
* 目标平均长度 35–45 个葡语词。
* 长度分布：
  - 15%：8–15 词。必须包含明确风险语义。
  - 30%：16–30 词。普通社媒评论。
  - 40%：31–55 词。主力样本，包含背景、主体、风险行为或动员意图。
  - 15%：56–90 词。模拟群转发、新闻长评、X 长帖。
* 不要为了变长而重复关键词；不要把一句话机械扩写成模板。
* 长样本必须增加真实上下文：谁在说、针对哪个机构、为什么、有何风险动作。

【禁止残片样本】
以下类型禁止生成：
✗ 单独实体：TSE / STF / Congresso / Militares / PCC / CV
✗ 半句话：Se o povo... / Se a justiça... / Na quebrada...
✗ 只有地点：Na Zona Oeste / Na favela / Em Brasília
✗ 只有泛口号：Não é golpe / Chega disso / O povo acordou
✗ 只有抽象判断：Democracia de mentira / Sistema podre / Tudo armado
每条样本必须不依赖外部上下文，也能判断为何属于 BR_State_Security_Democratic_Order。

【隐晦选举类锚点约束】
即使不用 TSE/urna/fraude 等直白词，也必须包含以下至少一个巴西选举锚点：
Brasil / brasileiro / Justiça Eleitoral / TSE / TRE / urna eletrônica / urnas /
apuração / boletim de urna / eleição brasileira / resultado no Brasil / Brasília /
voto auditável / sistema eleitoral brasileiro
✗ 禁止生成只有泛化表达、缺少巴西制度锚点的句子。

【选举合法性类增强规则（A 类）】
必须补充至少一个：
① 涉及哪个选举实体/流程：TSE / TRE / urnas eletrônicas / apuração / boletim de urna
② 风险动作：不承认结果、要求取消、号召抵抗、封路、去 quartel、要求军队介入
③ 系统性指控：声称系统性操控/盗票/造假，而非普通怀疑或要求透明
✗ 禁止生成：普通要求透明/审计 / 中立讨论选举制度 / 泛泛"我不信任政治"

【军事干预/政变/AI-5 增强规则（B 类）】
必须同时满足：
① 至少一个实体：Forças Armadas / Exército / quartel / generais / militares /
   AI-5 / STF / TSE / Congresso / Brasília / governo eleito
② 至少一个反宪法动作：intervenção militar / militares assumirem / fechar STF /
   fechar TSE / fechar Congresso / restaurar AI-5 / limpar Brasília / tirar os togados /
   derrubar o governo eleito / substituir as instituições
必须表达支持、呼吁、正当化或动员军事/反宪法接管。
✗ 禁止：只是历史讨论军政府 / 普通爱国口号 / 普通批评 STF/TSE

【三权机构暴力/8 de janeiro 增强规则（C 类）】
必须同时满足：
① 明确针对机构：STF / TSE / Congresso / Senado / Câmara / Palácio do Planalto /
   Praça dos Três Poderes / Brasília
② 至少一个反民主行为：invadir / ocupar / cercar / tomar / fechar / derrubar portas /
   quebrar tudo / impedir sessão / repetir 8 de janeiro / fazer de novo o 8/1 /
   tirar ministro de lá / ninguém sai do plenário
③ 语气上体现鼓励、组织、赞美、重复、正当化、威胁或动员
✗ 禁止：只是骂某个法官 / 只说 STF/TSE 很差 / 详细袭击教程/路线/武器

【PCC/CV/milícia 挑战国家权威增强规则（D 类）】
必须同时满足：
① 至少一个实体：PCC / CV / Comando Vermelho / milícia / firma / facção / comando / o movimento
② 明确体现替代国家治理：Estado não entra / polícia não manda / quem manda é a firma /
   toque de recolher imposto pela facção / prefeitura perdeu autoridade /
   a lei é do PCC/CV/milícia / comércio só abre com autorização da facção /
   morador obedece regra do comando / governo não manda na área /
   a ordem vem do morro, não do Estado
✗ 禁止：普通毒品/武器/洗钱/诈骗/普通犯罪威胁 / 只有 favela 但无派系+替代治理含义

【普通政治批评排除规则】
以下内容禁止作为正样本：
✗ Lula é corrupto / Bolsonaro é incompetente / STF é uma vergonha
✗ Não gosto do governo / O Congresso não trabalha / Alexandre de Moraes é autoritário
✗ O Brasil precisa de reforma política / Precisamos de mais transparência nas urnas
✗ Quero auditoria no sistema eleitoral / O governo mente para o povo
只有进一步涉及系统性选举造假/否认结果/号召抵抗/军事干预/关闭机构/占领三权/派系替代治理时，才可生成。

【关键词多样化规则】
同一 shard 内，以下词组不得反复出现：
"urna fraudada" / "TSE fraudou" / "intervenção militar já" / "fechar STF" / "PCC manda" / "CV manda"
必须使用多种表达方式：明确 / 隐晦 / 反问 / 群聊转发 / Telegram 广播 /
WhatsApp 家族群 / X/Twitter 帖 / TikTok 短评 / 新闻长评 / 阴谋论 / 愤怒动员 / 冷嘲热讽

【CSV 格式硬约束】
* 每条样本独占一行 CSV。
* text 字段内部禁止换行。
* text 字段不得包含：final_category / BR_State_Security_Democratic_Order /
  CSV header / 标签名 / markdown / 编号 / 解释
* 不要把多条样本拼进同一个 text 字段。
* text 含逗号/引号时，必须使用标准 CSV 转义（双引号包裹，内部引号加倍）。
* final_category 固定为：BR_State_Security_Democratic_Order

【生成前自检（每条）】
1. 少于 8 个词？→ 不生成
2. 只是实体词/半句/口号碎片？→ 不生成
3. 缺少巴西本地锚点？→ 不生成
4. 只是普通政治批评/普通辱骂/普通犯罪？→ 不生成
5. 重复上一条的句式或关键词组合？→ 改写
6. 无法不依赖外部上下文判断为 BR_State？→ 不生成
7. 包含标签泄漏/CSV header/多行拼接？→ 重写

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输出格式：
* 输出纯 CSV，第一行为表头：text,final_category
* 之后每行一条样本，不输出任何其他内容。
* text 含逗号/引号时使用标准 CSV 转义。
"""

SHARD_INSTRUCTIONS = {
    "001": """\
SHARD 001：选举合法性 - 明确型
* 主题集中在 urnas eletrônicas、TSE、apuração、resultado、eleição roubada。
* 每条必须包含巴西选举锚点（TSE/TRE/urnas/apuração 等）。
* 必须有风险动作或系统性指控，不只是"要求透明"。
* 包含短评、群聊转发、hashtag、反问句，但不能只有关键词。
* 优先 25–55 词，包含：主体 + 选举流程/实体 + 舞弊指控 + 后续态度或动员意图。
""",
    "002": """\
SHARD 002：选举合法性 - 隐晦/阴谋论型
* 主题仍是选举合法性，但减少直白关键词。
* 即使不用 TSE/urna/fraude，也必须包含至少一个巴西选举锚点。
* 多用阴谋论语气，质疑民主程序，号召不承认结果。
* 禁止生成：只有泛化表达（"o sistema não deixa o povo escolher"）但无巴西锚点的句子。
* 优先 25–55 词，可见：为什么不信任选举结果、指向哪个巴西制度实体、有什么抵抗意图。
""",
    "003": """\
SHARD 003：军事干预/政变/AI-5 - 明确型
* 主题集中在 intervenção militar、Forças Armadas、Exército、AI-5、fechar Congresso/STF/TSE。
* 每条必须明确出现军队/军事实体 + 反宪法动作。
* 必须表达支持、呼吁、正当化或动员反宪法接管。
* 不要只是历史讨论军政府；不要只是普通爱国口号。
* 优先 30–60 词，说清楚：为什么要求军队介入、要取代哪个机构、支持什么行动。
""",
    "004": """\
SHARD 004：军事干预/反宪法动员 - 隐晦/口号型
* 减少直接使用 golpe、intervenção，但必须仍能看出要求军队行动或关闭机构。
* 多用"quartel", "verde e amarelo", "os militares precisam agir", "limpar Brasília",
  "tirar os togados", "botar ordem na marra"等表达。
* 包含街头动员、群聊号召、转发口号，但每条要有足够上下文。
* 不要变成普通爱国口号，必须有取代民主机构或反宪法接管含义。
* 优先 30–55 词。
""",
    "005": """\
SHARD 005：三权机构暴力/8 de janeiro - 明确型
* 主题集中在 STF、TSE、Congresso、Planalto、Praça dos Três Poderes、8 de janeiro。
* 每条必须同时出现：具体民主机构 + 占领/破坏/围堵/重复8/1 等反民主行为。
* 语气体现赞美、正当化、鼓励重复、组织、动员。
* 不要给具体袭击教程、路线、武器、操作步骤。
* 优先 25–55 词，包含：机构 + 行为 + 动员或正当化理由。
""",
    "006": """\
SHARD 006：三权机构暴力/反机构威胁 - 变体型
* 减少"8 de janeiro"重复，用多种方式表达类似含义。
* 多用"invadir", "ocupar", "cercar", "tomar", "derrubar as portas",
  "tirar ministro de lá", "ninguém sai do plenário"等表达。
* 可以有怒评、直播弹幕、新闻评论、反问句、暗讽。
* 必须针对具体巴西民主机构，不要只是骂某个法官。
* 每条必须明确出现机构实体 + 反民主行为语义。
""",
    "007": """\
SHARD 007：PCC/CV/milícia 挑战国家权威 - 明确型
* 主题集中在 PCC、CV、Comando Vermelho、milícia。
* 每条必须同时包含：派系/民兵实体 + 替代国家治理/挑战国家权威的明确含义。
* 重点是"国家不再有权威，派系/民兵制定规则、控制社区/商业/居民行动"。
* 不要只是普通毒品交易、普通枪支、普通诈骗、普通犯罪威胁。
* 语言可以像：社区公告、恐吓标语、匿名评论、群消息、社媒评论。
* 优先 30–70 词。
""",
    "008": """\
SHARD 008：混合难例/边界增强
* 混合 A/B/C/D 四类，每类约各占 1/4，但每条都必须满足对应类别的硬约束。
* 重点生成容易和 Safe / Harassment / Politically_Sensitive_Topics / Dangerous_Content
  混淆但仍应归入 BR_State_Security_Democratic_Order 的样本：
  1. 看似普通政治评论，但实际否认选举合法性 + 有风险动作。
  2. 看似爱国口号，但实际要求军队取代民主机构。
  3. 看似骂 STF/TSE，但实际鼓励占领/关闭/围堵机构。
  4. 看似犯罪派系评论，但实际宣称派系/民兵替代国家治理。
* 每条最好 25–60 词，能清楚看出为何不是普通政治批评，而是威胁民主秩序/选举合法性/三权运行/国家权威。
* 不要生成明显 safe 的普通批评，不要生成普通辱骂。
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

def call_api(prompt: str, label: str) -> str:
    base_url = (
        os.getenv("NIM_BASE_URL") or os.getenv("NVIDIA_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    ).rstrip("/")
    api_key = (
        os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
        or os.getenv("NIW_API_KEY") or ""
    )
    if not api_key:
        raise RuntimeError("API key not found. Set NVIDIA_API_KEY.")

    for attempt in range(1, MAX_RETRIES + 1):
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": (
                    "You are a content-safety synthetic data generator. "
                    "Output only valid CSV as instructed. "
                    "No markdown, no explanation, no extra text."
                )},
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

_EN_FW = {"the","and","you","your","with","for","this","that","are","have",
          "can","will","want","just","what","when","but","not","its","been",
          "they","them","our","more","some","come","here","now","let","get","got"}
_PT_OK = {"live","chat","hot","vip","dm","link","story","stories","feed","post",
          "follow","like","bio","sex","sexy","pack","fake","real","ok","baby",
          "show","group","free","top","online"}

def _is_english(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    en = [w for w in words if w not in _PT_OK]
    return sum(1 for w in en if w in _EN_FW) >= 3

def _word_count(text: str) -> int:
    return len(text.split())

def parse_csv_output(raw: str, shard_id: str, label: str) -> list[dict]:
    raw = re.sub(r"```[^\n]*\n?", "", raw).strip()
    records = []
    header_skipped = False

    try:
        reader = csv.reader(io.StringIO(raw))
    except Exception:
        return records

    for row in reader:
        if not row:
            continue
        if not header_skipped and row[0].strip().lower() in ("text", '"text"'):
            header_skipped = True
            continue
        if len(row) < 2:
            print(f"  [{label}] Skip short row: {row}")
            continue

        text = row[0].strip().strip('"')
        if not text:
            continue

        # 硬性过滤
        wc = _word_count(text)
        if wc < 8:
            print(f"  [{label}] Too short ({wc}w): \"{text[:60]}\"")
            continue
        if _is_english(text):
            print(f"  [{label}] English rejected: \"{text[:60]}\"")
            continue
        # 禁止标签泄漏
        if "BR_State_Security_Democratic_Order" in text or "final_category" in text:
            print(f"  [{label}] Label leak: \"{text[:60]}\"")
            continue

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
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            w.writeheader()
        w.writerows(records)

# =====================================================================
# Shard 运行
# =====================================================================

def run_shard(shard_id: str, target: int, seen_texts: set[str], file_exists: bool) -> tuple[int, bool]:
    label = f"STATE-P/{shard_id}"
    collected = 0
    round_idx = 0

    print(f"\n[{label}] ▶ Start (补充目标={target})")

    while collected < target and round_idx < MAX_FILL_ROUNDS:
        needed = target - collected
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

        deduped = []
        for rec in records:
            t = rec["text"]
            if t not in seen_texts:
                seen_texts.add(t)
                deduped.append(rec)

        dup_removed = len(records) - len(deduped)
        if dup_removed:
            print(f"[{label}/r{round_idx}] Removed {dup_removed} dups, kept {len(deduped)}")

        if deduped:
            remaining = target - collected
            deduped = deduped[:remaining]
            append_records(deduped, file_exists)
            file_exists = True
            collected += len(deduped)
            print(f"[{label}/r{round_idx}] +{len(deduped)} written → 本轮合计 {collected}/{target}")
        else:
            print(f"[{label}/r{round_idx}] All duplicates, retrying...")

        round_idx += 1

    status = "✓ Complete" if collected >= target else f"⚠ Incomplete ({collected}/{target})"
    print(f"[{label}] {status}")
    return collected, file_exists

# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 62)
    print("  BR R3 STATE Patch Generator（补充轮）")
    print("=" * 62)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Output     : {OUTPUT_FILE}")
    total_target = sum(SHARD_TARGETS.values())
    print(f"  补充总目标 : {total_target} 行")
    for sid, t in SHARD_TARGETS.items():
        print(f"    shard {sid}: +{t}")
    print("=" * 62)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载已有数据 → 全局 seen_texts
    seen_texts: set[str] = set()
    file_exists = OUTPUT_FILE.exists()
    if file_exists:
        with open(OUTPUT_FILE, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                t = row.get("text", "").strip()
                if t:
                    seen_texts.add(t)
        print(f"\n  已有数据：{len(seen_texts)} 行（已加入去重集合）")

    total_added = 0
    for shard_id, target in SHARD_TARGETS.items():
        added, file_exists = run_shard(shard_id, target, seen_texts, file_exists)
        total_added += added

    print(f"\n{'=' * 62}")
    print(f"  本次新增：{total_added} 行")
    # 读取最终总行数
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8", newline="") as f:
            final_total = sum(1 for _ in csv.DictReader(f))
        print(f"  文件总行数：{final_total} 行")
    print("=" * 62)


if __name__ == "__main__":
    main()
