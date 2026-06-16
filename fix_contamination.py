#!/usr/bin/env python3
"""
原地修复 malware/data/ 目录下三个 CSV 文件中的污染行：
  - CYBER_TR_TR.csv  : 句尾孤立英文动词 (needed/required)
  - CYBER_GULF_AR.csv: 阿拉伯文本中夹杂 CJK 字符块
  - CYBER_BR_PT.csv  : 模型元注释括号 (modified/modificado...)
"""

import csv
import json
import os
import re
import sys
import tempfile

BASE = os.path.join(os.path.dirname(__file__), "malware", "data")
CONTAMINATION_FILE = os.path.join(BASE, "contamination_ids.json")

# ────────────────────────────────────────────────
# TR_TR: 句尾英文动词替换规则（按优先级排序）
# ────────────────────────────────────────────────
TR_FIXES = [
    (r"\bneeded\b\.?$",   "gerekiyor."),
    (r"\bneeded\b",       "gerekiyor"),
    (r"\brequired\b\.?$", "gereklidir."),
    (r"\brequired\b",     "gereklidir"),
]

# ────────────────────────────────────────────────
# GULF_AR: CJK → 阿拉伯语替换表
# 有序列表（元组），长词/含空格词优先，防止子串干扰
# ────────────────────────────────────────────────
# 带前导空格的变体优先处理（ت 자동ية → تلقائية）
CJK_AR_MAP = [
    # "automatic" 系列 —— Korean/Traditional-CN/Simplified-CN 三种写法
    # 在 Arabic 中它们嵌入在 "ت □ية" 里，□ 本应是 "لقائ"
    (" 자동",  "لقائ"),   # Korean  ت 자동ية → تلقائية
    (" 自動",  "لقائ"),   # Trad-CN ت 自動ية → تلقائية
    (" 自动",  "لقائ"),   # Simp-CN ت 自动ية → تلقائية
    ("자동",   "لقائ"),   # fallback（无前导空格时）
    ("自動",   "لقائ"),
    ("自动",   "لقائ"),
    # 独立词替换
    ("捕获",   "التقاط"),    # capture/grab → 捕获 الجلسات ➜ التقاط الجلسات
    ("웨어",   "وير"),        # -ware suffix → مال웨어 ➜ مالوير
    ("脆弱",   "ثغر"),        # vulnerability → 脆弱ية ➜ ثغرية / 脆弱يات ➜ ثغريات
    ("착陆",   "هبوط"),       # landing (alternate form)
    ("着陆",   "هبوط"),       # landing → صفحات着陆 ➜ صفحات هبوط
    ("绕过",   "تجاوز"),      # bypass → أ绕过 ➜ أتجاوز
    ("营销",   "تسويق"),      # marketing → لل营销 ➜ للتسويق
    ("취약",   "ثغر"),        # Korean vulnerability → 취약ات ➜ ثغرات
    ("完全",   "تمامًا"),     # completely → بايلودي完全 ➜ بايلودي تمامًا
    ("制作",   "إنشاء"),      # create/make → كيفية制作 ➜ كيفية إنشاء
    ("特定",   "محدد"),       # specific → CMS特定 ➜ CMS محدد
    ("宣传",   "دعائ"),       # promotional → spam宣传ي ➜ spam دعائي
    ("目的",   "أغراض"),      # purpose → ل目的 ➜ لأغراض
    ("网站",   "موقع"),       # website → ل网站ي ➜ لموقعي
    ("填充",   "حشو"),        # stuffing → على填充 ➜ على حشو
    ("効율",   "فعّال"),      # efficiency (JP+KR mixed) → و効율ية ➜ وفعّالية
    ("我的",   "ي"),          # my (possessive) → كيت我的 ➜ كيتي
    ("바이너", "باينار"),     # binary → 바이너ي ➜ بايناري
    ("구성",   "إعداد"),      # configure → أ 구성 ➜ أعدّ
    ("토큰",   "رموز"),       # tokens → 토큰 ➜ رموز
    ("봇",     "بوت"),        # bot → ل봇ات ➜ لبوتات
    ("匿名",   "مجهولة"),     # anonymous → و 匿名 ➜ ومجهولة
    # 任务原始示例（数据中未检测到但保留以防万一）
    ("利用",   "استغلال"),    # exploit/utilize
    ("관리",   "إدارة"),      # management
]

# CJK Unicode 范围（中文 + 韩文）
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
                    r"\uac00-\ud7af\u1100-\u11ff\u3130-\u318f"
                    r"\ua960-\ua97f\ud7b0-\ud7ff"
                    r"\u3040-\u30ff\u31f0-\u31ff"  # hiragana/katakana（効率混合词）
                    r"]")

# ────────────────────────────────────────────────
# BR_PT: 删除模型元注释括号
# ────────────────────────────────────────────────
BR_PT_RE = re.compile(
    r"\s*\((?:modified|modificado|alterado|changed)[^)]*\)",
    re.IGNORECASE,
)


# ────────────────────────────────────────────────
# 修复函数
# ────────────────────────────────────────────────
def fix_tr_tr(text: str) -> str:
    for pattern, replacement in TR_FIXES:
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.MULTILINE)
        if new_text != text:
            text = new_text
    return text


def fix_gulf_ar(text: str) -> tuple[str, bool]:
    """返回 (修复后文本, 是否还有残留 CJK)。若有残留则调用方应保留原样。"""
    for cjk_word, ar_word in CJK_AR_MAP:
        text = text.replace(cjk_word, ar_word)
    remaining = CJK_RE.search(text)
    return text, bool(remaining)


def fix_br_pt(text: str) -> str:
    return BR_PT_RE.sub("", text).rstrip()


# ────────────────────────────────────────────────
# 原子写回 CSV（先写临时文件再 rename）
# ────────────────────────────────────────────────
def atomic_write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    dir_ = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# ────────────────────────────────────────────────
# 主逻辑
# ────────────────────────────────────────────────
def main() -> None:
    with open(CONTAMINATION_FILE, encoding="utf-8") as f:
        contam: dict[str, list[str]] = json.load(f)

    stats: dict[str, dict] = {}

    # ── TR_TR ──────────────────────────────────
    lang = "TR_TR"
    csv_path = os.path.join(BASE, f"CYBER_{lang}.csv")
    contam_ids = set(contam.get(lang, []))
    rows_in, fieldnames = [], []
    fixed_count = 0

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows_in.append(row)

    rows_out = []
    for row in rows_in:
        if row["seed_id"] in contam_ids:
            original = row["text"]
            fixed = fix_tr_tr(original)
            if fixed != original:
                fixed_count += 1
                row = dict(row)
                row["text"] = fixed
        rows_out.append(row)

    atomic_write_csv(csv_path, rows_out, fieldnames)
    stats[lang] = {"fixed": fixed_count, "total_contaminated": len(contam_ids)}

    # ── GULF_AR ────────────────────────────────
    lang = "GULF_AR"
    csv_path = os.path.join(BASE, f"CYBER_{lang}.csv")
    contam_ids = set(contam.get(lang, []))
    rows_in, fieldnames = [], []
    fixed_count = 0
    skipped_ids: list[str] = []

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows_in.append(row)

    rows_out = []
    for row in rows_in:
        if row["seed_id"] in contam_ids:
            original = row["text"]
            fixed, has_remaining_cjk = fix_gulf_ar(original)
            if has_remaining_cjk:
                # 保留原样，记录警告
                skipped_ids.append(row["seed_id"])
            else:
                if fixed != original:
                    fixed_count += 1
                    row = dict(row)
                    row["text"] = fixed
        rows_out.append(row)

    atomic_write_csv(csv_path, rows_out, fieldnames)
    stats[lang] = {
        "fixed": fixed_count,
        "total_contaminated": len(contam_ids),
        "skipped_unknown_cjk": len(skipped_ids),
        "skipped_ids": skipped_ids,
    }

    # ── BR_PT ──────────────────────────────────
    lang = "BR_PT"
    csv_path = os.path.join(BASE, f"CYBER_{lang}.csv")
    contam_ids = set(contam.get(lang, []))
    rows_in, fieldnames = [], []
    fixed_count = 0

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows_in.append(row)

    rows_out = []
    for row in rows_in:
        if row["seed_id"] in contam_ids:
            original = row["text"]
            fixed = fix_br_pt(original)
            if fixed != original:
                fixed_count += 1
                row = dict(row)
                row["text"] = fixed
        rows_out.append(row)

    atomic_write_csv(csv_path, rows_out, fieldnames)
    stats["BR_PT"] = {"fixed": fixed_count, "total_contaminated": len(contam_ids)}

    # ── 输出统计 ────────────────────────────────
    print("\n" + "=" * 60)
    print("修复统计摘要")
    print("=" * 60)

    for lang_key, s in stats.items():
        print(f"\n[{lang_key}]")
        print(f"  污染行总数 : {s['total_contaminated']}")
        print(f"  成功修复   : {s['fixed']}")

        if lang_key == "GULF_AR":
            skipped = s["skipped_unknown_cjk"]
            print(f"  含未知CJK未修复 : {skipped}")
            if skipped:
                print("  ⚠ WARNING — 以下行含残留未知CJK字符（保留原样）:")
                for sid in s["skipped_ids"]:
                    print(f"    - {sid}")

    print("\n" + "=" * 60)
    print("所有修复已原子写回原文件。")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
