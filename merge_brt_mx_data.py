"""
Merge train_*.csv + *_new_data.csv for BR / TR / MX.

Keeps both original label schemas:
  - label_name  (single-label taxonomy from train CSV)
  - 7-dim raw   (hate_speech, false_info, violence, harassment, obscenity, illegal, national_security)

Outputs per-language merged CSV + distribution summary toward TARGET_N=5000 per label.
"""

import csv
import hashlib
import os
import re
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_N = 5000

LANG_CONFIG = {
    "BR": {
        "train": "BR/train_BR.csv",
        "new": "BR/BR_new_data.csv",
    },
    "TR": {
        "train": "TR/train_TR.csv",
        "new": "TR/TR_new_data.csv",
    },
    "MX": {
        "train": "MX/train_MX.csv",
        "new": "MX/MX_new_data.csv",
    },
}

RAW_LABEL_COLS = [
    "hate_speech",
    "false_info",
    "violence",
    "harassment",
    "obscenity",
    "illegal",
    "national_security",
]

MERGED_COLS = [
    "text",
    "language",
    "label_name",
    *RAW_LABEL_COLS,
    "train_source",
    "origin",
]

SUMMARY_COLS = [
    "language",
    "label_type",
    "label",
    "current_count",
    "need_to_5000",
    "pct_of_target",
]


def norm_text(x):
    x = str(x or "").strip()
    return re.sub(r"\s+", " ", x)


def text_key(x):
    return hashlib.md5(norm_text(x).encode("utf-8")).hexdigest()


def to_binary(value):
    if value is None:
        return 0
    v = str(value).strip().lower()
    if not v:
        return 0
    if v in {"1", "1.0", "true", "yes", "y", "t"}:
        return 1
    try:
        return 1 if float(v) == 1.0 else 0
    except ValueError:
        return 0


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_language(lang, cfg):
    train_rows = read_csv(os.path.join(BASE_DIR, cfg["train"]))
    new_rows = read_csv(os.path.join(BASE_DIR, cfg["new"]))

    merged = {}

    for row in train_rows:
        key = text_key(row.get("text"))
        merged[key] = {
            "text": norm_text(row.get("text")),
            "language": lang,
            "label_name": (row.get("label_name") or "").strip(),
            **{col: 0 for col in RAW_LABEL_COLS},
            "train_source": (row.get("source") or "").strip(),
            "origin": "train_only",
        }

    for row in new_rows:
        key = text_key(row.get("text"))
        raw_vals = {col: to_binary(row.get(col)) for col in RAW_LABEL_COLS}

        if key in merged:
            entry = merged[key]
            for col in RAW_LABEL_COLS:
                entry[col] = max(entry[col], raw_vals[col])
            entry["origin"] = "both"
        else:
            merged[key] = {
                "text": norm_text(row.get("text")),
                "language": lang,
                "label_name": "",
                **raw_vals,
                "train_source": "",
                "origin": "new_only",
            }

    rows = list(merged.values())
    return rows


def build_summary(all_rows_by_lang):
    summary = []

    for lang, rows in all_rows_by_lang.items():
        label_name_counts = Counter(
            r["label_name"] for r in rows if r["label_name"]
        )
        for label, count in sorted(label_name_counts.items(), key=lambda x: -x[1]):
            need = max(TARGET_N - count, 0)
            summary.append(
                {
                    "language": lang,
                    "label_type": "label_name",
                    "label": label,
                    "current_count": count,
                    "need_to_5000": need,
                    "pct_of_target": round(count / TARGET_N * 100, 1),
                }
            )

        for col in RAW_LABEL_COLS:
            count = sum(1 for r in rows if r[col] == 1)
            need = max(TARGET_N - count, 0)
            summary.append(
                {
                    "language": lang,
                    "label_type": "raw_label",
                    "label": col,
                    "current_count": count,
                    "need_to_5000": need,
                    "pct_of_target": round(count / TARGET_N * 100, 1),
                }
            )

        origin_counts = Counter(r["origin"] for r in rows)
        summary.append(
            {
                "language": lang,
                "label_type": "meta",
                "label": f"total_rows={len(rows)}",
                "current_count": len(rows),
                "need_to_5000": "",
                "pct_of_target": "",
            }
        )
        for origin, count in sorted(origin_counts.items()):
            summary.append(
                {
                    "language": lang,
                    "label_type": "meta",
                    "label": f"origin_{origin}",
                    "current_count": count,
                    "need_to_5000": "",
                    "pct_of_target": "",
                }
            )

    combined_label_name = Counter()
    combined_raw = {col: 0 for col in RAW_LABEL_COLS}
    for rows in all_rows_by_lang.values():
        for r in rows:
            if r["label_name"]:
                combined_label_name[r["label_name"]] += 1
            for col in RAW_LABEL_COLS:
                if r[col] == 1:
                    combined_raw[col] += 1

    for label, count in sorted(combined_label_name.items(), key=lambda x: -x[1]):
        need = max(TARGET_N - count, 0)
        summary.append(
            {
                "language": "ALL",
                "label_type": "label_name",
                "label": label,
                "current_count": count,
                "need_to_5000": need,
                "pct_of_target": round(count / TARGET_N * 100, 1),
            }
        )

    for col, count in sorted(combined_raw.items(), key=lambda x: -x[1]):
        need = max(TARGET_N - count, 0)
        summary.append(
            {
                "language": "ALL",
                "label_type": "raw_label",
                "label": col,
                "current_count": count,
                "need_to_5000": need,
                "pct_of_target": round(count / TARGET_N * 100, 1),
            }
        )

    return summary


def print_report(all_rows_by_lang, summary):
    print("=" * 70)
    print("BR / TR / MX 数据合并报告")
    print("=" * 70)

    for lang, rows in all_rows_by_lang.items():
        print(f"\n--- {lang} ---")
        print(f"  合并后总行数: {len(rows)}")
        origins = Counter(r["origin"] for r in rows)
        for o, c in sorted(origins.items()):
            print(f"    {o}: {c}")

        print(f"\n  label_name 分布 (目标 {TARGET_N}):")
        for label, count in Counter(
            r["label_name"] for r in rows if r["label_name"]
        ).most_common():
            gap = max(TARGET_N - count, 0)
            flag = " ✓" if count >= TARGET_N else f" 缺 {gap}"
            print(f"    {label}: {count}{flag}")

        print(f"\n  raw 7维标签分布 (目标 {TARGET_N}):")
        for col in RAW_LABEL_COLS:
            count = sum(1 for r in rows if r[col] == 1)
            gap = max(TARGET_N - count, 0)
            flag = " ✓" if count >= TARGET_N else f" 缺 {gap}"
            print(f"    {col}: {count}{flag}")

        both = [r for r in rows if r["origin"] == "both"]
        if both:
            mismatch = 0
            for r in both:
                has_raw = any(r[c] == 1 for c in RAW_LABEL_COLS)
                if r["label_name"] == "safe" and has_raw:
                    mismatch += 1
                elif r["label_name"] and r["label_name"] != "safe" and not has_raw:
                    mismatch += 1
            print(f"\n  口径不一致 (both来源, label_name vs raw): ~{mismatch}/{len(both)} 条")


def main():
    all_rows_by_lang = {}

    for lang, cfg in LANG_CONFIG.items():
        rows = merge_language(lang, cfg)
        all_rows_by_lang[lang] = rows

        out_path = os.path.join(BASE_DIR, lang, f"{lang}_merged.csv")
        write_csv(out_path, rows, MERGED_COLS)
        print(f"Saved {lang}: {out_path} ({len(rows)} rows)")

    summary = build_summary(all_rows_by_lang)
    summary_path = os.path.join(BASE_DIR, "BRT_MX_label_distribution_summary.csv")
    write_csv(summary_path, summary, SUMMARY_COLS)
    print(f"Saved summary: {summary_path}")

    print_report(all_rows_by_lang, summary)


if __name__ == "__main__":
    main()
