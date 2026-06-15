import csv
import hashlib
import os
import random
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_CSV = os.path.join(BASE_DIR, "sa_new_annotated_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "supplement_from_local_rawlabel_only")
OUT_CSV = os.path.join(OUT_DIR, "SA_local_candidates_for_llm_rawlabel_only.csv")
OUT_SUMMARY_CSV = os.path.join(
    OUT_DIR, "SA_local_candidates_for_llm_rawlabel_only_summary.csv"
)

TEXT_COL = "text"

RAW_LABEL_COLS = [
    "hate_speech",
    "false_info",
    "violence",
    "harassment",
    "obscenity",
    "illegal",
    "national_security",
]

TARGET_CURRENT = {
    "SA_State_Security_Royalty": 1803,
    "Dangerous_Content": 952,
    "Harassment": 703,
    "SA_Religious_Violation": 241,
    "Politically_Sensitive_Topics": 240,
    "SA_LGBTQ_Content": 163,
    "Sexually_Explicit_Information": 26,
    "Cybersecurity_Malware": 16,
}

TARGET_N = 5000

CANDIDATE_CONFIG = {
    "SA_State_Security_Royalty": {
        "raw_cols_any": ["national_security"],
        "oversample_ratio": 1.5,
    },
    "Politically_Sensitive_Topics": {
        "raw_cols_any": ["national_security", "false_info"],
        "oversample_ratio": 1.3,
    },
    "Dangerous_Content": {
        "raw_cols_any": ["violence", "false_info", "illegal"],
        "oversample_ratio": 1.3,
    },
    "Harassment": {
        "raw_cols_any": ["harassment"],
        "oversample_ratio": 1.2,
    },
    "Sexually_Explicit_Information": {
        "raw_cols_any": ["obscenity"],
        "oversample_ratio": 999,
    },
}

LABEL_ORDER = [
    "SA_State_Security_Royalty",
    "Dangerous_Content",
    "Politically_Sensitive_Topics",
    "Sexually_Explicit_Information",
    "Harassment",
]

RANDOM_STATE = 42


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


def sample_up_to(rows, n, random_state=42):
    if n <= 0:
        return []
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(random_state)
    return rng.sample(rows, n)


def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for col in RAW_LABEL_COLS:
                row[col] = to_binary(row.get(col))
            row["_norm_text"] = norm_text(row.get(TEXT_COL))
            row["_text_key"] = text_key(row["_norm_text"])
            row["_raw_label_count"] = sum(row[col] for col in RAW_LABEL_COLS)
            rows.append(row)
    return rows


def raw_cols_match(row, cols):
    return any(row[col] == 1 for col in cols)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pool = load_rows(SRC_CSV)
    print("source rows:", len(pool))
    print("rows used as pool:", len(pool))

    selected_parts = []
    summary_rows = []
    used_text_keys = set()

    for label in LABEL_ORDER:
        cfg = CANDIDATE_CONFIG[label]
        raw_cols = cfg["raw_cols_any"]
        ratio = cfg["oversample_ratio"]

        current_count = TARGET_CURRENT[label]
        need = max(TARGET_N - current_count, 0)

        cand = [row for row in pool if raw_cols_match(row, raw_cols)]
        raw_candidate_n = len(cand)

        cand = [row for row in cand if row["_text_key"] not in used_text_keys]
        after_cross_label_dedup_n = len(cand)

        if ratio >= 999:
            target_candidate_n = after_cross_label_dedup_n
        else:
            target_candidate_n = int(need * ratio)

        cand = sorted(
            cand,
            key=lambda row: (row["_raw_label_count"], len(str(row.get(TEXT_COL, "")))),
            reverse=True,
        )

        selected = sample_up_to(cand, target_candidate_n, random_state=RANDOM_STATE)

        for row in selected:
            row["_candidate_label"] = label
            row["_candidate_raw_cols"] = ",".join(raw_cols)
            row["_current_count"] = current_count
            row["_need_to_5000"] = need
            row["_candidate_target_n"] = target_candidate_n

        used_text_keys.update(row["_text_key"] for row in selected)
        selected_parts.extend(selected)

        summary_rows.append(
            {
                "target_label": label,
                "raw_cols_any": ",".join(raw_cols),
                "current_count": current_count,
                "need_to_5000": need,
                "oversample_ratio": ratio,
                "target_candidate_n": target_candidate_n,
                "raw_candidate_n": raw_candidate_n,
                "after_cross_label_dedup_n": after_cross_label_dedup_n,
                "selected_for_llm_n": len(selected),
                "selected_vs_need_ratio": round(len(selected) / need, 3) if need else "",
            }
        )

    keep_cols = [
        "text",
        "hate_speech",
        "false_info",
        "violence",
        "harassment",
        "obscenity",
        "illegal",
        "national_security",
    ]

    out_rows = []
    for row in selected_parts:
        out_row = {col: row.get(col, "") for col in keep_cols}
        out_rows.append(out_row)

    write_csv(OUT_CSV, out_rows, keep_cols)
    write_csv(
        OUT_SUMMARY_CSV,
        summary_rows,
        [
            "target_label",
            "raw_cols_any",
            "current_count",
            "need_to_5000",
            "oversample_ratio",
            "target_candidate_n",
            "raw_candidate_n",
            "after_cross_label_dedup_n",
            "selected_for_llm_n",
            "selected_vs_need_ratio",
        ],
    )

    print("\nSaved candidate CSV:", OUT_CSV)
    print("Saved summary CSV:", OUT_SUMMARY_CSV)
    print("\nTotal selected for LLM:", len(out_rows))

    print("\nSummary:")
    for row in summary_rows:
        print(row)

    print("\nCandidate label distribution:")
    counts = {}
    for row in selected_parts:
        label = row["_candidate_label"]
        counts[label] = counts.get(label, 0) + 1
    for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        print(label, count)


if __name__ == "__main__":
    main()
