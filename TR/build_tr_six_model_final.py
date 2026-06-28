#!/usr/bin/env python3
"""Build one TR annotation master CSV: six-model rows + legacy jury-unanimous rows."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FULL = PROJECT_ROOT / "TR" / "data" / "annotations_tr" / "tr_annotation_full.csv"
DEFAULT_QUALITY_PASS = PROJECT_ROOT / "3rd_annotaion" / "TR" / "tr_full_quality_pass.csv"
DEFAULT_TRIPLE = (
    PROJECT_ROOT
    / "3rd_annotaion"
    / "TR"
    / "TR_new_annotaion-6.22"
    / "TR_full_triple_annotation_pending_triple_voted.csv"
)
DEFAULT_OUT = PROJECT_ROOT / "TR" / "data" / "annotations_tr" / "tr_six_model_final.csv"
DEFAULT_CAND_ANN = (
    PROJECT_ROOT / "TR" / "data" / "annotations_tr" / "tr_new_candidates_round1_dedup_annotations.csv"
)
CANDIDATE_TRAIN_ID_START = 12132

JURY_LABEL_COLS = ["juror_a_category", "juror_b_category", "juror_c_category"]
TRIPLE_LABEL_COLS = ["primary_label", "secondary_label", "tertiary_label"]


def norm_label(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in ("", "none", "nan"):
        return "safe"
    return text


def map_train_id_to_candidate_id(content_id: str) -> str | None:
    match = re.match(r"tr_train_(\d+)", str(content_id or ""))
    if not match:
        return None
    index = int(match.group(1))
    if index < CANDIDATE_TRAIN_ID_START:
        return None
    return f"tr_new_round1_{index - CANDIDATE_TRAIN_ID_START:08d}"


def jury_unanimous_label(row: pd.Series) -> str:
    labels = [norm_label(row.get(col)) for col in JURY_LABEL_COLS]
    if len(labels) < 3 or not all(labels):
        return ""
    if len(set(labels)) != 1:
        return ""
    return labels[0]


def six_model_vote(labels: list) -> tuple[str, str, int, str]:
    normalized = [norm_label(label) for label in labels if norm_label(label)]
    if len(normalized) < 6:
        missing = 6 - len(normalized)
        return "", f"missing_{missing}_labels", 0, "|".join(normalized)

    counts = Counter(normalized)
    top, count = counts.most_common(1)[0]
    distribution = "|".join(f"{label}:{n}" for label, n in counts.most_common())

    if count >= 5:
        return top, "six_vote_5_or_6", count, distribution
    if count >= 4:
        return top, "six_vote_4", count, distribution
    if count == 3 and len(counts) == 2:
        return "need_review", "six_vote_3_3_tie", count, distribution
    if count <= 2:
        return "need_review", "six_vote_no_majority", count, distribution
    return "need_review", "six_vote_3_way", count, distribution


def build_jury_maps(full: pd.DataFrame, cand_ann: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    jury_detail = [col for col in full.columns if col.startswith("juror_")]
    jury_detail += [col for col in ("total_latency_ms", "max_latency_ms", "annotated_at") if col in full.columns]
    jury_cols = ["content_id", "clean_text", "source", "country", "language"] + jury_detail

    jury_from_full = full[jury_cols].drop_duplicates(subset=["content_id"], keep="last")
    jury_by_content_id = {row["content_id"]: row for _, row in jury_from_full.iterrows()}
    jury_by_candidate_id = {row["content_id"]: row for _, row in cand_ann.iterrows()}
    return jury_by_content_id, jury_by_candidate_id


def apply_jury_row(row: pd.Series, jury_by_content_id: dict, jury_by_candidate_id: dict, jury_detail: list[str]) -> pd.Series:
    content_id = str(row["content_id"])
    jury_row = jury_by_content_id.get(content_id)
    if jury_row is None:
        mapped = map_train_id_to_candidate_id(content_id)
        if mapped:
            jury_row = jury_by_candidate_id.get(mapped)
    if jury_row is None:
        return row

    for col in ["clean_text", "source", "country", "language"] + jury_detail:
        if col not in jury_row.index:
            continue
        current = row.get(col)
        if pd.isna(current) or (isinstance(current, str) and not str(current).strip()):
            row[col] = jury_row[col]
    return row


def build(
    full_path: Path,
    quality_pass_path: Path,
    triple_path: Path,
    out_path: Path,
    cand_ann_path: Path,
) -> pd.DataFrame:
    full = pd.read_csv(full_path, low_memory=False)
    quality_pass = pd.read_csv(quality_pass_path, low_memory=False)
    triple = pd.read_csv(triple_path, low_memory=False)
    cand_ann = pd.read_csv(cand_ann_path, low_memory=False)

    triple_ids = set(triple["content_id"])
    if not triple_ids.issubset(set(quality_pass["content_id"])):
        missing = triple_ids - set(quality_pass["content_id"])
        raise ValueError(f"triple rows missing from quality_pass: {len(missing)}")

    jury_detail = [col for col in full.columns if col.startswith("juror_")]
    jury_detail += [col for col in ("total_latency_ms", "max_latency_ms", "annotated_at") if col in full.columns]
    jury_by_content_id, jury_by_candidate_id = build_jury_maps(full, cand_ann)

    triple_cols = [col for col in triple.columns if col != "content_id"]
    triple_skip = set(
        jury_detail
        + ["clean_text", "source", "country", "language", "subreddit", "title"]
        + [col for col in triple_cols if col.startswith("juror_")]
    )
    triple_copy_cols = [col for col in triple_cols if col not in triple_skip]
    triple_by_id = triple.set_index("content_id")

    rows: list[dict] = []
    for _, base in quality_pass.iterrows():
        content_id = base["content_id"]
        row = {col: base.get(col) for col in quality_pass.columns}

        for col in jury_detail:
            if col not in row:
                row[col] = pd.NA

        if content_id in triple_by_id.index:
            triple_row = triple_by_id.loc[content_id]
            if isinstance(triple_row, pd.DataFrame):
                triple_row = triple_row.iloc[0]
            for col in triple_copy_cols:
                row[col] = triple_row.get(col)
            row["annotation_tier"] = "six_model"
        else:
            row["annotation_tier"] = "jury_unanimous_3"
            for col in TRIPLE_LABEL_COLS + [
                "primary_reason",
                "secondary_reason",
                "tertiary_reason",
                "primary_model",
                "secondary_model",
                "tertiary_model",
                "final_label",
                "final_source",
            ]:
                row.setdefault(col, pd.NA)

        row = apply_jury_row(pd.Series(row), jury_by_content_id, jury_by_candidate_id, jury_detail).to_dict()

        if row["annotation_tier"] == "six_model":
            row["six_models_complete"] = all(norm_label(row.get(col)) for col in JURY_LABEL_COLS + TRIPLE_LABEL_COLS)
            row["six_vote_labels"] = "|".join(norm_label(row.get(col)) for col in JURY_LABEL_COLS + TRIPLE_LABEL_COLS)
            final_label, final_source, top_count, distribution = six_model_vote(
                [row.get(col) for col in JURY_LABEL_COLS + TRIPLE_LABEL_COLS]
            )
            row["final_label_6"] = final_label
            row["final_source_6"] = final_source
            row["six_vote_top_count"] = top_count
            row["six_vote_distribution"] = distribution
        else:
            row["six_models_complete"] = False
            jury_label = jury_unanimous_label(pd.Series(row))
            row["six_vote_labels"] = "|".join(norm_label(row.get(col)) for col in JURY_LABEL_COLS)
            row["six_vote_top_count"] = 3 if jury_label else 0
            row["six_vote_distribution"] = f"{jury_label}:3" if jury_label else ""
            row["final_label_6"] = jury_label
            row["final_source_6"] = "jury_unanimous_3" if jury_label else "jury_incomplete"

        rows.append(row)

    merged = pd.DataFrame(rows)

    front = [
        "content_id",
        "country",
        "language",
        "source",
        "clean_text",
        "annotation_tier",
        "six_models_complete",
        "six_vote_labels",
        "six_vote_top_count",
        "six_vote_distribution",
        "final_label_6",
        "final_source_6",
        "final_label",
        "final_source",
    ]
    jury_block = [col for col in merged.columns if col.startswith("juror_")]
    triple_block = [
        "primary_label",
        "primary_reason",
        "secondary_label",
        "secondary_reason",
        "tertiary_label",
        "tertiary_reason",
        "primary_model",
        "secondary_model",
        "tertiary_model",
        "vote_labels",
        "vote_agreement",
        "legacy_juror_agreement",
        "triple_annotation_pending_reason",
    ]
    rest = [col for col in merged.columns if col not in front + jury_block + triple_block]
    merged = merged[[col for col in front + jury_block + triple_block + rest if col in merged.columns]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TR annotation master CSV")
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--quality-pass", type=Path, default=DEFAULT_QUALITY_PASS)
    parser.add_argument("--triple", type=Path, default=DEFAULT_TRIPLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cand-ann", type=Path, default=DEFAULT_CAND_ANN)
    args = parser.parse_args()

    merged = build(args.full, args.quality_pass, args.triple, args.output, args.cand_ann)
    print(f"Written: {args.output}")
    print(f"Rows: {len(merged)}")
    print("\nannotation_tier:")
    print(merged["annotation_tier"].value_counts().to_string())
    print(f"\nsix_models_complete: {int(merged['six_models_complete'].sum())}/{len(merged)}")
    print("\nfinal_label_6 (top):")
    print(merged["final_label_6"].value_counts().head(12).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
