#!/usr/bin/env python3
"""
Export quality-filtered rows from tr_annotation_full.csv.

Outputs:
  tr_full_quality_pass.csv
    Rows surviving hard filter + fake_agglut rejection (skip R4 need-review pool texts).

  TR_full_triple_annotation_pending.csv
    Subset of quality_pass that still needs NEW triple-model annotation (Llama+Gemma+Qwen).

    Include row if ANY of:
      A) unannotated        — juror_a/b/c_category all empty (never went through old jury)
      B) requires_review    — old pipeline flagged requires_review=True
      C) non_unanimous_juror — old jury ran, but juror_a/b/c_category are not all identical

    Exclude only when: old jury 3/3 category unanimous (4518 rows) → keep legacy label for now.

    8746 = 4191 (A) + 4555 (C); B is a subset of C (16 rows).
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from tr_phase_a_clean import has_phase_a_artifacts, phase_a_clean
from tr_phase_b_detect import detect, spam_tier
from tr_phase_c_judge import judge_row

FULL_CSV = Path("/Users/liushuyu/Desktop/Huawei/multilingual/TR/data/annotations_tr/tr_annotation_full.csv")
SKIP_CSV = SCRIPT_DIR / "TR" / "TR_R4_train_pool_need_review_annotated.csv"
OUT_PASS = SCRIPT_DIR / "TR" / "tr_full_quality_pass.csv"
OUT_REQUIRES_REVIEW = SCRIPT_DIR / "TR" / "tr_full_quality_pass_requires_review.csv"
OUT_TRIPLE_PENDING = SCRIPT_DIR / "TR" / "TR_full_triple_annotation_pending.csv"


def juror_categories(row: dict) -> list[str]:
    return [str(row.get(f"juror_{juror}_category") or "").strip() for juror in ("a", "b", "c")]


def is_unannotated(row: dict) -> bool:
    return len([label for label in juror_categories(row) if label]) < 3


def is_requires_review_flag(row: dict) -> bool:
    return str(row.get("requires_review") or "").strip().lower() == "true"


def is_non_unanimous_juror(row: dict) -> bool:
    labels = [label for label in juror_categories(row) if label]
    if len(labels) < 3:
        return False
    return len(set(labels)) != 1


def triple_annotation_pending_reasons(row: dict) -> list[str]:
    reasons: list[str] = []
    if is_unannotated(row):
        reasons.append("unannotated")
    if is_requires_review_flag(row):
        reasons.append("requires_review")
    if is_non_unanimous_juror(row):
        reasons.append("non_unanimous_juror")
    return reasons


def needs_triple_annotation(row: dict) -> bool:
    return bool(triple_annotation_pending_reasons(row))


def legacy_juror_agreement(row: dict) -> str:
    labels = [label for label in juror_categories(row) if label]
    if len(labels) < 3:
        return "no_juror_vote"
    unique = set(labels)
    if len(unique) == 1:
        return "unanimous_3"
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if max(counts.values()) == 2:
        return "majority_2_of_3"
    return "all_different_3"


def is_hard_reject(had_artifacts: bool, tier: str, exclude_reason: str) -> bool:
    if had_artifacts:
        return True
    if tier == "spam_likely":
        return True
    primary = (exclude_reason or "").split("|")[0]
    return primary == "seo_template_gibberish"


def should_keep_row(clean_text: str, skip_texts: set[str]) -> tuple[bool, str]:
    text = clean_text.strip()
    if not text:
        return False, "empty"
    if text in skip_texts:
        return False, "need_review_pool_skip"
    cleaned = phase_a_clean(text) if has_phase_a_artifacts(text) else text
    flags, _caps, _fake = detect(cleaned)
    tier = spam_tier(flags)
    work = {
        "clean_text": text,
        "clean_text_phaseA": cleaned,
        "phaseB_flags": "|".join(flags),
        "phaseB_caps_ratio": "0",
        "phaseB_fake_ratio": "0",
    }
    ok, reason, *_ = judge_row(work)
    if is_hard_reject(has_phase_a_artifacts(text), tier, reason):
        return False, "hard_filter"
    if not ok and reason and "fake_agglutination" in reason:
        return False, "fake_agglutination"
    return True, "pass"


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".csv", dir=str(path.parent))
    try:
        import os

        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        Path(tmp_name).replace(path)
    except Exception:
        import os

        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def run(
    full_csv: Path,
    skip_csv: Path,
    out_pass: Path,
    out_requires_review: Path,
    out_triple_pending: Path,
) -> dict[str, int]:
    skip_texts: set[str] = set()
    with skip_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("clean_text") or "").strip()
            if text:
                skip_texts.add(text)

    stats: dict[str, int] = {
        "full_total": 0,
        "pass": 0,
        "legacy_unanimous_3": 0,
        "triple_pending": 0,
        "criterion_unannotated": 0,
        "criterion_requires_review": 0,
        "criterion_non_unanimous_juror": 0,
    }
    pass_rows: list[dict] = []
    review_rows: list[dict] = []
    pending_rows: list[dict] = []
    fieldnames: list[str] = []
    extra_cols = [
        "legacy_juror_agreement",
        "triple_annotation_pending",
        "triple_annotation_pending_reason",
    ]

    with full_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for extra in extra_cols:
            if extra not in fieldnames:
                fieldnames.append(extra)

        for row in reader:
            stats["full_total"] += 1
            keep, _why = should_keep_row(str(row.get("clean_text") or ""), skip_texts)
            if not keep:
                continue

            stats["pass"] += 1
            row = dict(row)
            agreement = legacy_juror_agreement(row)
            reasons = triple_annotation_pending_reasons(row)
            pending = bool(reasons)

            row["legacy_juror_agreement"] = agreement
            row["triple_annotation_pending"] = "true" if pending else "false"
            row["triple_annotation_pending_reason"] = "|".join(reasons) if reasons else "legacy_unanimous_3"
            pass_rows.append(row)

            if pending:
                stats["triple_pending"] += 1
                pending_rows.append(row)
                if is_unannotated(row):
                    stats["criterion_unannotated"] += 1
                if is_requires_review_flag(row):
                    stats["criterion_requires_review"] += 1
                if is_non_unanimous_juror(row):
                    stats["criterion_non_unanimous_juror"] += 1
            else:
                stats["legacy_unanimous_3"] += 1

            if is_requires_review_flag(row):
                review_rows.append(row)

    write_csv_atomic(out_pass, fieldnames, pass_rows)
    write_csv_atomic(out_requires_review, fieldnames, review_rows)
    write_csv_atomic(out_triple_pending, fieldnames, pending_rows)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export quality-pass + triple-annotation-pending TR rows")
    parser.add_argument("--full-csv", type=Path, default=FULL_CSV)
    parser.add_argument("--skip-csv", type=Path, default=SKIP_CSV)
    parser.add_argument("--out-pass", type=Path, default=OUT_PASS)
    parser.add_argument("--out-requires-review", type=Path, default=OUT_REQUIRES_REVIEW)
    parser.add_argument("--out-triple-pending", type=Path, default=OUT_TRIPLE_PENDING)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = run(
        args.full_csv,
        args.skip_csv,
        args.out_pass,
        args.out_requires_review,
        args.out_triple_pending,
    )
    print("=== tr_annotation_full quality export ===")
    print(f"full_total: {stats['full_total']}")
    print(f"quality_pass: {stats['pass']}")
    print(f"legacy_unanimous_3 (skip new annotation): {stats['legacy_unanimous_3']}")
    print()
    print("=== TR_full_triple_annotation_pending selection (A | B | C) ===")
    print(f"  A unannotated:              {stats['criterion_unannotated']}")
    print(f"  B requires_review=True:     {stats['criterion_requires_review']}  (subset of C)")
    print(f"  C non_unanimous_juror:      {stats['criterion_non_unanimous_juror']}")
    print(f"  union A|B|C (pending rows): {stats['triple_pending']}")
    print(f"  check: {stats['criterion_unannotated']} + {stats['criterion_non_unanimous_juror']} = {stats['criterion_unannotated'] + stats['criterion_non_unanimous_juror']}")
    print()
    print(f"quality_pass csv:        {args.out_pass}")
    print(f"triple pending csv:      {args.out_triple_pending}")
    print(f"requires_review subset:  {args.out_requires_review} ({len(list(csv.DictReader(args.out_requires_review.open(encoding='utf-8-sig')))) if args.out_requires_review.exists() else 0} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
