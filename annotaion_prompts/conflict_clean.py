#!/usr/bin/env python3
"""
Create text-level deduplicated inputs from duplicate-text conflict CSVs.

The source conflict files intentionally keep every original row in a duplicate
text group. This script creates one row per clean_text so the next arbitration
pass can judge each text once, then later merge the text-level result back to
the original multi-row conflict files.

Outputs are written to:
  2nd_annotation/conflict/clean/{COUNTRY}_conflict_clean.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFLICT_DIR = PROJECT_ROOT / "2nd_annotation" / "conflict"
CLEAN_DIR = CONFLICT_DIR / "clean"

DEFAULT_INPUTS = {
    "MX": CONFLICT_DIR / "MX_R1_duplicate_text_label_conflict.csv",
    "SA": CONFLICT_DIR / "SA_R3_duplicate_text_label_conflicts.csv",
    "TR": CONFLICT_DIR / "TR_R2_duplicate_text_label_conflicts.csv",
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def unique_join(values: Iterable[object]) -> str:
    seen: list[str] = []
    for value in values:
        text = normalize_text(value)
        if text and text not in seen:
            seen.append(text)
    return " | ".join(seen)


def json_list(values: Iterable[object]) -> str:
    return json.dumps([int(v) if isinstance(v, int) else v for v in values], ensure_ascii=False)


def clean_one(country: str, input_path: Path, output_dir: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"[{country}] input not found: {input_path}")

    df = pd.read_csv(input_path)
    if "clean_text" not in df.columns:
        raise ValueError(f"[{country}] input missing clean_text column: {input_path}")
    if "final_category" not in df.columns:
        raise ValueError(f"[{country}] input missing final_category column: {input_path}")

    work = df.copy()
    work["_source_row_index"] = list(range(len(work)))
    work["_text_key"] = work["clean_text"].map(normalize_text)
    work = work[work["_text_key"].astype(bool)].copy()

    rows: list[dict] = []
    for text, group in work.groupby("_text_key", sort=False):
        labels = [normalize_text(value) for value in group["final_category"] if normalize_text(value)]
        unique_labels = sorted(set(labels))

        row = {
            "country": country,
            "clean_text": text,
            "conflict_source_file": input_path.name,
            "duplicate_count": int(len(group)),
            "label_nunique": int(len(unique_labels)),
            "original_labels": " | ".join(unique_labels),
            "original_source_row_indices": json_list(group["_source_row_index"].tolist()),
        }

        for source_col in ("source", "source_stage"):
            if source_col in group.columns:
                row[f"original_{source_col}s"] = unique_join(group[source_col])

        if "id" in group.columns:
            row["original_ids"] = unique_join(group["id"])
        if "_orig_row_order" in group.columns:
            row["original_row_orders"] = json_list(group["_orig_row_order"].dropna().astype(int).tolist())
        if "_text_raw" in group.columns:
            row["original_text_raw_variants"] = unique_join(group["_text_raw"])
        if "_text_norm" in group.columns:
            row["text_norm"] = normalize_text(group["_text_norm"].iloc[0])

        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{country}_conflict_clean.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        f"[{country}] {len(df):,} source rows -> {len(rows):,} unique clean_text rows: "
        f"{output_path}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="all", help="MX, SA, TR, or all")
    parser.add_argument("--input", type=Path, help="Override input path for a single country.")
    parser.add_argument("--output-dir", type=Path, default=CLEAN_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    country_arg = args.country.upper()
    countries = list(DEFAULT_INPUTS) if country_arg == "ALL" else [country_arg]

    unknown = [country for country in countries if country not in DEFAULT_INPUTS]
    if unknown:
        print(f"Unsupported country: {', '.join(unknown)}")
        return 2

    if args.input and len(countries) != 1:
        print("--input can only be used with a single country")
        return 2

    for country in countries:
        input_path = args.input or DEFAULT_INPUTS[country]
        clean_one(country, input_path, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
