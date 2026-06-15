#!/usr/bin/env python3
"""
Balance BR R2 training data to a minimum per-label count of 2000.

This script only appends existing candidate rows. It does not downsample,
delete original rows, call any model, or generate new text.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


TARGET_MIN_COUNT = 2000

GLOBAL_LABELS = [
    "Dangerous_Content",
    "Harassment",
    "Hate_Speech",
    "Sexually_Explicit_Information",
    "Politically_Sensitive_Topics",
    "Cybersecurity_Malware",
    "BR_State_Security_Democratic_Order",
    "safe",
]

LABEL_NORMALIZE_MAP = {
    "Politically Sensitive Topics": "Politically_Sensitive_Topics",
    "Sexually Explicit Information": "Sexually_Explicit_Information",
    "BR State Security Democratic Order": "BR_State_Security_Democratic_Order",
    "Cybersecurity Malware": "Cybersecurity_Malware",
}

DEFAULT_MAIN_PATH = Path(
    "/home/ma-user/work/Datasets/LoRA-code-2026-6-5/BR/BR_R2_data/"
    "br_annotation_R2_full.csv"
)
DEFAULT_DROPPED_SYN_PATH = Path(
    "/home/ma-user/work/Datasets/LoRA-code-2026-6-5/BR/BR_R2_data/"
    "br_annotation_R2_dropped_synthetic.csv"
)
DEFAULT_ADD_SYN_PATHS = [
    Path(
        "/home/ma-user/work/Datasets/LoRA-code-2026-6-5/BR/add_data/"
        "br_synthetic_R2_valid_dedup.csv"
    ),
    Path(
        "/home/ma-user/work/Datasets/LoRA-code-2026-6-5/BR/add_data/"
        "br_cybersecurity_malware_valid_dedup.csv"
    ),
]
DEFAULT_OUT_BALANCED_PATH = Path(
    "/home/ma-user/work/Datasets/LoRA-code-2026-6-5/BR/BR_R2_data/"
    "br_annotation_R2_full_balanced2000.csv"
)
DEFAULT_OUT_ADDED_PATH = Path(
    "/home/ma-user/work/Datasets/LoRA-code-2026-6-5/BR/BR_R2_data/"
    "br_annotation_R2_added_to_balance2000.csv"
)
DEFAULT_OUT_DIST_PATH = Path(
    "/home/ma-user/work/Datasets/LoRA-code-2026-6-5/BR/BR_R2_data/"
    "br_annotation_R2_balanced2000_label_distribution.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = str(PROJECT_ROOT / "synthetic_R2")
MODEL_NAME = "qwen/qwen3-next-80b-a3b-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append existing BR R2 synthetic candidates until each label has at least 2000 rows."
    )
    parser.add_argument("--main-path", type=Path, default=DEFAULT_MAIN_PATH)
    parser.add_argument("--dropped-synthetic-path", type=Path, default=DEFAULT_DROPPED_SYN_PATH)
    parser.add_argument("--add-synthetic-path", type=Path, nargs="*", default=DEFAULT_ADD_SYN_PATHS)
    parser.add_argument("--out-balanced-path", type=Path, default=DEFAULT_OUT_BALANCED_PATH)
    parser.add_argument("--out-added-path", type=Path, default=DEFAULT_OUT_ADDED_PATH)
    parser.add_argument("--out-dist-path", type=Path, default=DEFAULT_OUT_DIST_PATH)
    parser.add_argument("--target-min-count", type=int, default=TARGET_MIN_COUNT)
    return parser.parse_args()


def load_local_env_files() -> list[Path]:
    candidate_paths = [
        Path.cwd() / ".env",
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
        PROJECT_ROOT.parent / "MX" / ".env",
        PROJECT_ROOT.parent / "BR" / ".env",
        Path(BASE_DIR) / ".env",
    ]
    loaded_paths = []

    for env_path in candidate_paths:
        if not env_path.exists():
            continue

        loaded_paths.append(env_path)
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value

    return loaded_paths


def print_niw_api_key_status(loaded_env_paths: list[Path]) -> None:
    api_key_names = ["NIW_API_KEY", "NIM_API_KEY", "MX_API_KEY", "NVIDIA_API_KEY"]
    for key_name in api_key_names:
        value = os.getenv(key_name, "")
        if value:
            print(f"NIW API key loaded from env var: {key_name}")
            if loaded_env_paths:
                print("Env files loaded:")
                for env_path in loaded_env_paths:
                    print(f"  {env_path}")
            return

    print("WARNING: NIW/NIM API key not found in loaded env files.")


def read_csv_or_fail(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{source_name} not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace(LABEL_NORMALIZE_MAP)


def ensure_standard_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()

    if "clean_text" not in df.columns and "text" in df.columns:
        df["clean_text"] = df["text"]

    if "final_category" not in df.columns and "llm_label" in df.columns:
        df["final_category"] = df["llm_label"]

    missing = [col for col in ["clean_text", "final_category"] if col not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns after normalization: {missing}")

    df["clean_text"] = df["clean_text"].astype(str)
    df["final_category"] = normalize_label_series(df["final_category"])
    df["_clean_text_key"] = df["clean_text"].astype(str).str.strip()
    return df


def prepare_candidate_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = ensure_standard_columns(df, source_name)
    df = df[df["_clean_text_key"].ne("")]
    df = df[df["final_category"].isin(GLOBAL_LABELS)]
    return df


def label_distribution(df: pd.DataFrame) -> pd.Series:
    return df["final_category"].value_counts().reindex(GLOBAL_LABELS, fill_value=0)


def compute_deficits(dist: pd.Series, target_min_count: int) -> dict[str, int]:
    return {
        label: max(0, target_min_count - int(dist.get(label, 0)))
        for label in GLOBAL_LABELS
    }


def take_candidates(
    candidate_df: pd.DataFrame,
    current_text_keys: set[str],
    remaining_deficits: dict[str, int],
    source_bucket: str,
    source_file: Path,
) -> pd.DataFrame:
    picked_parts = []

    candidate_df = candidate_df.copy()
    candidate_df = candidate_df[~candidate_df["_clean_text_key"].isin(current_text_keys)]
    candidate_df = candidate_df.drop_duplicates(subset=["_clean_text_key"], keep="first")

    for label in GLOBAL_LABELS:
        need = remaining_deficits.get(label, 0)
        if need <= 0:
            continue

        label_candidates = candidate_df[candidate_df["final_category"].eq(label)]
        if label_candidates.empty:
            continue

        picked = label_candidates.head(need).copy()
        if picked.empty:
            continue

        picked["balance_source"] = source_bucket
        picked["balance_source_file"] = str(source_file)
        picked_parts.append(picked)

        picked_keys = set(picked["_clean_text_key"])
        current_text_keys.update(picked_keys)
        remaining_deficits[label] -= len(picked)

        candidate_df = candidate_df[~candidate_df["_clean_text_key"].isin(picked_keys)]

    if not picked_parts:
        return pd.DataFrame(columns=list(candidate_df.columns) + ["balance_source", "balance_source_file"])

    return pd.concat(picked_parts, ignore_index=True)


def print_label_counts(title: str, counts: pd.Series | dict[str, int]) -> None:
    print("\n" + "=" * 80)
    print(title)
    for label in GLOBAL_LABELS:
        print(f"{label}: {int(counts.get(label, 0))}")


def main() -> None:
    args = parse_args()
    loaded_env_paths = load_local_env_files()
    print(f"Configured BR model name: {MODEL_NAME}")
    print_niw_api_key_status(loaded_env_paths)

    main_df = read_csv_or_fail(args.main_path, "main R2 data")
    main_df = ensure_standard_columns(main_df, "main R2 data")

    main_duplicate_count = int(main_df["_clean_text_key"].duplicated().sum())
    if main_duplicate_count > 0:
        print(
            f"WARNING: main data has {main_duplicate_count} duplicate clean_text rows. "
            "They are kept unchanged."
        )

    original_dist = label_distribution(main_df)
    original_deficits = compute_deficits(original_dist, args.target_min_count)

    print_label_counts("Original R2 label distribution", original_dist)
    print_label_counts(f"Deficits to TARGET_MIN_COUNT={args.target_min_count}", original_deficits)

    remaining_deficits = original_deficits.copy()
    current_text_keys = set(main_df["_clean_text_key"])

    added_parts = []
    added_from_dropped = defaultdict(int)
    added_from_add_data = defaultdict(int)

    dropped_df = read_csv_or_fail(args.dropped_synthetic_path, "dropped synthetic")
    dropped_df = prepare_candidate_df(dropped_df, "dropped synthetic")
    picked_dropped = take_candidates(
        candidate_df=dropped_df,
        current_text_keys=current_text_keys,
        remaining_deficits=remaining_deficits,
        source_bucket="dropped_synthetic",
        source_file=args.dropped_synthetic_path,
    )

    if not picked_dropped.empty:
        added_parts.append(picked_dropped)
        for label, count in picked_dropped["final_category"].value_counts().items():
            added_from_dropped[label] += int(count)

    for add_path in args.add_synthetic_path:
        if all(need <= 0 for need in remaining_deficits.values()):
            break

        add_df = read_csv_or_fail(add_path, f"add_data synthetic: {add_path.name}")
        add_df = prepare_candidate_df(add_df, f"add_data synthetic: {add_path.name}")
        picked_add = take_candidates(
            candidate_df=add_df,
            current_text_keys=current_text_keys,
            remaining_deficits=remaining_deficits,
            source_bucket="add_data_synthetic",
            source_file=add_path,
        )

        if not picked_add.empty:
            added_parts.append(picked_add)
            for label, count in picked_add["final_category"].value_counts().items():
                added_from_add_data[label] += int(count)

    if added_parts:
        added_df = pd.concat(added_parts, ignore_index=True)
    else:
        added_df = pd.DataFrame(columns=list(main_df.columns) + ["balance_source", "balance_source_file"])

    main_output_cols = [col for col in main_df.columns if col != "_clean_text_key"]
    balanced_df = pd.concat(
        [
            main_df[main_output_cols],
            added_df.reindex(columns=main_output_cols),
        ],
        ignore_index=True,
    )

    added_detail_cols = main_output_cols + ["balance_source", "balance_source_file"]
    added_detail_df = added_df.reindex(columns=added_detail_cols)

    final_check_df = ensure_standard_columns(balanced_df, "balanced output")
    final_dist = label_distribution(final_check_df)
    dist_report_df = final_dist.rename_axis("final_category").reset_index(name="count")
    dist_report_df["target_min_count"] = args.target_min_count
    dist_report_df["gap_after_balance"] = dist_report_df["count"].apply(
        lambda count: max(0, args.target_min_count - int(count))
    )

    args.out_balanced_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_added_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_dist_path.parent.mkdir(parents=True, exist_ok=True)

    balanced_df.to_csv(args.out_balanced_path, index=False, encoding="utf-8-sig")
    added_detail_df.to_csv(args.out_added_path, index=False, encoding="utf-8-sig")
    dist_report_df.to_csv(args.out_dist_path, index=False, encoding="utf-8-sig")

    print_label_counts("Added from dropped synthetic", added_from_dropped)
    print_label_counts("Added from add_data synthetic", added_from_add_data)
    print_label_counts("Final label distribution", final_dist)

    print("\n" + "=" * 80)
    print("Final warnings")
    has_warning = False
    for label in GLOBAL_LABELS:
        final_count = int(final_dist.get(label, 0))
        if final_count < args.target_min_count:
            has_warning = True
            print(
                f"WARNING: {label} still below {args.target_min_count}: "
                f"{final_count}, missing {args.target_min_count - final_count}"
            )
    if not has_warning:
        print(f"All labels reached at least {args.target_min_count}.")

    print("\n" + "=" * 80)
    print("Saved files")
    print(f"Balanced training data: {args.out_balanced_path}")
    print(f"Added sample details:   {args.out_added_path}")
    print(f"Distribution report:    {args.out_dist_path}")


if __name__ == "__main__":
    main()
