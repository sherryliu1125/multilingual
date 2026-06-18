#!/usr/bin/env python3
"""Filter need_review rows from 3rd annotated CSVs; write beside source, never modify source."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

SOURCES = [
    SCRIPT_DIR / "BR" / "BR_train_pool_R5_review_with_id_annotated.csv",
    SCRIPT_DIR / "BR" / "golden_test_error_analysis_BR_annotated.csv",
    SCRIPT_DIR / "TR" / "TR_R4_train_pool_need_review_annotated.csv",
    SCRIPT_DIR / "SA" / "SA_train_pool_R4.5_review_annotated.csv",
]


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


def export_one(input_path: Path) -> tuple[Path, int, int]:
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    need_rows = [
        row for row in rows
        if str(row.get("final_label") or "").strip() == "need_review"
    ]
    output_path = input_path.with_name(f"{input_path.stem}_need_review.csv")
    write_csv_atomic(output_path, fieldnames, need_rows)
    return output_path, len(need_rows), len(rows)


def main() -> int:
    for input_path in SOURCES:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing: {input_path}")
        out, need, total = export_one(input_path)
        print(f"{input_path.relative_to(SCRIPT_DIR)}: {need}/{total} -> {out.relative_to(SCRIPT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
