#!/usr/bin/env python3
"""Merge TR BERT-ready CSVs into TR/data/6.22/tr_bert_train_merged.csv."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "TR" / "data" / "6.22"
OUT_CSV = OUT_DIR / "tr_bert_train_merged.csv"
SUMMARY = OUT_DIR / "merge_summary.json"

SOURCES = [
    {
        "name": "bert_train_ready_153",
        "path": ROOT / "3rd_annotaion/TR/TR_new_annotaion-6.22/TR_bert_train_ready_153.csv",
        "id_col": "id",
        "text_col": "clean_text",
        "label_col": "final_label",
        "priority": 1,
    },
    {
        "name": "tr_sex_r3_merged",
        "path": ROOT / "TR/TR_R3/TR_SEX_merged.csv",
        "id_col": "id",
        "text_col": "text",
        "label_col": "target_label",
        "priority": 2,
    },
    {
        "name": "tr_six_model_final",
        "path": ROOT / "TR/data/annotations_tr/tr_six_model_final-6.22.csv",
        "id_col": "content_id",
        "text_col": "clean_text",
        "label_col": "final_label_6",
        "priority": 3,
    },
]

FIELDS = [
    "content_id",
    "country",
    "language",
    "text",
    "label",
    "source_dataset",
    "source_row_id",
]


def norm_text(t: str) -> str:
    return " ".join((t or "").split()).lower()


def norm_label(v: str) -> str:
    t = (v or "").strip()
    if not t or t.lower() in {"none", "nan"}:
        return "safe"
    return t


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_text: dict[str, dict] = {}
    stats: dict = {"per_source": {}, "dedup_dropped": 0}

    for src in SOURCES:
        rows = list(csv.DictReader(src["path"].open(encoding="utf-8-sig")))
        added = 0
        skipped = 0
        dedup = 0
        for row in rows:
            text = (row.get(src["text_col"]) or "").strip()
            if not text:
                skipped += 1
                continue
            label = norm_label(row.get(src["label_col"], ""))
            nt = norm_text(text)
            rec = {
                "content_id": row.get(src["id_col"], "").strip(),
                "country": (row.get("country") or "TR").strip() or "TR",
                "language": (row.get("language") or "tr").strip() or "tr",
                "text": text,
                "label": label,
                "source_dataset": src["name"],
                "source_row_id": row.get(src["id_col"], "").strip(),
                "_priority": src["priority"],
            }
            prev = by_text.get(nt)
            if prev is None:
                by_text[nt] = rec
                added += 1
            elif src["priority"] < prev["_priority"]:
                dedup += 1
                stats["dedup_dropped"] += 1
                by_text[nt] = rec
                added += 1
            else:
                dedup += 1
                stats["dedup_dropped"] += 1
        stats["per_source"][src["name"]] = {
            "input_rows": len(rows),
            "added_new": added,
            "skipped_empty": skipped,
            "dedup_dropped": dedup,
        }

    merged = sorted(by_text.values(), key=lambda r: (r["source_dataset"], r["content_id"]))
    for r in merged:
        r.pop("_priority", None)

    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(OUT_DIR))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(merged)
        Path(tmp).replace(OUT_CSV)
    except Exception:
        os.unlink(tmp)
        raise

    summary = {
        "output": str(OUT_CSV),
        "total_rows": len(merged),
        "label_distribution": dict(Counter(r["label"] for r in merged)),
        "source_distribution": dict(Counter(r["source_dataset"] for r in merged)),
        **stats,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
