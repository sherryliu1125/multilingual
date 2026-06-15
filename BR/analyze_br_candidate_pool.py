#!/usr/bin/env python3
"""Analyze BR_new_data.csv old-label candidate pool for R-supplement annotation.

Goal: figure out how many candidates we can recall (per old label) to boost the
4 deficit target labels, after de-duplicating against already-annotated text.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
NEW_DATA = ROOT / "BR_new_data.csv"
ANNOT_FULL = ROOT / "data" / "annotations_br" / "br_annotation_full.csv"

OLD_LABELS = [
    "hate_speech", "false_info", "violence",
    "harassment", "obscenity", "illegal", "national_security",
]


def norm(s: object) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def main() -> None:
    print("== Loading BR_new_data ==")
    nd = pd.read_csv(NEW_DATA)
    print(f"rows: {len(nd):,}  cols: {list(nd.columns)}")

    for c in OLD_LABELS:
        nd[c] = pd.to_numeric(nd[c], errors="coerce").fillna(0.0)

    print("\n== Old-label positive counts (==1.0) ==")
    for c in OLD_LABELS:
        print(f"  {c:<18} {int((nd[c] == 1.0).sum()):>7}")

    pos_mask = (nd[OLD_LABELS] == 1.0).any(axis=1)
    print(f"\nany positive old label: {int(pos_mask.sum()):,}")
    print(f"all-zero rows         : {int((~pos_mask).sum()):,}")

    print("\n== Already-annotated text (br_annotation_full) ==")
    seen = set()
    if ANNOT_FULL.exists():
        af = pd.read_csv(ANNOT_FULL, usecols=lambda c: c in {"clean_text", "final_category"})
        print(f"annotated rows: {len(af):,}")
        seen = set(af["clean_text"].map(norm))
        seen.discard("")
        print(f"unique normalized clean_text: {len(seen):,}")
    else:
        print("!! br_annotation_full.csv not found")

    nd["_norm"] = nd["text"].map(norm)
    nd["_dup"] = nd["_norm"].isin(seen)
    nd["_empty"] = nd["_norm"] == ""

    print("\n== Candidate pool per old label AFTER dedup vs annotated & non-empty ==")
    avail = nd[pos_mask & ~nd["_dup"] & ~nd["_empty"]]
    print(f"total positive, deduped, non-empty: {len(avail):,}")
    for c in OLD_LABELS:
        n_raw = int((nd[c] == 1.0).sum())
        n_av = int(((nd[c] == 1.0) & ~nd["_dup"] & ~nd["_empty"]).sum())
        print(f"  {c:<18} raw={n_raw:>7}  available={n_av:>7}")

    print("\n== Deficit-target recall sources (available, deduped) ==")
    # SEI <- obscenity
    sei = (nd["obscenity"] == 1.0)
    # BR state security <- national_security + false_info
    brss = (nd["national_security"] == 1.0) | (nd["false_info"] == 1.0)
    # Politically sensitive <- false_info + national_security
    pol = (nd["false_info"] == 1.0) | (nd["national_security"] == 1.0)
    # Cyber <- illegal + false_info (phishing/hacking subset)
    cyber = (nd["illegal"] == 1.0) | (nd["false_info"] == 1.0)

    base = ~nd["_dup"] & ~nd["_empty"]
    print(f"  SEI  (obscenity)                 available={int((sei & base).sum()):>7}")
    print(f"  BRSS (national_security|false)   available={int((brss & base).sum()):>7}")
    print(f"  POL  (false_info|national_sec)   available={int((pol & base).sum()):>7}")
    print(f"  CYBER(illegal|false_info)        available={int((cyber & base).sum()):>7}")

    print("\n== Multi-label combos among available positives (top 15) ==")
    combo = avail[OLD_LABELS].apply(
        lambda r: "|".join([c for c in OLD_LABELS if r[c] == 1.0]), axis=1
    )
    print(combo.value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
