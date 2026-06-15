#!/usr/bin/env python3
"""Build BR candidate pool CSV for run_br_annotation.py.

Rule (per user):
  - candidate SOURCES = old labels: false_info, violence, illegal, national_security
  - EXCLUDED as sources: hate_speech, harassment, obscenity  (screenshotted)
  - EXCLUDED: all-zero rows (safe)
  - take up to 2000 per source, union + dedup vs already-annotated text
  - emit one model-input CSV the script reads directly (+ sidecar with old labels)

Old labels are NOT shown to jurors. The jury prompt's own 8-label taxonomy decides.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
NEW_DATA = ROOT / "BR_new_data.csv"
ANNOT_FULL = ROOT / "data" / "annotations_br" / "br_annotation_full.csv"
OUT_INPUT = ROOT / "data" / "annotations_br" / "br_supplement_candidates_input.csv"
OUT_SIDECAR = ROOT / "data" / "annotations_br" / "br_supplement_candidates_sidecar.csv"

OLD_LABELS = ["hate_speech", "false_info", "violence", "harassment", "obscenity", "illegal", "national_security"]
SOURCES = ["false_info", "violence", "illegal", "national_security"]  # candidate sources
CAP_PER_SOURCE = 2000


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else ""


def main() -> None:
    nd = pd.read_csv(NEW_DATA)
    for c in OLD_LABELS:
        nd[c] = pd.to_numeric(nd[c], errors="coerce").fillna(0.0)

    # dedup vs already-annotated text
    seen = set()
    if ANNOT_FULL.exists():
        af = pd.read_csv(ANNOT_FULL, usecols=lambda c: c == "clean_text")
        seen = set(af["clean_text"].map(norm)); seen.discard("")
    nd["_norm"] = nd["text"].map(norm)
    base = (nd["_norm"] != "") & ~nd["_norm"].isin(seen)

    # take up to CAP_PER_SOURCE from each source, then union
    picked_idx = set()
    per_source = {}
    for src in SOURCES:
        pool = nd[(nd[src] == 1.0) & base]
        take = pool.head(CAP_PER_SOURCE)
        per_source[src] = len(take)
        picked_idx.update(take.index.tolist())

    cand = nd.loc[sorted(picked_idx)].copy()
    cand["content_id"] = cand["index"].map(lambda i: f"br_new_{int(i):08d}")
    cand["source_labels"] = cand[OLD_LABELS].apply(
        lambda r: "|".join([c for c in OLD_LABELS if r[c] == 1.0]), axis=1
    )

    # MODEL INPUT (only safe fields; NO old labels shown to model)
    pd.DataFrame({
        "content_id": cand["content_id"],
        "country": "BR",
        "language": "pt",
        "source": "br_new_data",
        "subreddit": "",
        "title": "",
        "clean_text": cand["text"].astype(str),
    }).to_csv(OUT_INPUT, index=False)

    # SIDECAR (old labels for later analysis, joined by content_id)
    cand[["content_id", "index"] + OLD_LABELS + ["source_labels"]].rename(
        columns={"index": "original_index"}
    ).to_csv(OUT_SIDECAR, index=False)

    print(f"TOTAL candidate pool (deduped union): {len(cand):,}")
    print(f"  model input -> {OUT_INPUT}")
    print(f"  sidecar     -> {OUT_SIDECAR}")
    print("\nper-source taken (cap 2000, before union dedup):")
    for src, n in per_source.items():
        avail = int(((nd[src] == 1.0) & base).sum())
        print(f"  {src:<18} taken={n:>5}  (available={avail})")


if __name__ == "__main__":
    main()
