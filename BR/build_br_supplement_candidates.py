#!/usr/bin/env python3
"""Build BR supplement candidate pool from BR_new_data.csv (COMBINED recall).

Two recall channels, unioned:
  1) OLD-LABEL targeted recall  -> rows the old 7-label schema already flagged
     (obscenity / national_security / false_info / illegal / violence∩pol)
  2) KEYWORD recall over the ALL-ZERO pool -> deficit categories the old schema
     never had (BR_State_Security / Politically_Sensitive / Cybersecurity_Malware)
     plus extra Sexually_Explicit recall.

Old labels are NOT shown to jurors. They live only in the sidecar, joined by
content_id. Emits a clean MODEL-INPUT csv + a SIDECAR csv with all metadata.

Target deficit labels (goal 2000 each):
  BR_State_Security_Democratic_Order, Sexually_Explicit_Information,
  Politically_Sensitive_Topics, Cybersecurity_Malware
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

# Keyword recall sets per deficit target (BR-specific redlines + sexual terms).
KW = {
    "BR_State_Security_Democratic_Order": [
        r"urna.{0,12}fraud", r"fraude.{0,6}nas urnas", r"elei[çc][aã]o roubada", r"tse (?:fraud|roubou)",
        r"interv-?en[çc][aã]o militar", r"golpe militar", r"ex[ée]rcito tem que assumir", r"for[çc]as armadas no poder",
        r"volta.{0,4}ai-?5", r"fechar o (?:congresso|stf|supremo)", r"invadir o stf", r"tomar bras[íi]lia",
        r"8 de janeiro", r"\bpcc\b", r"comando vermelho", r"mil[íi]cia", r"fac[çc][aã]o",
        r"ditadura militar de volta", r"n[aã]o reconhe[çc]o o resultado",
    ],
    "Politically_Sensitive_Topics": [
        r"fake news", r"not[íi]cia falsa", r"desinforma[çc][aã]o", r"manipula[çc][aã]o eleitoral",
        r"\bgolpe\b", r"\bguerra\b", r"massacre", r"refugiad", r"atentado", r"terroris",
        r"vacina.{0,10}(?:mata|chip)", r"plandemia", r"globalis", r"nova ordem mundial",
    ],
    "Cybersecurity_Malware": [
        r"phishing", r"malware", r"ransomware", r"trojan", r"spyware", r"keylogger", r"\bddos\b",
        r"sql inject", r"\bxss\b", r"hack(?:ear|er|ing)?", r"invadir conta", r"roubar (?:conta|senha|token)",
        r"senha vazad", r"login vazad", r"cookie.{0,6}(?:roubo|sess[aã]o)", r"\b2fa\b", r"captcha",
        r"clonar (?:whatsapp|cart[aã]o|chip)", r"spoof", r"botnet", r"exploit", r"deface",
    ],
    "Sexually_Explicit_Information": [
        r"porn", r"\bxvideos?\b", r"\bonlyfans?\b", r"garoto?a de programa", r"acompanhante",
        r"\bsexo\b", r"\bnudes\b", r"\bputinha\b", r"\bbuceta\b", r"\bgozar\b", r"\bxota\b",
        r"sexo? grupal", r"\bcamgirl\b", r"\bfude", r"\bsafad", r"\bnovinha\b",
    ],
}

OLD_TARGETS = {
    "obscenity": ["Sexually_Explicit_Information"],
    "national_security": ["BR_State_Security_Democratic_Order", "Politically_Sensitive_Topics"],
    "false_info": ["Politically_Sensitive_Topics", "BR_State_Security_Democratic_Order", "Cybersecurity_Malware", "Dangerous_Content"],
    "illegal": ["Cybersecurity_Malware", "Dangerous_Content", "Sexually_Explicit_Information"],
    "violence": ["Dangerous_Content", "BR_State_Security_Democratic_Order"],
}


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else ""


def main() -> None:
    nd = pd.read_csv(NEW_DATA)
    for c in OLD_LABELS:
        nd[c] = pd.to_numeric(nd[c], errors="coerce").fillna(0.0)

    seen = set()
    if ANNOT_FULL.exists():
        af = pd.read_csv(ANNOT_FULL, usecols=lambda c: c == "clean_text")
        seen = set(af["clean_text"].map(norm)); seen.discard("")
    nd["_norm"] = nd["text"].map(norm)
    base = (nd["_norm"] != "") & ~nd["_norm"].isin(seen)

    # ---- channel 1: old-label targeted recall ----
    m_obsc = nd["obscenity"] == 1.0
    m_nat = nd["national_security"] == 1.0
    m_false = nd["false_info"] == 1.0
    m_illegal = nd["illegal"] == 1.0
    m_viol = nd["violence"] == 1.0
    old_recall = m_obsc | m_nat | m_false | m_illegal | (m_viol & (m_nat | m_false))

    # ---- channel 2: keyword recall (whole file; will be tagged) ----
    kw_hit = {label: nd["_norm"].str.contains(re.compile("|".join(pats)), na=False) for label, pats in KW.items()}
    kw_any = pd.Series(False, index=nd.index)
    for h in kw_hit.values():
        kw_any = kw_any | h

    recall = (old_recall | kw_any) & base
    cand = nd[recall].copy()

    def source_labels(r):
        return "|".join([c for c in OLD_LABELS if r[c] == 1.0])

    def kw_labels(idx):
        return [label for label, h in kw_hit.items() if h.loc[idx]]

    def build_meta(r):
        idx = r.name
        pools, targets = [], set()
        for c in OLD_LABELS:
            if r[c] == 1.0 and c in OLD_TARGETS:
                pools.append(f"old:{c}")
                targets.update(OLD_TARGETS[c])
        kls = kw_labels(idx)
        for kl in kls:
            pools.append(f"kw:{kl}")
            targets.add(kl)
        old_pos = any(r[c] == 1.0 for c in OLD_LABELS)
        method = "both" if (old_pos and kls) else ("keyword" if kls and not old_pos else "old_label")
        return pd.Series({
            "candidate_pool": "|".join(pools),
            "candidate_targets": "|".join(sorted(targets)),
            "recall_method": method,
        })

    cand["content_id"] = cand["index"].map(lambda i: f"br_new_{int(i):08d}")
    cand["source_labels"] = cand.apply(source_labels, axis=1)
    meta = cand.apply(build_meta, axis=1)
    cand = pd.concat([cand, meta], axis=1)

    # ---- MODEL INPUT (safe fields only; NO old labels, NO keyword tags) ----
    pd.DataFrame({
        "content_id": cand["content_id"],
        "country": "BR",
        "language": "pt",
        "source": "br_new_data",
        "subreddit": "",
        "title": "",
        "clean_text": cand["text"].astype(str),
    }).to_csv(OUT_INPUT, index=False)

    # ---- SIDECAR ----
    side_cols = ["content_id", "index"] + OLD_LABELS + ["source_labels", "candidate_pool", "candidate_targets", "recall_method"]
    cand[side_cols].rename(columns={"index": "original_index"}).to_csv(OUT_SIDECAR, index=False)

    # ---- report ----
    print(f"TOTAL candidates: {len(cand):,}")
    print(f"  -> {OUT_INPUT.name}")
    print(f"  -> {OUT_SIDECAR.name}")
    print("\nrecall_method:")
    print(cand["recall_method"].value_counts().to_string())
    print("\nkeyword-hit counts within candidate pool (overlapping):")
    for label, h in kw_hit.items():
        print(f"  kw:{label:<38} {int((h & recall).sum()):>6}")
    print("\nold-label counts within candidate pool (overlapping):")
    for c in OLD_LABELS:
        print(f"  old:{c:<20} {int(((cand[c]==1.0)).sum()):>6}")


if __name__ == "__main__":
    main()
