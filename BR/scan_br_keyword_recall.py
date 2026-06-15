#!/usr/bin/env python3
"""Estimate keyword-recall potential for the 4 deficit labels over BR_new_data,
especially the OLD-LABEL ALL-ZERO pool (which the old 7-label schema never
captured for cyber / political / state-security categories)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
NEW_DATA = ROOT / "BR_new_data.csv"
ANNOT_FULL = ROOT / "data" / "annotations_br" / "br_annotation_full.csv"
OLD_LABELS = ["hate_speech", "false_info", "violence", "harassment", "obscenity", "illegal", "national_security"]


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else ""


KW = {
    "BR_State_Security_Democratic_Order": [
        r"urna.{0,12}fraud", r"fraude.{0,6}nas urnas", r"elei[çc][aã]o roubada", r"tse (fraud|roubou)",
        r"interv-?en[çc][aã]o militar", r"golpe militar", r"ex[ée]rcito tem que assumir", r"for[çc]as armadas no poder",
        r"volta.{0,4}ai-?5", r"fechar o (congresso|stf|supremo)", r"invadir o stf", r"tomar bras[íi]lia",
        r"8 de janeiro", r"\bpcc\b", r"comando vermelho", r"\bcv\b", r"mil[íi]cia", r"fac[çc][aã]o",
        r"ditadura militar de volta", r"n[aã]o reconhe[çc]o o resultado",
    ],
    "Politically_Sensitive_Topics": [
        r"fake news", r"not[íi]cia falsa", r"desinforma[çc][aã]o", r"manipula[çc][aã]o eleitoral",
        r"golpe", r"guerra", r"massacre", r"refugiad", r"atentado", r"terroris",
        r"vacina.{0,10}(mata|chip)", r"plandemia", r"globalis", r"nova ordem mundial",
    ],
    "Cybersecurity_Malware": [
        r"phishing", r"malware", r"ransomware", r"trojan", r"spyware", r"keylogger", r"\bddos\b",
        r"sql inject", r"\bxss\b", r"hack(ear|er|ing)?", r"invadir conta", r"roubar (conta|senha|token)",
        r"senha vazad", r"login vazad", r"cookie.{0,6}(roubo|sess[aã]o)", r"2fa", r"captcha",
        r"clonar (whatsapp|cart[aã]o|chip)", r"spoof", r"botnet", r"exploit", r"deface",
    ],
    "Sexually_Explicit_Information": [
        r"porn", r"\bxvideos?\b", r"\bonlyfans?\b", r"\bgaroto?a de programa\b", r"acompanhante",
        r"sexo", r"nudes", r"\bputinha\b", r"\bbuceta\b", r"\bpau\b", r"\bgozar\b", r"\bxota\b",
        r"\bsexo? grupal\b", r"\bcam(s|girl)?\b", r"fude", r"\bsafad", r"\bnovinha\b",
    ],
}


def main():
    nd = pd.read_csv(NEW_DATA)
    for c in OLD_LABELS:
        nd[c] = pd.to_numeric(nd[c], errors="coerce").fillna(0.0)
    pos = (nd[OLD_LABELS] == 1.0).any(axis=1)
    zero = ~pos

    seen = set()
    if ANNOT_FULL.exists():
        af = pd.read_csv(ANNOT_FULL, usecols=lambda c: c == "clean_text")
        seen = set(af["clean_text"].map(norm)); seen.discard("")
    nd["_norm"] = nd["text"].map(norm)
    dup = nd["_norm"].isin(seen)
    nonempty = nd["_norm"] != ""

    print("== Funnel ==")
    print(f"total rows                : {len(nd):,}")
    print(f"any old-positive label    : {int(pos.sum()):,}")
    print(f"ALL-ZERO old labels       : {int(zero.sum()):,}   <-- 88% never flagged by old schema")
    print(f"already annotated (dedup) : {int(dup.sum()):,}")

    low = nd["_norm"]
    print("\n== Keyword recall over ALL-ZERO pool (deduped, non-empty) ==")
    base_zero = zero & ~dup & nonempty
    for label, pats in KW.items():
        rx = re.compile("|".join(pats))
        hit = low.str.contains(rx, na=False)
        print(f"  {label:<40} zero-pool hits={int((hit & base_zero).sum()):>6}   (whole-file hits={int((hit & ~dup & nonempty).sum()):>6})")

    print("\n== Combined extra candidates from ALL-ZERO pool (union of 4 deficit keyword sets) ==")
    rx_all = re.compile("|".join(p for pats in KW.values() for p in pats))
    hit_all = low.str.contains(rx_all, na=False)
    print(f"  zero-pool union hits     : {int((hit_all & base_zero).sum()):,}")


if __name__ == "__main__":
    main()
