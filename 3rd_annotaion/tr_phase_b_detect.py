#!/usr/bin/env python3
"""
Phase B: detect SEO / gibberish patterns on clean_text_phaseA.

Does not modify source files. Reads *_phaseA.csv and writes *_phaseB.csv
with detection flags only (no text deletion).
"""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = SCRIPT_DIR / "TR" / "TR_R4_train_pool_need_review_annotated_phaseA.csv"

CAM_SHOW_RE = re.compile(
    r"cam[-\s]*show|sậnậl|shỗw|sanal[-\s]*show|teldesex|görüntülü\s*show|"
    r"görüntülüshow|whatsappshow|ücretlishow|cam\s*show",
    re.I,
)
SHOW_HASHTAG_RE = re.compile(r"#ücretlishow|#whatsappshow", re.I)
SEX_SPAM_RE = re.compile(
    r"\b(sexting|satilik|ensest|türbanli|türbanlı|porno|sikiş|teldesex)\b",
    re.I,
)
DASH_SEGMENT_RE = re.compile(r"--[\wậỗ]+--", re.I)
SLASH_NOISE_RE = re.compile(r"(?:\s/+\s|//|\s/\s*/)")
FAKE_AGGLUT_RE = re.compile(
    r"(?:EBİLİRSİN|ECEKTİM|ECEKLER|IYORTULAR|MEMİŞLER|MAMIŞLAR|ACAKTIM|ACAKLAR|"
    r"YTIM|ABİLİRSİN|EECEKTİM|IYORUM|YOR(TULAR|TU)|MEMİŞ|mamış|ebilirsin|"
    r"ecektim|ecekler|iyortular|memişler|mamışlar|acaktım|acaklar|ytim|abilirsin)$",
    re.I,
)
DISTRICT_TAGS = (
    "beşiktaş", "ortaköy", "etiler", "şişli", "taksim", "bakırköy", "ataköy", "halkalı",
    "istanbul", "kadiköy", "kadıköy", "maltepe", "kartal", "gebze", "tuzla", "florya",
    "mecidiyeköy", "üsküdar", "pendik", "ümraniye", "arnavutköy", "sarıyer",
    "gaziosmanpaşa", "bayrampaşa", "başakşehir", "eyüp", "eyüpsultan", "okmeydanı",
    "kağıthane", "beyoğlu", "küçükçekmece", "sirinevler", "bağcılar",
)


def words(text: str) -> list[str]:
    return re.findall(r"[\wçğıöşüÇĞİÖŞÜ']+", text, re.I)


def is_shout_word(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    if len(letters) < 6:
        return False
    upperish = sum(1 for c in letters if c.isupper() or c in "İIÜÖÇŞĞ")
    return upperish / len(letters) >= 0.85


def district_hashtag_count(text: str) -> int:
    tags = re.findall(r"#(\w+)", text, re.I)
    count = 0
    for tag in tags:
        lower = tag.lower()
        if any(d in lower for d in DISTRICT_TAGS):
            count += 1
    return count


def detect(text: str) -> tuple[list[str], float, float]:
    flags: list[str] = []
    if CAM_SHOW_RE.search(text):
        flags.append("cam_show_pattern")
    if SHOW_HASHTAG_RE.search(text):
        flags.append("show_hashtag")
    if SEX_SPAM_RE.search(text):
        flags.append("sex_keyword_spam")
    if DASH_SEGMENT_RE.search(text):
        flags.append("dash_segment_spam")
    if SLASH_NOISE_RE.search(text):
        flags.append("slash_noise")

    ws = words(text)
    caps_ratio = 0.0
    fake_ratio = 0.0
    if ws:
        shout = [w for w in ws if is_shout_word(w)]
        fake = [w for w in ws if FAKE_AGGLUT_RE.search(w) or (len(w) > 18 and is_shout_word(w))]
        caps_ratio = len(shout) / len(ws)
        fake_ratio = len(fake) / len(ws)
        if caps_ratio >= 0.45:
            flags.append("high_caps_ratio")
        if fake_ratio >= 0.25:
            flags.append("fake_agglutination")

    if district_hashtag_count(text) >= 2:
        flags.append("district_hashtag_stack")

    return flags, round(caps_ratio, 3), round(fake_ratio, 3)


def spam_tier(flags: list[str]) -> str:
    strong = {
        "show_hashtag",
        "cam_show_pattern",
        "dash_segment_spam",
        "fake_agglutination",
        "sex_keyword_spam",
    }
    strong_hits = [f for f in flags if f in strong]
    if len(strong_hits) >= 2:
        return "spam_likely"
    if len(strong_hits) == 1:
        return "spam_possible"
    if "high_caps_ratio" in flags and "district_hashtag_stack" in flags:
        return "spam_possible"
    if flags:
        return "review"
    return "clean"


def default_output_for(input_path: Path) -> Path:
    if input_path.stem.endswith("_phaseA"):
        stem = input_path.stem[: -len("_phaseA")] + "_phaseB"
    else:
        stem = input_path.stem + "_phaseB"
    return input_path.with_name(f"{stem}{input_path.suffix}")


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


def run(input_path: Path, output_path: Path) -> dict[str, int]:
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    extra_cols = [
        "phaseB_flags",
        "phaseB_caps_ratio",
        "phaseB_fake_ratio",
        "phaseB_spam_tier",
    ]
    out_fieldnames = source_fieldnames + [c for c in extra_cols if c not in source_fieldnames]
    stats: dict[str, int] = {"rows": len(rows)}

    for row in rows:
        text = str(row.get("clean_text_phaseA") or row.get("clean_text") or "")
        flags, caps_ratio, fake_ratio = detect(text)
        tier = spam_tier(flags)
        row["phaseB_flags"] = "|".join(flags)
        row["phaseB_caps_ratio"] = str(caps_ratio)
        row["phaseB_fake_ratio"] = str(fake_ratio)
        row["phaseB_spam_tier"] = tier
        stats[tier] = stats.get(tier, 0) + 1
        for flag in flags:
            stats[f"flag:{flag}"] = stats.get(f"flag:{flag}", 0) + 1

    write_csv_atomic(output_path, out_fieldnames, rows)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B: flag SEO/gibberish patterns on clean_text_phaseA")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else SCRIPT_DIR / args.input
    output_path = args.output or default_output_for(input_path)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from input path.")

    stats = run(input_path, output_path)
    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"rows: {stats['rows']}")
    for key in ("clean", "review", "spam_possible", "spam_likely"):
        if key in stats:
            print(f"{key}: {stats[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
