#!/usr/bin/env python3
"""
Phase C: decide BERT trainability on clean_text_phaseA (readability / gibberish only).

Reads *_phaseB.csv, never modifies it. Writes *_phaseC.csv.

Policy: 宁可错杀不能错放 for gibberish; keep readable channel spam (sex/show/dash patterns).
Phase B spam flags alone do NOT auto-reject — only gibberish / unreadable text does.
"""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = SCRIPT_DIR / "TR" / "TR_R4_train_pool_need_review_annotated_phaseB.csv"

FAKE_AGGLUT_RE = re.compile(
    r"(?:EBİLİRSİN|ECEKTİM|ECEKLER|IYORTULAR|MEMİŞLER|MAMIŞLAR|ACAKTIM|ACAKLAR|"
    r"YTIM|ABİLİRSİN|EECEKTİM|IYORUM|YOR(TULAR|TU)|MEMİŞ|mamış|ebilirsin|"
    r"ecektim|ecekler|iyortular|memişler|mamışlar|acaktım|acaklar|ytim|abilirsin|"
    r"mştm|miyor|mekiyor|mekte|mekiyor|deliğiiyor|düzeltmekiyor|lenmek|memek|"
    r"ebilme|abilme|iyortular|memiş|mamış|acaklar|ecekler|MEKn|LIKe|IMde|"
    r"İYOR|IYOR|mekiyor|mekte)$",
    re.I,
)
SHOW_BOILERPLATE_RE = re.compile(
    r"görüntülü\s*cam.*(?:sậnậl|sanal).*?(?:whatsapp|shỗw|show)|"
    r"türbanli.*ensest.*cam.*sexting|"
    r"sậnậl--shỗw--|teldesex--|sậnậlshow|görüntülüshow|sanal\s*show\s*cam\s*sex",
    re.I,
)
OBFUSC_SHOW_RE = re.compile(r"sậnậl|shỗw|sanalshow|teldesex", re.I)
DASH_OBFUSC_RE = re.compile(r"--[\wậỗ\u0300-\u036f]+--", re.I)
PHONE_RE = re.compile(r"\b0?5?\d{7,12}\b")
DISTRICT_NAMES = (
    "ataköy", "maltepe", "bahçelievler", "merter", "çekmeköy", "sarıyer", "beşiktaş",
    "kartal", "kadıköy", "kadiköy", "üsküdar", "bağcılar", "bakırköy", "başakşehir",
    "avcılar", "bayrampaşa", "beykoz", "beylikdüzü", "küçükçekmece", "güngören",
    "taksim", "şişli", "sisli", "etiler", "florya", "beyoğlu", "pendik", "ümraniye",
    "sultangazi", "eyüp", "eyüpsultan", "kağıthane", "arnavutköy", "nişantaşı",
    "cevizlibağ", "yenibosna", "beykent", "acıbadem",
)
SENTENCE_LIKE_RE = re.compile(
    r"\b(ben|sen|biz|siz|bu|şu|o|ne|nasıl|neden|ama|değil|mi|mı|musun|musunuz|"
    r"diyor|dedi|gibi|için|var|yok|lan|amk|aq|ki|çok|bir|şey|olmak|isteyen|"
    r"var\s+mı|musun|misin|değil)\b",
    re.I,
)
PUNCT_RE = re.compile(r"[.!?…،]")


def words(text: str) -> list[str]:
    return re.findall(r"[\wçğıöşüÇĞİÖŞÜ']+", text, re.I)


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return default


def phase_b_flags(row: dict) -> set[str]:
    raw = str(row.get("phaseB_flags") or "").strip()
    if not raw:
        return set()
    return {part for part in raw.split("|") if part}


def has_fake_agglut_suffix(word: str) -> bool:
    if bool(FAKE_AGGLUT_RE.search(word)):
        return True
    # OCR glue: mixed case inside token (GÖZLETMEKn, SONRAKİmştm, ÇADIRCILIKe)
    if len(word) >= 8 and re.search(r"[a-zçğıöşü].*[A-ZÇĞİÖŞÜ]|[A-ZÇĞİÖŞÜ].*[a-zçğıöşü].*[A-ZÇĞİÖŞÜ]", word):
        letters = [c for c in word if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.4:
            return True
    return False


def district_name_count(text: str) -> int:
    lower = text.lower()
    return sum(1 for name in DISTRICT_NAMES if name in lower)


def show_template_signals(text: str, flags: set[str]) -> list[str]:
    hits: list[str] = []
    if SHOW_BOILERPLATE_RE.search(text):
        hits.append("show_boilerplate")
    if OBFUSC_SHOW_RE.search(text) and flags & {"show_hashtag", "cam_show_pattern", "dash_segment_spam"}:
        hits.append("obfuscated_show_template")
    if DASH_OBFUSC_RE.search(text) and flags & {"show_hashtag", "cam_show_pattern", "dash_segment_spam"}:
        hits.append("dash_obfuscated_show")
    return hits


def fake_word_density(text: str) -> float:
    token_list = words(text)
    if not token_list:
        return 0.0
    fake_count = sum(1 for token in token_list if has_fake_agglut_suffix(token))
    return fake_count / len(token_list)


def has_long_fake_agglut_word(text: str) -> bool:
    return any(len(token) > 18 and has_fake_agglut_suffix(token) for token in words(text))


def strip_fake_agglut_tokens(text: str) -> str:
    parts: list[str] = []
    for piece in re.split(r"(\s+)", text):
        if piece.strip() and not piece.isspace() and has_fake_agglut_suffix(piece):
            parts.append(" ")
        else:
            parts.append(piece)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def fake_agglut_stripped_incomplete(text: str) -> bool:
    """True when fake-agglut tokens exist and the remainder is not a usable sentence."""
    fake_tokens = [token for token in words(text) if has_fake_agglut_suffix(token)]
    if not fake_tokens:
        return False
    remainder = strip_fake_agglut_tokens(text)
    if not remainder:
        return True
    remainder_words = words(remainder)
    if len(remainder_words) < 4:
        return True
    return not is_sentence_like(remainder)


def is_sentence_like(text: str) -> bool:
    return bool(PUNCT_RE.search(text)) or bool(SENTENCE_LIKE_RE.search(text))


def hashtag_overload_signals(text: str, flags: set[str]) -> list[str]:
    """Reject when hashtags dominate and the remaining text is incomplete or fake."""
    hits: list[str] = []
    token_list = words(text)
    tag_list = re.findall(r"#\w+", text, re.I)
    tag_count = len(tag_list)
    if tag_count == 0:
        return hits

    stripped = re.sub(r"#\w+", " ", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    remainder_words = words(stripped)
    tag_ratio = tag_count / max(len(token_list), 1)
    remainder_fake = fake_word_density(stripped)
    remainder_sentence_like = is_sentence_like(stripped)

    if tag_count >= 3:
        hits.append("hashtag_count>=3")
    if tag_count >= 2 and tag_ratio >= 0.20:
        hits.append("hashtag_ratio_dominant")
    if tag_count >= 2 and len(remainder_words) < 8:
        hits.append("hashtag_stripped_too_few_words")
    if tag_count >= 2 and remainder_fake >= 0.12:
        hits.append("hashtag_fake_remainder")

    if "district_hashtag_stack" in flags and tag_count >= 2:
        if len(remainder_words) < 10 or not remainder_sentence_like:
            hits.append("district_hashtag_incomplete")
        if remainder_fake >= 0.10:
            hits.append("district_hashtag_fake_remainder")

    show_flags = {"show_hashtag", "cam_show_pattern", "dash_segment_spam"}
    if flags & show_flags and tag_count >= 1:
        if len(remainder_words) < 12 and not remainder_sentence_like:
            hits.append("show_hashtag_incomplete")

    return hits


def gibberish_policy_hits(row: dict, text: str) -> list[str]:
    """Return policy rule ids that fire (gibberish reject signals)."""
    hits: list[str] = []
    flags = phase_b_flags(row)
    fake_ratio = parse_float(row.get("phaseB_fake_ratio"))
    caps_ratio = parse_float(row.get("phaseB_caps_ratio"))
    fake_density = fake_word_density(text)
    sentence_like = is_sentence_like(text)

    if fake_density >= 0.15:
        hits.append("fake_density>=0.15")
    if fake_ratio >= 0.15:
        hits.append("fake_ratio>=0.15")
    if fake_agglut_stripped_incomplete(text):
        hits.append("fake_agglut_stripped_incomplete")

    if "high_caps_ratio" in flags and (fake_ratio >= 0.10 or fake_density >= 0.10):
        hits.append("high_caps_with_fake_signal")

    if "district_hashtag_stack" in flags and (
        fake_ratio >= 0.10 or "fake_agglutination" in flags
    ):
        hits.append("district_stack_with_fake_signal")

    template_flags = {"show_hashtag", "cam_show_pattern"}
    if flags & template_flags and fake_ratio >= 0.10:
        hits.append("show_template_with_fake_ratio")

    # Caps word salad mixed with show/cam template (fake_ratio may be 0 for short nonsense tokens).
    if flags & template_flags and caps_ratio >= 0.35 and not sentence_like:
        hits.append("show_template_caps_salad")

    if has_long_fake_agglut_word(text):
        hits.append("long_fake_agglut_word")

    if fake_agglut_stripped_incomplete(text):
        hits.append("fake_agglut_stripped_incomplete")

    high_caps = "high_caps_ratio" in flags or caps_ratio >= 0.45
    if not sentence_like and (
        fake_ratio >= 0.15
        or fake_density >= 0.15
        or fake_agglut_stripped_incomplete(text)
        or (high_caps and "district_hashtag_stack" in flags)
    ):
        hits.append("not_sentence_like_gibberish")

    hits.extend(show_template_signals(text, flags))

    if district_name_count(text) >= 8:
        hits.append("district_name_list_spam")

    if PHONE_RE.search(text) and district_name_count(text) >= 2 and not sentence_like:
        hits.append("phone_district_seo")

    if PHONE_RE.search(text) and len(words(text)) <= 12:
        hits.append("phone_short_seo")

    hits.extend(hashtag_overload_signals(text, flags))

    return hits


def judge_row(row: dict) -> tuple[bool, str, str, str, str]:
    text = str(row.get("clean_text_phaseA") or row.get("clean_text") or "")
    stripped = text.strip()
    token_list = words(stripped)

    reject_reasons: list[str] = []
    policy_notes: list[str] = []

    if not stripped:
        return False, "empty", "reject", "empty", "basic:empty"
    if len(stripped) < 20:
        reject_reasons.append("too_short")
    if len(token_list) < 4:
        reject_reasons.append("too_few_words")
    if len(stripped) > 512:
        reject_reasons.append("too_long")

    gib_hits = gibberish_policy_hits(row, stripped)
    if gib_hits:
        policy_notes.extend(gib_hits)
        template_hits = {
            "show_boilerplate",
            "obfuscated_show_template",
            "dash_obfuscated_show",
            "district_name_list_spam",
            "phone_district_seo",
            "phone_short_seo",
            "hashtag_count>=3",
            "hashtag_ratio_dominant",
            "hashtag_stripped_too_few_words",
            "hashtag_fake_remainder",
            "district_hashtag_incomplete",
            "district_hashtag_fake_remainder",
            "show_hashtag_incomplete",
        }
        if any(h.startswith("fake") or "fake" in h for h in gib_hits):
            reject_reasons.append("fake_agglutination")
        elif "long_fake_agglut_word" in gib_hits:
            reject_reasons.append("fake_agglutination")
        elif any(h in template_hits for h in gib_hits):
            reject_reasons.append("seo_template_gibberish")
        elif "not_sentence_like_gibberish" in gib_hits:
            reject_reasons.append("not_sentence_like")
        else:
            reject_reasons.append("gibberish_likely")

    if reject_reasons:
        if any(r in reject_reasons for r in ("fake_agglutination", "gibberish_likely", "not_sentence_like")):
            gib_tier = "gibberish_likely"
        elif "too_short" in reject_reasons or "too_few_words" in reject_reasons:
            gib_tier = "too_short"
        elif "too_long" in reject_reasons:
            gib_tier = "too_long"
        else:
            gib_tier = "gibberish_likely"
        policy = "|".join(policy_notes) if policy_notes else "basic"
        return False, "|".join(dict.fromkeys(reject_reasons)), "reject", gib_tier, policy

    flags = phase_b_flags(row)
    gib_tier = "readable" if flags else "clean"
    return True, "", "ok", gib_tier, "keep:readable_spam_ok"


def default_output_for(input_path: Path) -> Path:
    if input_path.stem.endswith("_phaseB"):
        stem = input_path.stem[: -len("_phaseB")] + "_phaseC"
    else:
        stem = input_path.stem + "_phaseC"
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
        "bert_text_ok",
        "bert_exclude_reason",
        "phaseC_verdict",
        "bert_gibberish_tier",
        "bert_reject_policy",
    ]
    out_fieldnames = source_fieldnames + [col for col in extra_cols if col not in source_fieldnames]
    stats: dict[str, int] = {"rows": len(rows), "bert_ok": 0, "bert_reject": 0}

    for row in rows:
        ok, reason, verdict, gib_tier, policy = judge_row(row)
        row["bert_text_ok"] = "true" if ok else "false"
        row["bert_exclude_reason"] = reason
        row["phaseC_verdict"] = verdict
        row["bert_gibberish_tier"] = gib_tier
        row["bert_reject_policy"] = policy
        if ok:
            stats["bert_ok"] += 1
        else:
            stats["bert_reject"] += 1
            if reason:
                primary = reason.split("|")[0]
                stats[f"reason:{primary}"] = stats.get(f"reason:{primary}", 0) + 1
            if policy and not ok:
                for part in policy.split("|"):
                    stats[f"policy:{part}"] = stats.get(f"policy:{part}", 0) + 1
        stats[f"gib_tier:{gib_tier}"] = stats.get(f"gib_tier:{gib_tier}", 0) + 1

    write_csv_atomic(output_path, out_fieldnames, rows)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C: BERT readability judgment on phaseB output")
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
    print(f"bert_text_ok=true: {stats['bert_ok']}")
    print(f"bert_text_ok=false: {stats['bert_reject']}")
    for key in sorted(k for k in stats if k.startswith("reason:")):
        print(f"  {key.replace('reason:', '')}: {stats[key]}")
    for key in sorted(k for k in stats if k.startswith("policy:")):
        print(f"  {key.replace('policy:', '')}: {stats[key]}")
    for key in sorted(k for k in stats if k.startswith("gib_tier:")):
        print(f"  {key.replace('gib_tier:', '')}: {stats[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
