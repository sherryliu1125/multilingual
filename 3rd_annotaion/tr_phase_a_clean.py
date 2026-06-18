#!/usr/bin/env python3
"""
Phase A: remove LINK / escort artifacts from clean_text only.

Does not modify the source CSV. Writes a sibling file with suffix _phaseA.csv
and adds clean_text_phaseA + phaseA_had_artifacts columns.
"""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = SCRIPT_DIR / "TR" / "TR_R4_train_pool_need_review_annotated.csv"

COMBINING_RE = re.compile(
    r"[\u0300-\u036f\u0483-\u0489\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]"
)
LINK_RE = re.compile(r"\bLINK\b|https?://\S+", re.I)
PARENS_BLOCK_RE = re.compile(r"\(\(\([^)]*?\)\)\)", re.S)
SLASH_BLOCK_RE = re.compile(r"//[^/\n]{0,120}//")
SPACED_ESCORT_RE = re.compile(r"\be\s+s\s+c\s+o\s+r\s+t\b", re.I)
ESCORT_HASHTAG_RE = re.compile(r"#\w*escort\w*", re.I)
PLAIN_ESCORT_RE = re.compile(r"\bescort\w*\b", re.I)


def is_obfuscated_escort_token(token: str) -> bool:
    if not COMBINING_RE.search(token):
        return False
    base = COMBINING_RE.sub("", token).lower().replace(" ", "")
    if "escort" in base:
        return True
    combining_count = len(COMBINING_RE.findall(token))
    return combining_count >= 2 and combining_count / max(len(token), 1) > 0.2


def strip_obfuscated_tokens(text: str) -> str:
    parts: list[str] = []
    for piece in re.split(r"(\s+)", text):
        if piece.strip() and not piece.isspace() and is_obfuscated_escort_token(piece):
            parts.append(" ")
        else:
            parts.append(piece)
    return "".join(parts)


def has_phase_a_artifacts(text: str) -> bool:
    if LINK_RE.search(text):
        return True
    if PARENS_BLOCK_RE.search(text) or SLASH_BLOCK_RE.search(text):
        return True
    if SPACED_ESCORT_RE.search(text):
        return True
    if ESCORT_HASHTAG_RE.search(text):
        return True
    return any(is_obfuscated_escort_token(token) for token in re.findall(r"\S+", text))


def phase_a_clean(text: str) -> str:
    cleaned = text
    cleaned = LINK_RE.sub(" ", cleaned)
    cleaned = PARENS_BLOCK_RE.sub(" ", cleaned)
    cleaned = SLASH_BLOCK_RE.sub(" ", cleaned)
    cleaned = SPACED_ESCORT_RE.sub(" ", cleaned)
    cleaned = ESCORT_HASHTAG_RE.sub(" ", cleaned)
    cleaned = strip_obfuscated_tokens(cleaned)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = COMBINING_RE.sub("", cleaned)
    cleaned = PLAIN_ESCORT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def default_output_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_phaseA{input_path.suffix}")


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


def run(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    out_fieldnames = source_fieldnames + ["clean_text_phaseA", "phaseA_had_artifacts"]
    changed = 0
    with_artifacts = 0

    for row in rows:
        original = str(row.get("clean_text") or "")
        had_artifacts = has_phase_a_artifacts(original)
        cleaned = phase_a_clean(original) if had_artifacts else original
        row["clean_text_phaseA"] = cleaned
        row["phaseA_had_artifacts"] = "true" if had_artifacts else "false"
        if cleaned != original:
            changed += 1
        if had_artifacts:
            with_artifacts += 1

    write_csv_atomic(output_path, out_fieldnames, rows)
    return len(rows), with_artifacts, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A: strip LINK/escort artifacts from TR clean_text")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source CSV (default: {DEFAULT_INPUT.relative_to(SCRIPT_DIR)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <input_stem>_phaseA.csv beside input)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else SCRIPT_DIR / args.input
    output_path = args.output
    if output_path is None:
        output_path = default_output_for(input_path)
    elif not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from input path.")

    total, with_artifacts, changed = run(input_path, output_path)
    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"rows: {total}")
    print(f"phaseA_had_artifacts=true: {with_artifacts}")
    print(f"clean_text changed: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
