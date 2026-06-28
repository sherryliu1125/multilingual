#!/usr/bin/env python3
"""Strip excess emoji from existing TR_R3 TR_SEX_*.csv (default: keep max 2 per text)."""
import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "TR_R3"
MAX_EMOJI = int(sys.argv[1] if len(sys.argv) > 1 else "2")

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)


def emoji_count(text: str) -> int:
    return sum(len(m) for m in _EMOJI_RE.findall(text))


def strip_excess(text: str) -> str:
    s = text
    while emoji_count(s) > MAX_EMOJI:
        s = re.sub(
            r"[\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            r"\U00002702-\U000027B0\U000024C2-\U0001F251"
            r"\U00002600-\U000026FF]+$",
            "",
            s,
        ).rstrip()
    return s


def main() -> None:
    changed = 0
    for path in sorted(OUT_DIR.glob("TR_SEX_*.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if not rows:
            continue
        file_changed = 0
        for row in rows:
            old = row.get("text", "")
            new = strip_excess(old)
            if new != old:
                row["text"] = new
                file_changed += 1
        if file_changed:
            with path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            print(f"{path.name}: cleaned {file_changed} rows")
            changed += file_changed
    print(f"done, {changed} rows updated (max {MAX_EMOJI} emoji kept)")


if __name__ == "__main__":
    main()
