#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMON_PATH = ROOT / "common_base.txt"
COUNTRIES_DIR = ROOT / "countries"
COMPILED_DIR = ROOT / "compiled"

PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")
HEADER_RE = re.compile(r"^([a-z_]+):\s*$")

REQUIRED_SECTIONS = {
    "country_header",
    "allowed_local_labels",
    "local_language_context",
    "country_code_or_context",
    "local_redline_summary",
    "local_language_difficulty",
    "harassment_local_terms",
    "hate_speech_local_identity_markers",
    "sexual_local_terms",
    "political_country_variables",
    "local_label_cards",
    "safe_country_boundaries",
    "local_rule",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def parse_patch(path: Path) -> dict[str, str]:
    text = read_text(path)
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in text.splitlines():
        match = HEADER_RE.match(line)
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)

    parsed = {key: "\n".join(value).strip() for key, value in sections.items()}
    missing = sorted(REQUIRED_SECTIONS - parsed.keys())
    if missing:
        raise ValueError(f"{path.name} missing required sections: {', '.join(missing)}")

    if not parsed["allowed_local_labels"]:
        raise ValueError(f"{path.name} has empty allowed_local_labels")

    return parsed


def country_code_from_patch(path: Path) -> str:
    match = re.match(r"^([A-Z]+)_patch\.txt$", path.name)
    if not match:
        raise ValueError(f"Unexpected patch filename: {path.name}")
    return match.group(1)


def build_replacements(sections: dict[str, str]) -> dict[str, str]:
    replacements = dict(sections)
    replacements["country_display_name"] = extract_required_value(
        sections["country_header"], "Country display name"
    )
    replacements.update(parse_political_variables(sections["political_country_variables"]))
    return replacements


def extract_required_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            if value:
                return value
    raise ValueError(f"country_header missing {key}:")


def parse_political_variables(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("country_specific:"):
            values["country_specific"] = line.split(":", 1)[1].strip()
        elif line.startswith("non_country:"):
            values["non_country"] = line.split(":", 1)[1].strip()
    missing = {"country_specific", "non_country"} - values.keys()
    if missing:
        raise ValueError(
            "political_country_variables missing: " + ", ".join(sorted(missing))
        )
    return values


def compile_prompt(common: str, replacements: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in replacements:
            raise ValueError(f"No replacement provided for {{{{{key}}}}}")
        return replacements[key]

    compiled = PLACEHOLDER_RE.sub(replace, common)
    leftovers = PLACEHOLDER_RE.findall(compiled)
    if leftovers:
        raise ValueError("Unreplaced placeholders: " + ", ".join(sorted(set(leftovers))))
    return compiled.rstrip() + "\n"


def main() -> None:
    common = read_text(COMMON_PATH)
    patch_paths = sorted(COUNTRIES_DIR.glob("*_patch.txt"))
    if not patch_paths:
        raise SystemExit("No country patch files found.")

    COMPILED_DIR.mkdir(exist_ok=True)

    written: list[Path] = []
    for patch_path in patch_paths:
        country_code = country_code_from_patch(patch_path)
        sections = parse_patch(patch_path)
        replacements = build_replacements(sections)
        compiled = compile_prompt(common, replacements)
        output_path = COMPILED_DIR / f"{country_code}_full.txt"
        output_path.write_text(compiled, encoding="utf-8")
        written.append(output_path)

    for path in written:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
