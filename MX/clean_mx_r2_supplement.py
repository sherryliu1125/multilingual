#!/usr/bin/env python3
"""
Conservative cleanup for MX R2 synthetic supplement.

This script cleans the generated supplement before it is used for R2 training.
It does not call NIW and does not generate new text.

Default high-precision actions:
- Drop Cybersecurity_Malware rows when the text clearly targets
  banking/payment/financial account theft or phishing.
- MX_Narco_Culture_And_Cartel_Glorification -> Dangerous_Content when the text
  is operational crime: drug sales/transport, weapon acquisition, laundering,
  evading police, recruitment for violence, or hit/attack tasks.
- Sexually_Explicit_Information -> Dangerous_Content when the text mentions
  minors, school age, coercion, blackmail, leaked packs, hidden cameras, or
  non-consensual sexual content.
- Politically_Sensitive_Topics -> Dangerous_Content only for explicit violence
  planning or weapon/attack operational content.

Rows that are only suspicious/template-like are kept but flagged in the review
queue. Use the review queue for manual/NIW audit if you want stricter cleanup.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "synthetic_R2" / "mx_r2_target2000_supplement_valid_dedup.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "synthetic_R2" / "mx_r2_target2000_add.csv"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "quality_audit"


FINANCIAL_TARGET_RE = re.compile(
    r"""
    \b(?:banco|bbva|banorte|santander|hsbc|azteca|banamex|citibanamex|
       mercado\s*pago|paypal|oxxo|coppel|tarjeta|cuenta\s+bancaria|
       banca\s+m[oó]vil|app\s+del\s+banco|login\s+del\s+banco)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

FINANCIAL_ABUSE_RE = re.compile(
    r"""
    \b(?:phishing|fake\s+login|login\s+falso|p[aá]gina\s+falsa|clon(?:e|ada)|
       robar|captur(?:a|ar)|sacar|extraer|credenciales?|contraseñ?a|password|
       token|cookie|sesi[oó]n|2fa|otp|keylogger|malware|kit)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

NARCO_OPERATIONAL_RE = re.compile(
    r"""
    \b(?:vendo|compro|busco|necesito)\b.{0,80}
    \b(?:droga|coca|cristal|fentanilo|mota|armas?|cuerno|rifle|granada|municiones?)\b
    |
    \b(?:pasar|mover|transportar|cruzar|subir|bajar)\b.{0,80}
    \b(?:paquete|carga|kilos?|gramos?|droga|coca|cristal|fentanilo|mota)\b
    |
    \b(?:ruta|rutas)\b.{0,80}\b(?:droga|coca|cristal|fentanilo|mota|carga|paquete)\b
    |
    \b(?:lavar\s+dinero|cobro\s+de\s+piso|plaza\s+vacante|jale\s+de\s+halc[oó]n|
       reclut(?:o|an|ando).{0,60}(?:halc[oó]n|sicario|plaza)|evadir.{0,40}(?:ret[eé]n|patrulla|polic[ií]a|checkpoint))\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

NARCO_GLORIFY_RE = re.compile(
    r"""
    \b(?:puro\s+cartel|arriba\s+el|saludos?\s+al\s+jefe|jefe|patr[oó]n|
       vida\s+de\s+narco|corrid[oó]n?|corridos?\s+tumbados?|narcocorrido|
       la\s+plaza|la\s+empresa|respeto\s+al\s+señ?or|manda\s+la\s+mafia|
       trae\s+billete|lujo|trocas?|cadena|poder|protege\s+el\s+barrio)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

SEX_DANGEROUS_RE = re.compile(
    r"""
    \b(?:menor(?:es)?|niñ[oa]s?|adolescent(?:e|es)?|secundaria|prepa|
       colegiala|schoolgirl|15\s*a[nñ]os|16\s*a[nñ]os|17\s*a[nñ]os|
       dormid[ao]|borrach[ao]|drograd[ao]|sin\s+consentimiento|forzar|
       violar|chantaje|extorsi[oó]n|c[aá]mara\s+oculta|pack\s+filtrado|
       pack\s+de\s+mi\s+ex|revenge\s+porn|sin\s+que\s+sepa)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

POLITICAL_OPERATIONAL_VIOLENCE_RE = re.compile(
    r"""
    \b(?:(?:lleven|traigan|traer)\s+armas?|molotov|explosiv[ao]s?|
       quememos\s+(?:casillas|palacio|ayuntamiento)|quemar\s+(?:casillas|palacio|ayuntamiento)|
       ataquemos\s+(?:casillas|palacio|ayuntamiento|polic[ií]a|militares?)|
       atacar\s+(?:casillas|palacio|ayuntamiento|polic[ií]a|militares?)|
       punto\s+de\s+reuni[oó]n|hora\s+exacta|ruta\s+de\s+escape|
       instrucciones\s+para\s+el\s+ataque)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

POLITICAL_NEUTRAL_RE = re.compile(
    r"""
    \b(?:noticia|reportaje|an[aá]lisis|historia|documental|debate|opini[oó]n|
       propuesta|pol[ií]tica\s+p[uú]blica|cr[ií]tica\s+al\s+gobierno)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean MX R2 synthetic supplement.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--drop-neutral-narco",
        action="store_true",
        help="Drop narco rows that lack obvious glorification and lack operational crime.",
    )
    parser.add_argument(
        "--drop-neutral-political",
        action="store_true",
        help="Drop political rows that look neutral/news-like by keyword heuristic.",
    )
    return parser.parse_args()


def text_of(row: pd.Series) -> str:
    return str(row.get("text", "") or "")


def is_narco_operational_crime(text: str) -> bool:
    lowered = text.lower()
    if "no compro armas" in lowered and ("cancion" in lowered or "canción" in lowered or "verso" in lowered):
        return False
    return bool(NARCO_OPERATIONAL_RE.search(text))


def is_sexual_dangerous(text: str) -> bool:
    # "no es/son para niños" is a common adult-content disclaimer, not a minor target.
    normalized = re.sub(r"\bno\s+(?:es|son)\s+para\s+niñ[oa]s\b", "", text, flags=re.IGNORECASE)
    return bool(SEX_DANGEROUS_RE.search(normalized))


def decide(row: pd.Series, args: argparse.Namespace) -> tuple[str, str, str | None]:
    label = str(row.get("llm_label", "")).strip()
    text = text_of(row)

    if label == "Cybersecurity_Malware":
        if FINANCIAL_TARGET_RE.search(text) and FINANCIAL_ABUSE_RE.search(text):
            return "drop", "cyber_financial_asset_theft_drop", None
        return "keep", "cyber_keep", None

    if label == "MX_Narco_Culture_And_Cartel_Glorification":
        if is_narco_operational_crime(text):
            return "relabel", "narco_operational_crime_to_dangerous", "Dangerous_Content"
        if args.drop_neutral_narco and not NARCO_GLORIFY_RE.search(text):
            return "drop", "narco_no_clear_glorification_drop", None
        return "keep", "narco_glorification_keep", None

    if label == "Sexually_Explicit_Information":
        if is_sexual_dangerous(text):
            return "relabel", "sexual_nonconsensual_or_minor_to_dangerous", "Dangerous_Content"
        return "keep", "sexual_keep", None

    if label == "Politically_Sensitive_Topics":
        if POLITICAL_OPERATIONAL_VIOLENCE_RE.search(text):
            return "relabel", "political_operational_violence_to_dangerous", "Dangerous_Content"
        if args.drop_neutral_political and POLITICAL_NEUTRAL_RE.search(text):
            return "drop", "political_neutral_news_drop", None
        return "keep", "political_keep", None

    return "keep", "label_not_target_keep", None


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    kept = []
    actions = []

    for idx, original in df.iterrows():
        row = original.copy()
        old_label = str(row.get("llm_label", "")).strip()
        action, reason, new_label = decide(row, args)
        if action == "relabel" and new_label:
            row["llm_label"] = new_label

        actions.append(
            {
                "row_index": idx,
                "action": action,
                "reason": reason,
                "old_label": old_label,
                "new_label": row.get("llm_label", "") if action != "drop" else "",
                "text": text_of(row),
            }
        )
        if action != "drop":
            kept.append(row)

    clean_df = pd.DataFrame(kept, columns=df.columns)
    actions_df = pd.DataFrame(actions)
    review_df = actions_df[actions_df["action"].ne("keep")].copy()

    clean_df.to_csv(args.output, index=False, encoding="utf-8")
    actions_path = args.audit_dir / "mx_r2_boundary_clean_actions.csv"
    review_path = args.audit_dir / "mx_r2_boundary_clean_review_queue.csv"
    dist_path = args.audit_dir / "mx_r2_boundary_clean_label_distribution.csv"

    actions_df.to_csv(actions_path, index=False, encoding="utf-8")
    review_df.to_csv(review_path, index=False, encoding="utf-8")
    clean_df["llm_label"].value_counts().rename_axis("llm_label").reset_index(name="count").to_csv(
        dist_path, index=False, encoding="utf-8"
    )

    print(f"input_rows={len(df)}")
    print(f"output_rows={len(clean_df)}")
    print(f"dropped_rows={int((actions_df['action'] == 'drop').sum())}")
    print(f"relabeled_rows={int((actions_df['action'] == 'relabel').sum())}")
    print(f"wrote_clean={args.output}")
    print(f"wrote_actions={actions_path}")
    print(f"wrote_review_queue={review_path}")
    print()
    print(actions_df["action"].value_counts().to_string())
    print()
    print(actions_df[actions_df["action"].ne("keep")]["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
