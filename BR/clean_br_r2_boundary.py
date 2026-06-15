#!/usr/bin/env python3
"""
Create a conservative BR R2-clean dataset by fixing the most obvious
LLM-generated boundary pollution.

Default policy:
- Keep all real/original rows unchanged.
- Keep historical llm_generated safe rows unless explicitly requested.
- Keep historical llm_generated Harassment rows unless explicitly requested.
- Relabel llm_generated Cybersecurity_Malware rows to Dangerous_Content when
  they clearly involve financial/payment/banking phishing or asset theft.
- Relabel obvious llm_generated safe rows to Dangerous_Content or
  BR_State_Security_Democratic_Order if --safe-action=relabel is used.

This script does not call NIW. It writes a review queue CSV that can be sent to
NIW or checked manually if you want to salvage dropped/flagged rows.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "BR_R2_data" / "br_annotation_R2_full_balanced2000.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "BR_R2_data" / "br_annotation_R2_full_balanced2000_boundary_clean.csv"
DEFAULT_ACTIONS = PROJECT_ROOT / "quality_audit" / "r2_boundary_clean_actions.csv"
DEFAULT_REVIEW = PROJECT_ROOT / "quality_audit" / "r2_boundary_clean_review_queue.csv"
DEFAULT_DIST = PROJECT_ROOT / "quality_audit" / "r2_boundary_clean_label_distribution.csv"


FINANCIAL_TARGET_RE = re.compile(
    r"""
    \b(?:nubank|caixa\s+econ[oô]mica|bradesco|itau|ita[uú]|santander|
       banco\s+do\s+brasil|banco\s+digital|banco\s+online|app\s+de\s+banco|
       site\s+do\s+banco|login\s+do\s+banco|conta\s+(?:de\s+)?banco|
       conta\s+banc[aá]ria|mercado\s*pago|paypal|cart[aã]o)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

FINANCIAL_ABUSE_RE = re.compile(
    r"""
    \b(?:phishing|fake\s+login|login\s+falso|p[aá]gina\s+falsa|clone|clonad[ao]|
       kit|roubar|steal|captur(?:a|ar)|pegar|coletar|credenciais?|senha|
       token|cookie|sess[aã]o|2fa|keylogger|malware|phishing\s+autom[aá]tico)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

PIX_SCAM_RE = re.compile(
    r"""
    \b(?:pix\s+(?:scam|golpe|falso|fake)|golpe\s+(?:do\s+)?pix|urubu\s+do\s+pix|
       pix\s+dobrad[ao]|dobrar\s+pix|fake\s+nubank|nubank\s+fake)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

MERCADO_LIVRE_ASSET_THEFT_RE = re.compile(
    r"""
    \bmercado\s*livre\b.{0,100}
    \b(?:phishing|fake\s+login|login\s+falso|token|cookie|sess[aã]o|2fa|credenciais?|senha)\b
    |
    \b(?:phishing|fake\s+login|login\s+falso|token|cookie|sess[aã]o|2fa|credenciais?|senha)\b
    .{0,100}\bmercado\s*livre\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

SELF_HARM_RE = re.compile(
    r"""
    final\s+post|farewell|adeus|[uú]ltim[ao]\s+(?:post|live|adeus)|
    n[aã]o\s+aguento\s+mais|n[aã]o\s+se\s+force\s+a\s+ficar|
    acabar\s+com\s+tudo|tirar\s+minha\s+vida|quero\s+morrer|
    jeito\s+(?:mais\s+)?(?:f[aá]cil|menos\s+doloroso).{0,80}(?:sumir|morrer|acabar)
    """,
    re.IGNORECASE | re.VERBOSE,
)

SCAM_RE = re.compile(
    r"""
    [uú]ltima\s+chance|desconto\s+de\s+\d+%|clique\s+aqui|link\s+falso|
    resgat(?:e|ar)|promo[cç][aã]o.{0,80}pix|pix.{0,80}dobr|
    ganhe\s+dinheiro|r\$\s?\d+|pr[eê]mio.{0,80}(?:pix|caixa|banco)
    """,
    re.IGNORECASE | re.VERBOSE,
)

BR_STATE_RE = re.compile(
    r"""
    (?:urna|urnas|tse|elei[cç][aã]o|resultado).{0,80}
    (?:fraud|roubad|farsa|manipulad|inv[aá]lid|n[aã]o\s+reconhe[cç]o)
    |
    (?:interven[cç][aã]o\s+militar|golpe\s+militar|ex[eé]rcito\s+.{0,40}(?:assum|tomar|entrar)|
       fechar\s+(?:o\s+)?(?:stf|congresso|tse)|volta\s+ai-?5|
       invadir\s+(?:o\s+)?(?:stf|congresso|bras[ií]lia|tr[eê]s\s+poderes))
    """,
    re.IGNORECASE | re.VERBOSE,
)

MINOR_SEXUAL_RE = re.compile(
    r"""
    \b(?:menor(?:es)?|novinh[ao]s?|garotinh[ao]s?|adolescent(?:e|es)?|
       ensino\s+m[eé]dio|2[ºo]\s*ano|escola)\b
    .{0,100}
    \b(?:pack|fotos?|v[ií]deo|sensual|ousad[ao]|adulto|link|inbox|corpo)\b
    |
    \b(?:pack|fotos?|v[ií]deo|link)\b
    .{0,100}
    \b(?:menor(?:es)?|novinh[ao]s?|garotinh[ao]s?|adolescent(?:e|es)?|
       ensino\s+m[eé]dio|2[ºo]\s*ano|escola)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

SEVERE_VIOLENCE_RE = re.compile(
    r"""
    \b(?:matar|linchar|enforcar|hang|strung\s+up|left\s+to\s+rot|
       fac[aã]o|fogo|queimar|incendiar|dar\s+um\s+recado\s+bem\s+claro|
       mostrar\s+quem\s+manda|bonde|fac[cç][aã]o|rival\s+fans)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BR R2-clean-v1 dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DIST)
    parser.add_argument(
        "--safe-action",
        choices=["keep", "drop_all", "relabel_obvious"],
        default="keep",
        help=(
            "Default keep avoids changing historical/generated safe rows that were "
            "not part of the R2 supplement. Use drop_all or relabel_obvious only "
            "for a broader upstream cleanup."
        ),
    )
    parser.add_argument(
        "--harassment-action",
        choices=["keep", "drop_all", "relabel_obvious"],
        default="keep",
        help=(
            "Default keep avoids changing historical/generated Harassment rows that "
            "were not part of the R2 supplement. Use drop_all or relabel_obvious "
            "only for a broader upstream cleanup."
        ),
    )
    parser.add_argument(
        "--cyber-financial-action",
        choices=["relabel_dangerous", "drop"],
        default="relabel_dangerous",
    )
    return parser.parse_args()


def is_llm_generated(row: pd.Series) -> bool:
    return str(row.get("source", "")).strip() == "llm_generated"


def text_of(row: pd.Series) -> str:
    return str(row.get("clean_text", "") or "")


def set_label(row: pd.Series, label: str) -> None:
    row["final_category"] = label
    row["final_violation"] = label != "safe"


def is_financial_asset_theft(text: str) -> bool:
    """High-precision boundary: financial/payment target + abusive acquisition."""
    return bool(
        (FINANCIAL_TARGET_RE.search(text) and FINANCIAL_ABUSE_RE.search(text))
        or PIX_SCAM_RE.search(text)
        or MERCADO_LIVRE_ASSET_THEFT_RE.search(text)
    )


def decide(row: pd.Series, args: argparse.Namespace) -> tuple[str, str, str | None]:
    """Return (action, reason, new_label)."""
    if not is_llm_generated(row):
        return "keep", "real_or_original_keep", None

    label = str(row.get("final_category", "")).strip()
    text = text_of(row)

    if label == "Cybersecurity_Malware" and is_financial_asset_theft(text):
        if args.cyber_financial_action == "drop":
            return "drop", "cyber_financial_asset_theft_drop", None
        return "relabel", "cyber_financial_asset_theft_to_dangerous", "Dangerous_Content"

    if label == "safe":
        if args.safe_action == "keep":
            return "keep", "generated_safe_keep_not_r2_supplement", None
        if args.safe_action == "drop_all":
            return "drop", "generated_safe_drop_all_for_clean_v1", None
        if SELF_HARM_RE.search(text) or SCAM_RE.search(text):
            return "relabel", "generated_safe_obvious_dangerous", "Dangerous_Content"
        if BR_STATE_RE.search(text):
            return "relabel", "generated_safe_obvious_br_state", "BR_State_Security_Democratic_Order"
        return "drop", "generated_safe_ambiguous_drop", None

    if label == "Harassment":
        if args.harassment_action == "keep":
            return "keep", "generated_harassment_keep_not_r2_supplement", None
        if args.harassment_action == "drop_all":
            return "drop", "generated_harassment_drop_all_for_clean_v1", None
        if MINOR_SEXUAL_RE.search(text) or SEVERE_VIOLENCE_RE.search(text):
            return "relabel", "generated_harassment_obvious_dangerous", "Dangerous_Content"
        return "keep", "generated_harassment_keep", None

    return "keep", "llm_generated_keep", None


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.actions.parent.mkdir(parents=True, exist_ok=True)
    args.review_queue.parent.mkdir(parents=True, exist_ok=True)
    args.distribution.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    kept_rows = []
    action_rows = []
    review_rows = []

    for idx, original in df.iterrows():
        row = original.copy()
        old_label = str(row.get("final_category", "")).strip()
        action, reason, new_label = decide(row, args)

        if action == "relabel" and new_label:
            set_label(row, new_label)

        action_rows.append(
            {
                "row_index": idx,
                "content_id": row.get("content_id", ""),
                "source": row.get("source", ""),
                "action": action,
                "reason": reason,
                "old_category": old_label,
                "new_category": row.get("final_category", old_label) if action != "drop" else "",
                "text": text_of(row),
            }
        )

        if action in {"drop", "relabel"}:
            review_rows.append(action_rows[-1])

        if action != "drop":
            kept_rows.append(row)

    clean_df = pd.DataFrame(kept_rows, columns=df.columns)
    clean_df.to_csv(args.output, index=False, encoding="utf-8")

    actions_df = pd.DataFrame(action_rows)
    actions_df.to_csv(args.actions, index=False, encoding="utf-8")

    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(args.review_queue, index=False, encoding="utf-8")

    dist = pd.crosstab(clean_df["final_category"], clean_df["source"], margins=True, dropna=False)
    dist.to_csv(args.distribution, encoding="utf-8")

    print(f"input_rows={len(df)}")
    print(f"output_rows={len(clean_df)}")
    print(f"dropped_rows={int((actions_df['action'] == 'drop').sum())}")
    print(f"relabeled_rows={int((actions_df['action'] == 'relabel').sum())}")
    print(f"wrote_clean={args.output}")
    print(f"wrote_actions={args.actions}")
    print(f"wrote_review_queue={args.review_queue}")
    print(f"wrote_distribution={args.distribution}")
    print()
    print(actions_df["action"].value_counts().to_string())
    print()
    print(actions_df[actions_df["action"] != "keep"]["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
