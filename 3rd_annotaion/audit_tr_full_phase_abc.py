#!/usr/bin/env python3
"""Audit tr_annotation_full.csv with Phase A/B/C rules, skipping need-review pool texts."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from tr_phase_a_clean import has_phase_a_artifacts, phase_a_clean
from tr_phase_b_detect import detect, spam_tier
from tr_phase_c_judge import judge_row


def is_hard_reject(had_artifacts: bool, spam_tier_name: str, exclude_reason: str) -> bool:
    if had_artifacts:
        return True
    if spam_tier_name == "spam_likely":
        return True
    primary = (exclude_reason or "").split("|")[0]
    return primary == "seo_template_gibberish"

FULL_CSV = Path("/Users/liushuyu/Desktop/Huawei/multilingual/TR/data/annotations_tr/tr_annotation_full.csv")
SKIP_CSV = SCRIPT_DIR / "TR" / "TR_R4_train_pool_need_review_annotated.csv"
OUT_REJECT = SCRIPT_DIR / "TR" / "tr_full_phase_abc_reject.csv"
OUT_SUMMARY = SCRIPT_DIR / "TR" / "tr_full_phase_abc_summary.txt"


def load_skip_texts(path: Path) -> set[str]:
    texts: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("clean_text") or "").strip()
            if text:
                texts.add(text)
    return texts


def audit() -> dict[str, object]:
    skip_texts = load_skip_texts(SKIP_CSV)

    stats: Counter[str] = Counter()
    reject_rows: list[dict[str, str]] = []

    with FULL_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            stats["full_total"] += 1
            text = (row.get("clean_text") or "").strip()
            if not text:
                stats["empty_text"] += 1
                continue
            if text in skip_texts:
                stats["skipped_need_review_pool"] += 1
                continue

            stats["audited"] += 1
            had_artifacts = has_phase_a_artifacts(text)
            if had_artifacts:
                stats["phaseA_had_artifacts"] += 1
            cleaned = phase_a_clean(text) if had_artifacts else text

            flags, caps_ratio, fake_ratio = detect(cleaned)
            tier = spam_tier(flags)
            stats[f"phaseB:{tier}"] += 1
            for flag in flags:
                stats[f"phaseB_flag:{flag}"] += 1

            work = {
                "clean_text": text,
                "clean_text_phaseA": cleaned,
                "phaseA_had_artifacts": "true" if had_artifacts else "false",
                "phaseB_flags": "|".join(flags),
                "phaseB_caps_ratio": str(caps_ratio),
                "phaseB_fake_ratio": str(fake_ratio),
                "phaseB_spam_tier": tier,
            }
            ok, reason, verdict, gib_tier, policy = judge_row(work)
            work["bert_text_ok"] = "true" if ok else "false"
            work["bert_exclude_reason"] = reason
            work["phaseC_verdict"] = verdict
            work["bert_gibberish_tier"] = gib_tier
            work["bert_reject_policy"] = policy

            if ok:
                stats["phaseC_ok"] += 1
                stats[f"gib_tier:{gib_tier}"] += 1
            else:
                stats["phaseC_reject"] += 1
                stats[f"gib_tier:{gib_tier}"] += 1
                if reason:
                    stats[f"reject:{reason.split('|')[0]}"] += 1
                if policy:
                    for part in policy.split("|"):
                        stats[f"policy:{part}"] += 1
                if is_hard_reject(had_artifacts, tier, reason):
                    stats["hard_filter_reject"] += 1
                elif "fake_agglut_stripped_incomplete" in (policy or ""):
                    stats["fake_agglut_stripped_reject"] += 1

                reject_rows.append(
                    {
                        "content_id": row.get("content_id", ""),
                        "final_category": row.get("final_category", ""),
                        "source": row.get("source", ""),
                        "requires_review": row.get("requires_review", ""),
                        "vote_method": row.get("vote_method", ""),
                        "clean_text": text,
                        "clean_text_phaseA": cleaned,
                        "phaseA_had_artifacts": work["phaseA_had_artifacts"],
                        "phaseB_flags": work["phaseB_flags"],
                        "phaseB_spam_tier": tier,
                        "bert_exclude_reason": reason,
                        "bert_reject_policy": policy,
                    }
                )

    out_cols = [
        "content_id",
        "final_category",
        "source",
        "requires_review",
        "vote_method",
        "clean_text",
        "clean_text_phaseA",
        "phaseA_had_artifacts",
        "phaseB_flags",
        "phaseB_spam_tier",
        "bert_exclude_reason",
        "bert_reject_policy",
    ]
    OUT_REJECT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_REJECT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(reject_rows)

    return {"stats": stats, "reject_rows": len(reject_rows)}


def format_summary(result: dict[str, object]) -> str:
    stats: Counter[str] = result["stats"]  # type: ignore[assignment]
    audited = stats["audited"]
    ok = stats["phaseC_ok"]
    reject = stats["phaseC_reject"]
    lines = [
        "TR tr_annotation_full.csv Phase A/B/C audit",
        f"full_csv: {FULL_CSV}",
        f"skip_pool: {SKIP_CSV}",
        "",
        f"full_total_rows: {stats['full_total']}",
        f"empty_text: {stats['empty_text']}",
        f"skipped_need_review_pool (by clean_text): {stats['skipped_need_review_pool']}",
        f"audited_rows: {audited}",
        "",
        "--- Phase A ---",
        f"phaseA_had_artifacts: {stats['phaseA_had_artifacts']} ({pct(stats['phaseA_had_artifacts'], audited)})",
        "",
        "--- Phase B spam tier ---",
    ]
    for key in ("clean", "review", "spam_possible", "spam_likely"):
        if stats[f"phaseB:{key}"]:
            lines.append(f"{key}: {stats[f'phaseB:{key}']} ({pct(stats[f'phaseB:{key}'], audited)})")
    lines.extend(["", "--- Phase B top flags ---"])
    flag_items = sorted(
        ((k.replace("phaseB_flag:", ""), v) for k, v in stats.items() if k.startswith("phaseB_flag:")),
        key=lambda x: (-x[1], x[0]),
    )
    for name, count in flag_items[:15]:
        lines.append(f"{name}: {count} ({pct(count, audited)})")

    lines.extend(
        [
            "",
            "--- Phase C (BERT text quality) ---",
            f"bert_text_ok=true: {ok} ({pct(ok, audited)})",
            f"bert_text_ok=false: {reject} ({pct(reject, audited)})",
            "",
            "--- Hard filter (PhaseA artifact | spam_likely | seo_template) ---",
            f"hard_filter_reject: {stats['hard_filter_reject']} ({pct(stats['hard_filter_reject'], audited)})",
            f"fake_agglut_stripped_incomplete (extra): {stats['fake_agglut_stripped_reject']} ({pct(stats['fake_agglut_stripped_reject'], audited)})",
            "",
            "reject reasons (primary):",
        ]
    )
    for key, count in sorted(
        ((k.replace("reject:", ""), v) for k, v in stats.items() if k.startswith("reject:")),
        key=lambda x: (-x[1], x[0]),
    ):
        lines.append(f"  {key}: {count} ({pct(count, audited)})")

    lines.extend(["", "top reject policies:"])
    for key, count in sorted(
        ((k.replace("policy:", ""), v) for k, v in stats.items() if k.startswith("policy:")),
        key=lambda x: (-x[1], x[0]),
    )[:20]:
        lines.append(f"  {key}: {count} ({pct(count, audited)})")

    lines.extend(
        [
            "",
            f"reject detail csv: {OUT_REJECT}",
            f"reject rows written: {result['reject_rows']}",
        ]
    )
    return "\n".join(lines)


def pct(n: int, total: int) -> str:
    if not total:
        return "0.0%"
    return f"{100.0 * n / total:.1f}%"


def main() -> int:
    result = audit()
    summary = format_summary(result)
    OUT_SUMMARY.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
