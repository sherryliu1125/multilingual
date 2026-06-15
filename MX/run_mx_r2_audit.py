#!/usr/bin/env python3
"""
3-juror audit for MX R2 LLM-generated supplement rows.

This script reuses the same NIM jury models and API rules as run_mx_annotation.py,
but loads MX/mx_audit_prompt.txt and audits current_label quality instead of
creating new training annotations.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from run_mx_annotation import (
    ALTERNATIVE_JURY,
    JURY_MODELS,
    NIM_MAX_RETRIES,
    NIM_RETRY_BACKOFF_SECONDS,
    call_nim_juror,
    majority_vote,
    parquet_engine_available,
    read_results_file,
    write_results_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "synthetic_R2" / "mx_r2_target2000_add.csv"
DEFAULT_AUDIT_PROMPT = PROJECT_ROOT / "mx_audit_prompt.txt"
DEFAULT_POLICY_PROMPT = PROJECT_ROOT / "mx_jury_prompt.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "quality_audit" / "mx_r2_generated_audit.csv"


def load_text(path: Path, name: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_audit_data(input_path: Path, limit: int = 0, labels: list[str] | None = None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False)

    if "clean_text" not in df.columns and "text" in df.columns:
        df["clean_text"] = df["text"]
    if "final_category" not in df.columns and "llm_label" in df.columns:
        df["final_category"] = df["llm_label"]
    if "content_id" not in df.columns:
        df["content_id"] = [f"mx_r2_audit_{i:08d}" for i in range(len(df))]
    else:
        empty_content_id = df["content_id"].astype(str).str.strip().eq("")
        if empty_content_id.any():
            generated_ids = [f"mx_r2_audit_{i:08d}" for i in range(len(df))]
            df.loc[empty_content_id, "content_id"] = [
                generated_ids[i] for i in range(len(df)) if bool(empty_content_id.iloc[i])
            ]
    if "language" not in df.columns:
        df["language"] = "es"
    if "source" not in df.columns:
        df["source"] = "llm_generated"
    if "subreddit" not in df.columns:
        df["subreddit"] = ""
    if "title" not in df.columns:
        df["title"] = ""

    missing = [col for col in ["content_id", "clean_text", "final_category"] if col not in df.columns]
    if missing:
        raise ValueError(f"input missing required columns after normalization: {missing}")

    df = df[df["clean_text"].astype(str).str.strip().ne("")]
    if labels:
        df = df[df["final_category"].isin(labels)]
    if limit and limit > 0:
        df = df.head(limit)
    return df.reset_index(drop=True)


def build_audit_prompts(row: pd.Series, jury_config: dict, audit_policy: str) -> dict:
    current_label = str(row.get("final_category", "")).strip()
    content = str(row.get("clean_text", "") or "")
    row_audit_policy = (
        audit_policy
        .replace("{current_label}", current_label)
        .replace("{text}", content)
    )
    user = f"""Metadata:
country: MX
language: {row.get("language", "es")}
source: {row.get("source", "llm_generated")}
subreddit: {row.get("subreddit", "")}
title: {row.get("title", "")}
current_label: {current_label}

Text:
{content}"""

    prompts = {}
    for key in ["A", "B", "C"]:
        cfg = jury_config[key]
        system = f"""You are Juror {key}: {cfg.get("role", "Mexico generated-data audit juror")}.

{row_audit_policy}
"""
        prompts[key] = {"system": system, "user": user}
    return prompts


def majority_item(values: list[str]) -> str:
    counts = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def majority_bool(values: list[object]) -> bool | None:
    normalized = []
    for value in values:
        if isinstance(value, bool):
            normalized.append(value)
        elif isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                normalized.append(True)
            elif lowered == "false":
                normalized.append(False)
    if not normalized:
        return None
    true_count = sum(1 for value in normalized if value is True)
    false_count = sum(1 for value in normalized if value is False)
    if true_count > false_count:
        return True
    if false_count > true_count:
        return False
    return None


def resolve_audit_action(current_label: str, vote_result: dict, verdicts: list[dict]) -> tuple[bool, bool, str, str, str]:
    final_category = vote_result["final_category"]
    label_match = final_category == current_label
    juror_usable = majority_bool([v.get("usable", None) for v in verdicts])
    usable = True if juror_usable is None else juror_usable

    if vote_result.get("requires_review"):
        return label_match, usable, "medium", "review", "jury_split_or_null"

    if not label_match:
        if usable:
            return False, True, "medium", "review", "category_mismatch_but_usable"
        return False, False, "high", "reject", "category_mismatch_unusable"

    juror_actions = [str(v.get("audit_action", "")).strip() for v in verdicts]
    juror_risks = [str(v.get("risk_level", "")).strip() for v in verdicts]
    action = majority_item(juror_actions)
    risk = majority_item(juror_risks)

    if not usable or action == "reject":
        return True, False, risk or "high", "reject", "juror_reject_despite_label_match"
    if action == "review" or risk in {"medium", "high"}:
        return True, usable, risk or "medium", "review", "juror_boundary_risk"
    return True, True, risk or "low", "accept", "label_match_low_risk"


def process_one_audit_row(
    row: pd.Series,
    jury_config: dict,
    audit_policy: str,
    juror_retries: int,
    retry_backoff: float,
) -> dict:
    prompts = build_audit_prompts(row, jury_config, audit_policy)

    juror_futures = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for key in ["A", "B", "C"]:
            cfg = jury_config[key]
            future = executor.submit(
                call_nim_juror,
                juror_key=key,
                model_name=cfg["name"],
                system_prompt=prompts[key]["system"],
                user_prompt=prompts[key]["user"],
                timeout=cfg.get("timeout", 45.0),
                max_retries=juror_retries,
                retry_backoff=retry_backoff,
            )
            juror_futures[future] = key

        verdicts = [future.result() for future in as_completed(juror_futures)]

    verdicts.sort(key=lambda v: v["juror"])
    vote_result = majority_vote(verdicts)
    current_label = str(row.get("final_category", "")).strip()
    label_match, usable, audit_risk_level, audit_action, audit_reason_code = resolve_audit_action(
        current_label,
        vote_result,
        verdicts,
    )

    record = {
        "content_id": row.get("content_id", ""),
        "source": row.get("source", ""),
        "language": row.get("language", "es"),
        "current_label": current_label,
        "clean_text": str(row.get("clean_text", ""))[:2000],
        "final_violation": vote_result["final_violation"],
        "final_category": vote_result["final_category"],
        "confirmed_label": vote_result["final_category"],
        "label_match": label_match,
        "usable": usable,
        "audit_risk_level": audit_risk_level,
        "audit_action": audit_action,
        "audit_reason_code": audit_reason_code,
        "vote_method": vote_result["method"],
        "requires_review": vote_result["requires_review"],
        "vote_agreement": vote_result["agreement"],
        "violation_count": vote_result["violation_count"],
        "clean_count": vote_result["clean_count"],
        "juror_a_model": verdicts[0]["model_name"],
        "juror_a_category": verdicts[0]["category"],
        "juror_a_confirmed_label": verdicts[0].get("confirmed_label", verdicts[0]["category"]),
        "juror_a_confidence": verdicts[0]["confidence"],
        "juror_a_usable": verdicts[0].get("usable", None),
        "juror_a_risk_level": verdicts[0].get("risk_level", ""),
        "juror_a_audit_action": verdicts[0].get("audit_action", ""),
        "juror_a_reason_code": verdicts[0].get("reason_code", ""),
        "juror_a_reasoning": verdicts[0]["reasoning"],
        "juror_a_error": verdicts[0]["error"],
        "juror_b_model": verdicts[1]["model_name"],
        "juror_b_category": verdicts[1]["category"],
        "juror_b_confirmed_label": verdicts[1].get("confirmed_label", verdicts[1]["category"]),
        "juror_b_confidence": verdicts[1]["confidence"],
        "juror_b_usable": verdicts[1].get("usable", None),
        "juror_b_risk_level": verdicts[1].get("risk_level", ""),
        "juror_b_audit_action": verdicts[1].get("audit_action", ""),
        "juror_b_reason_code": verdicts[1].get("reason_code", ""),
        "juror_b_reasoning": verdicts[1]["reasoning"],
        "juror_b_error": verdicts[1]["error"],
        "juror_c_model": verdicts[2]["model_name"],
        "juror_c_category": verdicts[2]["category"],
        "juror_c_confirmed_label": verdicts[2].get("confirmed_label", verdicts[2]["category"]),
        "juror_c_confidence": verdicts[2]["confidence"],
        "juror_c_usable": verdicts[2].get("usable", None),
        "juror_c_risk_level": verdicts[2].get("risk_level", ""),
        "juror_c_audit_action": verdicts[2].get("audit_action", ""),
        "juror_c_reason_code": verdicts[2].get("reason_code", ""),
        "juror_c_reasoning": verdicts[2]["reasoning"],
        "juror_c_error": verdicts[2]["error"],
        "total_latency_ms": sum(v.get("latency_ms", 0) for v in verdicts),
        "max_latency_ms": max(v.get("latency_ms", 0) for v in verdicts),
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }
    return record


def load_done(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        existing = read_results_file(output_path)
        if "content_id" not in existing.columns:
            return set()
        return set(existing["content_id"].astype(str))
    except Exception:
        return set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MX R2 generated supplement with the 3-juror NIM setup.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-prompt", type=Path, default=DEFAULT_AUDIT_PROMPT)
    parser.add_argument("--policy-prompt", type=Path, default=DEFAULT_POLICY_PROMPT)
    parser.add_argument("--limit", "-n", type=int, default=0)
    parser.add_argument("--labels", type=str, default="", help="Comma-separated labels to audit; empty means all.")
    parser.add_argument(
        "--jury-config",
        choices=["fast", "balanced", "thorough", "multilingual"],
        default="multilingual",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--juror-retries", type=int, default=NIM_MAX_RETRIES)
    parser.add_argument("--retry-backoff", type=float, default=float(NIM_RETRY_BACKOFF_SECONDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    df = load_audit_data(args.input.expanduser(), limit=args.limit, labels=labels)
    policy = load_text(args.policy_prompt.expanduser(), "MX jury policy")
    audit_template = load_text(args.audit_prompt.expanduser(), "MX audit prompt")
    audit_policy = (
        audit_template
        .replace("{{MX_POLICY}}", policy)
        .replace("{MX/mx_jury_prompt.txt}", policy)
    )
    jury_config = ALTERNATIVE_JURY.get(args.jury_config, JURY_MODELS)

    if args.output.suffix.lower() == ".parquet" and not parquet_engine_available():
        args.output = args.output.with_suffix(".csv")

    print(f"Loaded audit rows: {len(df):,} from {args.input}")
    print(f"Audit prompt: {args.audit_prompt}")
    print(f"Policy prompt: {args.policy_prompt}")
    print(f"Output: {args.output}")
    print(f"Jury config: {args.jury_config}")
    for key in ["A", "B", "C"]:
        print(f"  Juror {key}: {jury_config[key]['name']}")

    if args.dry_run:
        print("\nDry run sample:")
        for i, (_, row) in enumerate(df.head(5).iterrows(), start=1):
            print(f"[{i}] {row['content_id']} current_label={row['final_category']} text={row['clean_text'][:120]}...")
        return

    already_done = load_done(args.output) if args.resume else set()
    if already_done:
        print(f"Resume enabled; already audited: {len(already_done):,}")
        df = df[~df["content_id"].astype(str).isin(already_done)].reset_index(drop=True)
        print(f"Remaining rows: {len(df):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    t0 = time.monotonic()

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"[{i}/{len(df)}] {row['content_id']} current_label={row['final_category']} | {row['clean_text'][:80]}...")
        record = process_one_audit_row(
            row=row,
            jury_config=jury_config,
            audit_policy=audit_policy,
            juror_retries=args.juror_retries,
            retry_backoff=args.retry_backoff,
        )
        results.append(record)
        print(
            f"  -> {record['audit_action']} "
            f"final={record['final_category']} match={record['label_match']} "
            f"method={record['vote_method']} risk={record['audit_risk_level']}"
        )

        if i % args.checkpoint_interval == 0:
            new_df = pd.DataFrame(results)
            if args.output.exists():
                existing = read_results_file(args.output)
                out_df = pd.concat([existing, new_df], ignore_index=True)
                out_df = out_df.drop_duplicates(subset=["content_id"], keep="last")
            else:
                out_df = new_df
            write_results_file(out_df, args.output)
            results = []
            print(f"  checkpoint saved: {args.output}")

    if results:
        new_df = pd.DataFrame(results)
        if args.output.exists():
            existing = read_results_file(args.output)
            out_df = pd.concat([existing, new_df], ignore_index=True)
            out_df = out_df.drop_duplicates(subset=["content_id"], keep="last")
        else:
            out_df = new_df
        write_results_file(out_df, args.output)

    elapsed = time.monotonic() - t0
    final_df = read_results_file(args.output)
    print("\nAudit complete")
    print(f"Rows in output: {len(final_df):,}")
    if "audit_action" in final_df.columns:
        print(final_df["audit_action"].value_counts().to_string())
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
