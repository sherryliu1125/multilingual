#!/usr/bin/env python3
"""
Re-annotate TR R4 train-pool need_review rows with the dual-model pipeline.

Parallelism (both enabled by default):
  - row_parallel: up to --max-workers rows at once
  - model_parallel: primary + secondary called concurrently per row

Uses annotaion_prompts/compiled/TR_full.txt via run_dual_arbitration_annotation.py.

Examples:
  python run_tr_r4_need_review_dual_annotation.py --dry-run
  python run_tr_r4_need_review_dual_annotation.py --limit 30
  python run_tr_r4_need_review_dual_annotation.py --resume
  python run_tr_r4_need_review_dual_annotation.py --no-parallel-initial-models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_ROOT / "annotaion_prompts"
if str(PROMPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPTS_DIR))

from run_dual_arbitration_annotation import (  # noqa: E402
    COUNTRY_CONFIGS,
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRIES,
    run_country,
    with_model_overrides,
)

DEFAULT_INPUT = SCRIPT_DIR / "TR_R4_train_pool_need_review.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "TR_R4_train_pool_need_review_annotated.csv"
DEFAULT_MAX_WORKERS = 5
DEFAULT_BATCH_SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-model re-annotation for TR R4 train_pool need_review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only annotate first N rows")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per scheduling batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Concurrent rows in flight (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows already in output CSV")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument(
        "--parallel-initial-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Primary + secondary in parallel per row (default: on)",
    )
    parser.add_argument("--primary-model", help="Override primary model")
    parser.add_argument("--secondary-model", help="Override secondary model")
    parser.add_argument("--arbitrator-model", help="Override arbitrator model (audit only)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = with_model_overrides(COUNTRY_CONFIGS["TR"], args)

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    run_country(
        country_config=config,
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        batch_size=max(1, args.batch_size),
        max_workers=max(1, args.max_workers),
        resume=args.resume,
        dry_run=args.dry_run,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=max(0, args.retries),
        backoff_seconds=max(0.0, args.backoff_seconds),
        parallel_initial_models=args.parallel_initial_models,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
