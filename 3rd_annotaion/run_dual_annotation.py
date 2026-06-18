#!/usr/bin/env python3
"""
Dual-model re-annotation for any supported country (BR, MX, SA, TR, UAE).

Thin wrapper around annotaion_prompts/run_dual_arbitration_annotation.py.
Uses the compiled country prompt from annotaion_prompts/compiled/{COUNTRY}_full.txt.

Parallelism (both enabled by default):
  - row_parallel: up to --max-workers rows at once
  - model_parallel: primary + secondary called concurrently per row

Examples:
  # BR train pool review
  python run_dual_annotation.py --country BR \\
    --input BR/BR_train_pool_R5_review_with_id.csv \\
    --output BR/BR_train_pool_R5_review_with_id_annotated.csv

  # BR golden test error analysis
  python run_dual_annotation.py --country BR \\
    --input BR/golden_test_error_analysis_BR.csv \\
    --output BR/golden_test_error_analysis_BR_annotated.csv

  python run_dual_annotation.py --country TR --input TR_R4_train_pool_need_review.csv --dry-run
  python run_dual_annotation.py --country BR --input BR/foo.csv --limit 30 --resume
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

SUPPORTED_COUNTRIES = tuple(COUNTRY_CONFIGS)
DEFAULT_MAX_WORKERS = 5
DEFAULT_BATCH_SIZE = 10


def default_output_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_annotated{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-model re-annotation for BR / MX / SA / TR / UAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--country",
        required=True,
        choices=SUPPORTED_COUNTRIES,
        help=f"Country code; loads {{COUNTRY}}_full.txt prompt ({', '.join(SUPPORTED_COUNTRIES)})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV (must contain clean_text or equivalent text column)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <input_stem>_annotated.csv beside input)",
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
    country = args.country.upper()
    input_path = args.input if args.input.is_absolute() else SCRIPT_DIR / args.input
    output_path = args.output
    if output_path is None:
        output_path = default_output_for(input_path)
    elif not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path

    config = with_model_overrides(COUNTRY_CONFIGS[country], args)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 2
    if not config.prompt_path.exists():
        print(f"ERROR: prompt not found: {config.prompt_path}", file=sys.stderr)
        return 2

    run_country(
        country_config=config,
        input_path=input_path,
        output_path=output_path,
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
