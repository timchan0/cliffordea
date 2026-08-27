#!/usr/bin/env python3
"""Validate, smoke-test, or run the Sinter-backed MSC sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from cliffordep.symft_simulation.msc_framework import (
    DEFAULT_CALL_SHOTS,
    DEFAULT_MAX_SHOTS,
    DEFAULT_SMOKE_SHOTS,
    DEFAULT_TARGET_ERRORS,
    DEFAULT_NOISE_LEVELS,
    REFERENCE_PATH,
    collect_stats,
    smoke_sample,
    validate_all_variants,
)


def add_reference_arguments(parser: argparse.ArgumentParser) -> None:
    """Add reference-circuit selection to one simulation subcommand.

    :param parser: Subcommand parser receiving the shared arguments.
    :return: None.
    """
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--circuit-name")


def add_noise_levels_argument(parser: argparse.ArgumentParser) -> None:
    """Add configurable physical noise strengths to one subcommand.

    :param parser: Subcommand parser receiving the noise-level option.
    :return: None.
    """
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=DEFAULT_NOISE_LEVELS,
        metavar="P",
    )


def resolve_circuit_name(reference: Path, circuit_name: str | None) -> str:
    """Choose an explicit circuit name or derive it from the filename.

    :param reference: Selected reference circuit path.
    :param circuit_name: Optional user-supplied circuit identifier.
    :return: Circuit identifier stored in Sinter task metadata.
    """
    return reference.stem if circuit_name is None else circuit_name


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for simulation operations.

    :return: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="generate and parse all configured in-memory circuit variants",
    )
    add_reference_arguments(validate)
    add_noise_levels_argument(validate)

    smoke = subparsers.add_parser(
        "smoke",
        help="sample a tiny non-persisted T/S check at p=0.001 via Sinter",
    )
    add_reference_arguments(smoke)
    smoke.add_argument("--shots", type=int, default=DEFAULT_SMOKE_SHOTS)

    run = subparsers.add_parser(
        "run",
        help="resume the production sweep through sinter.collect",
    )
    add_reference_arguments(run)
    add_noise_levels_argument(run)
    run.add_argument("--stats", type=Path, required=True)
    run.add_argument("--target-errors", type=int, default=DEFAULT_TARGET_ERRORS)
    run.add_argument("--max-shots", type=int, default=DEFAULT_MAX_SHOTS)
    run.add_argument("--chunk-shots", type=int, default=DEFAULT_CALL_SHOTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected validation or sampling command.

    :param argv: Optional command-line arguments excluding the program name.
    :return: Process exit status.
    """
    args = build_parser().parse_args(argv)
    circuit_name = resolve_circuit_name(args.reference, args.circuit_name)
    if args.command == "validate":
        reference_text = args.reference.read_text(encoding="utf-8")
        for row in validate_all_variants(reference_text, args.noise_levels):
            print(
                json.dumps(
                    {"circuit_name": circuit_name, **row},
                    sort_keys=True,
                )
            )
        return 0

    if args.command == "smoke":
        smoke_sample(
            args.reference,
            args.shots,
            circuit_name,
        )
        return 0

    collect_stats(
        reference_path=args.reference,
        stats_path=args.stats,
        circuit_name=circuit_name,
        target_errors=args.target_errors,
        max_shots=args.max_shots,
        call_shots=args.chunk_shots,
        print_progress=True,
        noise_levels=args.noise_levels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
