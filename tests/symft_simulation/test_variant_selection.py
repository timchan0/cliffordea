"""Tests for selecting SymFT cultivation state variants."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import sinter

from cliffordep.symft_simulation import msc_framework, run_simulation
from cliffordep.symft_simulation.msc_framework import (
    REFERENCE_PATH,
    VARIANTS,
    build_tasks,
    collect_stats,
    smoke_sample,
    validate_all_variants,
)
from cliffordep.symft_simulation.run_simulation import build_parser


def test_default_selection_preserves_t_then_s_tasks() -> None:
    """Omitting a selection continues to validate and build both variants.

    :return: None.
    """
    reference_text = REFERENCE_PATH.read_text(encoding="utf-8")

    rows = validate_all_variants(reference_text, (0.001,))
    tasks = build_tasks(reference_text, noise_levels=(0.001,))

    assert [row["variant"] for row in rows] == ["T", "S"]
    assert [task.json_metadata["variant"] for task in tasks] == ["T", "S"]


@pytest.mark.parametrize(
    ("variants", "expected"),
    [
        (("T",), ["T"]),
        (("S",), ["S"]),
        (("S", "T", "S"), ["T", "S"]),
    ],
)
def test_selection_controls_tasks_in_canonical_order(
    variants: tuple[str, ...],
    expected: list[str],
) -> None:
    """Selections build only unique requested variants in T-then-S order.

    :param variants: Requested variants, including reordered duplicates.
    :param expected: Canonical variant order expected from the framework.
    :return: None.
    """
    reference_text = REFERENCE_PATH.read_text(encoding="utf-8")

    rows = validate_all_variants(reference_text, (0.001,), variants)
    tasks = build_tasks(
        reference_text,
        noise_levels=(0.001,),
        variants=variants,
    )

    assert [row["variant"] for row in rows] == expected
    assert [task.json_metadata["variant"] for task in tasks] == expected


@pytest.mark.parametrize("variants", [(), ("T", "unknown")])
def test_programmatic_selection_rejects_empty_or_unknown_variants(
    variants: tuple[str, ...],
) -> None:
    """Invalid selections fail instead of silently sampling another circuit.

    :param variants: Empty or unknown selection that must be rejected.
    :return: None.
    """
    reference_text = REFERENCE_PATH.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        validate_all_variants(reference_text, (0.001,), variants)
    with pytest.raises(ValueError):
        build_tasks(
            reference_text,
            noise_levels=(0.001,),
            variants=variants,
        )


def test_cli_defaults_and_accepts_variant_subsets(tmp_path: Path) -> None:
    """Every operation defaults to both and accepts explicit variant lists.

    :param tmp_path: Temporary path used for the required production CSV.
    :return: None.
    """
    parser = build_parser()

    validate_args = parser.parse_args(["validate"])
    smoke_args = parser.parse_args(
        ["smoke", "--variants", "S", "--seed", "7"]
    )
    run_args = parser.parse_args(
        [
            "run",
            "--stats",
            str(tmp_path / "stats.csv"),
            "--variants",
            "S",
            "T",
            "--seed",
            "8",
        ]
    )

    assert validate_args.variants == VARIANTS
    assert smoke_args.variants == ["S"]
    assert smoke_args.seed == 7
    assert run_args.variants == ["S", "T"]
    assert run_args.seed == 8
    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "--variants", "unknown"])


def test_cli_forwards_variants_to_every_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Parsed selections reach validation, smoke, and production operations.

    :param monkeypatch: Pytest fixture used to replace each operation.
    :param tmp_path: Temporary reference and statistics paths.
    :return: None.
    """
    reference_path = tmp_path / "reference.stim"
    reference_path.write_text("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    validate = Mock(return_value=[])
    smoke = Mock(return_value=None)
    collect = Mock(return_value=[])
    monkeypatch.setattr(run_simulation, "validate_all_variants", validate)
    monkeypatch.setattr(run_simulation, "smoke_sample", smoke)
    monkeypatch.setattr(run_simulation, "collect_stats", collect)

    assert run_simulation.main(
        ["validate", "--reference", str(reference_path), "--variants", "S"]
    ) == 0
    assert run_simulation.main(
        ["smoke", "--variants", "T", "--seed", "7"]
    ) == 0
    assert run_simulation.main(
        [
            "run",
            "--stats",
            str(tmp_path / "stats.csv"),
            "--variants",
            "T",
            "S",
            "--seed",
            "8",
        ]
    ) == 0

    assert validate.call_args.args[2] == ["S"]
    assert smoke.call_args.kwargs["variants"] == ["T"]
    assert smoke.call_args.kwargs["seed"] == 7
    assert collect.call_args.kwargs["variants"] == ["T", "S"]
    assert collect.call_args.kwargs["seed"] == 8


def test_smoke_sampling_builds_only_the_selected_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A T-only smoke check gives Sinter one T task and no S task.

    :param monkeypatch: Pytest fixture used to replace Sinter collection.
    :return: None.
    """
    collect = Mock(return_value=[])
    monkeypatch.setattr(sinter, "collect", collect)

    result = smoke_sample(REFERENCE_PATH, 100, variants=("T",))

    assert result is None
    tasks = collect.call_args.kwargs["tasks"]
    assert [task.json_metadata["variant"] for task in tasks] == ["T"]


def test_t_only_collection_accepts_historical_s_without_sampling_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resume validation keeps a compatible S row outside a T-only run.

    :param monkeypatch: Pytest fixture used to supply resume data and collection.
    :param tmp_path: Temporary location for the unused resume CSV path.
    :return: None.
    """
    reference_text = REFERENCE_PATH.read_text(encoding="utf-8")
    historical_task = build_tasks(
        reference_text,
        "resume",
        noise_levels=(0.001,),
        variants=("S",),
    )[0]
    historical_stat = sinter.TaskStats(
        strong_id=historical_task.strong_id(),
        decoder=str(historical_task.decoder),
        json_metadata=historical_task.json_metadata,
        shots=0,
        errors=0,
        discards=0,
        seconds=0,
    )
    read_stats = Mock(return_value=[historical_stat])
    collect = Mock(return_value=[])
    monkeypatch.setattr(msc_framework, "read_sinter_stats", read_stats)
    monkeypatch.setattr(sinter, "collect", collect)

    result = collect_stats(
        reference_path=REFERENCE_PATH,
        stats_path=tmp_path / "stats.csv",
        circuit_name="resume",
        target_errors=1,
        max_shots=1,
        call_shots=1,
        print_progress=False,
        noise_levels=(0.001,),
        variants=("T",),
    )

    assert result == []
    tasks = collect.call_args.kwargs["tasks"]
    assert [task.json_metadata["variant"] for task in tasks] == ["T"]
