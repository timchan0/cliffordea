# Sinter/SymFT S-versus-T MSC simulation

This directory runs injection-and-cultivation experiments through
`sinter.collect` using a custom SymFT sampler. It compares S- and T-state
cultivation, postselects every detector, and measures observable 0 on accepted
shots. The default reference is distance 3, and compatible distance-5
references can be selected explicitly.

The default **S-state** reference is
`cliffordea/sim/stim_files/d3a6_inject+cultivate_p1e-3.stim`.
Another compatible S reference can be selected on the command line. Reference
circuits may have any number of qubits, measurements, detectors, and
observables, but must define observable 0. T/T_DAG circuit text is made in
memory by replacing S/S_DAG gates; no non-Stim T circuit is saved. By default,
the physical noise strengths are `0.001`, `0.002`, `0.003`, `0.005`, `0.007`,
and `0.01`.

## Distance-five corrected references

Distance-five Bell growth leaves a Pauli frame that must be resolved before an
S/S_DAG proxy can be converted into a deterministic T/T_DAG circuit. The
`correct_d5_cultivation_circuit` function adds the required classical
feedforward and heals detectors that cross the correction boundary.

The coordinate-based validation rules, exact feedforward table, detector
healing, transformations outside its scope, and bundled corrected
references are documented in the
[distance-five correction specification](d5-correction.md).

## Environment

Run commands from the repository root after installing `cliffordea` and its
dependencies as described in the top-level README:

```bash
python -m cliffordea.sim.run_simulation validate
```

CPU sampling is the portable default. To sample on an NVIDIA GPU, install a
CUDA-enabled build of SymFT and pass `--cuda` to `smoke` or `run`.

## Validation and smoke sampling

Validate every configured in-memory SymFT circuit and its Stim-compatible S
proxy task without drawing shots. The defaults cover twelve combinations: two
variants at each of six physical noise strengths.

```bash
python -m cliffordea.sim.run_simulation validate
```

Exercise the full custom-Sampler and `sinter.collect` path with 100,000
non-persisted attempts for each variant at `p=0.001`:

```bash
python -m cliffordea.sim.run_simulation smoke
```

The `validate`, `smoke`, and `run` commands accept `--reference PATH`. By
default, `circuit_name` is the reference filename without its `.stim` suffix;
use `--circuit-name LABEL` when a stable or more descriptive dataset label is
needed:

```bash
python -m cliffordea.sim.run_simulation validate \
    --reference cliffordea/sim/stim_files/another.stim \
    --circuit-name another-cultivation-circuit
```

The `validate` and `run` commands accept one or more custom physical noise
strengths:

```bash
python -m cliffordea.sim.run_simulation validate \
    --noise-levels 0.001 0.004 0.01
```

Smoke sampling remains a fixed quick check at `p=0.001`.
All commands accept `--variants T`, `--variants S`, or
`--variants T S`; both variants are selected by default. The smoke shot count
can be changed with `--shots`.

Like `stim.CompiledDetectorSampler`, sampling uses system entropy when `--seed`
is omitted. Pass an unsigned 64-bit seed to make one invocation deterministic:

```bash
python -m cliffordea.sim.run_simulation smoke --seed 1234
```

The seed is a runtime option and is not saved in task metadata or results.
Reusing the same explicit seed with the same call sequence can repeat samples,
so omit it for independent production and top-up runs.

The SymFT adapter checks that accepted plus discarded equals attempted shots.
It also verifies the active CPU-worker count implied by the configured threads
and sample size, or the one host worker reported by SymFT's CUDA backend.

## Production sampling and resume

The production command uses one Sinter worker because each task owns its
SymFT sampler. It stops each task after 100 logical errors or 1,000,000,000
attempted shots and limits each persisted SymFT call to 10,000,000 shots:

```bash
python -m cliffordea.sim.run_simulation run \
    --stats results/stats.csv
```

## CUDA sampling

After installing CUDA-enabled SymFT and making an NVIDIA GPU available, add
`--cuda` to select the GPU backend. First run the GPU smoke check:

```bash
python -m cliffordea.sim.run_simulation smoke --cuda
```

Then run production sampling with a separate CUDA dataset identity:

```bash
python -m cliffordea.sim.run_simulation run \
    --cuda \
    --stats /path/to/cuda-stats.csv
```

The limits remain configurable:

```bash
python -m cliffordea.sim.run_simulation run \
    --reference cliffordea/sim/stim_files/another.stim \
    --circuit-name another-cultivation-circuit \
    --stats results/stats.csv \
    --noise-levels 0.001 0.004 0.01 \
    --target-errors 100 \
    --max-shots 1000000000 \
    --chunk-shots 10000000
```

`--stats` is required for production runs. Multiple references can append to
the same file: metadata records `circuit_name`, and that name participates in
Sinter's strong task ID. Rows belonging to other names are left alone. Reusing
one name for an incompatible circuit or sampler configuration is rejected
before collection, so circuit names must be unique dataset labels. A later run
may select a different subset of noise levels for the same circuit name;
compatible unselected points already present in the shared CSV are retained.

`sinter.collect` appends checkpoints and uses strong task IDs to aggregate and
resume counts. Every new invocation uses fresh entropy-derived SymFT streams by
default, including a later top-up of an existing CSV. The `custom_counts` CSV
column is empty. Do not run two writers against the same resume
file concurrently.

Task metadata records `schema_version=4`, `decoder_version`, `noise_level`,
`variant`, `circuit_name`, and `sampler`. Older schemas are rejected; start a
new CSV instead of reusing an older result file.

The production and smoke commands display Sinter's live progress. Detailed
statistics remain available in the production resume CSV.

## Plotting

The shared CSV can be loaded with `sinter.read_stats_from_csv_files` and plotted
with `sinter.plot_error_rate`.

To pool equivalent CPU and CUDA samples for a figure while preserving raw
Sinter resume data, use `read_plot_stats` instead. It creates plotting-only
rows keyed only by the decoder and the `schema_version`, `circuit_name`,
`noise_level`, and `variant` metadata. The returned rows retain decoder-version
and sampler details in `plot_provenance`, have synthetic IDs, and must never be
written back to the resume CSV or passed to a production collection:

```python
from pathlib import Path

from cliffordea.sim.msc_framework import read_plot_stats

plot_stats = read_plot_stats(
    Path("/path/to/cpu-stats.csv"),
    Path("/path/to/cuda-stats.csv"),
)
```

## Tests

Run the focused framework tests with:

```bash
python -m pytest -q tests/sim
```

Return to the [project README](../../README.md).
