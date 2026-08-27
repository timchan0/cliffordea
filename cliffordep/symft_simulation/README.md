# Sinter/SymFT S-versus-T MSC simulation

This directory runs the distance-3 injection-and-cultivation experiment through
`sinter.collect` using a custom SymFT sampler. It compares S- and T-state
cultivation, postselects every detector, and measures observable 0 on accepted
shots.

The default authoritative **S-state** reference is
`cliffordep/circuits/stim_files/full_circuits/d3a6_inject_cultivate_p1e-3.stim`.
Another compatible S reference can be selected on the command line. Reference
circuits may have any number of qubits, measurements, detectors, and
observables, but must define observable 0. The T circuit is made in memory by
swapping S and T gates; no derived circuit files are saved. By default, the
physical noise strengths are `0.001`, `0.002`, `0.003`, `0.005`, `0.007`, and
`0.01`.

## Environment

Run every command through the existing `cliffordep` Conda environment:

```bash
cd /Users/timchan0/repositories/cliffordep
conda run --no-capture-output -n cliffordep \
    python -m cliffordep.symft_simulation.run_simulation validate
```

No additional installation is required when `symft`, `stim`, `sinter`,
`cliffordep`, Matplotlib, and pytest are already available. To use ARC H100
jobs, install CUDA-enabled SymFT and pass `--cuda` to `smoke` or `run`.

## Validation and smoke sampling

Validate all twelve in-memory SymFT circuits and their Stim-compatible S proxy
tasks without drawing shots:

```bash
conda run --no-capture-output -n cliffordep \
    python -m cliffordep.symft_simulation.run_simulation validate
```

Exercise the full custom-Sampler and `sinter.collect` path with 100,000
non-persisted attempts for each variant at `p=0.001`:

```bash
conda run --no-capture-output -n cliffordep \
    python -m cliffordep.symft_simulation.run_simulation smoke
```

All three commands accept `--reference PATH`. By default, `circuit_name` is the
reference filename without its `.stim` suffix; use `--circuit-name LABEL` when
a stable or more descriptive dataset label is needed:

```bash
conda run --no-capture-output -n cliffordep \
    python -m cliffordep.symft_simulation.run_simulation validate \
    --reference cliffordep/circuits/stim_files/full_circuits/another.stim \
    --circuit-name another-cultivation-circuit
```

The `validate` command accepts one or more custom physical noise strengths:

```bash
conda run --no-capture-output -n cliffordep \
    python -m cliffordep.symft_simulation.run_simulation validate \
    --noise-levels 0.001 0.004 0.01
```

Smoke sampling remains a fixed quick check of both variants at `p=0.001`.

The SymFT adapter checks that accepted plus discarded equals attempted shots.
It also verifies eight active CPU workers for the CPU backend, or the one host
worker reported by SymFT's CUDA backend.

## Production sampling and resume

The production command uses one Sinter worker because each task owns its
SymFT sampler. It stops each task after 100 logical errors or 1,000,000,000
attempted shots and limits each persisted SymFT call to 10,000,000 shots:

```bash
caffeinate -i conda run --no-capture-output -n cliffordep \
    python -m cliffordep.symft_simulation.run_simulation run \
    --stats /Users/timchan0/Documents/PhD/Code/Python/cliffordep/Chan2026/results/stats.csv
```

## CUDA on ARC

After installing CUDA-enabled SymFT and requesting an NVIDIA GPU in an ARC HTC
job, add `--cuda` to select the GPU backend. First run the GPU smoke check:

```bash
python -m cliffordep.symft_simulation.run_simulation smoke --cuda
```

Then run production sampling with a separate CUDA dataset identity:

```bash
python -m cliffordep.symft_simulation.run_simulation run \
    --cuda \
    --stats /path/to/cuda-stats.csv
```

The limits remain configurable:

```bash
caffeinate -i conda run --no-capture-output -n cliffordep \
    python -m cliffordep.symft_simulation.run_simulation run \
    --reference cliffordep/circuits/stim_files/full_circuits/another.stim \
    --circuit-name another-cultivation-circuit \
    --stats /Users/timchan0/Documents/PhD/Code/Python/cliffordep/Chan2026/results/stats.csv \
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
resume them. Each row's custom counts record its SymFT stream ID and active
thread count. The runner derives the next unused stream from these counts after
a restart. Do not run two writers against the same resume file concurrently.

The production and smoke commands display Sinter's live progress. Detailed
statistics remain available in the production resume CSV.

## Plotting

The shared CSV can be loaded with `sinter.read_stats_from_csv_files` and plotted
with `sinter.plot_error_rate`.

## Tests

Run the focused framework tests with:

```bash
conda run --no-capture-output -n cliffordep \
    python -m pytest -q tests/symft_simulation
```
