# Sinter/SymFT S-versus-T MSC simulation

This directory runs injection-and-cultivation experiments through
`sinter.collect` using a custom SymFT sampler. It compares S- and T-state
cultivation, postselects every detector, and measures observable 0 on accepted
shots. The default reference is distance 3, and compatible distance-5
references can be selected explicitly.

The default authoritative **S-state** reference is
`cliffordep/circuits/stim_files/full_circuits/d3a6_inject_cultivate_p1e-3.stim`.
Another compatible S reference can be selected on the command line. Reference
circuits may have any number of qubits, measurements, detectors, and
observables, but must define observable 0. T/T_DAG circuit text is made in
memory by replacing S/S_DAG gates; no non-Stim T circuit is saved. By default,
the physical noise strengths are `0.001`, `0.002`, `0.003`, `0.005`, `0.007`,
and `0.01`.

## Distance-five corrected reference

The distance-5 Bell-growth circuit needs active feedforward before its S/S_DAG
proxy can be converted into a deterministic T/T_DAG circuit. Bell growth leaves
seven first-round check results that control a Pauli frame. An S circuit can
carry that frame without changing its later Clifford evolution, whereas the
corresponding T circuit has branch-dependent evolution unless the frame is
resolved before the next T/T_DAG layer.

`correct_d5_cultivation_circuit` performs this correction while the circuit is
still valid Stim text. It accepts either a noiseless or already-noisy S/S_DAG
proxy and applies the following rules.

### Rule 1: Resolve and validate the circuit by coordinates

All source measurements, correction targets, and structural landmarks are
resolved through `QUBIT_COORDS`, not integer qubit IDs. Reindexing the original
qubits or adding spectator qubits at otherwise unused coordinates therefore
does not change the correction.

Before modifying the circuit, the converter verifies:

- every required coordinate exists exactly once;
- the coordinate-normalized Bell-growth window has the expected structure;
- the first distance-5 Z- and X-syndrome measurements have the expected target
  coordinates and order; and
- detectors referencing the feedforward sources have the expected spatial and
  measurement-dependency profile.

Detector validation uses only the spatial `(x, y)` components of
`DETECTOR(x, y, ...)`. Additional coordinate components such as a detector's
time label are metadata and may change when flag interactions add ticks.
Detectors that do not reference a feedforward source are outside this profile,
so independent flag detectors are preserved without weakening validation of
the affected detectors.

The converter raises an error if these checks fail or if the detector profile
indicates that the correction was already applied. `REPEAT` blocks are not
supported and raise `NotImplementedError`; the committed cultivation references
do not contain them.

### Rule 2: Add Bell-growth feedforward

Immediately after the closing `TICK` of the first distance-5 syndrome round,
the converter emits these 17 classically controlled operations. `M` denotes a
Z-basis measurement and `MX` an X-basis measurement. Coordinates identify the
measured source and quantum target; each classical control is encoded as the
corresponding `rec` target.

| Measurement source | Correction | Target coordinates |
| --- | --- | --- |
| `MX(5, 0)` | `CZ` | `(7, 0)`, `(8, 0)` |
| `MX(3, 3)` | `CZ` | `(3, 4)`, `(4, 6)` |
| `M(6, 0)` | `CX` | `(7, 0)`, `(8, 0)`, `(7, 2)`, `(8, 0)` |
| `M(6, 2)` | `CX` | `(7, 0)`, `(8, 0)` |
| `M(4, 1)` | `CX` | `(3, 4)`, `(4, 6)` |
| `M(4, 3)` | `CX` | `(0, 0)`, `(3, 0)`, `(4, 6)` |
| `M(4, 5)` | `CX` | `(3, 4)`, `(4, 6)` |

The repeated `(8, 0)` entry controlled by `M(6, 0)` is intentional: this table
records the emitted instruction list exactly, including multiplicity.

### Rule 3: Heal detectors crossing the correction boundary

Active feedforward severs the old temporal correlation between a first-round
source result and later syndrome measurements. The converter therefore edits
only detectors that reference one of the seven feedforward sources:

- For local detectors, references to all seven source measurements are removed
  when the detector also contains other measurement records.
- If removing those references would leave a local detector empty, its original
  targets are retained. These source-only detectors remain useful
  initialization checks.
- For the long logical detector, only the Z-basis result `M(4, 3)` is removed.
  Its correction has odd overlap with the logical parity. The other source
  results are retained in that detector.

No detectors are added or deleted; only the affected measurement-record targets
are changed.

### Rule 4: Preserve the S proxy and its noise

The converter preserves all existing noise instructions, measurements,
S/S_DAG layers, observables, and the final logical
`S_DAG`–`MPP X_L`–`S` construction. In particular, it deliberately does **not**
copy these transformations from `clifft-paper`'s `convert_s_to_t.py`:

- S/S_DAG to T/T_DAG substitution;
- distance-5 errata gate flips; or
- final logical-Y measurement wrapping.

T/T_DAG substitution remains the responsibility of `make_variant_text`, after
the requested physical noise level has been substituted into the Stim-compatible
corrected proxy.

The committed corrected references are:

- `d5a19_inject_cultivate_corrected.stim`, generated from
  `d5a19_inject_cultivate.stim`; and
- `d5a19_inject_cultivate_corrected_p1e-3.stim`, generated from
  `d5a19_inject_cultivate_p1e-3.stim`.

Their filename stems are their default `circuit_name` values. The corrected
names and circuit hashes distinguish their schema-3 Sinter task identities from
the uncorrected references; no schema-version change is required.

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

Validate every configured in-memory SymFT circuit and its Stim-compatible S
proxy task without drawing shots. The defaults cover twelve combinations: two
variants at each of six physical noise strengths.

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

Task metadata records `schema_version=3`, `decoder_version`, `noise_level`,
`variant`, `circuit_name`, `circuit_sha256`, and `sampler`; older schemas must
be migrated before collection or plotting.

The production and smoke commands display Sinter's live progress. Detailed
statistics remain available in the production resume CSV.

## Plotting

The shared CSV can be loaded with `sinter.read_stats_from_csv_files` and plotted
with `sinter.plot_error_rate`.

To pool equivalent CPU and CUDA samples for a figure while preserving raw
Sinter resume data, use `read_plot_stats` instead. It creates plotting-only
rows keyed only by the decoder and the `schema_version`, `circuit_name`,
`circuit_sha256`, `noise_level`, and `variant` metadata. The returned rows
retain decoder-version and sampler details in `plot_provenance`, have synthetic
IDs, and must never be written back to the resume CSV or passed to a production
collection:

```python
from pathlib import Path

from cliffordep.symft_simulation.msc_framework import read_plot_stats

plot_stats = read_plot_stats(
    Path("/path/to/cpu-stats.csv"),
    Path("/path/to/cuda-stats.csv"),
)
```

## Tests

Run the focused framework tests with:

```bash
conda run --no-capture-output -n cliffordep \
    python -m pytest -q tests/symft_simulation
```
