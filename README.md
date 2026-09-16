# cliffordea (Clifford Error Analysis)

`cliffordea` is the Python package accompanying
*Diagnosing and Restoring the Degraded Fault Distance of Magic State
Cultivation* by Tim Chan, Armands Strikis, Zhu Sun, and Zhenyu Cai. It
contains the implementation of the paper's trivial-syndrome probability
algorithm, exact low-fault-count enumeration of logical-measurement circuits,
and the Sinter/SymFT Monte Carlo workflow.

## Choose a task

The public API is divided by the kind of analysis being performed:

| Task | Module | Guide |
| --- | --- | --- |
| Calculate acceptance probabilities and logical effects of Clifford errors | `cliffordea.accept` | [`cliffordea/accept/README.md`](cliffordea/accept/README.md) |
| Enumerate faults in logical-measurement circuits | `cliffordea.enum` | [`cliffordea/enum/README.md`](cliffordea/enum/README.md) |
| Run the Sinter/SymFT cultivation simulations | `cliffordea.sim` | [`cliffordea/sim/README.md`](cliffordea/sim/README.md) |

The root package exposes these three topic modules and no individual analysis
classes:

```python
from cliffordea import accept, enum, sim
```

## Installation

Python 3.10 or later is required. The Conda environment is the recommended way
to install the compiled and simulation dependencies:

```bash
conda env create -f environment.yml
conda activate cliffordea
python -m pip install -e . --no-build-isolation --no-deps
```

The final command installs the local checkout in editable mode while using the
dependencies already installed into the environment.

Alternatively, install the package and its declared dependencies with pip:

```bash
python -m pip install -e .
```

This pip-only route is not tested by the project and may require a
platform-specific SymFT installation.

## Quick start

### Acceptance probability

This complete example defines a three-qubit encoding frame with one stabilizer
degree of freedom and two logical degrees of freedom.
Applying a Hadamard to the stabilizer degree of freedom gives a
trivial-syndrome probability of one half:

```python
import stim

from cliffordea.accept import trivial_syndrome_probability

encoder = stim.Tableau(3)
error = stim.Tableau.from_named_gate("H") + stim.Tableau(2)

probability = trivial_syndrome_probability(
    encoder,
    error,
    logical_pauli_coefficients={"II": 1.0},
)
assert probability == 0.5
```

See the [acceptance guide](cliffordea/accept/README.md) for the encoder and
logical-state conventions and for the connection to the paper's algorithm.

### Fault enumeration

Construct a distance-three double check, add the paper's circuit-level noise
model, group elementary error events into canonical faults, and enumerate the
effects kept after circuit detection and final stabilizer postselection:

```python
from cliffordea import accept, enum

cultivated_states = ("S", "T")
circuit = enum.circuits.DoubleCheck(
    distance=3,
    ancilla_count=6,
)
noisy_circuit = enum.noise.uniformly_depolarize(
    circuit.INNER_CIRCUIT,
    noise_level=1e-3,
)
combinator = enum.FaultCombinator(noisy_circuit)
logical_analyzer = accept.CliffordLogicalAnalyzer(
    data_indices=circuit.DATA_INDICES,
    stabilizer_generators=circuit.STABILIZER_GENERATORS,
    logical_s=circuit.LOGICAL_S,
)

kept_effects = combinator.get_kept_effects(
    logical_analyzer=logical_analyzer,
    max_fault_count=3,
    cultivated_states=cultivated_states,
)
```

`kept_effects[state][fault_count]` maps each accepted packed Pauli effect to
its acceptance probability, logical fidelity, and canonical fault
configurations. The [enumeration guide](cliffordea/enum/README.md) explains the
result structure and continues to logical-error rates and configuration
inspection.

### Simulation validation

Generate and validate every default in-memory S- and T-cultivation circuit
without drawing any samples:

```bash
python -m cliffordea.sim.run_simulation validate
```

See the [simulation guide](cliffordea/sim/README.md) before drawing smoke or
production samples.

## Reproducing the paper analyses

| Paper component | Implementation or entry point | Verification |
| --- | --- | --- |
| Trivial-syndrome probability algorithm | `cliffordea.accept.trivial_syndrome_probability` | `python -m pytest -q tests/accept/test_probability.py` |
| Clifford-error acceptance and logical fidelity during enumeration | `cliffordea.accept.CliffordLogicalAnalyzer` | `python -m pytest -q tests/accept/logical_analyzers` |
| Low-fault-count analysis of the final logical-measurement circuit | `cliffordea.enum.FaultCombinator` and [`demo_notebooks/enum.ipynb`](demo_notebooks/enum.ipynb) | `python -m pytest -q tests/enum` |
| S- versus T-state cultivation Monte Carlo workflow | `python -m cliffordea.sim.run_simulation` | `python -m cliffordea.sim.run_simulation validate` |
| Distance-five feedforward and detector healing | `cliffordea.sim.correct_d5_cultivation_circuit` and the [correction specification](cliffordea/sim/d5-correction.md) | `python -m pytest -q tests/sim/test_d5_circuit_correction.py` |

Fault enumeration in this repository concerns the final logical-measurement
subcircuit represented by `DoubleCheck.INNER_CIRCUIT`; it is not an exhaustive
enumeration of every stage of the cultivation protocol. The simulation guide
documents the separate injection-and-cultivation Monte Carlo workflow.

## Tests

Run the full test suite from the repository root:

```bash
python -m pytest -q
```

Each module guide also gives its focused test command.

## Citation

When using this code, cite the accompanying paper and identify the version of
the code used. Until arXiv citation metadata is added, the source repository
can be cited as:

```bibtex
@software{Chan2026cliffordea,
  author = {Tim Chan},
  title = {cliffordea: Clifford Error Analysis},
  year = {2026},
  url = {https://github.com/timchan0/cliffordea}
}
```

## License

This repository does not currently include a license file. Until one is added,
the source remains under the default copyright restrictions; public
availability alone does not grant permission to copy, modify, or redistribute
it.
