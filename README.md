# cliffordea (Clifford Error Analysis)

The accompanying Python package for the Clifford-error-propagation paper. Its
public API is divided by task:

- `cliffordea.accept` computes trivial-syndrome probabilities and analyzes the
  logical effects of Clifford errors.
- `cliffordea.enum` enumerates faults in logical-measurement circuits.
- `cliffordea.sim` runs the Sinter/SymFT Monte Carlo workflow.

## Local Installation Instructions

### Using Conda
Create environment called `cliffordea`:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate cliffordea
```

Install the `cliffordea` package in editable mode (and its dependencies from pip):

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

### Using Pip (Untested)

Install the package in editable mode and its dependencies using pip:

```bash
pip install -e .
```

## Usage

Compute the acceptance probability from the paper's
`alg:probability_trivial_syndrome` directly:

```python
from cliffordea.accept import trivial_syndrome_probability

probability = trivial_syndrome_probability(
    encoder,
    error,
    logical_qubit_count=1,
    logical_coefficients={0b00: 1.0, 0b01: 1.0},
)
```

Construct and enumerate a double-check circuit through `cliffordea.enum`:

```python
from cliffordea import enum

circuit = enum.circuits.DoubleCheck(ancilla_count=6)
noisy_circuit = enum.noise.uniformly_depolarize(
    circuit.INNER_CIRCUIT,
    noise_level=1e-3,
)
faults = enum.FaultCombinator(noisy_circuit)
```

The enumerator groups equivalent error events into faults by detector signature
and resultant effect. Enumeration limits are expressed as a maximum fault count,
matching the terminology used in the paper.

A worked fault-enumeration example is available in
[`demo_notebooks/enum.ipynb`](demo_notebooks/enum.ipynb). The simulation
workflow, including validation, smoke sampling, production runs, and
reference-circuit selection, is documented in
[`cliffordea/sim/README.md`](cliffordea/sim/README.md).
