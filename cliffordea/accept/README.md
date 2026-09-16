# Clifford-error acceptance and logical analysis

`cliffordea.accept` implements the paper's algorithm for calculating the
probability that every stabilizer measurement returns a trivial syndrome after
an encoded state suffers a physical Clifford error. It also provides the
logical analyzers used by `cliffordea.enum` to classify accepted fault effects.

## Trivial-syndrome probability

The main standalone entry point is:

```python
from cliffordea.accept import trivial_syndrome_probability
```

It evaluates

```text
P(trivial syndrome | encoder, Clifford error, logical state).
```

The following runnable example has three physical qubits, one stabilizer
generator, and two logical qubits. The identity encoder makes the first input
axis the stabilizer axis and the final two axes the logical frame. A Hadamard on
the stabilizer axis produces a one-half overlap with the original codespace:

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

### Encoder convention

For an `n`-qubit `stim.Tableau` encoding `k` logical qubits:

- the first `n - k` input axes specify independent stabilizer generators;
- the final `k` input axes specify the chosen logical Pauli frame; and
- `error` is the physical Clifford error in encoded coordinates.

The logical-qubit count is inferred from the length of each key in
`logical_pauli_coefficients`. All keys therefore have the same length.

### Logical-state convention

The coefficient map gives the nonzero real coefficients `alpha_L` in the
logical Pauli expansion

```text
rho = 2**(-k) sum_L alpha_L L.
```

Keys are unsigned strings over `I`, `X`, `Y`, and `Z`, ordered by logical-qubit
index. Signs belong in the coefficient rather than the key. Common
one-logical-qubit states are:

| State | `logical_pauli_coefficients` |
| --- | --- |
| Maximally mixed | `{"I": 1.0}` |
| $\|0\rangle\langle0\|$ | `{"I": 1.0, "Z": 1.0}` |
| $\|+\rangle\langle+\|$ | `{"I": 1.0, "X": 1.0}` |
| S-state | `{"I": 1.0, "Y": 1.0}` |
| T-state | `{"I": 1.0, "X": 2**-0.5, "Y": 2**-0.5}` |

For example, `{"II": 1.0, "XI": 1.0}` describes $|+\rangle\langle+|$ on logical qubit
zero tensored with the maximally mixed state on logical qubit one.

## Connection to the paper algorithm

`trivial_syndrome_probability` is the public implementation of
`alg:probability_trivial_syndrome` in the companion paper. The implementation:

1. conjugates the physical error into the unencoded frame as `C^dag E C`;
2. transforms the unencoded stabilizer generators;
3. row-reduces their binary symplectic representation;
4. checks zero-row and kernel sign constraints; and
5. adds the logical Pauli contributions before applying the rank-dependent
   power-of-two prefactor.

The paper gives the mathematical derivation. The code
path begins in
[`probability.py`](probability.py), while
[`tests/accept/test_probability.py`](../../tests/accept/test_probability.py)
contains identity, multiple-logical-qubit, and distance-three regression cases.

## Logical analyzers used during enumeration

The analyzer classes answer a related, more specialized question: for a Pauli
effect immediately before the second layer of T gates in the double check, what is the
subsequent stabilizer-postselection probability and conditional logical
fidelity?

- `CliffordLogicalAnalyzer` uses the stabilizer-overlap algorithm and is the
  analyzer used for the paper's enumerations.
- `SuperpositionLogicalAnalyzer` expands the propagated Clifford error as a
  Pauli superposition. It provides a slower independent formulation used for
  cross-checking.
- `LogicalAnalyzer` is their abstract interface.

A typical analyzer for a committed double-check circuit is constructed as:

```python
from cliffordea import accept, enum

circuit = enum.circuits.DoubleCheck(
    distance=3,
    ancilla_count=6,
)
analyzer = accept.CliffordLogicalAnalyzer(
    data_indices=circuit.DATA_INDICES,
    stabilizer_generators=circuit.STABILIZER_GENERATORS,
    logical_s=circuit.LOGICAL_S,
)
```

Most users should pass this analyzer to
`cliffordea.enum.FaultCombinator.get_kept_effects` instead of calling
`analyze` directly. See the [enumeration guide](../enum/README.md) for the
complete workflow.

`PauliSum`, `LogicalVector`, `PauliMask`, and `SyndromeMask` are lower-level
building blocks exposed for analyzer development and result inspection. They
are not needed for the standalone probability function.

## Scope

- The standalone algorithm accepts a Clifford error represented by a
  `stim.Tableau`; it does not model a probabilistic mixture of errors in one
  call.
- The coefficient map represents a valid logical density operator supplied by
  the caller.
- Acceptance means that all stabilizer generators included by the algorithm
  return the trivial outcome. Logical fidelity is a separate quantity supplied
  by the analyzer workflow.

## Tests

Run the acceptance and logical-analysis tests from the repository root:

```bash
python -m pytest -q tests/accept
```

Return to the [project README](../../README.md).
