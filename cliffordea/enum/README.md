# Exact low-fault-count enumeration

`cliffordea.enum` enumerates faults in the logical-measurement circuits used by
the companion paper. It groups physically equivalent error events, finds
low-fault-count configurations with zero detector signature, and combines them
with `cliffordea.accept` to calculate state-dependent acceptance and logical
fidelity.

## Terminology

- An **error event** is one elementary Pauli branch of a noisy Stim
  instruction at a particular circuit location.
- A **fault** is a canonical class of error events with the same detector
  signature and resultant Pauli effect. The combinator records how many
  elementary branches from each noise-process class belong to each fault.
- A **fault configuration** is a combination of canonical faults. Its detector
  signature and Pauli effect are the XORs of its members' signatures and
  effects.
- An **undetected configuration** has zero circuit-detector signature.
- A **kept effect** is circuit-undetected and also has nonzero acceptance under
  the requested logical analyzer and cultivated state.

These distinctions matter for Clifford errors: a configuration can pass the
circuit detectors yet survive final stabilizer postselection only
probabilistically.

## Complete enumeration workflow

The following example analyzes the original distance-three double check
through two faults for both S- and T-state cultivation:

```python
from cliffordea import accept, enum

noise_level = 1e-3
circuit = enum.circuits.DoubleCheck(
    distance=3,
    ancilla_count=6,
)
noisy_circuit = enum.noise.uniformly_depolarize(
    circuit.INNER_CIRCUIT,
    noise_level=noise_level,
)

analyzer = accept.CliffordLogicalAnalyzer(
    data_indices=circuit.DATA_INDICES,
    stabilizer_generators=circuit.STABILIZER_GENERATORS,
    logical_s=circuit.LOGICAL_S,
)
combinator = enum.FaultCombinator(noisy_circuit)

kept_effects = combinator.get_kept_effects(
    logical_analyzer=analyzer,
    max_fault_count=2,
    cultivated_states=("S", "T"),
)

summary = combinator.summarize_contributions(kept_effects)
t_error_rate = combinator.error_rates_per_kept_shot(
    kept_effects["T"],
    noise_levels=[noise_level],
)[0]
```

`max_fault_count` is inclusive: the result contains separate entries for zero
through `max_fault_count` faults. Increasing it can sharply increase runtime
and memory use, so begin with a small bound.

## Reading `get_kept_effects`

The return value has the shape

```text
state -> fault count -> packed Pauli effect
      -> (acceptance probability, logical fidelity, configurations)
```

For example, `kept_effects["T"][2]` contains accepted T-state effects caused by
two-fault configurations. Each `configurations` entry is a list of tuples of
canonical fault indices. Use `combinator.index_to_events` to inspect the
physical error events represented by an index.

`summarize_contributions` converts the nested result into a Pandas table of
benign and malignant configuration counts and asymptotic weights.
`error_rates_per_kept_shot` evaluates the enumerated terms at one or more
uniform physical noise levels and returns the conditional logical-error rate
among kept shots.

Because enumeration stops at `max_fault_count`, a rate calculated this way is
the rate of the retained low-fault-count expansion, not an all-orders Monte
Carlo estimate. The first nonzero malignant contribution determines the fault
distance only within the enumerated circuit and checked range.

## Circuits and noise

The `enum.circuits` module loads committed Stim circuits and derives their data
qubits, stabilizer generators, logical operators, and transversal logical-S
implementation. It provides:

- `DoubleCheck` for the two logical-H_XY measurement circuits analyzed in the
  paper;
- `ShortSingleCheck` for a transversally measured GHZ-state check; and
- `LongSingleCheck` for a check with GHZ-state encoding and unencoding.

Each constructor documents the available `(distance, ancilla_count,
flag_count)` combinations. `INNER_CIRCUIT` is the logical-measurement
subcircuit between the two T layers. `FULL_CIRCUIT` adds the
measurements used to derive code properties.

`enum.noise.uniformly_depolarize` applies the circuit-level model used by the
paper. By default it adds noise to all timeslices except the first and last;
its optional arguments can restrict noisy timeslices or mark which qubits
are noisy/noiseless.

## Choosing a combinator

- `FaultCombinator` is the main paper-analysis path. It groups equivalent
  elementary events into canonical independent faults and integrates logical
  acceptance and fidelity.
- `ErrorEventCombinator` works directly with elementary error events.
- `DisjointFaultCombinator` treats error events as mutually exclusive events.

The latter two expose lower-level undetected-configuration analysis and were
used for correctness checks. They do not replace
`FaultCombinator.get_kept_effects` in the state-dependent workflow above.

## Inspecting configurations

`FaultCombinator.visualize_fault_configurations` inserts selected fault events
into a copy of the circuit and produces Stim diagrams. The worked notebook
uses it to display the four malignant two-fault configurations of the original
distance-three double check:

[`demo_notebooks/enum.ipynb`](../../demo_notebooks/enum.ipynb)

The notebook also demonstrates contribution summaries, logical-error-rate
curves, and configuration inspection for the distance-three and distance-five
circuits.

## Scope

The exact enumeration covers the selected logical-measurement circuit, such as
the final double check represented by `DoubleCheck.INNER_CIRCUIT`. It does not
by itself enumerate injection, growth, or escape stages of the full
cultivation protocol. Use the separate [simulation workflow](../sim/README.md)
for its injection-and-cultivation Monte Carlo experiments.

## Tests

Run the enumeration tests from the repository root:

```bash
python -m pytest -q tests/enum
```

Return to the [project README](../../README.md).
