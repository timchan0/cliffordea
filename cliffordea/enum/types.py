from collections import Counter

import stim

from cliffordea.accept.types import PauliMask


ErrorLocation = tuple[int, str, tuple[stim.GateTarget, ...]]
"""A noisy instruction represented by a tuple containing:
* `timeslice` the timeslice the error location occurs on.
* `name` the name of the Stim gate that produces noise, hence gives rise to faults.
This can be 'DEPOLARIZE1', 'DEPOLARIZE2',
or a member of `ONE_QUBIT_ERROR_EVENTS`.
* `targets` a tuple of `stim.GateTarget`s representing the qubits the error location acts on,
sorted by their qubit values.
It is a pair only for `name` 'DEPOLARIZE2'; else, a 1-tuple.
"""

ErrorEvent = tuple[int, str, tuple[stim.GateTarget, ...]]
"""An unintended event from the noise model
that occurs at a specific location in the circuit
with a probability proportional to
some characteristic noise level of the circuit
e.g. a Pauli X after an H gate.

Represented by a tuple containing:
* `timeslice` the timeslice the event occurs on.
* `name` the name of the event.
This can be 'E' or a member of `ONE_QUBIT_ERROR_EVENTS`.
* `targets` a tuple of `stim.GateTarget`s representing the qubits the event acts on,
sorted by their qubit values.
It is a pair only for `name` 'E'; else, a 1-tuple.
"""

DetectorSignatureMask = int
"""Packed detector signature ordered from least- to most-significant bit."""

MeasurementEventKey = tuple[int, tuple[int, ...]]
"""A measurement result keyed by timeslice and canonical target qubits."""

MeasurementLocation = tuple[int, int]
"""Instruction index and target-group index within a circuit timeslice."""

EffectMap = dict[str, Counter[ErrorLocation]]
"""A map from each effect (the resultant Pauli string of the fault
when propagated to the end of the circuit as an unsigned Pauli string)
to a counter mapping error locations to the number of disjoint error events
at that location that realize the fault.
"""

FaultBag = tuple[int, int, int]
"""The probability of a fault represented as
a multiset of independent events that contribute to the fault.
There are three types of events, corresponding to the three counts in the tuple.
The probability of each event depends on its class and the noise level in a nontrivial way.
The probability of the fault is the probability of an odd number of its events occurring.
"""

LogicalTriple = tuple[float, float, list[tuple[int, ...]]]
"""Acceptance probability, logical fidelity, and canonical fault configurations."""
