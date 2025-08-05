from collections import Counter

import stim

FaultSource = tuple[int, str, tuple[stim.GateTarget, ...]]
"""A noisy instruction represented by a tuple containing:
* `timeslice` the timeslice the fault source occurs on.
* `name` the name of the Stim gate that produces noise, hence gives rise to faults.
This can be 'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.
* `targets` a tuple of `stim.GateTarget`s representing the qubits the fault source acts on.
It is a pair only for `name` 'DEPOLARIZE2'; else, a 1-tuple.
"""

Fault = tuple[int, str, tuple[stim.GateTarget, ...]]
"""A Pauli error represented by a tuple containing:
* `timeslice` the timeslice the fault occurs on.
* `name` the name of the fault.
This can be 'E', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.
* `targets` a tuple of `stim.GateTarget`s representing the qubits the fault acts on.
It is a pair only for `name` 'E'; else, a 1-tuple.
"""

EffectMap = dict[str, Counter[FaultSource]]
"""A map from each effect (the resultant Pauli string of the fault
when propagated to the end of the circuit as an unsigned Pauli string)
to a map from each fault source to the number of its faults that cause that effect.
"""

MechanismBag = tuple[int, int, int]
"""The probability of an error mechanism represented as
a multiset of independent processes that contribute to the error mechanism.
There are three types of processes, corresponding to the three counts in the tuple.
The probability of each process depends on its class and the noise level in a nontrivial way.
The probability of the error mechanism is the probability of an odd number of its processes occurring.
"""