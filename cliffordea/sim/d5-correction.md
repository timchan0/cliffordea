# Distance-five cultivation-circuit correction

The implementation is based largely on `clifft-paper`'s
[`convert_s_to_t.py`](https://github.com/unitaryfoundation/clifft-paper/blob/main/magic_state_cultivation/convert_s_to_t.py).
This document specifies the feedforward and detector-healing transformation
implemented by `correct_d5_cultivation_circuit`. For simulation commands and
dataset semantics, see the [Sinter/SymFT simulation guide](README.md).

## Why the correction is required

The distance-five Bell-growth circuit needs active feedforward before its
S/S_DAG proxy can be converted into a deterministic T/T_DAG circuit. Bell
growth leaves seven first-round check results that control a Pauli frame. An S
circuit can carry that frame without changing its later Clifford evolution,
whereas the corresponding T circuit has branch-dependent evolution unless the
frame is resolved before the next T/T_DAG layer.

`correct_d5_cultivation_circuit` performs this correction while the circuit is
still valid Stim text. It accepts either a noiseless or already-noisy S/S_DAG
proxy and applies the following rules.

## Rule 1: Resolve and validate the circuit by coordinates

All source measurements, correction targets, and structural landmarks are
resolved through `QUBIT_COORDS`, not integer qubit IDs. Reindexing the original
qubits or adding spectator qubits at otherwise unused coordinates therefore
does not change the correction.

Before modifying the circuit, the converter verifies:

- every required coordinate exists exactly once;
- the coordinate-normalized Bell-growth window has the expected structure;
- the first distance-five Z- and X-syndrome measurements have the expected
  target coordinates and order; and
- detectors referencing the feedforward sources have the expected spatial and
  measurement-dependency profile.

Detector validation uses only the spatial `(x, y)` components of
`DETECTOR(x, y, ...)`. Additional coordinate components such as a detector's
time label are metadata and may change when flag interactions add ticks.
Detectors that do not reference a feedforward source are outside this profile,
so independent flag detectors are preserved without weakening validation of
the affected detectors.

The converter raises `RuntimeError` if these checks fail or if the detector
profile indicates that the correction was already applied. `REPEAT` blocks
are not supported and raise `NotImplementedError`; the bundled cultivation
references do not contain them.

## Rule 2: Add Bell-growth feedforward

Immediately after the closing `TICK` of the first distance-five syndrome
round, the converter emits these 17 classically controlled operations. `M`
denotes a Z-basis measurement and `MX` an X-basis measurement. Coordinates
identify the measured source and quantum target; each classical control is
encoded as the corresponding `rec` target.

| Measurement source | Correction | Target coordinates |
| --- | --- | --- |
| `MX(5, 0)` | `CZ` | `(7, 0)`, `(8, 0)` |
| `MX(3, 3)` | `CZ` | `(3, 4)`, `(4, 6)` |
| `M(6, 0)` | `CX` | `(7, 0)`, `(8, 0)`, `(7, 2)`, `(8, 0)` |
| `M(6, 2)` | `CX` | `(7, 0)`, `(8, 0)` |
| `M(4, 1)` | `CX` | `(3, 4)`, `(4, 6)` |
| `M(4, 3)` | `CX` | `(0, 0)`, `(3, 0)`, `(4, 6)` |
| `M(4, 5)` | `CX` | `(3, 4)`, `(4, 6)` |

The repeated `(8, 0)` entry controlled by `M(6, 0)` is intentional: this
table records the emitted instruction list exactly, including multiplicity.

## Rule 3: Heal detectors crossing the correction boundary

Active feedforward severs the old temporal correlation between a first-round
source result and later syndrome measurements. The converter therefore edits
only detectors that reference one of the seven feedforward sources:

- For local detectors, references to all seven source measurements are removed
  when the detector also contains other measurement records.
- If removing those references would leave a local detector empty, its
  original targets are retained. These source-only detectors remain useful
  initialization checks.
- For the long logical detector, only the Z-basis result `M(4, 3)` is removed.
  Its correction has odd overlap with the logical parity. The other source
  results are retained in that detector.

No detectors are added or deleted; only the affected measurement-record
targets are changed.

## Rule 4: Preserve the S proxy and its noise

The converter preserves all existing noise instructions, measurements,
S/S_DAG layers, observables, and the final logical
`S_DAG`–`MPP X_L`–`S` construction. Unlike the converter from `clifft-paper`,
its scope is limited to feedforward and detector healing; it does
**not** perform:

- conversion from S- to T-state cultivation, meaning S/S_DAG to T/T_DAG
  substitution;
- distance-five gate flips mentioned as an erratum in the original Gidney et
  al. paper; or
- final logical-Y measurement wrapping.

T/T_DAG substitution remains the responsibility of `make_variant_text`, after
the requested physical noise level has been substituted into the
Stim-compatible corrected proxy. The other transformations, where required,
remain separate preprocessing steps.

## Bundled references

The bundled corrected distance-five references comprise the unflagged
`d5a19` and 13-flag `d5a19f13` families, each in its original three-round and
four-round `r4` forms. The four-round forms already contain the feedforward and
detector healing described above.

For the three-round families, explicit `_uncorrected` source circuits are kept
beside their corrected counterparts in both noiseless and `_p1e-3` forms.
Corrected and explicitly uncorrected filename stems distinguish their Sinter
`circuit_name` values and therefore their task identities. Uncorrected `r4`
sources are not bundled.

The [bundled Stim circuit catalogue](stim_files/README.md) lists every file and
explains the round, flag, correction, and noise suffixes.

## Usage and tests

```python
import stim

from cliffordea.sim import correct_d5_cultivation_circuit

source = stim.Circuit.from_file(
    "cliffordea/sim/stim_files/"
    "d5a19_inject+cultivate_uncorrected.stim"
)
corrected = correct_d5_cultivation_circuit(source)
```

The returned circuit remains an S/S_DAG proxy. The simulation framework later
performs variant substitution in memory.

Run the correction regression tests from the repository root:

```bash
python -m pytest -q tests/sim/test_d5_circuit_correction.py
```
