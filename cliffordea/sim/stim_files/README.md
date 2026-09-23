# Bundled injection-and-cultivation circuits

These Stim files describe the injection and cultivation stages of magic-state
cultivation; they stop before escape. They are S-state proxy circuits containing
`S` and `S_DAG` gates. The simulation framework creates the corresponding
T-state circuits in memory by replacing those gates with `T` and `T_DAG`.

For commands that run these circuits, see the
[Sinter/SymFT simulation guide](../README.md). The classical feedforward and
detector changes in the corrected distance-5 circuits are specified in the
[distance-5 correction document](../d5-correction.md).

## Filename notation

The filenames follow
`d<distance>a<ancillas>[r<rounds>][f<flags>]_inject+cultivate[_uncorrected][_p1e-3].stim`:

- `d` gives the colour-code distance.
- `a` gives the number of ancilla qubits in the final double check.
- `r4` identifies distance-5 circuits with four distance-5 colour-code
  stabiliser measurement rounds between the distance-3 and distance-5
  double checks. Distance-five names without `r4` use three rounds.
- `f` gives the number of added Z-flag qubits in the final double check.
- `inject+cultivate` indicates that injection and cultivation are included but
  escape is not.
- `_uncorrected` identifies a distance-5 source circuit before the
  Bell-growth feedforward and detector healing are applied.
- `_p1e-3` identifies a noisy S-state reference at physical noise strength
  $p=10^{-3}$. The simulation framework replaces this value in memory with
  each requested noise strength.

The `_p1e-3` files are the references intended for the `validate`, `smoke`, and
`run` simulation commands. Their noiseless companions support circuit
inspection, deterministic checks, and correction regression tests. Every
bundled circuit defines logical observable 0.

## Circuit catalogue

| File | Circuit family | Z flags | D5 stabiliser rounds | Corrected | Noise | Intended use |
| --- | --- | ---: | ---: | --- | --- | --- |
| `d3a6_inject+cultivate.stim` | Distance 3 | 0 | -- | N/A | None | Noiseless unflagged reference |
| `d3a6_inject+cultivate_p1e-3.stim` | Distance 3 | 0 | -- | N/A | $p=10^{-3}$ | Unflagged simulation template and package default |
| `d3a6f2_inject+cultivate.stim` | Distance 3 | 2 | -- | N/A | None | Noiseless Z-flagged reference |
| `d3a6f2_inject+cultivate_p1e-3.stim` | Distance 3 | 2 | -- | N/A | $p=10^{-3}$ | Z-flagged simulation template |
| `d5a19_inject+cultivate_uncorrected.stim` | Distance 5, original schedule | 0 | 3 | No | None | Noiseless input to the correction procedure |
| `d5a19_inject+cultivate_uncorrected_p1e-3.stim` | Distance 5, original schedule | 0 | 3 | No | $p=10^{-3}$ | Noisy input to the correction procedure |
| `d5a19_inject+cultivate.stim` | Distance 5, original schedule | 0 | 3 | Yes | None | Noiseless corrected reference |
| `d5a19_inject+cultivate_p1e-3.stim` | Distance 5, original schedule | 0 | 3 | Yes | $p=10^{-3}$ | Corrected simulation template |
| `d5a19f13_inject+cultivate_uncorrected.stim` | Distance 5, original schedule | 13 | 3 | No | None | Noiseless Z-flagged input to the correction procedure |
| `d5a19f13_inject+cultivate_uncorrected_p1e-3.stim` | Distance 5, original schedule | 13 | 3 | No | $p=10^{-3}$ | Noisy Z-flagged input to the correction procedure |
| `d5a19f13_inject+cultivate.stim` | Distance 5, original schedule | 13 | 3 | Yes | None | Noiseless corrected Z-flagged reference |
| `d5a19f13_inject+cultivate_p1e-3.stim` | Distance 5, original schedule | 13 | 3 | Yes | $p=10^{-3}$ | Corrected Z-flagged simulation template |
| `d5a19r4_inject+cultivate.stim` | Distance 5, formal four-round schedule | 0 | 4 | Yes | None | Noiseless corrected four-round reference |
| `d5a19r4_inject+cultivate_p1e-3.stim` | Distance 5, formal four-round schedule | 0 | 4 | Yes | $p=10^{-3}$ | Corrected four-round simulation template |
| `d5a19r4f13_inject+cultivate.stim` | Distance 5, formal four-round schedule | 13 | 4 | Yes | None | Noiseless corrected four-round Z-flagged reference |
| `d5a19r4f13_inject+cultivate_p1e-3.stim` | Distance 5, formal four-round schedule | 13 | 4 | Yes | $p=10^{-3}$ | Corrected four-round Z-flagged simulation template |

## Three- and four-round distance-5 circuits

The original distance-5 source circuit uses three distance-5 stabiliser
measurement rounds and was intentionally designed to have fault distance 4.
The `r4` circuits add a fourth round between the distance-3 and
distance-5 double checks, as required by the formal fault-distance-5 S-state
cultivation protocol. Each `r4` circuit otherwise corresponds to the circuit
with the same name after removing `r4`.

This distinction describes the stabilisation schedule of the S-state proxy. It
does not assert that every T-state circuit generated from an `r4` reference has
fault distance 5; the final T-state double check can have the degraded fault
distance analysed in the companion paper.
