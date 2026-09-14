# EgoTouch calibration audit — 2026-09-14

There is strong empirical evidence of future-dependent normalization in the released
pressure grids. Our downstream extractor does not apply whole-clip p5 calibration, but
that does not make the upstream data causal. The ActionSense/OpenTouch TRAIN-only fix
does not repair EgoTouch's released grids.

## Downstream code

- `scripts/egotouch/extract_egotouch_states.py:99-114` reads the released left/right
  grids, stacks hands, and replaces invalid values with zero. It does not read or undo
  the normalization metadata.
- At line 165 it calls `clip_states(clip, baseline_pct=None)`: no additional p5 fit.
- `configs/egotouch/tactile_map.yaml` sets `baseline_frames: 0`, which the current
  shared map loader honors. Its old comment claiming otherwise is stale.
- `scripts/egotouch/train_tactile_map.py:139-166` fits target/map normalization on
  TRAIN and performs model selection and sigma calibration on VAL.

Thus the specific downstream whole-recording p5 issue is absent here. The unresolved
boundary is how the released `pressure_grids.npz` was generated from sensor readings.

## Upstream evidence

The [authors' paper, Appendix 8.2](https://arxiv.org/html/2605.13083v1)
describes optional first-frame baseline subtraction, spatial repair of broken columns,
and separate normalization of tactile and bending sensors. It does not specify the
time/split scope of the normalization denominators. First-frame subtraction alone is
not evidence of future leakage. Spatial interpolation is also different from temporal
interpolation.

Two public episodes were downloaded to `/private/tmp` for inspection, with their raw
`jq_pressure.json` and processed `pressure_grids.npz`:

| Episode | Frames | `tactile_max` | `bend_max` |
| --- | ---: | ---: | ---: |
| [Home/arrange_pillow/20260412_101136_379](https://huggingface.co/datasets/zhouzhoujy/EgoTouch/tree/main/Home/arrange_pillow/20260412_101136_379) | 1652 | 51 | 90 |
| [Workbench/change_screwdriver_bit/20260319_113351_072](https://huggingface.co/datasets/zhouzhoujy/EgoTouch/tree/main/Workbench/change_screwdriver_bit/20260319_113351_072) | 267 | 53 | 59 |

Both declare `separate_normalization=True` and baseline correction for both hands.
Using `configs/touchanything/pressure_position_mapping_left.json`, every one of the
217 mapped left-hand cells matches one of these two formulas over the complete episode,
with maximum absolute error at most `1e-6`:

```text
corrected[t, sensor] = max(raw[t, sensor] - raw[0, sensor], 0)
grid[t, row, col] = corrected[t, mapped_sensor] / tactile_max
                or corrected[t, mapped_sensor] / bend_max
```

For the pillow episode, 216 mapped cells are nonzero at some time. Among cells matching
the tactile divisor, the full-episode corrected maximum is exactly 51, first reached
at frame 355; the maximum through frame 100 is only 49. Among those matching the bending
divisor, the full maximum is exactly 90, first reached at frame 289; through frame 100
it is only 26. Frame indices here are zero-based.

The screwdriver episode provides another check: matched right-hand tactile cells reach
53 at frame 146, while their maximum through frame 100 is 37. Matched right-hand bending
cells reach 59 at frame 55. Right-hand reconstruction is partial: 26 cells in the first
episode and 23 in the second do not match simple mirrored mapping plus division. The
paper's spatial repair is a possible explanation; that repair was not reconstructed in
this audit. Left-hand reconstruction has no mapped-cell mismatches in either episode.

**Inference:** the changing per-episode divisors, exact grid reconstruction, and agreement
with later within-episode maxima strongly support whole-episode normalization upstream.
The generator source was not found in the public repository tree at commit
`d74f9ef5c189a957b7ff72781a0c998e41b45a56`. These checks do not establish the preprocessing
rule for every released episode or quantify its effect on our reported model scores.

Under that inferred rule, changing a future maximum changes earlier normalized pressure
despite unchanged past raw readings. Those earlier grids feed both map inputs and
physical-state extraction. A common scalar cancels in CoP, but separate tactile/bending
scales can alter their relative spatial weights; force/pressure mass is directly scaled.
Scale-free metrics do not by themselves establish causal availability of the input.

## Required next step

Treat EgoTouch's current results as using released offline preprocessing, not as verified
online causal forecasting. Before changing its protocol, obtain/verify the grid generator
or reconstruct it from raw sensor JSON, mapping and modality masks, including spatial
repairs and the first-frame decision. Then use scales fitted on TRAIN or fixed sensor
constants, freeze them for held-out sequences, rebuild maps/states and retrain. Simply
adding another TRAIN-only baseline to the normalized NPZ does not remove its upstream
future dependence.

## Can we avoid the released normalization?

Yes. Raw `jq_pressure.json` is available for the two inspected episodes and contains
`sensor_left` / `sensor_right`, 256 readings per hand, plus timestamps/frame indices.
The preferred replacement is to build grids from these readings using the fixed mapping,
verify any spatial repair and baseline policy, and fit any amplitude scale on TRAIN only.
The official trajectory split can remain unchanged if the same source episodes are kept.
Whole-corpus raw-JSON availability still needs checking.

There is also a narrower inverse when the correct modality mask and stored scale are
known: multiply tactile cells by `tactile_max` and bending cells by `bend_max`. This
undoes that division; a blanket multiplication by `tactile_max` is incorrect for bending
cells. The two inspected left-hand grids recover `max(raw - raw_first, 0)` with maximum
absolute errors `3.814697265625e-06` and `1.9073486328125e-06`. Multiplying every cell by
the tactile scale instead gives maximum errors of 39 and about 5.49 sensor counts.
These checks used the cell/divisor correspondence inferred in the preceding audit;
a production inverse must verify a fixed modality mask rather than infer it from a
held-out episode's complete signals. Repaired right-hand cells remain unverified.

Multiplication restores the pre-division, corrected readings, not all original raw data.
First-frame subtraction followed by clipping at zero loses negative differences; spatial
repair can overwrite measured cells. A stored baseline flag is not the original baseline
vector. The two samples contain 21,570 and 8,897 mapped left-hand frame/cell entries below
their first-frame values, demonstrating that this clipping is not merely hypothetical.

In exact arithmetic, undoing `y = x / s_episode` with `x = y * s_episode` cancels that
scale dependence. Merely having a future-derived scale in restoration metadata is not
itself proof that the restored input leaks; the important issue is whether cancellation
is complete and the remaining processing is causal. Float storage, mixed modalities,
clipping and unknown spatial repair prevent treating this as a verified full inverse
without the raw comparisons. Rebuilding from raw avoids those ambiguities. Maps, physical
states, normalization and models must then be regenerated together.

## Download integrity

SHA256, in episode order:

```text
pillow pressure_grids.npz d7bcacd5b80677473f50b988024ad0aad288ba9850fa82783bd1f95a2e2d4271
pillow jq_pressure.json  03142e979adc50119c21cf8e42f16106064538249f6fb28f612303728db5ba40
screw  pressure_grids.npz 7e13941060c7018cd715b2774c163aa2ca861cc6cfd596291977489f2ba03742
screw  jq_pressure.json  36f6740a39ed0d094f18a9aeaf5d19c846a41f61a7e418db8c485ed609a66927
```
