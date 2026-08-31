## How to read it

**Down a column is safe; across the D1 boundary is not.** `raw` and `df` score an
uncorrected target that was 99.78% constant in F, so persistence coasted and every skill is
depressed. The correction shrank the target roughly 350-fold. Skill is a ratio on one
target, so it survives that; absolute MSE does not, and is not shown.

**`raw` → `df`: the dF/dt input changed nothing** (probGRU F 0.203 → 0.207). The feature
ablation is a null result.

**`d1` → `d1_mse`: the selection criterion changed nothing either.** probGRU moves by at
most 0.004, and in opposite directions between the two skill conventions. Aligning selection
with the reported metric was right in principle and worth nothing in practice.

**`d1_map2` → `d1_map3` is a REPLICATE, and its disagreement is the noise floor.** Same
seed, data and scope; cnn moved 4.9e−3 and flatten 4.2e−3, while map_aggregate held to
1.5e−4 and the classical baselines were bitwise identical. Any ranking whose margin is not
several times that is not established — currently one: cnn over flatten on CoPy, 0.011
against a combined 0.009.

**AR is the strongest arm on OpenTouch, on every channel.** A linear autoregression on F and
CoP beats a CNN reading the full 16×16 map, and beats the probGRU copied from ActionSense
verbatim. The ordering falls monotonically with how much raw spatial detail an arm is
handed: AR, GRU-aggregate, CNN, flatten. The same ordering holds on ActionSense, and it
holds on OpenTouch's VAL split as well as its TEST split.

**The probGRU row cannot be compared across sensors.** ActionSense's predicts the FAST
component against persistence-of-fast; OpenTouch's predicts the RAW target and is scored by
the harness. Its ActionSense column is left empty rather than filled with a number that
would invite the comparison.

**seasonal is at or below zero everywhere**, on both sensors and in every run: it falls back
to persistence when it finds no cycle, which is nearly always.

## probGRU and GRU-aggregate are two backbones, and they fail differently

Both read the same inputs and emit a Gaussian per (step, channel) trained by the same NLL.
Everything below differs, and each difference has a visible consequence.

| | **GRU-aggregate** (`Seq2Seq`) | **probGRU** |
|---|---|---|
| decoding | **one-shot**: all H steps from one linear read of the encoder state | **autoregressive**: H sequential decoder steps, each fed the previous MEAN |
| predicts | **residual** over the last observed value | **absolute** value |
| action label | none | **8-dim embedding**, vocabulary from TRAIN only |
| hyperparameters | d 64 / hidden 64 | hidden 48 |
| origin | ActionSense `tactile_map/` | ActionSense `action_dynamics.py` |

### Why one looks jagged and the other looks flat

**probGRU smooths itself.** Each decoder step is fed the previous step's mean, and a mean is
by construction the noise-free part, so every step removes a little more variation. By step
30 the decoder is running entirely on its own smooth output. The trajectory is a contraction
toward a smooth path — it cannot stay rough even if the data is.

**Seq2Seq has nothing coupling its steps.** The H outputs are H independent linear functions
of one hidden vector. Nothing smooths them, so they can move freely step to step.

**Part of what looks like detail is the anchor, not the model.** Seq2Seq predicts a residual
and adds it to the last observed value, which jumps at every forecast origin the way
persistence does. Some of the visible structure is that staircase, not a predicted
fluctuation.

### The two metrics disagree, and consistently

Frame-pooled MSE skill prefers probGRU; Hausdorff prefers Seq2Seq, by +0.11 to +0.23 on every
one of the three inputs and three channels — a backbone effect at least as large as the
difference between input representations within a backbone (0.08 within Seq2Seq, 0.13 within
probGRU).

Ranked by shape, **every Seq2Seq arm beats every probGRU arm**, and `seasonal` — the only
model that loses to persistence on MSE — beats them all. The common cause is the same one the
forecast figures have shown since 2026-08-20: MSE is pointwise and rewards a flat line
through the middle of an oscillation, while Hausdorff charges it roughly the amplitude. A
model that emits a wrongly-phased waveform is penalised heavily by the first and lightly by
the second.

So "which backbone is better" has no answer without saying better at what. **Point accuracy:
probGRU, frame-pooled. Curve shape: Seq2Seq, decisively. Point accuracy per clip on F:
Seq2Seq** — see the convention table above.

### It does not transfer across sensors

On ActionSense the same backbone swap costs 0.063 to 0.072 of mean skill, the opposite sign
from OpenTouch's frame-pooled result. Two candidate causes are open and one control is
pending:

- the absolute head has no persistence prior, so it starts further back and may simply need
  more epochs — `probgru_agg_e150` tests this;
- the ActionSense Seq2Seq column it is compared against was written before `_predict` was
  changed to return the persistence reference rather than assume zeros, so part of the gap is
  unattributed — `seq2seq_agg_recheck` closes that.

**Until both land, the cross-sensor reversal is an observation and not a finding.**

## d256 is ActionSense reprocessed, not a fourth collection

The ICLR page presents `d256.zip` as its own dataset. The evidence says it is ActionSense,
decimated 5x in time and cut into 16-frame sliding windows with paired video:

* All 20 class strings match ActionSense's verbatim ("Slice a cucumber", "Get/replace items
  from refrigerator/cabinets/drawers", ...).
* The sensor suite is identical -- two tactile gloves, Myo EMG and accelerometer,
  joint-position, both hand poses -- and the subject codes S01-S05 are the same.
* Pairing recordings by label and taking the three classes where the counts match 1:1, the
  length ratio over **15 independent recordings** is **4.948 +- 0.085**. Independently
  collected recordings of the same activity do not come out to a constant 4.95x.
* 6 Hz is 30 Hz / 5, and ActionSense is natively 30 Hz.

This is inference from length ratios, not frame-level alignment. The check that would settle
it: cross-correlate a d256 F(t) against the matching ActionSense F(t) decimated by 5 --
same recording means correlation near 1 once aligned, since the rescaling to [0,1] changes
amplitude but not shape. Not yet run.

It matters for reading the tables. d256 and ActionSense are not independent evidence about
tactile forecasting; they are the same recordings at two rates, under two protocols
(leave-one-subject-out against a stratified split). Their agreement is not corroboration.
