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

## Definitions

Notation: a forecast origin `t` produces `H` steps. `y_{t,h,c}` is the true value of channel
`c` at target time `t+h`, `ŷ` the forecast. `k` indexes clips, `n_k` the number of valid
(origin, step) pairs a clip contributes to that channel after the force mask.

### Skill against persistence — TWO estimators, not interchangeable

Both are `1 − MSE_model / MSE_reference` with persistence as the reference. They differ in
what is averaged before the ratio is taken, and on F they disagree in SIGN (see the table
above), so every number has to say which it is.

**Frame-pooled** (`opentouch_cv4*.csv`, written by the driver; `src/opentouch/metrics.py`):

```
                     Σ_k Σ_{t,h} (ŷ − y)²                    MSE_model,c
    MSE_model,c  =  ───────────────────────      skill_c = 1 ────────────
                          Σ_k n_k                             MSE_pers,c
```

Every valid (frame, channel) counts once, so long clips and high-variance clips carry more
weight than short quiet ones.

**Clip-balanced** (`opentouch_report*.csv`; `aggregate.clip_equal_ratio`): each clip's mean
squared error is formed first, and the ratio is of the two clip-averaged means —

```
                   (1/K) Σ_k [ SSE_model,k,c / n_{k,c} ]
    skill_c = 1 − ───────────────────────────────────────
                   (1/K) Σ_k [ SSE_pers,k,c  / n_{k,c} ]
```

Every clip counts once regardless of length. Clips with `n_{k,c} = 0` are dropped from BOTH
numerator and denominator so the two are always over the same clip set; a channel with no
usable clip yields NaN rather than 0 or 1.

`skill = 0` ties persistence, `1` is perfect, `< 0` is worse than assuming nothing changes.

### R² against the class mean (`opentouch_report*.csv`)

Same clip-balanced averaging, but the reference is the mean of the class being scored rather
than a forecaster, so it answers "better than knowing only this class's average level".

### Hausdorff distance between forecast and truth curves

`src/shape_metrics.py`, one implementation shared by both sensors. Each forecast from one
origin, in one channel, is treated as a SET of `H` points in (time, value):

```
    σ  = std_h( y_{t,h} )                     the truth's own spread over this horizon
    p_i = ( i/H , ŷ_{t,i}/σ )                 forecast points,  i = 0..H−1
    q_j = ( j/H ,  y_{t,j}/σ )                truth points,     j = 0..H−1

    d(p_i, q_j) = sqrt( (i−j)²/H²  +  (ŷ_{t,i} − y_{t,j})²/σ² )

    HD_t = max{  max_i min_j d(p_i,q_j) ,  max_j min_i d(p_i,q_j)  }
```

Reported per channel as the mean of `HD_t` over origins within a clip, then over clips, and
as a ratio to the same quantity for persistence. Lower is better; `1.00x` ties persistence.

Three properties matter for how it is read:

- **Time is scaled to [0,1] and value by σ.** Both are conventions. They are identical for
  every model, so ratios between models mean something even though the absolute number does
  not have units.
- **Invariant to shifting both curves together.** Adding the same constant to `ŷ` and `y`
  moves every point of both sets and leaves all pairwise distances unchanged, which is why it
  may be computed on residual-over-persistence targets exactly, not approximately.
- **`σ = 0` yields NaN, not 0.** A truth that is constant over the horizon has no shape to
  compare against; scoring that as a perfect match would flatter every model.

### Why both metrics are reported

MSE is pointwise, so a flat forecast through the middle of an oscillation is charged only its
distance from the centre. Hausdorff asks how far the WORST point of one curve is from the
whole of the other, so the same flat forecast is charged roughly the amplitude. On a unit
sine over one horizon: a perfect forecast scores 0.000, one shifted 0.4 rad scores 0.301, and
the mean scores 0.995.

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
