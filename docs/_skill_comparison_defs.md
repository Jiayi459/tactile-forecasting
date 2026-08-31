## Definitions

A forecast origin $t$ produces $H$ steps. $y_{t,h,c}$ is the true value of channel $c$ at
target time $t+h$ and $\hat y_{t,h,c}$ the forecast; persistence predicts $y_{t,c}$, the value
at the origin, for every $h$. $k$ indexes clips and $n_{k,c}$ is the number of valid
(origin, step) pairs clip $k$ contributes to channel $c$ after the force mask.

### Skill against persistence — two estimators, not interchangeable

Both are $1-\mathrm{MSE}_{\text{model}}/\mathrm{MSE}_{\text{persistence}}$. They differ in
what is averaged before the ratio is taken, and on F they disagree in **sign**, so every
number has to say which one it is.

**Frame-pooled** — written by the driver into `opentouch_cv4*.csv`:

$$
\mathrm{skill}_c \;=\; 1-\frac{\sum_k\sum_{t,h}\bigl(\hat y_{t,h,c}-y_{t,h,c}\bigr)^2}
{\sum_k\sum_{t,h}\bigl(y_{t,c}-y_{t,h,c}\bigr)^2}
$$

Every valid (frame, channel) counts once, so long and high-variance clips carry more weight.

**Clip-balanced** — written by the report into `opentouch_report*.csv`:

$$
\mathrm{skill}_c \;=\; 1-
\frac{\dfrac{1}{K}\sum_k \dfrac{1}{n_{k,c}}\sum_{(t,h)\in k}\bigl(\hat y_{t,h,c}-y_{t,h,c}\bigr)^2}
{\dfrac{1}{K}\sum_k \dfrac{1}{n_{k,c}}\sum_{(t,h)\in k}\bigl(y_{t,c}-y_{t,h,c}\bigr)^2}
$$

Each clip's mean error is formed first, so every clip counts once regardless of length. Clips
with $n_{k,c}=0$ are dropped from numerator and denominator together, keeping both over the
same clip set; a channel with no usable clip yields NaN rather than 0 or 1.

$\mathrm{skill}=0$ ties persistence, $1$ is perfect, $<0$ is worse than assuming nothing
changes.

### The same name across the three datasets

`skill` always means "against persistence" and never "against the mean" -- the mean is the
denominator of $R^2$, which the report prints in a separate table and which is a different
quantity. What does differ between the datasets is how the errors are averaged and which
frames are counted:

| | reference | averaging | force mask |
|---|---|---|---|
| OpenTouch, `opentouch_cv4*.csv` | persistence | frame-pooled | yes |
| OpenTouch, `opentouch_report*.csv` | persistence | clip-balanced | yes |
| OpenTouch $R^2$ | the scored subset's own mean | clip-balanced | yes |
| ActionSense, `tactile_map` | persistence | frame-pooled per fold, then averaged over folds | **no** |
| d256 | — | — | — |

Two consequences worth carrying:

- **OpenTouch and ActionSense skill are not computed over the same frames.** OpenTouch drops
  frames below a force threshold (`masking.valid_mask`); ActionSense counts all of them. A
  cross-sensor skill difference therefore includes a difference in which frames were scored.
- **ActionSense computes in z-normalised residual space and OpenTouch in raw absolute units,
  and that one does NOT matter.** Skill is a ratio, so a per-channel constant scale cancels
  from numerator and denominator alike.

**d256 has no skill number**, and the reason is worth stating precisely because the loose
version of it is wrong. A skill number is COMPUTABLE there -- pick a horizon in frames, hold
the last observation, take the ratio. What is not computable is a skill number belonging in
this table, and there are three separate obstacles of quite different weight:

1. **No forecaster exists.** `src/d256.py` reads the archive and `scripts/d256/` probes it;
   nothing trains. This is work not yet done, not an impossibility.
2. **The sampling rate is unknown and not recoverable from the archive.** Clips carry no
   timestamps; only the relative 1:2:3 stride between the three signal groups is known. Every
   number in this document is at a ONE-SECOND horizon, and one second cannot be located in a
   stream whose rate nobody knows. A d256 column would be "skill at H frames" against
   "skill at 1 s" elsewhere, which is not the same quantity. Resolving this needs the
   ActionSense release or the authors, not more computation.
3. **The split unit is forced and the obvious one leaks.** Neighbouring clips overlap 15/16,
   so a clip-level random split puts near-duplicates on both sides. The sound unit is the
   reconstructed recording, of which there are 94 -- and a recording is a SEGMENT found by
   testing whether consecutive clips actually advance by one frame, not a directory: the
   stream jumps inside cells, and splicing across a jump would train a forecaster on an
   artefact. `scripts/d256/extract_d256_states.py` does this correctly; the point is that
   the mistake is available and silent.

Only (2) is a genuine barrier. It is also the one that no amount of care on our side removes.

### Hausdorff distance between forecast and truth curves

One forecast, from one origin, in one channel, is treated as a **set** of $H$ points in
(time, value). Time is scaled to $[0,1]$ over the horizon and value by the truth's own spread
there, so the two axes are commensurate:

$$
\sigma_t=\operatorname{std}_h\bigl(y_{t,h}\bigr),\qquad
p_i=\Bigl(\tfrac{i}{H},\ \tfrac{\hat y_{t,i}}{\sigma_t}\Bigr),\qquad
q_j=\Bigl(\tfrac{j}{H},\ \tfrac{y_{t,j}}{\sigma_t}\Bigr),\qquad i,j=0,\dots,H-1
$$

$$
d(p_i,q_j)=\sqrt{\frac{(i-j)^2}{H^2}+\frac{\bigl(\hat y_{t,i}-y_{t,j}\bigr)^2}{\sigma_t^2}}
$$

$$
\mathrm{HD}_t=\max\Bigl\{\ \max_i\min_j d(p_i,q_j),\ \ \max_j\min_i d(p_i,q_j)\ \Bigr\}
$$

Reported per channel as the mean of $\mathrm{HD}_t$ over origins within a clip, then over
clips, and as a ratio to the same quantity for persistence. Lower is better; $1.00\times$
ties persistence.

Three properties govern how it is read:

- **Both axis scalings are conventions.** They are identical for every model, so ratios
  between models mean something even though the absolute number carries no units.
- **Invariant to shifting both curves together.** Adding the same constant to $\hat y$ and
  $y$ moves every point of both sets and leaves all pairwise distances unchanged — which is
  why it may be computed on residual-over-persistence targets exactly, not approximately.
- **$\sigma_t=0$ yields NaN, not 0.** A truth that is constant over the horizon has no shape
  to compare against, and scoring that as a perfect match would flatter every model.

### Why both are reported

MSE is pointwise, so a flat forecast through the middle of an oscillation is charged only its
distance from the centre. Hausdorff asks how far the worst point of one curve is from the
whole of the other, so the same flat forecast is charged roughly the amplitude. On a unit
sine over one horizon: a perfect forecast scores $0.000$, one shifted by $0.4$ rad scores
$0.301$, and the mean scores $0.995$.
