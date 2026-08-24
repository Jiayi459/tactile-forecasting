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
