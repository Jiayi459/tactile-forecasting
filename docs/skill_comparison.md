# ActionSense vs OpenTouch — skill against persistence

Skill = 1 − MSE(model) / MSE(persistence) at the full 1 s horizon; positive is better than
"assume nothing changes". Only the RIGHT hand is shown: OpenTouch instruments one hand, so
ActionSense's `_R` channels are the closest match its two-handed target allows.

**The two columns are not the same measurement.** Read the caveats before the numbers.

## The table

| model | ActionSense F_R | CoPx_R | CoPy_R | OpenTouch raw F_R | CoPx_R | CoPy_R | OpenTouch D1 F_R | CoPx_R | CoPy_R |
|---|---|---|---|---|---|---|---|---|---|
| seasonal | 0.000 | 0.000 | 0.000 | −0.019 | −0.015 | −0.016 | −0.038 | −0.033 | −0.009 |
| AR | 0.200 | 0.254 | 0.194 | 0.149 | 0.214 | 0.171 | **0.367** | **0.431** | **0.476** |
| GRU-aggregate | 0.181 | 0.233 | 0.175 | — | — | — | — | — | — |
| flatten (map) | −0.042 | −0.002 | −0.051 | — | — | — | — | — | — |
| CNN (map) | 0.138 | 0.011 | 0.045 | — | — | — | — | — | — |
| probGRU | (0.378) | (0.461) | (0.392) | 0.203 | 0.288 | 0.221 | **0.386** | **0.427** | **0.472** |

ActionSense: harness baselines from `docs/actionsense/harness_baselines.csv`; map arms are 10-fold CV
means at their best history from `docs/tactile_map_cv_results{,_aggregate}.csv`.
OpenTouch: 4-fold location-held-out means from `docs/opentouch_cv4{,_d1}.csv`.
Dashes are arms not yet run on OpenTouch, not zeros.

## What differs between the two columns

1. **The probGRU row is parenthesised because it is not the same quantity.** ActionSense's
   probGRU predicts the FAST (high-passed) component and is scored against
   persistence-of-fast; OpenTouch's predicts the RAW target and is scored by the harness.
   Both are "probGRU skill", and they answer different questions. Nothing in that row
   should be compared across columns.

2. **The target differs in the D1 column.** OpenTouch D1 removes a per-taxel baseline; the
   ActionSense harness target does not (its own DC-offset issue, P4, is unresolved). The
   comparable pair is ActionSense vs **OpenTouch raw**; the D1 column shows what removing
   the DC did, not how the sensors compare.

3. **The split differs.** ActionSense cross-validates by recording; OpenTouch holds out
   whole LOCATIONS, which is stricter -- test clips come from environments, objects and
   habits never seen in training. Location-held-out numbers should be lower for the same
   underlying predictability, so OpenTouch is not being flattered here.

4. **The horizon is 1 s in both, but the rate is not.** ActionSense runs at 10 Hz (10
   steps), OpenTouch at 30 Hz (30 steps). Same physical lookahead, three times the steps to
   roll out.

5. **Clip length differs by an order of magnitude.** ActionSense recordings support a 10 s
   history; OpenTouch's median clip is 2.80 s, so its sweep stops at 3 s and even that is
   mostly zero-padding for three quarters of the clips.

## What the numbers say

**AR transfers, and the DC was hiding how well.** On raw OpenTouch, AR scores below its
ActionSense counterpart (0.149 vs 0.200 on F). After the baseline correction it more than
doubles, to 0.367, and CoP goes from 0.17-0.21 to 0.43-0.48. The classical baseline was
never the limitation; 99.78% of F was a constant, and persistence was coasting on it.

**probGRU and AR end up tied on OpenTouch once the DC is gone** (F 0.386 vs 0.367, CoP
within 0.005 of each other), where on raw the GRU led by 5-7 points. Whatever advantage the
GRU had was largely an advantage at reproducing a constant.

**seasonal is at or below zero everywhere.** On ActionSense it is exactly 0.000 -- it falls
back to persistence when no cycle is found, which is every group. On OpenTouch it is
slightly negative, i.e. the few groups where a period was detected were hurt by using it.

**The map arms are the open question.** On ActionSense, flatten is *worse* than persistence
(−0.042 on F) while CNN is clearly positive (0.138), and the aggregate encoder beats both
(0.181) -- so on that sensor the raw map hurt unless a convolution exploited its structure,
and even then it did not beat using the summary signal alone. Those arms exist for
OpenTouch now but have not been run; the D1 correction makes them worth running, since the
map's DC was the same 99.78%.
