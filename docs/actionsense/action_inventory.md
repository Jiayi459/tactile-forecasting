# ActionSense action inventory

299 recordings, 14 distinct verbs, from `/users/jhao3/TouchAnything/data/actionsense_states`.
One activity per recording, so the verb is available per window -- which is what
lets a probGRU arm carry the same action embedding here as on OpenTouch.

| verb | recordings | frames | objects | example label |
|---|---|---|---|---|
| `clean` | 60 | 18960 | sponge, towel | Clean a plate with a sponge |
| `slice` | 45 | 51162 | bread, cucumber, potato | Slice a cucumber |
| `get` | 30 | 56075 | refrigerator/cabinets/drawers, utensils | Get items from refrigerator/cabinets/drawers |
| `peel` | 30 | 46086 | cucumber, potato | Peel a cucumber |
| `spread` | 30 | 18312 | slice | Spread almond butter on a bread slice |
| `clear` | 28 | 21852 | board | Clear cutting board |
| `pour` | 25 | 5139 | glass | Pour water from a pitcher into a glass |
| `get/replace` | 15 | 26285 | refrigerator/cabinets/drawers | Get/replace items from refrigerator/cabinets/drawers |
| `open/close` | 9 | 3682 | butter | Open/close a jar of almond butter |
| `open` | 6 | 2624 | butter | Open a jar of almond butter |
| `set` | 6 | 19988 | utensils | Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils |
| `stack` | 5 | 6088 | bowls | Stack on table: 3 each large/small plates, bowls |
| `load` | 5 | 20655 | utensils | Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils |
| `unload` | 5 | 23401 | utensils | Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils |

## Next step

Audit each verb against the rubric in `src/opentouch/trait.py` (Layer 1) and
commit the verdict BEFORE scoring anything by it. Note that a training-free
predictability probe over these actions already exists
(`docs/actionsense/predictability_by_category*.csv`), so the audit has to stand on
the physical rubric alone and be readable as such by someone who has seen it.
