# Experiment log

Each script writes its results to a JSON / JSONL file next to it, so every number
quoted in the methodology document traces back to a recorded run.

| # | script / artefact | question it answers |
| --- | --- | --- |
| 0 | — | all-singleton floor: predicting nothing for every entity scores the singleton rate (~0.056), the baseline any real model must beat |
| 1 | `exp01_blocking.py` → `exp01_blocking_log.jsonl` | raw candidate recall and cost of the full family set |
| 2 | `exp02_family_ablation.py` | per-family recall/cost, greedy marginal contribution, and what blocking misses |
| 3 | `exp03_sweep.py` → `exp03_sweep_log.jsonl` | blocking-configuration sweep and, critically, recall **after** top-K selection |
| 4 | `exp04_prescore.py` → `exp04_prescore_log.jsonl` | learned cheap ranker vs hand-tuned weights at recall@K |
| 5 | `src/train.py` → `exp05_model_comparison.json`, `exp05_feature_importance.json` | model family comparison and the entity-context feature ablation |
| 6 | `src/tune_decision.py` → `exp06_decision_tuning.json` | staged optimisation of the decision rule against validation F0.5 |
| 7 | `src/error_analysis.py` → `../reports/error_analysis.json` | where false merges and missed matches actually come from |

## Findings that changed the design

1. **Address beats name as the primary blocking signal.** The noise process can
   replace a business name outright; a matched record whose name was pure
   gibberish still shared its full address. Address-token families (AA, NA) carry
   the most recall.
2. **Loosening blocking hurt end-to-end recall.** Raising the block-size cap
   improved raw recall from 0.9416 to 0.9540 but *lowered* recall@16 from 0.9026
   to 0.8972, because the extra weak candidates displaced true ones inside a fixed
   candidate budget. The tighter configuration was kept.
3. **The selection stage was the binding constraint, not the index.** Swapping
   hand-tuned selection weights for a small learned ranker moved recall@12 from
   0.8955 to 0.9373 against a blocking ceiling of 0.9412 — at the same cost.
4. **Rarity features dominate.** The top cheap-ranker features are the rarest
   shared token id for address, number and name (`at_minid`, `an_minid`,
   `mt_minid`), which is what motivated the document-frequency-ordered token id
   representation.
5. **Precision dominates F0.5.** A false positive on a three-match entity costs
   about 0.21 of that entity's score, whereas lifting recall from 93.7% to 96%
   with perfect precision is worth about 0.005. Effort went to precision and
   singleton detection rather than to chasing the recall ceiling.
