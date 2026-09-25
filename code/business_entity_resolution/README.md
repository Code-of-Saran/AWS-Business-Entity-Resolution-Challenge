# Business Entity Resolution — ML Challenge 2026

Matches Source-2 and Source-3 business records to the Source-1 reference entities
they describe, optimised for the leaderboard metric: macro-averaged
F<sub>0.5</sub> over Source-1 entities.

Everything is trained from scratch on the provided TSVs. No pretrained model, no
external dataset, no network access at any stage.

## Layout

```
code/business_entity_resolution/
├── run_pipeline.sh          end-to-end reproduction (8 steps)
├── requirements.txt         pinned versions + licence audit
└── src/
    ├── data_loader.py       TSV readers (tab separator, quoting disabled)
    ├── preprocessing.py     name/address normalisation (vectorised, country-agnostic)
    ├── prepare.py           stage 1 — normalise once, cache as Parquet
    ├── eda.py               dataset profiling -> reports/eda_report.json
    ├── blocking.py          stage 2 — rarity-ordered multi-family inverted index
    ├── prescore.py          stage 2b — cheap set-overlap features
    ├── train_prescore.py    stage 2c — train the cheap top-K ranker (fold 0)
    ├── pipeline.py          shared funnel: per-country index + chunked probing
    ├── features.py          stage 3 — pairwise feature engineering (74 features)
    ├── build_dataset.py     stage 3b — materialise train/validation pairs
    ├── train.py             stage 4 — fit and compare matching models
    ├── decision.py          stage 5 — threshold / singleton / exclusivity rules
    ├── tune_decision.py     stage 5b — optimise the decision rule on folds 8-9
    ├── evaluation.py         the F_0.5 metric and its diagnostics
    ├── inference.py         stage 6 — score every candidate, persist to Parquet
    └── submission.py        stage 7 — write the two TSVs
```

`utils.py` from the suggested skeleton is intentionally absent: its would-be
contents (path handling, chunking, fold assignment) live with the code that uses
them — `data_loader.py` for paths and `pipeline.py` for chunking and folds — which
avoids a grab-bag module.

## Reproducing

```bash
cd code/business_entity_resolution
./run_pipeline.sh
```

The dataset is read from `../../dataset/{train,test}` by default; override with
`BER_DATA_ROOT`. Intermediate artefacts go to `../../artifacts` (override with
`BER_ARTIFACTS`) and the two submission files to `../../output`.

Each stage is independently runnable and writes its own artefacts, so a later
stage can be re-run without repeating the earlier ones — re-tuning the decision
rule, for example, does not require re-scoring the test set.

## Pipeline

```
train/test TSVs
   │
   ├─ prepare.py ............ normalise names and addresses once, cache Parquet
   │
   ├─ blocking.py ........... 8 key families over rarity-ordered token ids
   │                          ~120 candidates per Source-1 entity
   ├─ prescore.py ........... cheap integer set-overlap features
   │  train_prescore.py ..... LightGBM ranker -> top-20 per entity
   │
   ├─ features.py ........... 74 pairwise features (rapidfuzz + set overlap)
   ├─ train.py .............. LightGBM / XGBoost / HistGB / ExtraTrees / LogReg
   │
   ├─ decision.py ........... threshold, relative threshold, singleton gate,
   │  tune_decision.py ...... match cap, per-source thresholds, exclusivity
   │
   └─ submission.py ......... output/matching_results.tsv
                              output/candidate_pairs.tsv
```

`candidate_pairs.tsv` is written from the persisted top-K scored set — the exact
input the matching model ran inference over — so `matching_results.tsv` is a
strict subset of it by construction, not by coincidence.

## Determinism

Fixed seeds throughout: fold assignment hashes `entity_id` with seed `0xC0FFEE`
(`pipeline.FOLD_SEED`), all samplers use explicit seeds, all models use
`random_state=0`, and every top-K / conflict-resolution tie is broken on a
deterministic id so repeated runs produce byte-identical output.

## Entity-level splits

Source-1 entities are partitioned into 10 folds by a stable hash of
`entity_id`, keeping each entity's whole ground-truth match group on one side of
every split:

| folds | use |
| --- | --- |
| 0 | train the cheap candidate ranker |
| 1–7 | train the matching model |
| 8–9 | validation: model selection, threshold and decision tuning |

No entity is ever used to fit a component and then to evaluate it.
