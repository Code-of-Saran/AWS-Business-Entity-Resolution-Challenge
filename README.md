# ML Challenge 2026 — Business Entity Resolution

Submission package. Matches Source-2 / Source-3 business records to the Source-1
reference entity they describe, optimised for macro-averaged F<sub>0.5</sub>.

```
project/
├── dataset/                  -> symlink to the untouched challenge data
│   ├── train/                   (train_source1/2/3.tsv, train_ground_truth.tsv)
│   └── test/                    (test_source1/2/3.tsv)
├── output/
│   ├── matching_results.tsv     final matches (leaderboard file)
│   └── candidate_pairs.tsv      the candidate set the model scored
├── code/business_entity_resolution/
│   ├── src/                     the pipeline
│   ├── run_pipeline.sh          end-to-end reproduction
│   ├── README.md                architecture and run instructions
│   └── requirements.txt         pinned versions + licence audit
├── Documentation_template.md    filled-in methodology write-up
├── reports/                     EDA output
├── experiments/                 experiment scripts and their logged results
└── artifacts/                   generated caches, models, scored pairs (not shipped)
```

The original challenge files under `../student_resource/` are never modified;
`dataset/` is a read-only symlink to them.

## Quick start

```bash
cd code/business_entity_resolution
./run_pipeline.sh
```

## Approach in one paragraph

A four-stage funnel. Names and addresses are normalised once with a
country-agnostic pipeline that folds accents, strips alias markers
(`X a/k/a Y`), un-concatenates domain-style names and removes literal `null`
placeholders. Candidates come from an inverted index over eight key families
built on token ids that are numbered in ascending document-frequency order, so
"the rarest tokens in this record" is a list sort and "is this shared token rare"
is one integer comparison. A small LightGBM ranker over integer set-overlap
features cuts ~120 candidates per entity to the top 20 while keeping 99.6% of the
blocking recall ceiling. Those survivors get 74 pairwise features (rapidfuzz
string metrics over four normalised views of the name, address metrics, and
rarity-bucketed overlaps) and a LightGBM matcher. Finally, because the metric is
per-entity and precision-weighted, the decision layer is tuned as a unit: a
global threshold, a threshold relative to each entity's best candidate, an
explicit singleton gate, a match cap, and global conflict resolution that
exploits a property verified across all 7,638,365 training pairs — every matched
Source-2/3 record has exactly one true Source-1 owner.

No external data, no pretrained weights, no network access. Country is treated as
an open set of strings throughout; the unseen test country (France) flows through
exactly the same code path as the training countries.
