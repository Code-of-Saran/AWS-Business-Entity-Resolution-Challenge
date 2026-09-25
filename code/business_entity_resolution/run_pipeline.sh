#!/usr/bin/env bash
# End-to-end reproduction: raw TSVs -> output/matching_results.tsv + candidate_pairs.tsv
#
# Run from this directory (code/business_entity_resolution). Expects the challenge
# dataset at ../../dataset (train/ and test/), or set BER_DATA_ROOT.
# Total runtime on a 10-core / 16 GB machine: roughly 2.5 hours.
set -euo pipefail
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-10}"

echo "== 1/8 exploratory data analysis (reports/eda_report.json)"
python3 src/eda.py

echo "== 2/8 normalise + cache every source file as Parquet"
python3 src/prepare.py --split both

echo "== 3/8 train the cheap candidate ranker (Source-1 fold 0)"
python3 src/train_prescore.py

echo "== 4/8 build the pairwise train/validation dataset (folds 1-7 / 8-9)"
python3 src/build_dataset.py --train-per-country 150000 --valid-per-country 75000 --k 20

echo "== 5/8 train and compare matching models"
python3 src/train.py --models lgbm,xgb,hgb,extratrees,logreg

echo "== 6/8 score the full validation folds, then tune the decision rule"
python3 src/inference.py --split train --folds 8,9
python3 src/tune_decision.py

echo "== 7/8 score the test set"
python3 src/inference.py --split test

echo "== 8/8 write the submission files"
python3 src/submission.py --split test --config ../../artifacts/models/decision_config.json

# The official validator ships with the challenge bundle, not with this repo, so
# on a clean clone it is absent. Skip it with a notice rather than failing after a
# successful 8-stage run; point BER_VALIDATOR at it to re-enable the check.
echo "== validate"
VALIDATOR="${BER_VALIDATOR:-../../../student_resource/utils/validate_submission.py}"
if [ -f "$VALIDATOR" ]; then
    python3 "$VALIDATOR" \
        --matching ../../output/matching_results.tsv \
        --candidate ../../output/candidate_pairs.tsv \
        --test-dir "${BER_DATA_ROOT:-../../dataset}/test"
else
    echo "   skipped: no validator at $VALIDATOR"
    echo "   (set BER_VALIDATOR to the challenge bundle's validate_submission.py)"
fi
