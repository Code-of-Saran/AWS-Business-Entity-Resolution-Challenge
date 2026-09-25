#!/usr/bin/env bash
# Build <team>_submission.zip in the structure the challenge requires.
# Archives straight from the project root so the paths inside the zip are already
# output/..., code/business_entity_resolution/..., Documentation_template.md --
# no staging directory and no deletions.
set -euo pipefail
TEAM="${1:-CodeTitans}"
ZIP="$(pwd)/${TEAM}_submission.zip"

# Gate: never package a structurally-valid but empty submission.
n_match=$(awk -F'\t' 'NR>1 && $2!=""' output/matching_results.tsv | wc -l | tr -d ' ')
n_cand=$(awk -F'\t' 'NR>1 && $2!=""' output/candidate_pairs.tsv | wc -l | tr -d ' ')
echo "non-empty match rows=$n_match  non-empty candidate rows=$n_cand"
if [ "$n_match" -eq 0 ] || [ "$n_cand" -eq 0 ]; then
  echo "REFUSING TO BUILD: outputs contain no predictions. Run the pipeline first." >&2
  exit 1
fi
if grep -q "\[Your Team Name\]" Documentation_template.md; then
  echo "REFUSING TO BUILD: Documentation_template.md still has template placeholders." >&2
  exit 1
fi

zip -q -r -X "$ZIP" \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution \
    Documentation_template.md \
    -x '*__pycache__*' '*.pyc' '*.DS_Store' '*.log'
echo "built $ZIP ($(du -h "$ZIP" | cut -f1))"
unzip -l "$ZIP"
