"""Stage 7: write matching_results.tsv and candidate_pairs.tsv.

Both files are generated from the same persisted scored-candidate set, so the
subset property the validator checks holds by construction rather than by luck.

Guarantees enforced here:

* exactly one row per Source-1 entity in ``test_source1.tsv``, in file order;
* ``matched_entity_ids`` empty for predicted singletons;
* Source-2/3 ids only, deduplicated within each list;
* tab-separated, UTF-8, no quoting (addresses and id lists both contain commas,
  which is why the challenge uses TSV).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader as dl     # noqa: E402
import decision as dc        # noqa: E402
from inference import score_dir  # noqa: E402

OUT_DIR = os.path.join(dl.PROJECT_ROOT, "output")


def load_scored(split: str) -> pl.DataFrame:
    return pl.read_parquet(os.path.join(score_dir(split), "*.parquet"))


def _agg_lists(pairs: pl.DataFrame, value_col: str, out_col: str) -> pl.DataFrame:
    """Collapse pairs to one comma-joined id list per Source-1 entity."""
    return (pairs.group_by("a_entity_id")
                 .agg(pl.col(value_col).unique().sort().str.join(",").alias(out_col)))


def write_submission(split: str, cfg: dc.DecisionConfig,
                     out_dir: str = OUT_DIR) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    scored = load_scored(split)
    required = (pl.scan_parquet(
        os.path.join(dl.ARTIFACT_ROOT, "prepared", f"{split}_source1.parquet"))
        .select("entity_id").collect().rename({"entity_id": "a_entity_id"}))

    # The decision rule needs integer keys for its window functions.
    keyed = dc.add_integer_keys(scored)
    accepted = dc.apply(keyed, cfg, score_col="p")
    matches = keyed.join(accepted, on=["a_idx", "b_idx"], how="semi")

    m = _agg_lists(matches, "b_entity_id", "matched_entity_ids")
    c = _agg_lists(scored, "b_entity_id", "candidate_entity_ids")

    match_tsv = (required.join(m, on="a_entity_id", how="left")
                 .with_columns(pl.col("matched_entity_ids").fill_null(""))
                 .rename({"a_entity_id": "source1_entity_id"}))
    cand_tsv = (required.join(c, on="a_entity_id", how="left")
                .with_columns(pl.col("candidate_entity_ids").fill_null(""))
                .rename({"a_entity_id": "source1_entity_id"}))

    mp = os.path.join(out_dir, "matching_results.tsv")
    cp = os.path.join(out_dir, "candidate_pairs.tsv")
    match_tsv.write_csv(mp, separator="\t", include_header=True,
                        quote_style="never")
    cand_tsv.write_csv(cp, separator="\t", include_header=True,
                       quote_style="never")

    n_nonempty = int((match_tsv["matched_entity_ids"] != "").sum())
    info = {"split": split, "decision": cfg.to_dict(),
            "rows": match_tsv.height,
            "entities_with_matches": n_nonempty,
            "entities_predicted_singleton": match_tsv.height - n_nonempty,
            "predicted_pairs": matches.height,
            "candidate_pairs": scored.height,
            "matching_results": mp, "candidate_pairs_file": cp}
    with open(os.path.join(out_dir, "submission_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--threshold-s2", type=float, default=None)
    ap.add_argument("--threshold-s3", type=float, default=None)
    ap.add_argument("--rel-threshold", type=float, default=0.0)
    ap.add_argument("--min-best", type=float, default=0.0)
    ap.add_argument("--max-matches", type=int, default=0)
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--config", default="", help="JSON file with a decision config")
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = dc.DecisionConfig(**json.load(f))
    else:
        cfg = dc.DecisionConfig(
            threshold=args.threshold, threshold_s2=args.threshold_s2,
            threshold_s3=args.threshold_s3, rel_threshold=args.rel_threshold,
            min_best=args.min_best, max_matches=args.max_matches,
            exclusive=args.exclusive)
    write_submission(args.split, cfg, args.out_dir)


if __name__ == "__main__":
    main()
