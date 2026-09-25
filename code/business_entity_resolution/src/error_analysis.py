"""Stage 8: error analysis on the validation folds.

Answers two questions the aggregate metric hides:

* Where do false merges come from? They are the expensive error under F_0.5, so
  they are bucketed by *which* evidence misled the model (an identical name at a
  different address, an identical address under a different name, and so on).
* Where are missed matches lost? Separated into candidate-generation misses
  (blocking never retrieved the record, so no threshold could recover it) and
  scoring misses (the record was scored but fell below the decision rule).

Run after ``inference.py --split train --folds 8,9``.

Usage:  python3 src/error_analysis.py [--examples 8]
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
import pipeline as pp        # noqa: E402
from tune_decision import load_validation  # noqa: E402

REPORT_DIR = os.path.join(dl.PROJECT_ROOT, "reports")
MODEL_DIR = os.path.join(dl.ARTIFACT_ROOT, "models")

TEXT = ["entity_id", "business_name", "business_address", "country",
        "name_key", "addr_norm", "name_nonlatin"]


def _text_table(split: str) -> pl.DataFrame:
    parts = [pp.load_prepared(split, s, columns=TEXT) for s in (1, 2, 3)]
    return pl.concat(parts)


def bucket_errors(df: pl.DataFrame) -> pl.DataFrame:
    """Label each error row with the evidence pattern that produced it."""
    name_same = pl.col("a_name_key") == pl.col("b_name_key")
    addr_same = pl.col("a_addr_norm") == pl.col("b_addr_norm")
    return df.with_columns(
        bucket=pl.when(name_same & addr_same).then(pl.lit("identical_name_and_address"))
        .when(name_same).then(pl.lit("identical_name_different_address"))
        .when(addr_same).then(pl.lit("identical_address_different_name"))
        .when(pl.col("b_name_nonlatin")).then(pl.lit("transliterated_name"))
        .when(pl.col("b_addr_norm").str.len_chars() == 0).then(pl.lit("empty_address"))
        .when(pl.col("nm_sim") >= 0.85).then(pl.lit("near_identical_name"))
        .when(pl.col("ad_sim") >= 0.85).then(pl.lit("near_identical_address"))
        .otherwise(pl.lit("weak_on_both")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "decision_config.json")) as f:
        cfg = dc.DecisionConfig(**json.load(f))
    scored, truth_counts, truth_pairs = load_validation()
    accepted = dc.apply(scored, cfg, score_col="p")
    scored = (scored.join(accepted.with_columns(pred=pl.lit(1, pl.Int8)),
                          on=["a_idx", "b_idx"], how="left")
                    .with_columns(pl.col("pred").fill_null(0)))

    fp = scored.filter((pl.col("pred") == 1) & (pl.col("y") == 0))
    fn_scored = scored.filter((pl.col("pred") == 0) & (pl.col("y") == 1))
    in_cand = scored.select("a_entity_id", "b_entity_id")
    fn_blocking = truth_pairs.join(in_cand, on=["a_entity_id", "b_entity_id"],
                                   how="anti")

    from rapidfuzz import fuzz, process
    txt = _text_table("train")
    acols = txt.select(a_entity_id="entity_id", a_name="business_name",
                       a_addr="business_address", a_name_key="name_key",
                       a_addr_norm="addr_norm", a_country="country")
    bcols = txt.select(b_entity_id="entity_id", b_name="business_name",
                       b_addr="business_address", b_name_key="name_key",
                       b_addr_norm="addr_norm", b_name_nonlatin="name_nonlatin")

    def enrich(d: pl.DataFrame) -> pl.DataFrame:
        d = d.join(acols, on="a_entity_id", how="inner").join(
            bcols, on="b_entity_id", how="inner")
        if d.height == 0:
            return d
        nm = process.cpdist(d["a_name_key"].to_list(), d["b_name_key"].to_list(),
                            scorer=fuzz.token_set_ratio, workers=-1) / 100.0
        ad = process.cpdist(d["a_addr_norm"].to_list(), d["b_addr_norm"].to_list(),
                            scorer=fuzz.token_set_ratio, workers=-1) / 100.0
        return bucket_errors(d.with_columns(nm_sim=pl.Series("nm_sim", nm),
                                            ad_sim=pl.Series("ad_sim", ad)))

    report: dict = {"decision_config": cfg.to_dict(),
                    "counts": {"accepted_pairs": accepted.height,
                               "false_positive_pairs": fp.height,
                               "true_pairs": truth_pairs.height,
                               "missed_in_scoring": fn_scored.height,
                               "missed_in_blocking": fn_blocking.height}}

    pl.Config.set_fmt_str_lengths(140)
    for label, d in (("false_positives", enrich(fp)),
                     ("false_negatives_scored", enrich(fn_scored)),
                     ("false_negatives_blocking", enrich(fn_blocking))):
        if d.height == 0:
            report[label] = {"n": 0}
            continue
        counts = (d.group_by("bucket").len().sort("len", descending=True))
        report[label] = {"n": d.height,
                         "buckets": {r[0]: int(r[1]) for r in counts.iter_rows()},
                         "bucket_pct": {r[0]: round(100.0 * r[1] / d.height, 2)
                                        for r in counts.iter_rows()},
                         "by_country": {r[0]: int(r[1]) for r in
                                        d.group_by("a_country").len().iter_rows()}}
        print("=" * 100)
        print(f"{label}: n={d.height:,}")
        for r in counts.iter_rows():
            print(f"   {r[0]:<34} {r[1]:>9,}  ({100.0*r[1]/d.height:5.2f}%)")
        ex = d.sample(n=min(args.examples, d.height), seed=5)
        for r in ex.iter_rows(named=True):
            print("-" * 96)
            print(f"  [{r['bucket']}] p={r.get('p', float('nan')):.4f} "
                  f"nm={r['nm_sim']:.3f} ad={r['ad_sim']:.3f} ({r['a_country']})")
            print(f"   S1 {r['a_entity_id']:<14} {r['a_name']}\n"
                  f"      {'':<14} {r['a_addr']}")
            print(f"   -> {r['b_entity_id']:<14} {r['b_name']}\n"
                  f"      {'':<14} {r['b_addr']}")

    out = os.path.join(REPORT_DIR, "error_analysis.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[error] wrote {out}")


if __name__ == "__main__":
    main()
