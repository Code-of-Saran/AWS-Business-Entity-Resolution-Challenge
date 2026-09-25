"""Stage 5a: optimise the decision rule against validation macro F_0.5.

Runs on the *complete* validation folds (8 and 9) rather than a subsample, because
two of the rules -- global exclusivity and the singleton gate -- depend on how many
Source-1 entities compete for the same Source-2/3 record, and that density is
under-represented in a small sample.

The sweep is deliberately staged rather than a full grid: a global threshold
first, then each refinement is accepted only if it improves validation F_0.5 on
top of the already-chosen rules. That keeps the number of decisions made against
the validation set small, which limits how much of the gain can be validation
overfitting.

Usage:
  python3 src/inference.py --split train --folds 8,9   # produce scored pairs
  python3 src/tune_decision.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader as dl     # noqa: E402
import decision as dc        # noqa: E402
import evaluation as ev      # noqa: E402
import pipeline as pp        # noqa: E402
from inference import score_dir  # noqa: E402

EXP_DIR = os.path.join(dl.PROJECT_ROOT, "experiments")
MODEL_DIR = os.path.join(dl.ARTIFACT_ROOT, "models")


def load_validation(folds=pp.FOLDS_VALID):
    """Scored candidate pairs plus per-entity truth counts and labels."""
    scored = pl.read_parquet(os.path.join(score_dir("train"), "*.parquet"))
    gt = dl.read_ground_truth()
    fold = pp.fold_of(pl.col("source1_entity_id"))
    gtv = gt.filter(fold.is_in(list(folds)))

    truth_counts = gtv.select(
        a_entity_id="source1_entity_id",
        n_true=pl.when(pl.col("matched_entity_ids").str.strip_chars() == "")
                 .then(0)
                 .otherwise(pl.col("matched_entity_ids").str.split(",").list.len())
                 .cast(pl.Int32))
    truth_pairs = (gtv.select("source1_entity_id",
                              pl.col("matched_entity_ids").str.split(",").alias("m"))
                      .explode("m").drop_nulls("m").filter(pl.col("m") != "")
                      .select(a_entity_id="source1_entity_id", b_entity_id="m"))
    scored = (scored.join(truth_pairs.with_columns(y=pl.lit(1, pl.Int8)),
                          on=["a_entity_id", "b_entity_id"], how="left")
                    .with_columns(pl.col("y").fill_null(0)))
    scored = dc.add_integer_keys(scored)
    return scored, truth_counts, truth_pairs


def evaluate(scored: pl.DataFrame, truth_counts: pl.DataFrame,
             cfg: dc.DecisionConfig) -> dict:
    accepted = dc.apply(scored, cfg, score_col="p")
    rows = scored.join(accepted, on=["a_idx", "b_idx"], how="semi")
    return ev.score_labeled(rows, truth_counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EXP_DIR, "exp06_decision_tuning.json"))
    args = ap.parse_args()
    os.makedirs(EXP_DIR, exist_ok=True)

    scored, truth_counts, truth_pairs = load_validation()
    n_ent = truth_counts.height
    n_cand_hit = int(scored["y"].sum())
    print(f"[tune] validation entities={n_ent:,}  scored pairs={scored.height:,}  "
          f"true pairs={truth_pairs.height:,}  in-candidates={n_cand_hit:,} "
          f"(candidate recall {n_cand_hit/truth_pairs.height:.5f})")
    print(f"[tune] singletons={int((truth_counts['n_true']==0).sum()):,} "
          f"({100*float((truth_counts['n_true']==0).mean()):.2f}%)")

    log: dict = {"n_entities": n_ent, "n_scored_pairs": scored.height,
                 "candidate_recall": n_cand_hit / truth_pairs.height,
                 "stages": []}

    def record(stage: str, cfg: dc.DecisionConfig, m: dict) -> None:
        log["stages"].append({"stage": stage, "config": cfg.to_dict(),
                              **{k: m[k] for k in
                                 ("macro_f05", "micro_precision", "micro_recall",
                                  "singleton_accuracy", "macro_f05_singletons",
                                  "macro_f05_nonsingletons",
                                  "mean_pred_per_entity", "false_positive_pairs")}})

    # ---- stage 1: global threshold ----
    print("\n[tune] stage 1: global threshold")
    best = (None, -1.0)
    for t in [round(x, 3) for x in np.arange(0.05, 0.99, 0.025)]:
        cfg = dc.DecisionConfig(threshold=t)
        m = evaluate(scored, truth_counts, cfg)
        record("global_threshold", cfg, m)
        if m["macro_f05"] > best[1]:
            best = (t, m["macro_f05"])
        print(f"   t={t:<6} F0.5={m['macro_f05']:.5f}  P={m['micro_precision']:.4f} "
              f"R={m['micro_recall']:.4f}  sing_acc={m['singleton_accuracy']:.4f} "
              f"pred/ent={m['mean_pred_per_entity']:.2f}")
    t_glob = best[0]
    base = dc.DecisionConfig(threshold=t_glob)
    base_f = best[1]
    print(f"[tune] best global threshold={t_glob} F0.5={base_f:.5f}")

    # ---- stage 2: exclusivity ----
    print("\n[tune] stage 2: global exclusivity (one owner per S2/S3 record)")
    cfg = dc.DecisionConfig(threshold=t_glob, exclusive=True)
    m = evaluate(scored, truth_counts, cfg)
    record("exclusive", cfg, m)
    print(f"   exclusive=True F0.5={m['macro_f05']:.5f} (delta {m['macro_f05']-base_f:+.5f})")
    if m["macro_f05"] > base_f:
        base, base_f = cfg, m["macro_f05"]

    # ---- stage 3: relative threshold ----
    print("\n[tune] stage 3: relative threshold vs best candidate")
    for rel in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        cfg = dc.DecisionConfig(**{**base.to_dict(), "rel_threshold": rel})
        m = evaluate(scored, truth_counts, cfg)
        record("rel_threshold", cfg, m)
        print(f"   rel={rel:<5} F0.5={m['macro_f05']:.5f}  P={m['micro_precision']:.4f} "
              f"R={m['micro_recall']:.4f}")
        if m["macro_f05"] > base_f:
            base, base_f = cfg, m["macro_f05"]

    # ---- stage 4: singleton gate ----
    print("\n[tune] stage 4: singleton gate (drop all when best candidate is weak)")
    for mb in [0.0, t_glob, t_glob + 0.05, t_glob + 0.1, t_glob + 0.15,
               t_glob + 0.2, t_glob + 0.3]:
        cfg = dc.DecisionConfig(**{**base.to_dict(), "min_best": round(mb, 3)})
        m = evaluate(scored, truth_counts, cfg)
        record("min_best", cfg, m)
        print(f"   min_best={mb:<6.3f} F0.5={m['macro_f05']:.5f} "
              f"sing_acc={m['singleton_accuracy']:.4f} R={m['micro_recall']:.4f}")
        if m["macro_f05"] > base_f:
            base, base_f = cfg, m["macro_f05"]

    # ---- stage 5: per-source thresholds ----
    print("\n[tune] stage 5: per-source thresholds")
    for d2 in (-0.1, -0.05, 0.0, 0.05, 0.1):
        for d3 in (-0.1, -0.05, 0.0, 0.05, 0.1):
            cfg = dc.DecisionConfig(**{**base.to_dict(),
                                       "threshold_s2": round(t_glob + d2, 3),
                                       "threshold_s3": round(t_glob + d3, 3)})
            m = evaluate(scored, truth_counts, cfg)
            record("per_source", cfg, m)
            if m["macro_f05"] > base_f:
                base, base_f = cfg, m["macro_f05"]
                print(f"   s2={cfg.threshold_s2} s3={cfg.threshold_s3} "
                      f"F0.5={m['macro_f05']:.5f}  (new best)")

    # ---- stage 6: match cap ----
    print("\n[tune] stage 6: maximum matches per entity")
    for cap in (0, 4, 5, 6, 7, 8, 9):
        cfg = dc.DecisionConfig(**{**base.to_dict(), "max_matches": cap})
        m = evaluate(scored, truth_counts, cfg)
        record("max_matches", cfg, m)
        print(f"   cap={cap:<3} F0.5={m['macro_f05']:.5f}")
        if m["macro_f05"] > base_f:
            base, base_f = cfg, m["macro_f05"]

    final = evaluate(scored, truth_counts, base)
    log["final"] = {"config": base.to_dict(), **final}
    print("\n[tune] FINAL decision config:", json.dumps(base.to_dict()))
    print("[tune] FINAL validation metrics:", json.dumps(
        {k: v for k, v in final.items() if not isinstance(v, dict)}, indent=2))
    with open(args.out, "w") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(MODEL_DIR, "decision_config.json"), "w") as f:
        json.dump(base.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
