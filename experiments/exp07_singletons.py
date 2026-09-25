"""Experiment 7: why are true singletons being merged, and what can recover them?

Singletons are ~5.6% of entities and each one correctly left empty is worth a
full 1.0 under the macro metric, so mishandling them is the single largest
identified loss. This script asks three questions in order:

1. How separable are singleton and non-singleton entities from the *shape* of
   their candidate score distribution (best score, gap to runner-up, count)?
   If they separate well, a threshold-style gate suffices; if not, no gate will.
2. What does a singleton's strongest candidate actually look like? Printed as
   text, because the answer determines whether the failure is a modelling gap or
   a deliberately hard negative in the data.
3. What do the candidate gate rules actually buy, measured end to end?

Run after: python3 src/inference.py --split train --folds 8,9
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import polars as pl

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "code", "business_entity_resolution", "src")
sys.path.insert(0, os.path.abspath(SRC))
import decision as dc
import evaluation as ev
import pipeline as pp
from tune_decision import load_validation

TEXT = ["entity_id", "business_name", "business_address", "country"]


def entity_table(scored: pl.DataFrame, truth_counts: pl.DataFrame) -> pl.DataFrame:
    """One row per entity: the shape of its candidate score distribution."""
    g = scored.group_by("a_entity_id").agg(
        n_cands=pl.len(),
        best=pl.col("p").max(),
        second=pl.col("p").top_k(2).min(),
        psum=pl.col("p").sum(),
        n_over_50=(pl.col("p") > 0.5).sum(),
        n_over_90=(pl.col("p") > 0.9).sum(),
        best_cheap=pl.col("cheap_score").max())
    t = (truth_counts.join(g, on="a_entity_id", how="left")
         .with_columns(pl.col("n_cands").fill_null(0), pl.col("best").fill_null(0.0),
                       pl.col("second").fill_null(0.0), pl.col("psum").fill_null(0.0),
                       pl.col("n_over_50").fill_null(0), pl.col("n_over_90").fill_null(0),
                       pl.col("best_cheap").fill_null(0.0)))
    return t.with_columns(
        gap=pl.col("best") - pl.col("second"),
        is_singleton=(pl.col("n_true") == 0).cast(pl.Int8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=10)
    args = ap.parse_args()

    scored, truth_counts, truth_pairs = load_validation()
    ent = entity_table(scored, truth_counts)
    S = ent.filter(pl.col("is_singleton") == 1)
    N = ent.filter(pl.col("is_singleton") == 0)
    print(f"entities={ent.height:,}  singletons={S.height:,} "
          f"({100*S.height/ent.height:.2f}%)  non-singletons={N.height:,}")

    print("\n--- distribution of the entity's BEST candidate probability ---")
    qs = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print("        " + "".join(f"{'q'+str(int(q*100)):>9}" for q in qs) + f"{'mean':>9}")
    for nm, d in (("singleton", S), ("non-single", N)):
        print(f"{nm:<8}" + "".join(f"{float(d['best'].quantile(q)):>9.4f}" for q in qs)
              + f"{float(d['best'].mean()):>9.4f}")
    print("\n--- gap between best and second-best ---")
    for nm, d in (("singleton", S), ("non-single", N)):
        print(f"{nm:<8}" + "".join(f"{float(d['gap'].quantile(q)):>9.4f}" for q in qs)
              + f"{float(d['gap'].mean()):>9.4f}")
    print("\n--- number of candidates scoring > 0.5 ---")
    for nm, d in (("singleton", S), ("non-single", N)):
        print(f"{nm:<8}" + "".join(f"{float(d['n_over_50'].quantile(q)):>9.2f}" for q in qs)
              + f"{float(d['n_over_50'].mean()):>9.3f}")

    # separability of the entity-level signals
    from sklearn.metrics import roc_auc_score
    y = ent["is_singleton"].to_numpy()
    print("\n--- singleton detectability from single entity-level signals (AUC) ---")
    for col, sign in (("best", -1), ("gap", -1), ("psum", -1), ("n_over_50", -1),
                      ("n_over_90", -1), ("best_cheap", -1), ("n_cands", -1)):
        v = ent[col].to_numpy().astype(np.float64) * sign
        print(f"   {col:<12} AUC={roc_auc_score(y, v):.4f}")

    # a small entity-level model over the same signals
    feats = ["n_cands", "best", "second", "gap", "psum", "n_over_50",
             "n_over_90", "best_cheap"]
    X = ent.select(feats).to_numpy().astype(np.float32)
    rng = np.random.default_rng(3)
    m = rng.random(len(y)) < 0.5
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.06, num_leaves=63,
                             min_child_samples=40, random_state=0, n_jobs=10,
                             verbose=-1)
    clf.fit(X[m], y[m])
    ps = clf.predict_proba(X[~m])[:, 1]
    print(f"   entity-level LightGBM over all of the above: "
          f"AUC={roc_auc_score(y[~m], ps):.4f}")
    print("   importance: " + ", ".join(
        f"{k}({v})" for k, v in sorted(zip(feats, clf.feature_importances_),
                                       key=lambda t: -t[1])))

    # what the gate rules buy, end to end
    print("\n--- end-to-end effect of the singleton gate (threshold fixed at 0.625) ---")
    rows = []
    for mb in [0.0, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        cfg = dc.DecisionConfig(threshold=0.625, min_best=mb)
        acc = dc.apply(scored, cfg, score_col="p")
        pred = scored.join(acc, on=["a_idx", "b_idx"], how="semi")
        r = ev.score_labeled(pred, truth_counts)
        rows.append({"min_best": mb, **r})
        print(f"   min_best={mb:<5} F0.5={r['macro_f05']:.5f} "
              f"sing_acc={r['singleton_accuracy']:.4f} "
              f"P={r['micro_precision']:.4f} R={r['micro_recall']:.4f} "
              f"F_sing={r['macro_f05_singletons']:.4f} "
              f"F_non={r['macro_f05_nonsingletons']:.4f}")

    # ceiling: what would a perfect singleton oracle be worth?
    cfg = dc.DecisionConfig(threshold=0.625)
    acc = dc.apply(scored, cfg, score_col="p")
    pred = scored.join(acc, on=["a_idx", "b_idx"], how="semi")
    base = ev.score_labeled(pred, truth_counts)
    nonsingle = truth_counts.filter(pl.col("n_true") > 0).select("a_entity_id")
    pred_oracle = pred.join(nonsingle, on="a_entity_id", how="semi")
    orc = ev.score_labeled(pred_oracle, truth_counts)
    print(f"\n   baseline F0.5           = {base['macro_f05']:.5f}")
    print(f"   with a PERFECT singleton oracle = {orc['macro_f05']:.5f} "
          f"(headroom {orc['macro_f05']-base['macro_f05']:+.5f})")

    # text of singleton entities whose best candidate scored high
    print("\n--- singleton entities with a high-scoring best candidate ---")
    txt = pl.concat([pp.load_prepared("train", s, columns=TEXT) for s in (1, 2, 3)])
    hot = S.filter(pl.col("best") > 0.9).sample(
        n=min(args.examples, S.filter(pl.col("best") > 0.9).height), seed=4)
    top = (scored.join(hot.select("a_entity_id"), on="a_entity_id", how="semi")
                 .sort(["a_entity_id", "p"], descending=[False, True])
                 .group_by("a_entity_id", maintain_order=True).head(3))
    A = txt.select(a_entity_id="entity_id", an="business_name", aa="business_address",
                   ac="country")
    B = txt.select(b_entity_id="entity_id", bn="business_name", ba="business_address")
    top = top.join(A, on="a_entity_id", how="inner").join(B, on="b_entity_id", how="inner")
    for eid in hot["a_entity_id"].to_list():
        d = top.filter(pl.col("a_entity_id") == eid)
        if d.height == 0:
            continue
        r0 = d.row(0, named=True)
        print("=" * 100)
        print(f"  SINGLETON {eid} [{r0['ac']}]  (truth: no matches)")
        print(f"     NAME: {r0['an']}\n     ADDR: {r0['aa']}")
        for r in d.iter_rows(named=True):
            print(f"   -> p={r['p']:.4f} {r['b_entity_id']:<14} {r['bn']}")
            print(f"      {'':<21} {r['ba']}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "exp07_singletons.json")
    with open(out, "w") as f:
        json.dump({"n_entities": ent.height, "n_singletons": S.height,
                   "gate_sweep": [{k: v for k, v in r.items()
                                   if not isinstance(v, dict)} for r in rows],
                   "baseline_f05": base["macro_f05"],
                   "oracle_f05": orc["macro_f05"]}, f, indent=2)
    print(f"\n[exp07] wrote {out}")


if __name__ == "__main__":
    main()
