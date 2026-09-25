"""Experiment 4: does a learned cheap ranker beat the heuristic at recall@K?

Blocking is run once; the rarity-aware cheap features are then ranked three ways
(heuristic weights, logistic regression, LightGBM) and recall@K compared.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import polars as pl

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "code", "business_entity_resolution", "src")
sys.path.insert(0, os.path.abspath(SRC))
import blocking as bk
import prescore as ps
import data_loader as dl
from prepare import prepared_path

KS = (8, 12, 16, 20, 24, 32, 48)


def load(split, source, country):
    return (pl.scan_parquet(prepared_path(split, source))
              .filter(pl.col("country") == country).collect())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample", type=int, default=60000)
    args = ap.parse_args()
    cfg = bk.BlockConfig(n_addr_tok=5, n_name_tok=4, max_block=60)

    a_full = load("train", 1, args.country)
    b_raw = pl.concat([load("train", 2, args.country), load("train", 3, args.country)])
    a_raw = a_full.sample(n=min(args.sample, a_full.height), seed=42).with_row_index("idx")
    b_raw = b_raw.with_row_index("idx")

    vat = bk.build_vocab([a_full, b_raw], "addr_toks")
    van = bk.build_vocab([a_full, b_raw], "addr_nums")
    vmt = bk.build_vocab([a_full, b_raw], "name_toks")
    thr_at, thr_an, thr_mt = (ps.rarity_thresholds(v) for v in (vat, van, vmt))
    print("rarity id thresholds  addr:", thr_at, " num:", thr_an, " name:", thr_mt)

    a = bk.prepare_side(a_raw, vat, van, vmt, cfg)
    b = bk.prepare_side(b_raw, vat, van, vmt, cfg)
    t0 = time.time()
    cand = bk.generate_candidates(a, b, cfg)
    print(f"blocking {time.time()-t0:.1f}s -> {cand.height:,} pairs "
          f"({cand.height/a_raw.height:.1f}/S1)")

    t0 = time.time()
    sc = ps.attach_cheap_features(cand, a, b, thr_at, thr_an, thr_mt)
    print(f"cheap features {time.time()-t0:.1f}s")

    gt = dl.read_ground_truth()
    truth = (gt.join(a_raw.select("idx", "entity_id"),
                     left_on="source1_entity_id", right_on="entity_id", how="inner")
               .select("idx", pl.col("matched_entity_ids").str.split(",").alias("m"))
               .explode("m").drop_nulls("m").filter(pl.col("m") != "")
               .join(b_raw.select(b_idx="idx", m="entity_id"), on="m", how="inner")
               .select(a_idx="idx", b_idx="b_idx"))
    n_true = truth.height
    sc = sc.join(truth.with_columns(y=pl.lit(1, pl.Int8)),
                 on=["a_idx", "b_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    print(f"true pairs={n_true:,}  in-candidate={int(sc['y'].sum()):,} "
          f"(raw recall {sc['y'].sum()/n_true:.4f})")

    # entity-level split so no S1 entity appears in both halves
    rng = np.random.default_rng(7)
    ids = a_raw["idx"].to_numpy()
    tr_mask = rng.random(len(ids)) < 0.5
    tr_ids = pl.Series("a_idx", ids[tr_mask])
    va_ids = pl.Series("a_idx", ids[~tr_mask])
    tr = sc.filter(pl.col("a_idx").is_in(tr_ids))
    va = sc.filter(pl.col("a_idx").is_in(va_ids))
    truth_va = truth.filter(pl.col("a_idx").is_in(va_ids))
    n_true_va = truth_va.height

    F = ps.CHEAP_FEATURES
    Xtr = tr.select(F).to_numpy().astype(np.float32); ytr = tr["y"].to_numpy()
    Xva = va.select(F).to_numpy().astype(np.float32)

    scores = {}
    scores["heuristic"] = va.select(ps.heuristic_score().alias("s"))["s"].to_numpy()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, C=1.0))
    t0 = time.time(); lr.fit(Xtr, ytr)
    scores["logreg"] = lr.predict_proba(Xva)[:, 1]
    print(f"logreg fit {time.time()-t0:.1f}s")

    import lightgbm as lgb
    t0 = time.time()
    gbm = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.1, num_leaves=63,
                             min_child_samples=50, subsample=0.8, colsample_bytree=0.9,
                             random_state=0, n_jobs=10, verbose=-1)
    gbm.fit(Xtr, ytr)
    scores["lightgbm"] = gbm.predict_proba(Xva)[:, 1]
    print(f"lightgbm fit {time.time()-t0:.1f}s")

    print(f"\nvalidation half: S1={len(ids)-tr_mask.sum():,} true_pairs={n_true_va:,}")
    print("{:<11}".format("ranker") + " ".join(f"{'R@'+str(k):>8}" for k in KS))
    results = {}
    for nm, s in scores.items():
        v = va.with_columns(s=pl.Series("s", s))
        row = []
        for k in KS:
            tk = ps.top_k(v, k, "s")
            h = truth_va.join(tk.select("a_idx", "b_idx"), on=["a_idx", "b_idx"],
                              how="semi").height
            row.append(h / n_true_va)
        results[nm] = row
        print("{:<11}".format(nm) + " ".join(f"{x:>8.4f}" for x in row))

    imp = sorted(zip(F, gbm.feature_importances_), key=lambda t: -t[1])[:12]
    print("\ntop cheap-ranker features:", ", ".join(f"{k}({v})" for k, v in imp))

    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp04_prescore_log.jsonl")
    with open(log, "a") as f:
        f.write(json.dumps({"country": args.country, "ks": list(KS),
                            "raw_recall": sc["y"].sum() / n_true,
                            "results": {k: list(map(float, v)) for k, v in results.items()}}) + "\n")


if __name__ == "__main__":
    main()
