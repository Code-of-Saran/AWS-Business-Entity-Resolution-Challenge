"""Stage 4: train and compare matching models, then optimise the decision rule.

Several model families are fitted on the same features and compared on the same
held-out entity folds, because the brief (rightly) does not assume a winner. Every
number reported comes from ``evaluation.score_labeled`` on folds 8-9, which no
model and no ranker was fitted on.

Selection is by macro F_0.5 *after* a threshold sweep, not by AUC or log-loss: a
model that ranks slightly worse but separates the top of each entity's candidate
list more cleanly is the better model for this metric.

Usage:  python3 src/train.py --models lgbm,xgb,hgb,extratrees,logreg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader as dl     # noqa: E402
import decision as dc        # noqa: E402
import evaluation as ev      # noqa: E402
import features as ft        # noqa: E402

DATASET_DIR = os.path.join(dl.ARTIFACT_ROOT, "dataset")
MODEL_DIR = os.path.join(dl.ARTIFACT_ROOT, "models")
EXP_DIR = os.path.join(dl.PROJECT_ROOT, "experiments")


def load_dataset():
    """Features, pair metadata, and per-entity truth counts.

    The truth files define each split's *scope* -- every sampled entity, including
    ones blocking produced no candidate for, since those still contribute their
    zero to the macro average. ``n_true`` itself is re-derived by joining the
    ground truth on entity_id rather than trusted from the file.
    """
    X = np.load(os.path.join(DATASET_DIR, "X.npy"), mmap_mode="r")
    meta = pl.read_parquet(os.path.join(DATASET_DIR, "meta.parquet"))
    scope = pl.concat([
        pl.read_parquet(os.path.join(DATASET_DIR, f))
        for f in sorted(os.listdir(DATASET_DIR)) if f.startswith("truth_")])
    scope = scope.select(a_entity_id="entity_id", country="country", split="split")
    gt = dl.read_ground_truth().select(
        a_entity_id="source1_entity_id",
        n_true=pl.when(pl.col("matched_entity_ids").str.strip_chars() == "")
                 .then(0)
                 .otherwise(pl.col("matched_entity_ids").str.split(",").list.len())
                 .cast(pl.Int32))
    truth = (scope.join(gt, on="a_entity_id", how="left")
                  .with_columns(pl.col("n_true").fill_null(0)))
    return X, meta, truth


def build_models(names: list[str], n_jobs: int = 10) -> dict:
    zoo: dict = {}
    if "lgbm" in names:
        import lightgbm as lgb
        zoo["lgbm"] = lgb.LGBMClassifier(
            n_estimators=900, learning_rate=0.05, num_leaves=127,
            min_child_samples=40, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.85, reg_lambda=1.0, random_state=0,
            n_jobs=n_jobs, verbose=-1)
    if "lgbm_deep" in names:
        import lightgbm as lgb
        zoo["lgbm_deep"] = lgb.LGBMClassifier(
            n_estimators=1600, learning_rate=0.03, num_leaves=255,
            min_child_samples=30, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=2.0, random_state=0,
            n_jobs=n_jobs, verbose=-1)
    if "xgb" in names:
        import xgboost as xgb
        zoo["xgb"] = xgb.XGBClassifier(
            n_estimators=900, learning_rate=0.05, max_depth=8,
            min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, tree_method="hist", random_state=0,
            n_jobs=n_jobs, eval_metric="logloss")
    if "hgb" in names:
        from sklearn.ensemble import HistGradientBoostingClassifier
        zoo["hgb"] = HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.06, max_leaf_nodes=127,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    if "extratrees" in names:
        from sklearn.ensemble import ExtraTreesClassifier
        zoo["extratrees"] = ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, max_features="sqrt",
            n_jobs=n_jobs, random_state=0)
    if "randomforest" in names:
        from sklearn.ensemble import RandomForestClassifier
        zoo["randomforest"] = RandomForestClassifier(
            n_estimators=250, min_samples_leaf=5, max_features="sqrt",
            n_jobs=n_jobs, random_state=0)
    if "logreg" in names:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        zoo["logreg"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=600, C=0.5, n_jobs=n_jobs))
    return zoo


def sweep_threshold(valid: pl.DataFrame, truth_valid: pl.DataFrame,
                    thresholds=None) -> tuple[float, dict, list]:
    """Global-threshold sweep; returns the best threshold and the full curve."""
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.arange(0.05, 0.98, 0.025)]
    curve = []
    best = (None, -1.0, None)
    for t in thresholds:
        pred = dc.apply(valid, dc.DecisionConfig(threshold=t), score_col="p")
        rows = valid.join(pred, on=["a_idx", "b_idx"], how="semi")
        m = ev.score_labeled(rows, truth_valid)
        curve.append({"threshold": t, **{k: m[k] for k in
                      ("macro_f05", "micro_precision", "micro_recall",
                       "singleton_accuracy", "mean_pred_per_entity")}})
        if m["macro_f05"] > best[1]:
            best = (t, m["macro_f05"], m)
    return best[0], best[2], curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgbm,xgb,hgb,extratrees,logreg")
    ap.add_argument("--feature-sets", default="all",
                    help="comma list of: all, base (base = without the "
                         "entity-level group features, for ablation)")
    ap.add_argument("--save-best", default="matcher")
    ap.add_argument("--n-jobs", type=int, default=10,
                    help="worker threads per model; lower it when another stage "
                         "is running concurrently")
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    X, meta, truth = load_dataset()
    is_tr = (meta["split"] == "train").to_numpy()
    is_va = ~is_tr
    y = meta["y"].to_numpy()
    print(f"[train] X={X.shape}  train_rows={int(is_tr.sum()):,}  "
          f"valid_rows={int(is_va.sum()):,}  pos_rate={y.mean():.4f}")

    Xtr = np.ascontiguousarray(X[is_tr]); ytr = y[is_tr]
    Xva = np.ascontiguousarray(X[is_va]); 
    valid_meta = meta.filter(pl.Series(is_va))
    truth_valid = truth.filter(pl.col("split") == "valid").select(
        "a_entity_id", "n_true")
    print(f"[train] valid entities={truth_valid.height:,} "
          f"(singletons={int((truth_valid['n_true']==0).sum()):,})")

    # Column subsets for the feature-set ablation.
    all_idx = list(range(len(ft.FEATURE_NAMES)))
    base_idx = [i for i, n in enumerate(ft.FEATURE_NAMES)
                if n not in ft.PRESCORE_GROUP_FEATURES]
    subsets = {"all": all_idx, "base": base_idx}

    results = {}
    best_name, best_f, best_model, best_cols = None, -1.0, None, all_idx
    for fs in args.feature_sets.split(","):
        cols = subsets[fs]
        Xtr_s = Xtr if cols == all_idx else np.ascontiguousarray(Xtr[:, cols])
        Xva_s = Xva if cols == all_idx else np.ascontiguousarray(Xva[:, cols])
        fnames = [ft.FEATURE_NAMES[i] for i in cols]
        for name, model in build_models(args.models.split(","), args.n_jobs).items():
            tag = f"{name}|{fs}"
            t0 = time.time()
            model.fit(Xtr_s, ytr)
            fit_s = time.time() - t0
            t0 = time.time()
            p = model.predict_proba(Xva_s)[:, 1].astype(np.float32)
            pred_s = time.time() - t0
            valid = valid_meta.with_columns(p=pl.Series("p", p))
            t, m, curve = sweep_threshold(valid, truth_valid)
            results[tag] = {"features": fs, "n_features": len(cols),
                            "fit_secs": round(fit_s, 1),
                            "pred_secs": round(pred_s, 1),
                            "best_threshold": t, **m, "curve": curve}
            print(f"[train] {tag:<20} F0.5={m['macro_f05']:.5f} @t={t}  "
                  f"P={m['micro_precision']:.4f} R={m['micro_recall']:.4f} "
                  f"sing_acc={m['singleton_accuracy']:.4f}  fit={fit_s:.0f}s",
                  flush=True)
            if m["macro_f05"] > best_f:
                best_name, best_f, best_model, best_cols = tag, m["macro_f05"], model, cols
                best_fnames = fnames

    print(f"\n[train] best model: {best_name}  macro F0.5={best_f:.5f}")
    import joblib
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{args.save_best}.joblib"))
    with open(os.path.join(MODEL_DIR, f"{args.save_best}_meta.json"), "w") as f:
        json.dump({"model": best_name, "feature_set": results[best_name]["features"],
                   "features": best_fnames,
                   "feature_columns": best_cols,
                   "best_threshold": results[best_name]["best_threshold"],
                   "valid_macro_f05": best_f}, f, indent=2)
    with open(os.path.join(EXP_DIR, "exp05_model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Feature importance for the winner, when the model exposes it.
    if hasattr(best_model, "feature_importances_"):
        imp = sorted(zip(best_fnames,
                         [float(v) for v in best_model.feature_importances_]),
                     key=lambda t: -t[1])
        print("[train] top features: " + ", ".join(f"{k}({v:.0f})" for k, v in imp[:20]))
        with open(os.path.join(EXP_DIR, "exp05_feature_importance.json"), "w") as f:
            json.dump(imp, f, indent=2)


if __name__ == "__main__":
    main()
