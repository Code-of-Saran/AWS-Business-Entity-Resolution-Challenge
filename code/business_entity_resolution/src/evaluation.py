"""The competition metric and the diagnostics used to steer the pipeline.

The leaderboard metric is F-beta with beta = 0.5, computed *per Source-1 entity*
and then macro-averaged over every Source-1 entity in the evaluation set:

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

Per-entity edge cases follow the problem statement exactly:

* true set empty and prediction empty  -> 1.0  (a correctly kept singleton)
* true set empty and prediction non-empty -> 0.0
* true set non-empty and prediction empty -> 0.0

Because the average is over entities rather than over pairs, a single false merge
on a singleton costs a full 1.0, while a false merge on an entity with three true
matches costs about 0.21. That asymmetry is why every decision rule downstream is
tuned for precision.
"""
from __future__ import annotations

import numpy as np
import polars as pl

BETA2 = 0.25          # beta^2 for beta = 0.5
ONE_PLUS_BETA2 = 1.25


def fbeta_from_counts(tp: np.ndarray, n_pred: np.ndarray,
                      n_true: np.ndarray) -> np.ndarray:
    """Per-entity F_0.5 from true-positive / predicted / truth counts."""
    tp = tp.astype(np.float64)
    n_pred = n_pred.astype(np.float64)
    n_true = n_true.astype(np.float64)

    out = np.zeros_like(tp)
    both_empty = (n_true == 0) & (n_pred == 0)
    out[both_empty] = 1.0

    live = (n_true > 0) & (n_pred > 0)
    if live.any():
        p = tp[live] / n_pred[live]
        r = tp[live] / n_true[live]
        denom = BETA2 * p + r
        f = np.zeros_like(p)
        ok = denom > 0
        f[ok] = ONE_PLUS_BETA2 * p[ok] * r[ok] / denom[ok]
        out[live] = f
    return out


def score_predictions(entity_ids: pl.Series, pred: pl.DataFrame,
                      truth: pl.DataFrame, id_col: str = "a_idx",
                      target_col: str = "b_idx") -> dict:
    """Macro F_0.5 plus the diagnostics needed for error analysis.

    ``entity_ids`` must list *every* Source-1 entity in the evaluation scope,
    including entities for which no candidate was generated -- they still count
    towards the macro average.
    """
    base = pl.DataFrame({id_col: entity_ids}).unique()
    n_ent = base.height

    np_cnt = pred.group_by(id_col).len().rename({"len": "n_pred"})
    nt_cnt = truth.group_by(id_col).len().rename({"len": "n_true"})
    tp_cnt = (pred.join(truth, on=[id_col, target_col], how="semi")
                  .group_by(id_col).len().rename({"len": "tp"}))

    j = (base.join(np_cnt, on=id_col, how="left")
             .join(nt_cnt, on=id_col, how="left")
             .join(tp_cnt, on=id_col, how="left")
             .fill_null(0))
    tp = j["tp"].to_numpy()
    npd = j["n_pred"].to_numpy()
    ntr = j["n_true"].to_numpy()
    f = fbeta_from_counts(tp, npd, ntr)

    tot_tp, tot_pred, tot_true = int(tp.sum()), int(npd.sum()), int(ntr.sum())
    is_single = ntr == 0
    pred_empty = npd == 0
    return {
        "macro_f05": float(f.mean()),
        "n_entities": n_ent,
        "micro_precision": (tot_tp / tot_pred) if tot_pred else 0.0,
        "micro_recall": (tot_tp / tot_true) if tot_true else 0.0,
        "n_pred_pairs": tot_pred, "n_true_pairs": tot_true, "n_tp_pairs": tot_tp,
        "mean_pred_per_entity": float(npd.mean()),
        "singletons_true": int(is_single.sum()),
        "singleton_accuracy": float(pred_empty[is_single].mean()) if is_single.any() else None,
        "singleton_false_merge_rate": float((~pred_empty)[is_single].mean()) if is_single.any() else None,
        "nonsingleton_all_missed_pct": float(pred_empty[~is_single].mean() * 100)
                                        if (~is_single).any() else None,
        "macro_f05_singletons": float(f[is_single].mean()) if is_single.any() else None,
        "macro_f05_nonsingletons": float(f[~is_single].mean()) if (~is_single).any() else None,
        "false_positive_pairs": tot_pred - tot_tp,
        "pair_fp_rate": (tot_pred - tot_tp) / tot_pred if tot_pred else 0.0,
    }


def candidate_recall(cand: pl.DataFrame, truth: pl.DataFrame,
                     id_col: str = "a_idx", target_col: str = "b_idx") -> dict:
    n_true = truth.height
    hit = truth.join(cand.select(id_col, target_col), on=[id_col, target_col],
                     how="semi").height
    per = cand.group_by(id_col).len()["len"]
    return {
        "candidate_recall": hit / n_true if n_true else None,
        "n_candidate_pairs": cand.height,
        "cands_mean": float(per.mean()) if per.len() else 0.0,
        "cands_median": float(per.median()) if per.len() else 0.0,
        "cands_p95": float(per.quantile(0.95)) if per.len() else 0.0,
    }


def score_labeled(pred: pl.DataFrame, truth_counts: pl.DataFrame,
                  id_col: str = "a_entity_id", label_col: str = "y") -> dict:
    """Macro F_0.5 when each predicted row already carries its own label.

    ``pred`` holds only the accepted pairs, each with ``label_col`` in {0,1}.
    ``truth_counts`` must list *every* entity in the evaluation scope with its
    ``n_true`` -- including entities that produced no candidate at all, and
    entities whose true matches blocking never retrieved. Both still contribute
    their score to the macro average, so omitting them would inflate the result.
    """
    agg = pred.group_by(id_col).agg(
        n_pred=pl.len().cast(pl.Int64),
        tp=pl.col(label_col).sum().cast(pl.Int64))
    j = (truth_counts.select(id_col, "n_true")
         .join(agg, on=id_col, how="left").fill_null(0))
    tp = j["tp"].to_numpy()
    npd = j["n_pred"].to_numpy()
    ntr = j["n_true"].to_numpy()
    f = fbeta_from_counts(tp, npd, ntr)

    tot_tp, tot_pred, tot_true = int(tp.sum()), int(npd.sum()), int(ntr.sum())
    single = ntr == 0
    empty = npd == 0
    return {
        "macro_f05": float(f.mean()),
        "n_entities": int(j.height),
        "micro_precision": (tot_tp / tot_pred) if tot_pred else 0.0,
        "micro_recall": (tot_tp / tot_true) if tot_true else 0.0,
        "mean_pred_per_entity": float(npd.mean()),
        "n_pred_pairs": tot_pred, "n_tp_pairs": tot_tp, "n_true_pairs": tot_true,
        "false_positive_pairs": tot_pred - tot_tp,
        "singletons_true": int(single.sum()),
        "singleton_accuracy": float(empty[single].mean()) if single.any() else None,
        "macro_f05_singletons": float(f[single].mean()) if single.any() else None,
        "macro_f05_nonsingletons": float(f[~single].mean()) if (~single).any() else None,
        "nonsingleton_predicted_empty_pct": float(empty[~single].mean() * 100)
                                            if (~single).any() else None,
    }
