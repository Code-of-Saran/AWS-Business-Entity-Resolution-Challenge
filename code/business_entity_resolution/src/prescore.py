"""Stage 2: cheap pre-scoring, blocking output -> top-K candidates per S1 entity.

Broad blocking yields ~120 candidates per Source-1 entity; the expensive pairwise
feature stage cannot afford that over 1.7M test entities. This stage ranks
candidates using only set statistics over integer token-id lists, which Polars
evaluates without any Python loop or string comparison.

The rarity trick: token ids are assigned in ascending document-frequency order,
so "is this shared token rare?" is a single integer comparison against a
threshold id. That gives IDF-aware overlap counts at the cost of a comparison,
with no per-token frequency lookup.

A small gradient-boosted ranker is trained on these features (see
train_prescore_ranker) because hand-tuned weights measurably lost recall@K to a
learned combination.
"""
from __future__ import annotations

import polars as pl

# Document-frequency levels at which a shared token counts as "rare" / "mid".
DF_LEVELS = (5, 50, 1000)

CHEAP_FEATURES = [
    "n_keys", "fam_mask",
    "at_inter", "at_jac", "at_cont", "at_r0", "at_r1", "at_r2", "at_minid",
    "at_alen", "at_blen",
    "mt_inter", "mt_jac", "mt_cont", "mt_r0", "mt_r1", "mt_r2", "mt_minid",
    "mt_alen", "mt_blen",
    "an_inter", "an_jac", "an_cont", "an_r0", "an_r1", "an_minid",
    "an_alen", "an_blen",
]


def rarity_thresholds(vocab: pl.DataFrame,
                      levels: tuple = DF_LEVELS) -> list[int]:
    """Token-id cut-offs: ids below ``out[i]`` have document frequency <= levels[i].

    Valid because ``build_vocab`` sorts the vocabulary by ascending df before
    assigning ids.
    """
    return [int((vocab["df"] <= lv).sum()) for lv in levels]


def _set_feats(prefix: str, a: str, b: str, thr: list[int],
               n_rare: int = 3) -> list[pl.Expr]:
    inter = pl.col(a).list.set_intersection(pl.col(b))
    ni = inter.list.len()
    la, lb = pl.col(a).list.len(), pl.col(b).list.len()
    union = la + lb - ni
    out = [
        ni.cast(pl.Int32).alias(f"{prefix}_inter"),
        pl.when(union > 0).then(ni / union).otherwise(0.0).alias(f"{prefix}_jac"),
        pl.when(pl.min_horizontal(la, lb) > 0)
          .then(ni / pl.min_horizontal(la, lb)).otherwise(0.0).alias(f"{prefix}_cont"),
        la.cast(pl.Int32).alias(f"{prefix}_alen"),
        lb.cast(pl.Int32).alias(f"{prefix}_blen"),
        # Rarest shared token; a large sentinel when nothing is shared.
        inter.list.min().fill_null(2 ** 30).log1p().alias(f"{prefix}_minid"),
    ]
    for i in range(n_rare):
        out.append(inter.list.eval(pl.element() < thr[i]).list.sum()
                   .cast(pl.Int32).alias(f"{prefix}_r{i}"))
    return out


def attach_cheap_features(cand: pl.DataFrame, a: pl.DataFrame, b: pl.DataFrame,
                          thr_at: list[int], thr_an: list[int],
                          thr_mt: list[int]) -> pl.DataFrame:
    """Join token-id lists onto candidate pairs and compute the cheap features."""
    acols = a.select(a_idx="idx", a_at="at_all", a_an="an_all", a_mt="mt_all")
    bcols = b.select(b_idx="idx", b_at="at_all", b_an="an_all", b_mt="mt_all")
    c = cand.join(acols, on="a_idx", how="inner").join(bcols, on="b_idx", how="inner")
    c = c.with_columns(
        *_set_feats("at", "a_at", "b_at", thr_at),
        *_set_feats("mt", "a_mt", "b_mt", thr_mt),
        *_set_feats("an", "a_an", "b_an", thr_an, n_rare=2),
    )
    return c.drop(["a_at", "a_an", "a_mt", "b_at", "b_an", "b_mt"])


def top_k(scored: pl.DataFrame, k: int, score_col: str = "cheap_score",
          ) -> pl.DataFrame:
    """Keep the k highest-scoring candidates per S1 entity (deterministic ties)."""
    return (scored.sort(["a_idx", score_col, "b_idx"],
                        descending=[False, True, False])
                  .group_by("a_idx", maintain_order=True).head(k))


def heuristic_score() -> pl.Expr:
    """Fallback ranking used before the learned ranker is available."""
    return (3.0 * pl.col("at_cont") + 2.5 * pl.col("mt_cont")
            + 1.0 * pl.col("an_cont") + 0.10 * pl.col("n_keys")
            + 0.6 * pl.col("at_r0") + 0.5 * pl.col("mt_r0")
            + 0.25 * pl.col("at_r1") + 0.2 * pl.col("mt_r1"))
