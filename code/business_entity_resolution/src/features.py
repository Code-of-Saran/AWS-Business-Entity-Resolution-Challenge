"""Stage 3: pairwise feature engineering for surviving candidate pairs.

Three groups of features are produced.

1. String-similarity features, computed with rapidfuzz ``process.cpdist`` over
   whole columns at once (multi-threaded, ~0.05 us/pair) rather than per-pair
   Python calls. Several normalised views of the name are compared because the
   noise generator attacks them differently: ``name_core`` (legal suffixes
   removed) for ordinary typos and word order, ``name_key`` (sorted tokens) for
   transposition, ``name_core_nospace`` for concatenated domain forms
   ("mumbaicure.com" vs "Mumbai Cure"), and ``name_norm`` for suffix evidence.

2. Set-overlap features carried over from the cheap pre-scoring stage, including
   the rarity-bucketed shared-token counts.

3. Group-context features, added after scoring: a candidate's rank and margin
   within its Source-1 entity, and the competition on the Source-2/3 side. The
   latter exploits a structural property verified in the training ground truth --
   every matched Source-2/3 record belongs to exactly one Source-1 entity -- so a
   record that several Source-1 entities want is evidence against all but the
   best of them.

No feature encodes a country identity. Country-level adaptation is supplied only
through unsupervised descriptors (index size, non-ASCII rate, address length)
that are defined for any country string, so an unseen country such as France is
handled by the same model without a special case.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein, Prefix

import prescore as ps

# (output name, a column, b column, scorer, scale)
_STRING_FEATURES = [
    ("nm_ratio",        "a_name_core", "b_name_core", fuzz.ratio, 0.01),
    ("nm_tokset",       "a_name_core", "b_name_core", fuzz.token_set_ratio, 0.01),
    ("nm_toksort",      "a_name_core", "b_name_core", fuzz.token_sort_ratio, 0.01),
    ("nm_partial",      "a_name_core", "b_name_core", fuzz.partial_ratio, 0.01),
    ("nm_jw",           "a_name_core", "b_name_core", JaroWinkler.normalized_similarity, 1.0),
    ("nm_lev",          "a_name_core", "b_name_core", Levenshtein.normalized_similarity, 1.0),
    ("nm_prefix",       "a_name_core", "b_name_core", Prefix.normalized_similarity, 1.0),
    ("nmkey_ratio",     "a_name_key", "b_name_key", fuzz.ratio, 0.01),
    ("nmkey_jw",        "a_name_key", "b_name_key", JaroWinkler.normalized_similarity, 1.0),
    ("nmns_ratio",      "a_name_core_nospace", "b_name_core_nospace", fuzz.ratio, 0.01),
    ("nmns_partial",    "a_name_core_nospace", "b_name_core_nospace", fuzz.partial_ratio, 0.01),
    ("nmns_jw",         "a_name_core_nospace", "b_name_core_nospace", JaroWinkler.normalized_similarity, 1.0),
    ("nmfull_ratio",    "a_name_norm", "b_name_norm", fuzz.ratio, 0.01),
    ("nmfull_tokset",   "a_name_norm", "b_name_norm", fuzz.token_set_ratio, 0.01),
    ("ad_ratio",        "a_addr_norm", "b_addr_norm", fuzz.ratio, 0.01),
    ("ad_tokset",       "a_addr_norm", "b_addr_norm", fuzz.token_set_ratio, 0.01),
    ("ad_toksort",      "a_addr_norm", "b_addr_norm", fuzz.token_sort_ratio, 0.01),
    ("ad_partial",      "a_addr_norm", "b_addr_norm", fuzz.partial_ratio, 0.01),
    ("ad_jw",           "a_addr_norm", "b_addr_norm", JaroWinkler.normalized_similarity, 1.0),
    ("ad_lev",          "a_addr_norm", "b_addr_norm", Levenshtein.normalized_similarity, 1.0),
]

_DERIVED = [
    "nm_exact", "nmkey_exact", "nmns_exact", "ad_exact", "both_exact",
    "nm_len_a", "nm_len_b", "nm_len_diff", "nm_len_ratio",
    "ad_len_a", "ad_len_b", "ad_len_diff", "ad_len_ratio",
    "b_addr_empty", "b_name_nonlatin", "b_addr_nonlatin", "is_s3",
    "nm_ad_prod", "nm_ad_max", "nm_ad_min", "nm_ad_wsum",
    "strong_nm_weak_ad", "weak_nm_strong_ad",
    "ctry_index_log", "ctry_nonlatin_rate", "ctry_addr_tok_mean",
]

# Entity-level context features. Derived only from the cheap ranker's score and
# from within-pair similarities, never from the matching model's own output, so
# they carry no leakage: the cheap ranker is fitted on Source-1 fold 0 only, while
# the matcher trains on folds 1-7 and is validated on folds 8-9.
#
# Deliberately a-side only. A b-side "how many entities want this record" feature
# would be well defined at full inference but not on a sampled training set, where
# competitor density depends on the sampling rate; that structural constraint is
# instead applied in the decision layer, where it can be enforced globally and
# identically for training-validation and test.
PRESCORE_GROUP_FEATURES = [
    "g_cheap", "g_n_cands", "g_rank", "g_cheap_best", "g_cheap_margin",
    "g_cheap_gap2", "g_cheap_share", "g_nm_best", "g_nm_margin",
    "g_ad_best", "g_ad_margin",
]

GROUP_FEATURES = [
    "grp_n_cands", "grp_rank", "grp_margin_best", "grp_best_score",
    "grp_nm_rel", "grp_ad_rel",
    "bside_n_competitors", "bside_rank", "bside_margin_best",
]

BASE_FEATURE_NAMES = ([n for n, *_ in _STRING_FEATURES] + _DERIVED
                      + ps.CHEAP_FEATURES)
FEATURE_NAMES = BASE_FEATURE_NAMES + PRESCORE_GROUP_FEATURES


def string_feature_block(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Run every rapidfuzz metric over the whole chunk at once."""
    cache: dict[str, list] = {}

    def col(name: str) -> list:
        if name not in cache:
            cache[name] = df[name].fill_null("").to_list()
        return cache[name]

    out: dict[str, np.ndarray] = {}
    for name, ca, cb, scorer, scale in _STRING_FEATURES:
        v = process.cpdist(col(ca), col(cb), scorer=scorer, workers=-1)
        out[name] = (v.astype(np.float32) * np.float32(scale))
    return out


def derived_feature_block(df: pl.DataFrame,
                          ctry: dict[str, tuple[float, float, float]],
                          sf: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Exact-match flags, length ratios, interactions and country descriptors."""
    f32 = np.float32

    def s(name: str) -> np.ndarray:
        return df[name].to_numpy()

    nm_a = df["a_name_core"].fill_null("").str.len_chars().to_numpy().astype(f32)
    nm_b = df["b_name_core"].fill_null("").str.len_chars().to_numpy().astype(f32)
    ad_a = df["a_addr_norm"].fill_null("").str.len_chars().to_numpy().astype(f32)
    ad_b = df["b_addr_norm"].fill_null("").str.len_chars().to_numpy().astype(f32)

    out: dict[str, np.ndarray] = {}
    out["nm_exact"] = (df["a_name_norm"] == df["b_name_norm"]).to_numpy().astype(f32)
    out["nmkey_exact"] = (df["a_name_key"] == df["b_name_key"]).to_numpy().astype(f32)
    out["nmns_exact"] = (df["a_name_core_nospace"]
                         == df["b_name_core_nospace"]).to_numpy().astype(f32)
    out["ad_exact"] = (df["a_addr_norm"] == df["b_addr_norm"]).to_numpy().astype(f32)
    out["both_exact"] = out["nm_exact"] * out["ad_exact"]

    out["nm_len_a"], out["nm_len_b"] = nm_a, nm_b
    out["nm_len_diff"] = np.abs(nm_a - nm_b)
    out["nm_len_ratio"] = np.minimum(nm_a, nm_b) / np.maximum(np.maximum(nm_a, nm_b), 1.0)
    out["ad_len_a"], out["ad_len_b"] = ad_a, ad_b
    out["ad_len_diff"] = np.abs(ad_a - ad_b)
    out["ad_len_ratio"] = np.minimum(ad_a, ad_b) / np.maximum(np.maximum(ad_a, ad_b), 1.0)

    out["b_addr_empty"] = (ad_b < 1).astype(f32)
    out["b_name_nonlatin"] = s("b_name_nonlatin").astype(f32)
    out["b_addr_nonlatin"] = s("b_addr_nonlatin").astype(f32)
    out["is_s3"] = s("is_s3").astype(f32)

    nm = np.maximum(sf["nm_tokset"], sf["nmns_ratio"])
    ad = np.maximum(sf["ad_tokset"], sf["ad_ratio"])
    out["nm_ad_prod"] = nm * ad
    out["nm_ad_max"] = np.maximum(nm, ad)
    out["nm_ad_min"] = np.minimum(nm, ad)
    out["nm_ad_wsum"] = f32(0.45) * nm + f32(0.55) * ad
    out["strong_nm_weak_ad"] = ((nm > 0.9) & (ad < 0.6)).astype(f32)
    out["weak_nm_strong_ad"] = ((nm < 0.6) & (ad > 0.9)).astype(f32)

    cvals = df["country"].to_list()
    desc = np.array([ctry.get(c, (0.0, 0.0, 0.0)) for c in cvals], dtype=f32)
    out["ctry_index_log"] = desc[:, 0]
    out["ctry_nonlatin_rate"] = desc[:, 1]
    out["ctry_addr_tok_mean"] = desc[:, 2]
    return out


def prescore_group_block(df: pl.DataFrame, nm: np.ndarray,
                         ad: np.ndarray) -> dict[str, np.ndarray]:
    """Entity-level context, computed over complete candidate groups.

    Safe because chunking slices *entities*: every candidate of an entity is in
    the same chunk, so each group is complete here.
    """
    t = df.select("a_idx", "cheap_score").with_columns(
        _nm=pl.Series("_nm", nm), _ad=pl.Series("_ad", ad))
    c = pl.col("cheap_score")
    t = t.with_columns(
        g_n_cands=pl.len().over("a_idx").cast(pl.Float32),
        g_rank=c.rank("ordinal", descending=True).over("a_idx").cast(pl.Float32),
        g_cheap_best=c.max().over("a_idx"),
        g_cheap_sum=c.sum().over("a_idx"),
        # Second-best score in the group: top_k(2).min() is the runner-up, and
        # collapses to the best itself for a single-candidate group (gap 0).
        g_second=c.top_k(2).min().over("a_idx"),
        g_nm_best=pl.col("_nm").max().over("a_idx"),
        g_ad_best=pl.col("_ad").max().over("a_idx"),
    )
    t = t.with_columns(
        g_cheap=c,
        g_cheap_margin=c - pl.col("g_cheap_best"),
        g_cheap_gap2=pl.col("g_cheap_best") - pl.col("g_second").fill_null(0.0),
        g_cheap_share=pl.when(pl.col("g_cheap_sum") > 0)
                        .then(c / pl.col("g_cheap_sum")).otherwise(0.0),
        g_nm_margin=pl.col("_nm") - pl.col("g_nm_best"),
        g_ad_margin=pl.col("_ad") - pl.col("g_ad_best"),
    )
    return {n: t[n].to_numpy().astype(np.float32) for n in PRESCORE_GROUP_FEATURES}


def build_feature_matrix(df: pl.DataFrame,
                         ctry: dict[str, tuple[float, float, float]],
                         ) -> np.ndarray:
    """Assemble the (n_pairs, n_features) float32 matrix for one chunk."""
    sf = string_feature_block(df)
    dv = derived_feature_block(df, ctry, sf)
    gp = prescore_group_block(df, dv["nm_ad_max"], sf["ad_tokset"])
    blocks = {**sf, **dv, **gp}
    cols = [blocks[n] if n in blocks
            else df[n].to_numpy().astype(np.float32) for n in FEATURE_NAMES]
    return np.ascontiguousarray(np.stack(cols, axis=1, dtype=np.float32))


def country_descriptors(b_frames: list[pl.DataFrame]) -> dict:
    """Unsupervised per-country descriptors, defined for any country string."""
    parts = [f.select("country", "name_nonlatin",
                      pl.col("addr_toks").list.len().alias("ntok"))
             for f in b_frames]
    g = (pl.concat(parts).group_by("country")
         .agg(n=pl.len(), nl=pl.col("name_nonlatin").mean(),
              tk=pl.col("ntok").mean()))
    return {r[0]: (float(np.log1p(r[1])), float(r[2]), float(r[3]))
            for r in g.iter_rows()}


def add_group_features(df: pl.DataFrame, score_col: str,
                       nm_col: str = "nm_ad_max", ad_col: str = "ad_tokset",
                       ) -> pl.DataFrame:
    """Rank / margin features within each Source-1 entity and each Source-2/3 record.

    ``bside_*`` encode the competition for a Source-2/3 record. Because a matched
    Source-2/3 record has exactly one true Source-1 owner (verified on the full
    training ground truth), a candidate that is not the strongest claim on its
    record is much more likely to be a false merge.
    """
    s_ = pl.col(score_col)
    return df.with_columns(
        grp_n_cands=pl.len().over("a_idx").cast(pl.Int32),
        grp_rank=s_.rank("ordinal", descending=True).over("a_idx").cast(pl.Int32),
        grp_best_score=s_.max().over("a_idx"),
        grp_margin_best=s_ - s_.max().over("a_idx"),
        grp_nm_rel=pl.col(nm_col) - pl.col(nm_col).max().over("a_idx"),
        grp_ad_rel=pl.col(ad_col) - pl.col(ad_col).max().over("a_idx"),
        bside_n_competitors=pl.len().over("b_idx").cast(pl.Int32),
        bside_rank=s_.rank("ordinal", descending=True).over("b_idx").cast(pl.Int32),
        bside_margin_best=s_ - s_.max().over("b_idx"),
    )
