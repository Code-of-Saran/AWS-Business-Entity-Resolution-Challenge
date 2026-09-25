"""Shared orchestration: per-country index, chunked probing, candidate funnel.

The funnel is identical for training and inference, which is what keeps the two
distributions matched:

    prepared records
      -> blocking index (Source 2 + Source 3 of one country)
      -> probe with a chunk of Source-1 entities        (~120 candidates / entity)
      -> cheap rarity-aware features
      -> learned cheap ranker -> top-K                  (K candidates / entity)
      -> expensive pairwise features
      -> matching model -> decision

Country partitioning is by whatever distinct country strings occur in the data.
Verified on the full training ground truth: a true match never crosses a country
boundary (7,638,365 / 7,638,365 pairs share the country string), so partitioning
costs no recall. An unseen country is simply another partition.

Source-1 entities are processed in chunks so that no stage ever materialises a
whole country's raw candidate set; the top-K survivors are the only thing that
accumulates.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import polars as pl

import blocking as bk
import data_loader as dl
import features as ft
import prescore as ps
from prepare import prepared_path

FOLD_SEED = 0xC0FFEE
N_FOLDS = 10
# Deterministic, reproducible entity-level split. Fold 0 trains the cheap ranker;
# folds 1-7 train the matching model; folds 8-9 are held out for validation.
FOLD_PRESCORE = (0,)
FOLDS_TRAIN = (1, 2, 3, 4, 5, 6, 7)
FOLDS_VALID = (8, 9)

TEXT_COLS = ["name_norm", "name_core", "name_key", "name_core_nospace",
             "addr_norm", "name_nonlatin", "addr_nonlatin"]


def fold_of(col: pl.Expr) -> pl.Expr:
    """Stable fold id for a Source-1 entity, from a hash of its entity_id."""
    return (col.hash(seed=FOLD_SEED) % N_FOLDS).cast(pl.UInt8)


def load_prepared(split: str, source: int, country: str | None = None,
                  columns: list[str] | None = None) -> pl.DataFrame:
    lf = pl.scan_parquet(prepared_path(split, source))
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    if columns is not None:
        lf = lf.select(columns)
    return lf.collect()


def countries(split: str) -> list[str]:
    """Distinct country strings present in a split's Source-1 file."""
    s = load_prepared(split, 1, columns=["country"])["country"]
    return sorted(s.unique().to_list())


@dataclass
class CountryIndex:
    """Everything needed to probe one country's Source-2/3 records."""
    country: str
    b: pl.DataFrame
    kb: pl.DataFrame
    vat: pl.DataFrame
    van: pl.DataFrame
    vmt: pl.DataFrame
    thr_at: list
    thr_an: list
    thr_mt: list
    cfg: bk.BlockConfig
    ctry_desc: dict


def build_country_index(split: str, country: str, cfg: bk.BlockConfig,
                        a_full: pl.DataFrame | None = None) -> CountryIndex:
    """Load a country's Source-2/3 records and build the blocking index.

    The vocabularies are fitted on Source 1 + Source 2 + Source 3 of this split
    and country, so token rarity reflects the corpus actually being matched.
    """
    if a_full is None:
        a_full = load_prepared(split, 1, country)
    b2 = load_prepared(split, 2, country)
    b3 = load_prepared(split, 3, country)
    b = pl.concat([b2.with_columns(is_s3=pl.lit(0, pl.Int8)),
                   b3.with_columns(is_s3=pl.lit(1, pl.Int8))])
    del b2, b3
    b = b.with_row_index("idx")

    vat = bk.build_vocab([a_full, b], "addr_toks")
    van = bk.build_vocab([a_full, b], "addr_nums")
    vmt = bk.build_vocab([a_full, b], "name_toks")
    b = bk.prepare_side(b, vat, van, vmt, cfg)
    kb = bk.build_index_keys(b, cfg)
    return CountryIndex(
        country=country, b=b, kb=kb, vat=vat, van=van, vmt=vmt,
        thr_at=ps.rarity_thresholds(vat), thr_an=ps.rarity_thresholds(van),
        thr_mt=ps.rarity_thresholds(vmt), cfg=cfg,
        ctry_desc=ft.country_descriptors([b]))


def prepare_a_chunk(a_chunk: pl.DataFrame, ix: CountryIndex) -> pl.DataFrame:
    return bk.prepare_side(a_chunk, ix.vat, ix.van, ix.vmt, ix.cfg)


def candidates_for_chunk(a_chunk: pl.DataFrame, ix: CountryIndex,
                         ranker, k: int) -> pl.DataFrame:
    """Blocking + cheap features + cheap ranker + top-K for one Source-1 chunk."""
    a = prepare_a_chunk(a_chunk, ix)
    cand = bk.probe(a, ix.kb, ix.cfg)
    if cand.height == 0:
        return cand
    sc = ps.attach_cheap_features(cand, a, ix.b, ix.thr_at, ix.thr_an, ix.thr_mt)
    if ranker is None:
        sc = sc.with_columns(cheap_score=ps.heuristic_score())
    else:
        X = sc.select(ps.CHEAP_FEATURES).to_numpy().astype(np.float32)
        sc = sc.with_columns(
            cheap_score=pl.Series("cheap_score", ranker.predict_proba(X)[:, 1]))
    return ps.top_k(sc, k)


def attach_text(cand: pl.DataFrame, a: pl.DataFrame,
                ix: CountryIndex) -> pl.DataFrame:
    """Join the text columns both sides need for the expensive features."""
    acols = a.select(["idx"] + TEXT_COLS + ["entity_id", "country"]).rename(
        {c: "a_" + c for c in TEXT_COLS} | {"idx": "a_idx",
                                            "entity_id": "a_entity_id"})
    bcols = ix.b.select(["idx"] + TEXT_COLS + ["entity_id", "is_s3"]).rename(
        {c: "b_" + c for c in TEXT_COLS} | {"idx": "b_idx",
                                           "entity_id": "b_entity_id"})
    return cand.join(acols, on="a_idx", how="inner").join(bcols, on="b_idx",
                                                          how="inner")


def chunks(df: pl.DataFrame, size: int):
    """Yield successive row-slices of a frame."""
    n = df.height
    for start in range(0, n, size):
        yield df.slice(start, size)


def truth_pairs(a_with_idx: pl.DataFrame, b: pl.DataFrame,
                gt: pl.DataFrame | None = None) -> pl.DataFrame:
    """True (a_idx, b_idx) pairs restricted to the given Source-1 entities."""
    if gt is None:
        gt = dl.read_ground_truth()
    return (gt.join(a_with_idx.select("idx", "entity_id"),
                    left_on="source1_entity_id", right_on="entity_id",
                    how="inner")
              .select("idx", pl.col("matched_entity_ids").str.split(",").alias("m"))
              .explode("m").drop_nulls("m").filter(pl.col("m") != "")
              .join(b.select(b_idx="idx", m="entity_id"), on="m", how="inner")
              .select(a_idx="idx", b_idx="b_idx"))


def truth_counts(a_with_idx: pl.DataFrame,
                 gt: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-entity number of true matches, independent of what blocking found.

    Needed because an entity whose true matches all fall outside the index still
    contributes its (zero) score to the macro average.
    """
    if gt is None:
        gt = dl.read_ground_truth()
    j = gt.join(a_with_idx.select("idx", "entity_id"),
                left_on="source1_entity_id", right_on="entity_id", how="inner")
    # entity_id is returned explicitly: a join does not preserve the caller's row
    # order, so attaching ids positionally afterwards silently misaligns counts
    # against entities while leaving their distribution unchanged -- invisible to
    # any aggregate sanity check.
    return j.select(
        a_idx="idx",
        entity_id=pl.col("source1_entity_id"),
        n_true=pl.when(pl.col("matched_entity_ids").str.strip_chars() == "")
                 .then(0)
                 .otherwise(pl.col("matched_entity_ids").str.split(",").list.len())
                 .cast(pl.Int32))
