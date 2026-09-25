"""Candidate generation (blocking).

Design
------
Comparing 1.7M Source-1 entities against ~10M Source-2/3 records exhaustively is
1.7e13 pairs, so candidates come from a multi-family inverted index.

The central trick: every token is assigned an integer id in ascending
document-frequency order, so *smaller id == rarer token*. Selecting a record's
k rarest tokens is then just ``list.sort().list.head(k)`` on a u32 list — no
per-token frequency lookup, no 50M-row sort. Blocking keys are built from those
rare-token ids and packed into a single u64 so the whole index is one hash join.

Key families (each contributes several keys per record):

======  =========================================================  ==============
family  key                                                        targets
======  =========================================================  ==============
AA      pairs among the 4 rarest address tokens                    shared street/city words
NA      3 rarest address numbers x 2 rarest address tokens         house number + street
MA      2 rarest name tokens x 2 rarest address tokens             name + location
MM      pairs among the 3 rarest name tokens                       distinctive names
MN      2 rarest name tokens x rarest address number               name + house number
NK      hash of the order-invariant normalised name key            exact name match
AK      hash of the full sorted address signature                  exact address match
======  =========================================================  ==============

Requiring a *pair* of rare tokens (rather than a single shared token) keeps block
sizes small without the enormous intermediate a single-token join would produce.
Blocks larger than ``max_block`` are dropped: an over-large block is by
definition uninformative, and keeping it would dominate the candidate budget.

Nothing here branches on a country value. Partitioning is done by whatever
distinct country strings appear in the data, so an unseen country is handled
exactly like a seen one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

# Bit layout of a packed key: [fam:6][a:29][b:29]
_ID_BITS = 29
_ID_SPAN = 1 << _ID_BITS          # ids must stay below this
_FAM_SPAN = 1 << (2 * _ID_BITS)   # stride between family blocks

FAM_AA, FAM_NA, FAM_MA, FAM_MM, FAM_MN, FAM_NK, FAM_AK, FAM_NS = range(8)
_FAM_BITS = {f: 1 << f for f in range(8)}
FAMILY_NAMES = {FAM_AA: "AA", FAM_NA: "NA", FAM_MA: "MA", FAM_MM: "MM",
                FAM_MN: "MN", FAM_NK: "NK", FAM_AK: "AK", FAM_NS: "NS"}


@dataclass
class BlockConfig:
    """Tunable blocking budget. Defaults are the validated configuration."""
    n_addr_tok: int = 4       # rarest address tokens kept
    n_addr_num: int = 3       # rarest address numbers kept
    n_name_tok: int = 3       # rarest name tokens kept
    max_block: int = 100      # default cap on a key's Source-2/3 block size
    # High-precision families (an exact normalised name or a full address
    # signature) earn a much larger cap: a big block there is still informative,
    # whereas a big rare-token-pair block is not.
    max_block_by_family: dict = field(default_factory=lambda: {
        FAM_NK: 3000, FAM_AK: 3000, FAM_NS: 3000})
    families: tuple = (FAM_AA, FAM_NA, FAM_MA, FAM_MM, FAM_MN, FAM_NK,
                       FAM_AK, FAM_NS)
    max_candidates: int = 0   # 0 = unlimited; else keep top-N per S1 by key hits
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #
def build_vocab(frames: list[pl.DataFrame], col: str) -> pl.DataFrame:
    """Token -> id, ids ordered so that a smaller id means a rarer token."""
    parts = [f.select(pl.col(col).alias("tok")).explode("tok") for f in frames]
    vocab = (pl.concat(parts)
             .drop_nulls("tok")
             .group_by("tok").len().rename({"len": "df"})
             .sort(["df", "tok"])
             .with_row_index("tid"))
    return vocab.with_columns(pl.col("tid").cast(pl.UInt32))


def map_token_ids(df: pl.DataFrame, col: str, vocab: pl.DataFrame,
                  out: str) -> pl.DataFrame:
    """Replace a list-of-strings column with the matching sorted list of ids."""
    ex = (df.select("idx", pl.col(col).alias("tok")).explode("tok")
            .drop_nulls("tok")
            .join(vocab.select("tok", "tid"), on="tok", how="inner")
            .group_by("idx").agg(pl.col("tid").sort().alias(out)))
    return df.join(ex, on="idx", how="left").with_columns(
        pl.col(out).fill_null(pl.lit([], dtype=pl.List(pl.UInt32))))


# --------------------------------------------------------------------------- #
# key construction
# --------------------------------------------------------------------------- #
def _slot(col: str, i: int) -> pl.Expr:
    return pl.col(col).list.get(i, null_on_oob=True).cast(pl.UInt64)


def _pack(fam: int, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Pack an unordered id pair. Polars Exprs have no shift/or, so the bit
    layout [fam:6][hi:29][lo:29] is expressed as arithmetic."""
    lo = pl.min_horizontal(a, b) % _ID_SPAN
    hi = pl.max_horizontal(a, b) % _ID_SPAN
    return (pl.lit(fam, dtype=pl.UInt64) * _FAM_SPAN + hi * _ID_SPAN + lo)


def _pack_ordered(fam: int, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Packing for cross-vocabulary pairs, where the two ids are not comparable."""
    return (pl.lit(fam, dtype=pl.UInt64) * _FAM_SPAN
            + (a % _ID_SPAN) * _ID_SPAN + (b % _ID_SPAN))


def _hash_key(fam: int, expr: pl.Expr) -> pl.Expr:
    return (pl.lit(fam, dtype=pl.UInt64) * _FAM_SPAN
            + expr.hash(seed=0x5EED) % _FAM_SPAN)


def make_keys(df: pl.DataFrame, cfg: BlockConfig) -> pl.DataFrame:
    """Return a long (idx, key, fam) frame: every blocking key of every record."""
    exprs: list[pl.Expr] = []
    na, nn, nm = cfg.n_addr_tok, cfg.n_addr_num, cfg.n_name_tok

    if FAM_AA in cfg.families:
        for i in range(na):
            for j in range(i + 1, na):
                exprs.append(_pack(FAM_AA, _slot("at", i), _slot("at", j))
                             .alias(f"k_aa_{i}{j}"))
    if FAM_NA in cfg.families:
        for i in range(nn):
            for j in range(min(2, na)):
                exprs.append(_pack_ordered(FAM_NA, _slot("an", i), _slot("at", j))
                             .alias(f"k_na_{i}{j}"))
    if FAM_MA in cfg.families:
        for i in range(min(2, nm)):
            for j in range(min(2, na)):
                exprs.append(_pack_ordered(FAM_MA, _slot("mt", i), _slot("at", j))
                             .alias(f"k_ma_{i}{j}"))
    if FAM_MM in cfg.families:
        for i in range(nm):
            for j in range(i + 1, nm):
                exprs.append(_pack(FAM_MM, _slot("mt", i), _slot("mt", j))
                             .alias(f"k_mm_{i}{j}"))
    if FAM_MN in cfg.families:
        for i in range(min(2, nm)):
            exprs.append(_pack_ordered(FAM_MN, _slot("mt", i), _slot("an", 0))
                         .alias(f"k_mn_{i}"))
    if FAM_NK in cfg.families:
        exprs.append(pl.when(pl.col("name_key").str.len_chars() >= 3)
                     .then(_hash_key(FAM_NK, pl.col("name_key")))
                     .alias("k_nk"))
    if FAM_NS in cfg.families:
        exprs.append(pl.when(pl.col("name_core_nospace").str.len_chars() >= 6)
                     .then(_hash_key(FAM_NS, pl.col("name_core_nospace")))
                     .alias("k_ns"))
    if FAM_AK in cfg.families:
        # Full order-invariant address signature (not just the rarest slots).
        sig = (pl.col("addr_toks").list.sort().list.join(" ") + pl.lit(" # ")
               + pl.col("addr_nums").list.sort().list.join(" "))
        exprs.append(pl.when(pl.col("addr_toks").list.len() >= 2)
                     .then(_hash_key(FAM_AK, sig))
                     .alias("k_ak"))

    wide = df.select(pl.col("idx"), *exprs)
    keys = (wide.unpivot(index="idx", variable_name="src", value_name="key")
                .drop_nulls("key"))
    keys = keys.with_columns(
        fam=(pl.col("key") // _FAM_SPAN).cast(pl.UInt8)).drop("src")
    return keys.unique(subset=["idx", "key"])


# --------------------------------------------------------------------------- #
# candidate generation
# --------------------------------------------------------------------------- #
def prepare_side(df: pl.DataFrame, vocab_at: pl.DataFrame, vocab_an: pl.DataFrame,
                 vocab_mt: pl.DataFrame, cfg: BlockConfig) -> pl.DataFrame:
    """Attach rarest-k token-id slots (``at``/``an``/``mt``) to one side."""
    df = map_token_ids(df, "addr_toks", vocab_at, "at_all")
    df = map_token_ids(df, "addr_nums", vocab_an, "an_all")
    df = map_token_ids(df, "name_toks", vocab_mt, "mt_all")
    # The full id lists are kept (not just the rarest-k slots) because the cheap
    # pre-scoring stage computes set overlaps on them, and integer set ops are far
    # cheaper than the equivalent on string lists.
    return df.with_columns(
        at=pl.col("at_all").list.head(cfg.n_addr_tok),
        an=pl.col("an_all").list.head(cfg.n_addr_num),
        mt=pl.col("mt_all").list.head(cfg.n_name_tok),
    )


def build_index_keys(b: pl.DataFrame, cfg: BlockConfig) -> pl.DataFrame:
    """Build the Source-2/3 inverted index once, dropping over-large blocks.

    Separated from probing so a country's index is built a single time and then
    probed with successive Source-1 chunks -- the memory shape that makes the full
    run fit: a chunked probe never materialises all ~150M candidate pairs of a
    large country at once.
    """
    kb = make_keys(b, cfg)
    caps = pl.DataFrame(
        {"fam": [f for f in cfg.families],
         "cap": [cfg.max_block_by_family.get(f, cfg.max_block) for f in cfg.families]},
        schema={"fam": pl.UInt8, "cap": pl.UInt32})
    sizes = kb.group_by(["key", "fam"]).len()
    good = (sizes.join(caps, on="fam", how="inner")
                 .filter(pl.col("len") <= pl.col("cap")).select("key"))
    return kb.join(good, on="key", how="inner")


def probe(a: pl.DataFrame, kb: pl.DataFrame, cfg: BlockConfig) -> pl.DataFrame:
    """Candidates for one Source-1 chunk against a prebuilt index.

    Returns (a_idx, b_idx, n_keys, fam_mask) where ``n_keys`` counts how many
    blocking keys the pair shares and ``fam_mask`` is a bitmask of the families
    that fired; both are reused as model features.
    """
    ka = make_keys(a, cfg)
    pairs = (ka.join(kb.select("key", b_idx="idx", fam_b="fam"), on="key",
                     how="inner")
               .select(a_idx="idx", b_idx="b_idx",
                       bit=pl.col("fam_b").cast(pl.UInt32)
                           .replace_strict(_FAM_BITS, return_dtype=pl.UInt32)))
    # Summing the *distinct* family bits is a bitwise OR, which Polars can do in
    # an aggregation whereas a shift-based mask cannot.
    out = (pairs.group_by(["a_idx", "b_idx"])
                .agg(n_keys=pl.len().cast(pl.Int32),
                     fam_mask=pl.col("bit").unique().sum()))
    if cfg.max_candidates:
        out = (out.sort(["a_idx", "n_keys"], descending=[False, True])
                  .group_by("a_idx", maintain_order=True)
                  .head(cfg.max_candidates))
    return out


def generate_candidates(a: pl.DataFrame, b: pl.DataFrame,
                        cfg: BlockConfig) -> pl.DataFrame:
    """Convenience wrapper: build the index and probe it in one call."""
    return probe(a, build_index_keys(b, cfg), cfg)
