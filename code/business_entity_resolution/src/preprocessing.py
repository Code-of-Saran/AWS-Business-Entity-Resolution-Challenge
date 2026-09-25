"""Text normalisation for business names and addresses.

Everything here is vectorised with Polars expressions so it runs over the full
~26M-row corpus in minutes rather than hours. The design is deliberately
language- and country-agnostic: no branch anywhere keys off a specific country
value, so an unseen country (e.g. France in the test set) flows through the same
code path as the training countries.

Observed noise the normalisation targets (all verified in the training data):

name
    legal-suffix churn (LLC / L.L.C. / Limited / Ltd / Pvt / Private / SARL ...),
    punctuation injection ("Federal-Construction"), bracketing ("[Inc.]"),
    junk prefixes ("--", "<<", "##", "@"), alias markers
    ("X a/k/a Y", "X f/k/a Y", "X DBA: Y"), domain forms ("acme.com"),
    accents, token transposition, typos, and Indic-script transliteration.

address
    component reordering, street-type abbreviation (Rd/Road, St/Street),
    state name vs. postal abbreviation vs. native script, literal "null" /
    "<NULL>" placeholders, inserted components (PO Box, 1/2, Hn 835-, ##),
    house-number ranges (512-516), and typos.
"""
from __future__ import annotations

import polars as pl

# Combining diacritical marks left behind by NFKD decomposition.
_COMBINING = r"[̀-ͯٓ-ٰٕ]"

# Alias markers: the *real* business name follows the marker in every observed
# case ("Wexdeltadova a/k/a Federal Construction Associates LLC").
_ALIAS_SPLIT = r"(?:\ba/?k/?a\b|\bf/?k/?a\b|\bd/?b/?a\b|\bformerly known as\b|\bnow known as\b|\btrading as\b|\bt/a\b)"

# Legal-form tokens. Kept deliberately broad and multi-lingual rather than
# country-gated; unknown suffixes simply survive normalisation unchanged.
LEGAL_TOKENS = {
    "llc", "l l c", "inc", "incorporated", "corp", "corporation", "co",
    "company", "ltd", "limited", "pvt", "pvt.", "private", "llp", "l l p",
    "lp", "plc", "pllc", "gmbh", "sarl", "sas", "sa", "sasu", "sci", "snc",
    "eurl", "scp", "selarl", "bv", "nv", "ag", "kg", "ohg", "oy", "ab", "as",
    "spa", "srl", "sl", "pte", "bhd", "sdn", "opc", "huf", "trust", "society",
    "association", "assn", "and", "the", "of",
}

# Tokens that carry no discriminative signal in an address.
ADDRESS_STOP = {
    "null", "none", "na", "n a", "nil", "po", "box", "po box", "c o", "co",
    "near", "opp", "opposite", "behind", "beside", "no", "nos", "number",
    "unit", "apt", "apartment", "suite", "ste", "fl", "floor", "bldg",
    "building", "block", "phase", "sector", "plot", "hn", "h no", "door",
    "the", "of", "at", "de", "du", "la", "le", "les", "des", "rue", "r",
    "road", "rd", "street", "st", "avenue", "ave", "av", "drive", "dr",
    "lane", "ln", "boulevard", "blvd", "bd", "court", "ct", "circle", "cir",
    "highway", "hwy", "place", "pl", "square", "sq", "terrace", "ter",
    "trail", "trl", "way", "parkway", "pkwy", "colony", "nagar", "marg",
}


def _strip_accents(col: pl.Expr) -> pl.Expr:
    return col.str.normalize("NFKD").str.replace_all(_COMBINING, "")


def normalize_name(col: pl.Expr) -> pl.Expr:
    """Full normalised name: accent-folded, de-aliased, punctuation-free."""
    c = _strip_accents(col).str.to_lowercase()
    # Keep only the segment after an alias marker when one is present.
    c = c.str.replace_all(r"^.*?" + _ALIAS_SPLIT + r"\s*:?\s*", "")
    # Domain / handle forms: "acme.com" -> "acme", "@acme - 12345" -> "acme".
    c = c.str.replace_all(r"\bwww\.", "")
    c = c.str.replace_all(r"\.(?:com|net|org|co|in|fr|io|biz|info)\b", " ")
    c = c.str.replace_all(r"&", " and ")
    c = c.str.replace_all(r"[^0-9a-zऀ-෿਀-૿ ]+", " ")
    return c.str.replace_all(r"\s+", " ").str.strip_chars()


def normalize_address(col: pl.Expr) -> pl.Expr:
    """Full normalised address: accent-folded, placeholder- and punctuation-free."""
    c = _strip_accents(col).str.to_lowercase()
    c = c.str.replace_all(r"<\s*null\s*>", " ")
    c = c.str.replace_all(r"&", " and ")
    c = c.str.replace_all(r"[^0-9a-zऀ-෿਀-૿ ]+", " ")
    c = c.str.replace_all(r"\b(?:null|none|nil)\b", " ")
    return c.str.replace_all(r"\s+", " ").str.strip_chars()


def name_content_tokens(col: pl.Expr) -> pl.Expr:
    """Content tokens of a normalised name, in original order (duplicates kept)."""
    return (col.str.split(" ")
              .list.eval(pl.element().filter(
                  ~pl.element().is_in(list(LEGAL_TOKENS)) & (pl.element() != ""))))


def name_tokens(col: pl.Expr) -> pl.Expr:
    """Deduplicated content tokens of a normalised name, for blocking keys."""
    return name_content_tokens(col).list.unique()


def address_tokens(col: pl.Expr) -> pl.Expr:
    """Discriminative alphabetic tokens of a normalised address."""
    return (col.str.split(" ")
              .list.eval(pl.element().filter(
                  ~pl.element().is_in(list(ADDRESS_STOP))
                  & (pl.element().str.len_chars() >= 2)
                  & (~pl.element().str.contains(r"^[0-9]+$"))))
              .list.unique())


def address_numbers(col: pl.Expr) -> pl.Expr:
    """Numeric components of an address, leading zeros stripped, deduplicated.

    House-number ranges ("512-516") and fractional forms ("764 1/2") are split by
    the punctuation rule above, so both halves are retained; matching on *any*
    shared number is what the blocking stage needs.
    """
    return (col.str.extract_all(r"[0-9]+")
              .list.eval(pl.element().str.strip_chars_start("0")
                         .filter(pl.element() != ""))
              .list.unique())


def add_normalized_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Attach every normalised representation used downstream."""
    df = df.with_columns(
        name_norm=normalize_name(pl.col("business_name")),
        addr_norm=normalize_address(pl.col("business_address")),
    )
    df = df.with_columns(
        name_toks=name_tokens(pl.col("name_norm")),
        addr_toks=address_tokens(pl.col("addr_norm")),
        addr_nums=address_numbers(pl.col("addr_norm")),
    )
    # Order-invariant name key: content tokens, deduplicated and sorted.
    # name_core keeps word order (needed for prefix/suffix and sequence-based
    # similarity); name_core_nospace collapses spaces so that a concatenated
    # domain form ("mumbaicure.com") lands on the same key as the spaced name
    # ("Mumbai Cure") -- a miss category confirmed in the training data.
    df = df.with_columns(
        name_core=name_content_tokens(pl.col("name_norm")).list.join(" "),
        name_key=pl.col("name_toks").list.sort().list.join(" "),
    )
    df = df.with_columns(
        name_core_nospace=pl.col("name_core").str.replace_all(" ", ""),
        name_nospace=pl.col("name_norm").str.replace_all(" ", ""),
        # Source 1 is always Latin script; Source 2/3 names are transliterated
        # into an Indic script in ~6-9% of records. Flagging that lets the model
        # learn to fall back on address evidence instead of an unusable name
        # comparison, without any rule keyed to a country.
        name_nonlatin=pl.col("business_name").str.contains(r"[^\x00-\x7f]"),
        addr_nonlatin=pl.col("business_address").str.contains(r"[^\x00-\x7f]"),
    )
    return df
