"""Build the analysis-ready household table from IHDS-II (ICPSR 36151, DS0002).

Lazy Polars port of the original ``src/build_dataset.py``. See
``walkthrough/dataset_construction.md`` for the reasoning behind every mapping.

Facts this module encodes (verified against ``36151-0002-Codebook.pdf``):

* ``CO1X``-``CO33`` are 30-day recall; ``CO34``-``CO52`` are annual recall.
  ``12 * monthly + annual`` approximately reproduces IHDS's own ``COTOTAL``.
* ``CO4X`` is kerosene (a fuel), so it belongs in Utilities.
* ``CO18`` is paan/tobacco/intoxicants; eating out is ``CO20``.
* ``NADULT*`` counts members aged 21+ (elders included), ``NTEEN*`` 15-20,
  ``NCHILD*`` 0-14 and ``NELDER*`` 60+. ``Age_Dependents`` is the standard
  age-dependency count (0-14 plus 60+); 15-20-year-olds are working age.
* Missing values are blank strings in the TSV and are read as null.

Two totals exist and they are not the same thing. The target uses the survey's
own ``COTOTAL``; the composition shares use ``Category_Total``, the sum of our
11 categories. They agree to within 1% for most households (see
``reconcile_totals``) but not exactly, so "Savings is an identity over the
expense columns" is only approximately true.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from savings_goal.config import EXPENSE_CATEGORIES, THRESHOLD

# Values are (monthly_items, annual_items). Monthly items are multiplied by 12.
CATEGORY_MAP: dict[str, tuple[list[str], list[str]]] = {
    "Groceries": (
        [f"CO{i}X" for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)]
        + ["CO15", "CO16", "CO17", "CO19"],
        [],
    ),
    "Eating_Out": (["CO20"], []),
    "Utilities": (["CO4X", "CO21", "CO22", "CO24"], []),
    "Rent": (["CO30"], []),
    "Transport": (["CO28", "CO29"], ["CO45"]),
    "Healthcare": (["CO33"], ["CO34", "CO46"]),
    "Education": ([], ["CO35", "CO36", "CO37"]),
    "Entertainment": (["CO23"], ["CO43", "CO51"]),
    "Insurance": ([], ["CO50"]),
    "Clothing_Footwear": ([], ["CO38", "CO39"]),
    "Miscellaneous": (
        ["CO18", "CO25", "CO26", "CO27", "CO31", "CO32"],
        ["CO40", "CO41", "CO42", "CO44", "CO47", "CO48", "CO49", "CO52"],
    ),
}
assert list(CATEGORY_MAP) == EXPENSE_CATEGORIES

URBAN4_LABELS = {
    0: "Metro_Urban",
    1: "Other_Urban",
    2: "Developed_Village",
    3: "Less_Developed_Village",
}
CASTE_LABELS = {
    1: "Brahmin",
    2: "Forward_Caste",
    3: "OBC",
    4: "Scheduled_Caste",
    5: "Scheduled_Tribe",
    6: "Other",
}
RELIGION_LABELS = {
    1: "Hindu",
    2: "Muslim",
    3: "Christian",
    4: "Sikh",
    5: "Buddhist",
    6: "Jain",
    7: "Tribal",
    8: "Other",
    9: "No_Religion",  # not "None": pandas reads that string back as NaN
}

# Worker-count columns -> occupation label, in tie-break priority order. A
# household with equal salaried and farm workers is labelled Salaried: the first
# source in this list holding the maximum count wins. The order runs from the
# most to the least regular cash income stream, on the reasoning that the more
# regular stream dominates the household's ability to plan saving.
OCCUPATION_SOURCES: list[tuple[str, str]] = [
    ("NWKSALARY", "Salaried"),
    ("NWKBUSINESS", "Business"),
    ("NWKNONAG", "Non_Ag_Labour"),
    ("NWKFARM", "Farm"),
    ("NWKAGLAB", "Ag_Labour"),
]

ID_COLS = ["STATEID", "DISTID", "PSUID", "IDPSU", "HHID", "HHSPLITID", "IDHH"]

DEMOGRAPHIC_COLS = [
    "NPERSONS",
    "NADULTM",
    "NADULTF",
    "NCHILDM",
    "NCHILDF",
    "NELDERM",
    "NELDERF",
    "NTEENM",
    "NTEENF",
    "MHEADAGE",
    "FHEADAGE",
    "HHEDUC",
    "URBAN2011",
    "URBAN4_2011",
    "METRO",
    "ID11",
    "ID13",
    "DB5",
    "DB9C",
    "DB9D",
    "DB9E",
    "DB9G",
    "DB9H",
    "DB9I",
    "POOR",
    "ASSETS",
    "WT",
    "INCOME",
    "COTOTAL",
    "COPC",
    "INCOMEPC",
] + [src for src, _ in OCCUPATION_SOURCES]

SAVINGS_INDICATORS = {
    "DB9C": "Has_Securities",
    "DB9D": "Has_Fixed_Deposit",
    "DB9E": "Has_Bank_Savings",
    "DB9G": "Has_Post_Office_Account",
    "DB9H": "Has_Pension_LIC",
    "DB9I": "Has_Gold_Jewellery",
}

OUTPUT_COLUMNS: list[str] = (
    ID_COLS
    + [
        "WT",
        "INCOME",
        "Log_Income",
        "Household_Size",
        "Age_Dependents",
        "Head_Age",
        "Max_Adult_Education",
        "Occupation",
        "Area_Type",
        "Caste_Group",
        "Religion",
        "Debt_To_Income",
        "Debt_Missing",
        "Occupation_Tie",
    ]
    + [f"{c}_Share" for c in EXPENSE_CATEGORIES]
    + list(SAVINGS_INDICATORS.values())
    + EXPENSE_CATEGORIES
    + ["Category_Total", "COTOTAL", "Savings", "Savings_Rate", "Goal_Met"]
)


def monthly_items() -> list[str]:
    return sorted({c for m, _ in CATEGORY_MAP.values() for c in m})


def annual_items() -> list[str]:
    return sorted({c for _, a in CATEGORY_MAP.values() for c in a})


def required_columns() -> list[str]:
    return sorted(set(monthly_items() + annual_items() + ID_COLS + DEMOGRAPHIC_COLS))


def scan_raw(tsv: Path) -> pl.LazyFrame:
    """Lazily read DS0002, keeping only the needed columns, all cast to Float64.

    ICPSR writes missing values as a single space, so every column is read as
    text, stripped, and cast non-strictly (unparseable -> null).
    """
    with tsv.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
    keep = required_columns()
    missing = set(keep) - set(header)
    if missing:
        raise KeyError(f"IHDS file is missing expected columns: {sorted(missing)}")
    return (
        pl.scan_csv(tsv, separator="\t", infer_schema=False, quote_char=None)
        .select(keep)
        .with_columns(pl.col(c).str.strip_chars().cast(pl.Float64, strict=False) for c in keep)
    )


def _category_exprs() -> list[pl.Expr]:
    exprs = []
    for name, (monthly, annual) in CATEGORY_MAP.items():
        parts: list[pl.Expr] = []
        if monthly:
            parts.append(pl.sum_horizontal(pl.col(monthly).fill_null(0)) * 12)
        if annual:
            parts.append(pl.sum_horizontal(pl.col(annual).fill_null(0)))
        exprs.append(pl.sum_horizontal(parts).alias(name))
    return exprs


def _occupation_exprs() -> list[pl.Expr]:
    sources = [src for src, _ in OCCUPATION_SOURCES]
    counts = [pl.col(s).fill_null(0) for s in sources]
    top = pl.max_horizontal(counts)
    label: pl.Expr = pl.lit(None, dtype=pl.String)
    # Build the chain from lowest to highest priority so the first source wins.
    for src, name in reversed(OCCUPATION_SOURCES):
        label = pl.when(pl.col(src).fill_null(0) == top).then(pl.lit(name)).otherwise(label)
    n_at_top = pl.sum_horizontal([(c == top).cast(pl.Int32) for c in counts])
    any_worker = pl.sum_horizontal(counts) > 0
    return [
        pl.when(any_worker).then(label).otherwise(pl.lit("No_Regular_Worker")).alias("Occupation"),
        (any_worker & (n_at_top > 1)).cast(pl.Int8).alias("Occupation_Tie"),
    ]


def _map(col: str, mapping: dict[int, str]) -> pl.Expr:
    return pl.col(col).replace_strict(
        {float(k): v for k, v in mapping.items()}, default=None, return_dtype=pl.String
    )


def build_lazy(raw: pl.LazyFrame, threshold: float = THRESHOLD) -> pl.LazyFrame:
    """Raw DS0002 columns -> one analysis row per household (lazy)."""
    share_exprs = [
        (pl.col(c) / pl.col("Category_Total")).alias(f"{c}_Share") for c in EXPENSE_CATEGORIES
    ]
    return (
        raw.with_columns(_category_exprs())
        .with_columns(pl.sum_horizontal(EXPENSE_CATEGORIES).alias("Category_Total"))
        # Savings rate is undefined for INCOME <= 0; COTOTAL is missing for a few.
        .filter((pl.col("INCOME") > 0) & pl.col("COTOTAL").is_not_null() & (pl.col("COTOTAL") > 0))
        .with_columns(
            (pl.col("INCOME") - pl.col("COTOTAL")).alias("Savings"),
            *share_exprs,
            pl.col("INCOME").log().alias("Log_Income"),
            pl.coalesce("MHEADAGE", "FHEADAGE").alias("Head_Age"),
            pl.sum_horizontal(
                pl.col(["NCHILDM", "NCHILDF", "NELDERM", "NELDERF"]).fill_null(0)
            ).alias("Age_Dependents"),
            pl.col("NPERSONS").alias("Household_Size"),
            *_occupation_exprs(),
            _map("URBAN4_2011", URBAN4_LABELS).alias("Area_Type"),
            _map("ID13", CASTE_LABELS).alias("Caste_Group"),
            _map("ID11", RELIGION_LABELS).alias("Religion"),
            pl.col("HHEDUC").alias("Max_Adult_Education"),
            # A missing DB5 is unknown debt, not zero debt.
            (pl.col("DB5") / pl.col("INCOME")).alias("Debt_To_Income"),
            pl.col("DB5").is_null().cast(pl.Int8).alias("Debt_Missing"),
            *[pl.col(src).alias(name) for src, name in SAVINGS_INDICATORS.items()],
        )
        .with_columns((pl.col("Savings") / pl.col("INCOME")).alias("Savings_Rate"))
        .with_columns((pl.col("Savings_Rate") >= threshold).cast(pl.Int8).alias("Goal_Met"))
        .with_columns(pl.col(ID_COLS).cast(pl.Int64))
        .select(OUTPUT_COLUMNS)
    )


@dataclass(frozen=True)
class BuildReport:
    n_raw: int
    n_final: int
    goal_met_rate: float
    threshold: float
    occupation_ties: int
    debt_missing: int

    @property
    def dropped(self) -> int:
        return self.n_raw - self.n_final

    def __str__(self) -> str:
        return (
            f"raw households:        {self.n_raw:,}\n"
            f"dropped (income<=0 / missing consumption): {self.dropped:,}\n"
            f"final households:      {self.n_final:,}\n"
            f"Goal_Met rate @ {self.threshold:.0%}:  {self.goal_met_rate:.4f}\n"
            f"occupation ties broken by priority: {self.occupation_ties:,}\n"
            f"debt (DB5) missing:    {self.debt_missing:,}"
        )


def _report(df: pl.DataFrame, n_raw: int, threshold: float) -> BuildReport:
    stats = df.select(
        pl.col("Goal_Met").mean().alias("rate"),
        pl.col("Occupation_Tie").sum().alias("ties"),
        pl.col("Debt_Missing").sum().alias("debt"),
    ).row(0, named=True)
    return BuildReport(
        n_raw=n_raw,
        n_final=df.height,
        goal_met_rate=float(stats["rate"]),
        threshold=threshold,
        occupation_ties=int(stats["ties"]),
        debt_missing=int(stats["debt"]),
    )


def _n_raw(tsv: Path) -> int:
    n: int = scan_raw(tsv).select(pl.len()).collect().item()
    return n


def build(tsv: Path, threshold: float = THRESHOLD) -> tuple[pl.DataFrame, BuildReport]:
    """Eager build: returns the household table and a summary report."""
    df = build_lazy(scan_raw(tsv), threshold).collect()
    return df, _report(df, _n_raw(tsv), threshold)


def write(tsv: Path, out: Path, threshold: float = THRESHOLD) -> BuildReport:
    """Stream the build straight to Parquet (or CSV, by extension)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lazy = build_lazy(scan_raw(tsv), threshold)
    if out.suffix == ".csv":
        lazy.sink_csv(out)
        df = pl.read_csv(out)
    else:
        lazy.sink_parquet(out)
        df = pl.read_parquet(out)
    return _report(df, _n_raw(tsv), threshold)


def reconcile_totals(df: pl.DataFrame, threshold: float = THRESHOLD) -> dict[str, float]:
    """How far the 11-category sum is from the survey's COTOTAL, and what it does to the label."""
    rel = (pl.col("Category_Total") - pl.col("COTOTAL")).abs() / pl.col("COTOTAL")
    alt_label = ((pl.col("INCOME") - pl.col("Category_Total")) / pl.col("INCOME")) >= threshold
    out = df.select(
        pl.corr("Category_Total", "COTOTAL").alias("pearson_r"),
        (pl.col("Category_Total") - pl.col("COTOTAL")).median().alias("median_diff"),
        (rel < 0.01).mean().alias("within_1pct"),
        (rel < 0.05).mean().alias("within_5pct"),
        (alt_label.cast(pl.Int8) == pl.col("Goal_Met")).mean().alias("label_agreement"),
    ).row(0, named=True)
    return {k: float(v) for k, v in out.items()}
