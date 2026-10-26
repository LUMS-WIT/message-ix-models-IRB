"""Aggregate basin (PID)-level population/GDP into this project's six
IRB country groups (``AFGHAN_SOUTH``, ``AFGHAN_NORTH``, ``CHINA``,
``INDIA_WEST``, ``INDIA_EAST``, ``PAKISTAN`` -- the group string embedded in
every ``BCU_name``, see :func:`region_group_of_irb` in
:mod:`build_irrigation_demand_lgrip30`).

Feeds :mod:`build_irrigation_demand_growth_IRB`, which needs a
population/GDP-per-capita trajectory *per region group* to compute a
GCAM-AgLU-style demand-growth index
(:func:`build_irrigation_demand_lgrip30.gcam_demand_index`). That function
takes plain ``region,year,value`` long CSVs -- national SSP tables (e.g.
OECD SSP) would need their own re-aggregation to this project's 6 groups
(which split AFG/IND/CHN by sub-national basin cluster, not by country) to
be usable directly; ``PID_pop_gdp.csv`` (below) is already basin/PID
resolved, at this project's own node granularity, so aggregating it up is a
more direct route than reconciling a national table.

Input
-----
``PID_pop_gdp.csv`` (committed here; sourced from the same NEST Indus
project folder as this repo's other raw-data ports, e.g. :mod:`storage`'s
GRanD/WAPDA CSVs). Columns: ``PID, year, total_pop, urban_gdp, rural_gdp,
total_gdp`` -- one row per (basin PID, decadal year in {2010, 2020, ...,
2060}). All 24 PIDs present, decadal only (no interpolation done here --
:mod:`build_irrigation_demand_growth_IRB` interpolates to annual
resolution once it also has the index calculation in view, rather than
baking one interpolation choice into this intermediate file).

Output
------
Two long CSVs, ``region,year,value``, region = the 6 IRB country groups:

``population_group_IRB.csv``
    Summed ``total_pop`` per group/year (persons) -- see "Pakistan
    refinement" below for a caveat on the ``PAKISTAN`` rows specifically.
``gdp_per_capita_group_IRB.csv``
    Summed ``total_gdp`` / summed ``total_pop`` per group/year (same
    currency/units as the source ``total_gdp`` column, per person) --
    group-level GDP per capita, not a population-weighted average of each
    basin's own per-capita figure (the two differ when population is
    unevenly split across a group's basins; summing both numerators and
    dividing is the standard way to get a genuine group-wide rate).

Pakistan refinement (optional, ``PAKISTAN`` rows only)
-------------------------------------------------------
If ``Pakistan_country_population_all_SSP.csv`` is present (a national,
multi-SSP population projection to 2100, 5-year steps -- newer and more
directly SSP-scenario-resolved than anything in ``PID_pop_gdp.csv``), the
``PAKISTAN`` group's population series is refined using it, filtered to
``SSP2`` (this repo's existing scenario convention, e.g. ``demands.r``'s
own ``SSP2`` filter). It is **not** a straight swap-in: that file is a
*national* total (all of Pakistan, including the non-Indus Balochistan
coast etc.), not the Indus-basin-clipped population ``PID_pop_gdp.csv``'s
PAK_* rows represent -- splicing the two directly would introduce a level
jump at the seam (different physical quantities), which would show up as
a spurious spike/dip in the demand-growth index right at the splice year.

Instead this **rebases** the national series onto the existing basin-level
one: at :data:`PAK_SPLICE_YEAR` (2025, the new data's first year), the
basin-level series is linearly interpolated (between its own surrounding
decadal points, e.g. 2020/2030) to get a real anchor value in the same
basin-population units the rest of this pipeline already uses; every later
point is then ``anchor * (national_pop[year] / national_pop[splice_year])``
-- i.e. the *absolute level* still comes from the basin data (unchanged),
but the *shape of future growth* comes from the newer, SSP-explicit
national projection. This keeps the pre-2025 numbers (including the 2015
point later scripts anchor a demand_index of 1.0 to) completely unchanged,
and preserves continuity at the seam by construction (the rebased value
*equals* the interpolated basin anchor exactly at 2025).

Capped at :data:`PAK_CAP_YEAR` (2060, matching ``PID_pop_gdp.csv``'s own
horizon and this repo's model horizon) even though the source national
data runs to 2100 -- :func:`~build_irrigation_demand_growth_IRB.gcam_demand_index`
requires population and GDP-per-capita to share the same year index, and
there is no equivalently updated GDP series to match a longer horizon.

Not done here (real follow-up, not automated by this script): a proper
district/province-level spatial join of the new data's finer-resolution
files (``Pakistan_district_population_all_SSP.csv`` /
``..._province_population_all_SSP.csv``, also delivered alongside the
country-level file) against the 13 Pakistan basin polygons, which would
give a genuine basin-specific (not merely rebased-national) refinement.
Skipped for the same reason :mod:`build_irrigation_demand_lgrip30`'s own
spatial step is optional -- it needs basin polygon geometry, i.e.
geopandas, which this script's own dependency-avoidance convention (see
``build_irrigation_demand_growth_IRB.py``'s docstring) steers away from
unless a task specifically needs it.

Usage
-----
::

    python build_pop_gdp_group_IRB.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from _pid_crosswalk import CROSSWALK

log = logging.getLogger(__name__)

_CROPS_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "crops"
DEFAULT_INPUT = _CROPS_DATA_DIR / "PID_pop_gdp.csv"
DEFAULT_PAK_SSP_CSV = _CROPS_DATA_DIR / "Pakistan_country_population_all_SSP.csv"
DEFAULT_POP_OUT = _CROPS_DATA_DIR / "population_group_IRB.csv"
DEFAULT_GDP_PC_OUT = _CROPS_DATA_DIR / "gdp_per_capita_group_IRB.csv"

#: Year the national SSP series takes over from the basin-level series --
#: the new data's own first year.
PAK_SPLICE_YEAR = 2025
#: Last year kept from the (longer, to 2100) national series -- matches
#: PID_pop_gdp.csv's / this pipeline's existing 2060 horizon (see module
#: docstring's "Pakistan refinement" section).
PAK_CAP_YEAR = 2060
PAK_SSP_SCENARIO = "SSP2"


def region_group_of_irb(bcu_name: str) -> str:
    """Same convention as
    :func:`build_irrigation_demand_lgrip30.region_group_of_irb` (duplicated
    rather than imported: that module lives one level up conceptually --
    it is the consumer of this script's output, not a shared dependency,
    and importing it here would pull in its geopandas-only top-level
    surface for one string-split helper)."""
    return bcu_name.split("|", 1)[1]


def _refine_pakistan_population(
    population: pd.DataFrame,
    pak_ssp_csv: Path,
    scenario: str = PAK_SSP_SCENARIO,
    splice_year: int = PAK_SPLICE_YEAR,
    cap_year: int = PAK_CAP_YEAR,
) -> pd.DataFrame:
    """Rebase the ``PAKISTAN`` group's population onto the newer national
    SSP series, per the module docstring's "Pakistan refinement" section.
    A no-op (with a logged warning) if `pak_ssp_csv` doesn't exist, so this
    stays an optional enhancement, not a hard dependency."""
    if not pak_ssp_csv.exists():
        log.warning(
            "No %s found; PAKISTAN population left as basin-summed only "
            "(no national-SSP refinement).",
            pak_ssp_csv.name,
        )
        return population

    national = pd.read_csv(pak_ssp_csv)
    national = national[national["ssp"] == scenario]
    if national.empty:
        log.warning("%s has no %s rows; skipping refinement.", pak_ssp_csv.name, scenario)
        return population
    national = national.set_index("year")["population"].sort_index()

    if splice_year not in national.index:
        log.warning(
            "%s has no row at splice year %d; skipping refinement.",
            pak_ssp_csv.name, splice_year,
        )
        return population

    pak = population[population["region"] == "PAKISTAN"].set_index("year")["value"].sort_index()
    # Interpolate the existing (decadal) basin-level series to get a real
    # anchor value in basin-population units at the splice year -- pandas
    # can interpolate directly off an index that doesn't yet contain
    # `splice_year` by reindexing to include it first.
    anchor_series = pak.reindex(sorted(set(pak.index) | {splice_year})).interpolate(method="index")
    anchor = float(anchor_series.loc[splice_year])

    new_years = [y for y in national.index if splice_year <= y <= cap_year]
    rebased = anchor * (national.loc[new_years] / national.loc[splice_year])

    kept = pak[pak.index < splice_year]
    combined = pd.concat([kept, rebased]).sort_index()

    refined_rows = pd.DataFrame({"region": "PAKISTAN", "year": combined.index, "value": combined.values})
    other_regions = population[population["region"] != "PAKISTAN"]
    result = pd.concat([other_regions, refined_rows], ignore_index=True)
    log.info(
        "PAKISTAN population refined with %s (%s): anchor=%.0f at %d, %d years replaced (%d-%d)",
        pak_ssp_csv.name, scenario, anchor, splice_year, len(new_years), splice_year, cap_year,
    )
    return result


def build(
    input_csv: Path = DEFAULT_INPUT,
    pak_ssp_csv: Path = DEFAULT_PAK_SSP_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(population, gdp_per_capita)``, each a long DataFrame with
    columns ``region, year, value``. `pak_ssp_csv`: see the module
    docstring's "Pakistan refinement" section; pass ``None`` to skip it."""
    df = pd.read_csv(input_csv)
    df["node"] = df["PID"].map(CROSSWALK)
    unmapped = df[df["node"].isna()]
    if not unmapped.empty:
        log.warning(
            "%d row(s) have a PID not in the crosswalk (PIDs: %s) -- dropped",
            len(unmapped), sorted(unmapped["PID"].unique().tolist()),
        )
    df = df.dropna(subset=["node"])
    df["region"] = df["node"].map(region_group_of_irb)

    grouped = df.groupby(["region", "year"], as_index=False)[["total_pop", "total_gdp"]].sum()

    population = grouped[["region", "year", "total_pop"]].rename(columns={"total_pop": "value"})
    gdp_per_capita = grouped.assign(value=grouped["total_gdp"] / grouped["total_pop"])[
        ["region", "year", "value"]
    ]

    if pak_ssp_csv is not None:
        population = _refine_pakistan_population(population, pak_ssp_csv)

    return population, gdp_per_capita


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pop_df, gdp_pc_df = build()
    pop_df.to_csv(DEFAULT_POP_OUT, index=False)
    gdp_pc_df.to_csv(DEFAULT_GDP_PC_OUT, index=False)
    print(f"Wrote {len(pop_df)} rows to {DEFAULT_POP_OUT}")
    print(f"Wrote {len(gdp_pc_df)} rows to {DEFAULT_GDP_PC_OUT}")
