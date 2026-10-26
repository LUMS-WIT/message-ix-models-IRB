"""Build a per-(country-group, year) irrigation-demand growth index --
the temporal-trend piece of :mod:`build_irrigation_demand_lgrip30`, run
standalone without its spatial (LGRIP30) piece.

Why split off from :mod:`build_irrigation_demand_lgrip30`
-----------------------------------------------------------
That script's own docstring notes the LGRIP30-derived ``country_shares``
never multiplies its output value -- it only produces ``share_flag``, a
diagnostic ("does the satellite-observed spatial split disagree with the
historical split by >2x"), because the real anchor is already basin-
specific (``historical_irrigation_withdrawals_IRB.csv``). That means the
actual per-node MCM number needs no raster processing at all -- and by
extension, no ``geopandas``/``rasterio`` install (a real risk on this
project's Windows setup, see the LGRIP30 script's own dependency note).

So rather than reimplement the anchor-and-multiply step here too, this
script produces *only* the dimensionless growth index --
:func:`build_irrigation_demand_lgrip30.gcam_demand_index`, reused directly
(pure pandas, no geopandas/rasterio import at module level) -- at country-
group granularity (:func:`~build_irrigation_demand_lgrip30.region_group_of_irb`,
6 groups). :func:`~..crops.irrigation_tech.add_irrigation_techs` already
has the real, basin-specific historical MCM figure in hand
(``hist_lookup``, from :func:`~..crops.common.read_crop_irrigation_water`'s
sibling reader) when it builds each basin's ``hist_irr_withdrawal_<basin>``
relation floor -- multiplying that by this index at lookup time is a
smaller, more direct change than routing a second per-node MCM CSV through
the pipeline.

Input
-----
``population_group_IRB.csv`` / ``gdp_per_capita_group_IRB.csv``
(:mod:`build_pop_gdp_group_IRB`'s output; decadal, 2010-2060, 6 IRB country
groups).

Method
------
Both series are reindexed to annual resolution over their full decadal
range and linearly interpolated (population/GDP trajectories are smooth by
construction -- SSP scenarios themselves are built from decadal points the
same way), then combined via
``gcam_demand_index = (pop_y/pop_0) * (gdp_pc_y/gdp_pc_0)**income_elasticity``,
indexed to 1.0 at :data:`BASE_YEAR` (2015 -- this project's existing
historical/baseline year convention, e.g. ``crop_irr_land_2015``).

Output
------
``irrigation_demand_growth_IRB.csv``: columns ``region`` (country group),
``year``, ``demand_index`` (1.0 at :data:`BASE_YEAR`).

Usage
-----
::

    python build_irrigation_demand_growth_IRB.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from build_irrigation_demand_lgrip30 import DEFAULT_INCOME_ELASTICITY, gcam_demand_index

log = logging.getLogger(__name__)

_CROPS_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "crops"
DEFAULT_POP_CSV = _CROPS_DATA_DIR / "population_group_IRB.csv"
DEFAULT_GDP_PC_CSV = _CROPS_DATA_DIR / "gdp_per_capita_group_IRB.csv"
DEFAULT_OUT = _CROPS_DATA_DIR / "irrigation_demand_growth_IRB.csv"

#: This project's existing historical/baseline year convention (e.g.
#: ``crop_irr_land_2015`` in crop_input_data_IRB.csv) -- the index is 1.0
#: here, matching :mod:`~..crops.irrigation_tech`'s existing historical
#: floor exactly at this year, and growing/shrinking from it afterward.
BASE_YEAR = 2015


def _annualize(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long (region, year, value) frame to year-indexed columns per
    region, reindexed to every year in its own min..max range and linearly
    interpolated."""
    wide = long_df.pivot(index="year", columns="region", values="value")
    full_years = range(int(wide.index.min()), int(wide.index.max()) + 1)
    return wide.reindex(full_years).interpolate(method="linear")


def build(
    population_csv: Path = DEFAULT_POP_CSV,
    gdp_per_capita_csv: Path = DEFAULT_GDP_PC_CSV,
    income_elasticity: float = DEFAULT_INCOME_ELASTICITY,
    base_year: int = BASE_YEAR,
) -> pd.DataFrame:
    pop = _annualize(pd.read_csv(population_csv))
    gdp_pc = _annualize(pd.read_csv(gdp_per_capita_csv))

    missing = set(pop.columns).symmetric_difference(gdp_pc.columns)
    if missing:
        raise ValueError(f"population/gdp_per_capita region mismatch: {missing}")

    rows = []
    for region in pop.columns:
        index = gcam_demand_index(
            pop[region], gdp_pc[region], income_elasticity=income_elasticity, base_year=base_year
        )
        rows.append(pd.DataFrame({"region": region, "year": index.index, "demand_index": index.values}))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = build()
    result.to_csv(DEFAULT_OUT, index=False)
    print(f"Wrote {len(result)} rows covering {result['region'].nunique()} regions to {DEFAULT_OUT}")
