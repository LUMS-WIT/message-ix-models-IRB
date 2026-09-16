"""Estimate a basin-level irrigation water demand trajectory from LGRIP30
(spatial pattern) + a GCAM-AgLU-style population/GDP demand-growth index
(temporal trend), anchored to this repo's existing historical withdrawal
total (physical scale) -- a stylized substitute for either of the two
irrigation-demand mechanisms already in this repo (see
:mod:`~message_ix_models.model.water.data.crops.irrigation_tech`'s module
docstring): the bypassed GLOBIOM ``land_input`` linkage (too heavy -- needs
a real GLOBIOM run) and the current ``hist_irr_withdrawal_<basin>``
relation (a floor only -- never rewards water efficiency, see the
discussion this script grew out of).

Method
------
1. **Where** (spatial pattern): for each basin polygon, compute irrigated-
   vs-rainfed-vs-total-cropland area from `LGRIP30
   <https://www.earthdata.nasa.gov/data/catalog/lpdaac-ecs-lgrip30-001>`_ (a
   30 m, Landsat-derived global cropland product), then each basin's SHARE
   of its country group's total irrigated area. This is an independent,
   modern, remote-sensing-based spatial key -- an upgrade over silently
   trusting whatever spatial split the committed historical CSV happens to
   already encode, and a genuine alternative to population-weighted
   downscaling (population is a poor proxy for *where* irrigated
   agriculture actually sits; this uses observed cropland instead).
2. **How much, nationally** (temporal trend): a GCAM-AgLU-style demand
   index, ``(pop_t/pop_0) * (gdp_pc_t/gdp_pc_0)**income_elasticity`` --
   population growth scales demand linearly, GDP-per-capita growth scales
   it via a literature income elasticity (see
   :data:`DEFAULT_INCOME_ELASTICITY`). Deliberately NOT a full land-market
   solve (GCAM's own AgLU module is this simple at its core; the land
   allocation/production-response machinery on top of it is what's
   expensive, and is exactly what this script is trying to avoid needing).
3. **Anchor** (physical units): basin demand at year `y` = basin's real
   historical annual withdrawal (``historical_irrigation_withdrawals_IRB.csv``,
   already committed) x the national demand index at `y` (relative to the
   historical data's own base year) -- i.e. hold the LGRIP30-derived
   spatial share fixed and let the temporal index carry the growth, so the
   output stays anchored to a real number instead of a purely synthetic one.

Generalized to any basin set: nothing here is hardcoded to Indus/IRB except
the *default* input paths (shapefile, historical CSV, PID crosswalk) --
`build()` accepts `region_group_of`, `shapefile_path`, `historical_csv`,
and `pid_crosswalk` for a different basin delineation, and the LGRIP30 /
population / GDP inputs are always caller-supplied (nothing about them is
Pakistan/Indus-specific to begin with).

Deliberately NOT done here (real follow-up decisions, not automated by this
script): whether the output below replaces or supplements
``hist_irr_withdrawal_<basin>``'s relation_lower in
:func:`~..crops.irrigation_tech.add_irrigation_techs`, and whether it should
be written as a ``demand`` parameter (a hard requirement, like
urban_mw/rural_mw) or a softer relation. Either way, once it does drive a
real requirement, ``irr_<method>_<crop>``'s existing per-method cost/water
data (unchanged by this script) is what lets the model actually trade off
cheap-and-thirsty vs. expensive-and-efficient methods against it.

Inputs this script does NOT bundle (unlike the fully-committed raw data
other ``build_*_IRB.py`` scripts use) -- supply your own:

- One or more LGRIP30 GeoTIFF tiles covering the basin set (download from
  NASA LP DAAC/Earthdata, or the Google Earth Engine community asset
  ``projects/sat-io/open-datasets/GFSAD/LGRIP30``). Legend (v1, 4 classes;
  see :data:`LGRIP30_CLASSES`): 0 ocean/water, 1 non-cropland, 2 irrigated
  cropland, 3 rainfed cropland.
- A population trajectory and a GDP-per-capita trajectory, per country
  group and year -- e.g. exported from the IIASA SSP database (keeps this
  consistent with the SSP2 framework the rest of this repo already uses)
  or FAOSTAT/World Bank. Plain long CSVs: ``region,year,value``.

New dependency, unlike this directory's other scripts: needs
``rasterio`` + ``geopandas`` (real polygon geometry and raster zonal
statistics -- not reasonably hand-rollable the way :mod:`_shapefile` avoids
that dependency for simple area weighting). Both were available in the
environment this was written and tested against; confirm before relying on
this in one that isn't (see e.g. ``build_fertilizer_weights_IRB.py``'s
docstring for why every *other* script here avoided them).

Usage
-----
::

    python build_irrigation_demand_lgrip30.py \\
        --lgrip30 /path/to/lgrip30_tile*.tif \\
        --population pop.csv --gdp-per-capita gdp_pc.csv \\
        --out irrigation_demand_lgrip30_IRB.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_CROPS_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "crops"
_NETWORK_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "network"

#: Default IRB inputs -- already committed in-repo.
DEFAULT_SHAPEFILE = _NETWORK_DATA_DIR / "raw" / "Indus_bcu.shp"
DEFAULT_HISTORICAL_CSV = _CROPS_DATA_DIR / "historical_irrigation_withdrawals_IRB.csv"

#: LGRIP30 v1 legend, confirmed against the GLAD/GFSAD community
#: documentation for the GEE asset ``projects/sat-io/open-datasets/GFSAD/LGRIP30``
#: (https://gee-community-catalog.org/projects/lgrip30/). LGRIP30 v2 (2024)
#: splits class 2 further into single/double/continuous-crop irrigated
#: sub-classes -- if using v2, remap those sub-classes to 2 before calling
#: :func:`lgrip30_basin_shares`, or extend this dict.
LGRIP30_CLASSES: dict[int, str] = {
    0: "ocean_water",
    1: "non_cropland",
    2: "irrigated_cropland",
    3: "rainfed_cropland",
}

#: Day-weighting for monthly MCM/day -> annual MCM/year, mirrored from
#: :data:`~..crops.common.DAYS_IN_MONTH` (not imported: this script lives in
#: pre_processing, one layer below the crops/ runtime package by this
#: repo's existing convention -- pre_processing produces the CSVs crops/
#: reads, not the other way around).
DAYS_IN_MONTH: dict[int, int] = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}

#: USDA ERS / FAO international food-demand elasticity studies put staple
#: cereal income elasticity of demand in roughly the 0.1-0.4 range for
#: middle-income countries (higher for more discretionary crops); 0.3 is a
#: reasonable single default for a stylized, all-crops-combined estimate --
#: NOT a precise, crop-specific literature value. Override per crop if you
#: have better numbers.
DEFAULT_INCOME_ELASTICITY = 0.3


def region_group_of_irb(bcu_name: str) -> str:
    """Default ``region_group_of``: this basin's country group, from the
    ``BCU_name`` convention (``"<id>|<REGION>"``) every IRB data file
    already uses -- e.g. ``"12|PAKISTAN"`` -> ``"PAKISTAN"``. The national
    demand index is computed per country group, since population/GDP
    trajectories are national, not per-basin.
    """
    return bcu_name.split("|", 1)[1]


def read_basin_geometries(
    shapefile_path: Path = DEFAULT_SHAPEFILE,
    pid_crosswalk: dict[str, str] | None = None,
):
    """Read basin polygons and attach each one's real ``node`` (BCU_name).

    `pid_crosswalk` maps the shapefile's own ``PID`` attribute to `node`;
    defaults to this repo's already-verified IRB crosswalk
    (:mod:`_pid_crosswalk`) -- see that module's docstring for why this
    must be a real, checked mapping (outlet-coordinate matched against the
    R-model source) and never assumed from row order or PID's embedded
    number, both of which have been found to NOT correspond (e.g. shapefile
    ``PAK_2`` is basin 13, not "the 2nd Pakistan basin" in the model's own
    numbering).

    Returns a :class:`geopandas.GeoDataFrame` with columns ``node``
    (BCU_name) and ``geometry``, one row per basin the crosswalk covers.
    """
    import geopandas as gpd

    if pid_crosswalk is None:
        from _pid_crosswalk import CROSSWALK as pid_crosswalk

    gdf = gpd.read_file(shapefile_path)
    if "PID" not in gdf.columns:
        raise ValueError(
            f"{shapefile_path} has no 'PID' column -- pass a matching "
            "pid_crosswalk for this shapefile's own basin-id attribute"
        )
    gdf["node"] = gdf["PID"].map(pid_crosswalk)
    unmapped = gdf[gdf["node"].isna()]
    if not unmapped.empty:
        log.warning(
            "%d/%d basin polygons have no entry in pid_crosswalk (PIDs: %s) "
            "-- dropped, not silently included with a guessed id",
            len(unmapped), len(gdf), sorted(unmapped["PID"].tolist()),
        )
    return gdf.dropna(subset=["node"])[["node", "geometry"]].reset_index(drop=True)


def _pixel_area_km2(row_lat_deg: np.ndarray, pixel_w_deg: float, pixel_h_deg: float) -> np.ndarray:
    """Approximate per-pixel area (km^2) on a geographic (lat/lon) grid via
    the standard spherical-earth cosine correction -- accurate to a
    fraction of a percent at 30 m resolution, and avoids reprojecting a
    potentially large raster mosaic just to get areas (this script only
    needs RELATIVE shares between basins, not survey-grade absolute area).
    """
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(row_lat_deg))
    return (pixel_h_deg * km_per_deg_lat) * (pixel_w_deg * km_per_deg_lon)


def lgrip30_basin_shares(basins, raster_paths: list[Path]) -> pd.DataFrame:
    """Zonal statistics of LGRIP30 class area per basin.

    Parameters
    ----------
    basins :
        As returned by :func:`read_basin_geometries`.
    raster_paths :
        One or more LGRIP30 GeoTIFF tiles. A basin spanning more than one
        tile is handled by masking each tile against it and summing --
        no mosaicking step needed since areas simply add.

    Returns
    -------
    pd.DataFrame
        Columns ``node``, ``irrigated_km2``, ``rainfed_km2``,
        ``cropland_km2`` (irrigated + rainfed).
    """
    import rasterio
    from rasterio.mask import mask
    from rasterio.errors import WindowError

    rows = []
    for _, basin in basins.iterrows():
        class_km2 = dict.fromkeys(LGRIP30_CLASSES.values(), 0.0)
        for raster_path in raster_paths:
            with rasterio.open(raster_path) as src:
                geom = [basin.geometry.__geo_interface__]
                # Reproject the geometry to the raster's CRS if they differ
                # (LGRIP30 tiles and this project's basin shapefiles are
                # both typically EPSG:4326, but never assume).
                if basins.crs is not None and src.crs is not None and str(basins.crs) != str(src.crs):
                    import geopandas as gpd
                    geom = gpd.GeoSeries([basin.geometry], crs=basins.crs).to_crs(src.crs)
                    geom = [geom.iloc[0].__geo_interface__]
                try:
                    data, transform = mask(src, geom, crop=True, filled=True, nodata=255)
                except (ValueError, WindowError):
                    continue  # this tile doesn't overlap this basin
                band = data[0]
                pixel_w = abs(transform.a)
                pixel_h = abs(transform.e)
                # Latitude of each row's centre, for the cosine area correction.
                row_lats = transform.f + transform.e * (np.arange(band.shape[0]) + 0.5)
                row_areas = _pixel_area_km2(row_lats, pixel_w, pixel_h)  # per-row km^2/pixel
                for code, name in LGRIP30_CLASSES.items():
                    mask_2d = band == code
                    if mask_2d.any():
                        class_km2[name] += float((mask_2d * row_areas[:, None]).sum())

        rows.append({
            "node": basin["node"],
            "irrigated_km2": class_km2["irrigated_cropland"],
            "rainfed_km2": class_km2["rainfed_cropland"],
            "cropland_km2": class_km2["irrigated_cropland"] + class_km2["rainfed_cropland"],
        })

    result = pd.DataFrame(rows)
    if result.empty or result["irrigated_km2"].sum() == 0:
        log.warning(
            "lgrip30_basin_shares: zero irrigated area found across all "
            "basins/tiles -- check raster/basin CRS and tile coverage "
            "before trusting downstream output"
        )
    return result


def add_country_share(
    shares: pd.DataFrame, region_group_of=region_group_of_irb
) -> pd.DataFrame:
    """Add ``country_share``: each basin's fraction of its country group's
    total irrigated area (from :func:`lgrip30_basin_shares`'s output) --
    the spatial downscaling key. A country group with zero total irrigated
    area gets ``country_share = 0`` for all its basins (not NaN/divide
    error) -- genuinely no irrigation there per LGRIP30, not missing data.
    """
    out = shares.copy()
    out["region_group"] = out["node"].map(region_group_of)
    totals = out.groupby("region_group")["irrigated_km2"].transform("sum")
    out["country_share"] = np.where(totals > 0, out["irrigated_km2"] / totals, 0.0)
    return out


def gcam_demand_index(
    population: pd.Series,
    gdp_per_capita: pd.Series,
    income_elasticity: float = DEFAULT_INCOME_ELASTICITY,
    base_year: int | None = None,
) -> pd.Series:
    """GCAM-AgLU-style demand growth index, indexed to 1.0 at `base_year`.

    ``index_y = (pop_y / pop_0) * (gdp_pc_y / gdp_pc_0) ** income_elasticity``

    Both `population` and `gdp_per_capita` are indexed by year (same index,
    e.g. a national or country-group trajectory -- this function doesn't
    care about spatial scale, only that both series share one). Pure
    population/GDP-response function; no land market, no crop choice --
    that's the whole point (see module docstring).
    """
    if population.index.symmetric_difference(gdp_per_capita.index).size:
        raise ValueError("population and gdp_per_capita must share the same year index")
    base_year = base_year if base_year is not None else population.index.min()
    pop_index = population / population.loc[base_year]
    gdp_index = (gdp_per_capita / gdp_per_capita.loc[base_year]) ** income_elasticity
    return (pop_index * gdp_index).rename("demand_index")


def read_historical_annual(csv_path: Path = DEFAULT_HISTORICAL_CSV) -> pd.DataFrame:
    """Historical monthly irrigation withdrawal (MCM/day-style), day-weighted
    to a real annual total per basin, summed across ``tec``
    (``irrigation_sw_diversion``/``irrigation_gw_diversion`` --
    :mod:`~..crops.irrigation_tech`'s own documented simplification is to
    treat these combined, matched here for consistency).

    Returns columns ``node``, ``value`` (MCM/year).
    """
    df = pd.read_csv(csv_path)
    df = df.assign(_annual=df["value"] * df["time"].map(DAYS_IN_MONTH))
    return df.groupby("node", as_index=False)["_annual"].sum().rename(columns={"_annual": "value"})


def estimate_irrigation_demand(
    historical_annual: pd.DataFrame,
    country_shares: pd.DataFrame,
    demand_index_by_region: dict[str, pd.Series],
    years: list[int],
) -> pd.DataFrame:
    """Combine the three pieces into a basin x year irrigation demand
    estimate, anchored to `historical_annual` at each series' own base year.

    ``demand_t[node] = historical_annual[node] * demand_index_by_region[region_group(node)][t]``

    The LGRIP30-derived `country_shares` is used only to flag basins whose
    modern satellite-observed irrigated-area share diverges sharply from
    what the historical baseline alone would imply (see ``share_flag``
    below) -- it does not directly multiply the output, because
    `historical_annual` already encodes a real, basin-specific spatial
    split; LGRIP30 here is a validation/adjustment signal, not a second
    independent driver layered on top (that would double-count the spatial
    pattern). If you want LGRIP30 to actively override the historical
    split instead of just flagging disagreement, redistribute
    `historical_annual`'s country-group total using `country_shares``
    before calling this function.

    Returns columns ``node``, ``year``, ``value`` (MCM/year), ``unit``,
    ``share_flag`` (True where the LGRIP30 country_share and the
    historical-share implied by `historical_annual` disagree by >2x, worth
    a manual look before trusting that basin's number).
    """
    merged = historical_annual.merge(country_shares[["node", "region_group", "country_share"]], on="node", how="left")
    hist_region_total = merged.groupby("region_group")["value"].transform("sum")
    merged["historical_share"] = np.where(hist_region_total > 0, merged["value"] / hist_region_total, 0.0)
    ratio = merged["country_share"] / merged["historical_share"].replace(0, np.nan)
    merged["share_flag"] = ~ratio.between(0.5, 2.0)
    merged["share_flag"] = merged["share_flag"].fillna(False)

    rows = []
    for _, r in merged.iterrows():
        index = demand_index_by_region.get(r["region_group"])
        if index is None:
            log.warning("No demand index for region_group=%s (node=%s) -- skipped", r["region_group"], r["node"])
            continue
        for year in years:
            if year not in index.index:
                continue
            rows.append({
                "node": r["node"], "year": year,
                "value": r["value"] * index.loc[year],
                "unit": "MCM/year", "share_flag": bool(r["share_flag"]),
            })
    return pd.DataFrame(rows)


def build(
    lgrip30_rasters: list[Path],
    population_csv: Path,
    gdp_per_capita_csv: Path,
    years: list[int],
    shapefile_path: Path = DEFAULT_SHAPEFILE,
    historical_csv: Path = DEFAULT_HISTORICAL_CSV,
    pid_crosswalk: dict[str, str] | None = None,
    region_group_of=region_group_of_irb,
    income_elasticity: float = DEFAULT_INCOME_ELASTICITY,
) -> pd.DataFrame:
    """Run the full pipeline. `population_csv`/`gdp_per_capita_csv` are
    plain long CSVs with columns ``region,year,value`` -- `region` must
    match `region_group_of`'s output (e.g. "PAKISTAN", "INDIA_WEST", ...).
    """
    basins = read_basin_geometries(shapefile_path, pid_crosswalk)
    shares = lgrip30_basin_shares(basins, lgrip30_rasters)
    shares = add_country_share(shares, region_group_of)

    historical_annual = read_historical_annual(historical_csv)

    pop = pd.read_csv(population_csv).pivot(index="year", columns="region", values="value")
    gdp_pc = pd.read_csv(gdp_per_capita_csv).pivot(index="year", columns="region", values="value")
    demand_index_by_region = {
        region: gcam_demand_index(pop[region], gdp_pc[region], income_elasticity, base_year=pop.index.min())
        for region in pop.columns
    }

    result = estimate_irrigation_demand(historical_annual, shares, demand_index_by_region, years)
    flagged = result.loc[result["share_flag"], "node"].unique()
    if len(flagged):
        log.warning(
            "%d basin(s) where LGRIP30's irrigated-area share disagrees with "
            "the historical split by >2x -- worth a manual look: %s",
            len(flagged), sorted(flagged),
        )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lgrip30", nargs="+", required=True, type=Path, help="LGRIP30 GeoTIFF tile(s)")
    parser.add_argument("--population", required=True, type=Path, help="CSV: region,year,value")
    parser.add_argument("--gdp-per-capita", required=True, type=Path, help="CSV: region,year,value")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Model years to compute")
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--historical-csv", type=Path, default=DEFAULT_HISTORICAL_CSV)
    parser.add_argument("--income-elasticity", type=float, default=DEFAULT_INCOME_ELASTICITY)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = build(
        lgrip30_rasters=args.lgrip30,
        population_csv=args.population,
        gdp_per_capita_csv=args.gdp_per_capita,
        years=args.years,
        shapefile_path=args.shapefile,
        historical_csv=args.historical_csv,
        income_elasticity=args.income_elasticity,
    )
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows covering {df['node'].nunique()} basins to {args.out}")
