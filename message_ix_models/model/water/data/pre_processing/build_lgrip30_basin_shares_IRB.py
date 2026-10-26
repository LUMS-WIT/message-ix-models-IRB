"""Extract real LGRIP30 irrigated/rainfed/cropland area per Indus basin
polygon (all 24 real BCUs -- Afghanistan, China, India and Pakistan
portions of the river basin, not just the country of Pakistan), from
user-downloaded LGRIP30 GeoTIFF tiles.

This is the raster-zonal-statistics half of
:mod:`build_irrigation_demand_lgrip30`'s pipeline (its
``read_basin_geometries`` + ``lgrip30_basin_shares`` + ``add_country_share``
steps, reused here rather than reimplemented) run on its own, producing a
committed CSV -- the concrete first step of the "LGRIP30 spatial validation"
that :mod:`~..crops.irrigation_tech`'s and
:mod:`build_irrigation_demand_lgrip30`'s own module docstrings both flag as
previously deferred for lack of real satellite data. It deliberately stops
before that script's temporal (population/GDP demand-index) step -- this
script's job is only "how much irrigated/rainfed cropland does LGRIP30
actually show per basin," which stands on its own as a real validation
signal regardless of whether/how it's later wired into a demand mechanism.

Needs ``rasterio`` + ``geopandas`` (see
:mod:`build_irrigation_demand_lgrip30`'s docstring for why that's a
deliberate, confirmed exception to this directory's usual
dependency-free convention) -- both must be installed separately; not a
committed requirement of the base package.

Usage
-----
::

    python build_lgrip30_basin_shares_IRB.py \\
        --lgrip30 C:/Users/IST/Downloads/LGRIP30_.../LGRIP30_2015_N20E60_*.tif \\
                  C:/Users/IST/Downloads/LGRIP30_.../LGRIP30_2015_N20E70_*.tif \\
                  ... \\
        --out lgrip30_basin_shares_IRB.csv

The Indus basin's real extent (all 24 BCUs, from
``network/raw/Indus_bcu.shp``'s own bounds) is lon 66.16-82.45E,
lat 24.04-37.10N -- i.e. LGRIP30's 10-degree tiles N20E60, N20E70, N20E80,
N30E60, N30E70, N30E80. All six are required for full basin coverage; a
subset will still run, but basins not covered by any supplied tile get
0 km^2 everywhere (see the "coverage" column this script adds, beyond what
:func:`build_irrigation_demand_lgrip30.lgrip30_basin_shares` itself
reports, to make missing-tile gaps visible rather than silently reading as
"zero irrigation").
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# build_irrigation_demand_lgrip30.py lives alongside this script and does
# its own `from _pid_crosswalk import CROSSWALK` sibling import -- both only
# resolve if this directory is on sys.path, same requirement that file's own
# __main__ block already has.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_irrigation_demand_lgrip30 import (  # noqa: E402
    DEFAULT_SHAPEFILE,
    add_country_share,
    lgrip30_basin_shares,
    read_basin_geometries,
    region_group_of_irb,
)

log = logging.getLogger(__name__)


def build(
    lgrip30_rasters: list[Path],
    shapefile_path: Path = DEFAULT_SHAPEFILE,
    pid_crosswalk: dict[str, str] | None = None,
    region_group_of=region_group_of_irb,
):
    """Run just the spatial (raster zonal-statistics) half of
    :mod:`build_irrigation_demand_lgrip30`'s pipeline.

    Returns
    -------
    pandas.DataFrame
        Columns ``node``, ``irrigated_km2``, ``rainfed_km2``,
        ``cropland_km2``, ``region_group``, ``country_share``, ``covered``
        (whether at least one supplied tile overlapped this basin at all --
        distinct from a real 0 km^2 result).
    """
    basins = read_basin_geometries(shapefile_path, pid_crosswalk)
    log.info("Loaded %d real basin polygons from %s", len(basins), shapefile_path)

    shares = lgrip30_basin_shares(basins, lgrip30_rasters)
    shares = add_country_share(shares, region_group_of)

    # lgrip30_basin_shares() silently returns 0.0 for a basin no supplied
    # tile overlaps at all (same as a basin with genuinely zero cropland) --
    # recompute coverage explicitly here so the two cases are distinguishable
    # in the output CSV, since a missing tile is an input-coverage problem,
    # not a real finding about that basin.
    import geopandas as gpd
    import rasterio

    tile_bounds = []
    for raster_path in lgrip30_rasters:
        with rasterio.open(raster_path) as src:
            tile_bounds.append(gpd.GeoSeries([_box(src.bounds)], crs=src.crs))

    covered = []
    for _, basin in basins.iterrows():
        geom = gpd.GeoSeries([basin.geometry], crs=basins.crs)
        covered.append(any(geom.iloc[0].intersects(tb.iloc[0]) for tb in tile_bounds))
    shares["covered"] = covered

    uncovered = shares.loc[~shares["covered"], "node"].tolist()
    if uncovered:
        log.warning(
            "%d/%d basin(s) not covered by any supplied LGRIP30 tile -- their "
            "km^2 values are NOT real zeros, just missing input: %s",
            len(uncovered), len(shares), sorted(uncovered),
        )
    return shares


def _box(bounds):
    from shapely.geometry import box

    return box(bounds.left, bounds.bottom, bounds.right, bounds.top)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lgrip30", nargs="+", required=True, type=Path, help="LGRIP30 GeoTIFF tile(s)")
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = build(lgrip30_rasters=args.lgrip30, shapefile_path=args.shapefile)
    result.to_csv(args.out, index=False)
    print(f"Wrote {len(result)} basin rows to {args.out}")
    print(result.to_string(index=False))
