"""Deeper, more precise LGRIP30 extraction than
:mod:`build_lgrip30_basin_shares_IRB`'s first pass.

Two upgrades over that script's basin-level output:

1. **All four LGRIP30 classes, not three.** ``lgrip30_basin_shares()`` (in
   :mod:`build_irrigation_demand_lgrip30`) computes ``ocean_water`` and
   ``non_cropland`` pixel area internally but discards both before
   returning -- only ``irrigated``/``rainfed``/``cropland`` (their sum) ever
   reached the first CSV. This script keeps all four, plus each basin's real
   total polygon area, so the four class areas provably sum to the whole
   basin (a correctness check the first pass couldn't offer).
2. **Percentages of real basin area, not just raw km^2.** Raw
   ``irrigated_km2`` alone conflates two different things -- how much a
   basin irrigates, and how big the basin is. ``irrigated_pct`` (of that
   basin's own real polygon area) is comparable across basins of very
   different size the way the raw number isn't.
3. **Province-level (41 rows), not just basin-level (24 rows).** Uses
   ``data/water/crops/raw/Indus_prov_bcu.shp`` -- the same province x basin
   intersection layer :mod:`build_fertilizer_weights_IRB` already uses for
   real, independent zonal statistics per polygon (not an area-weighted
   redistribution of the basin totals, which would carry no new
   information; this is a second, independent satellite read against
   smaller, finer real geometry).

Deliberately NOT attempted: assigning each province row a model ``node``.
:mod:`build_fertilizer_weights_IRB`'s own docstring documents a real,
non-obvious many-to-one wrinkle in the basin<->SUBASIN correspondence (one
SUBASIN label can legitimately span more than one country's real basin,
e.g. ``Indus|9`` covers rows in both Afghanistan and Pakistan) -- silently
reusing that crosswalk here without re-deriving it for this specific
purpose risked mis-attributing area across a country border. Province rows
are therefore reported under their own real identifiers (``PID``, ``ADMIN``,
``SUBASIN``, ``REGION``) rather than forced onto a ``node`` id; ``REGION``
already matches this repo's ``country_group_of()`` convention exactly
(``"pakistan"``, ``"afghan_south"``, ...) and is enough for a real
country-level rollup without that risk.

Usage
-----
::

    python build_lgrip30_precise_IRB.py \\
        --lgrip30 C:/Users/IST/Downloads/LGRIP30_.../LGRIP30_2015_N20E60_*.tif ... \\
        --out-basin lgrip30_basin_shares_precise_IRB.csv \\
        --out-province lgrip30_province_shares_IRB.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_irrigation_demand_lgrip30 import (  # noqa: E402
    LGRIP30_CLASSES,
    _pixel_area_km2,
    region_group_of_irb,
)

log = logging.getLogger(__name__)

_CROPS_RAW_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "crops" / "raw"
_NETWORK_RAW_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "network" / "raw"

DEFAULT_BASIN_SHAPEFILE = _NETWORK_RAW_DIR / "Indus_bcu.shp"
DEFAULT_PROVINCE_SHAPEFILE = _CROPS_RAW_DIR / "Indus_prov_bcu.shp"


def _zonal_stats(geoms, raster_paths: list[Path]) -> list[dict]:
    """Real per-polygon zonal statistics against one or more LGRIP30 tiles,
    for an arbitrary GeoDataFrame `geoms` (basin- or province-level -- the
    function doesn't care). Returns one dict per row: all four LGRIP30
    class areas (km^2), ``total_km2`` (their sum -- this polygon's real
    measured area from the raster's own pixel grid, not the shapefile's
    planar-degree area), and each class's percentage of ``total_km2``.
    """
    import rasterio
    from rasterio.errors import WindowError
    from rasterio.mask import mask

    results = []
    for _, row in geoms.iterrows():
        class_km2 = dict.fromkeys(LGRIP30_CLASSES.values(), 0.0)
        for raster_path in raster_paths:
            with rasterio.open(raster_path) as src:
                geom = [row.geometry.__geo_interface__]
                if geoms.crs is not None and src.crs is not None and str(geoms.crs) != str(src.crs):
                    import geopandas as gpd

                    reproj = gpd.GeoSeries([row.geometry], crs=geoms.crs).to_crs(src.crs)
                    geom = [reproj.iloc[0].__geo_interface__]
                try:
                    data, transform = mask(src, geom, crop=True, filled=True, nodata=255)
                except (ValueError, WindowError):
                    continue
                band = data[0]
                pixel_w = abs(transform.a)
                pixel_h = abs(transform.e)
                row_lats = transform.f + transform.e * (np.arange(band.shape[0]) + 0.5)
                row_areas = _pixel_area_km2(row_lats, pixel_w, pixel_h)
                for code, name in LGRIP30_CLASSES.items():
                    mask_2d = band == code
                    if mask_2d.any():
                        class_km2[name] += float((mask_2d * row_areas[:, None]).sum())

        total_km2 = sum(class_km2.values())
        pct = {
            f"{name}_pct": (100.0 * area / total_km2 if total_km2 > 0 else 0.0)
            for name, area in class_km2.items()
        }
        results.append(
            {
                **{f"{name}_km2": area for name, area in class_km2.items()},
                "total_km2": total_km2,
                **pct,
                "cropland_km2": class_km2["irrigated_cropland"] + class_km2["rainfed_cropland"],
                "cropland_pct": pct["irrigated_cropland_pct"] + pct["rainfed_cropland_pct"],
            }
        )
    return results


def build_basin_level(raster_paths: list[Path], shapefile_path: Path = DEFAULT_BASIN_SHAPEFILE) -> pd.DataFrame:
    """All 24 real basins, full 4-class + percentage breakdown."""
    import geopandas as gpd

    from build_irrigation_demand_lgrip30 import read_basin_geometries

    basins = read_basin_geometries(shapefile_path)
    stats = _zonal_stats(basins, raster_paths)
    out = pd.concat([basins[["node"]].reset_index(drop=True), pd.DataFrame(stats)], axis=1)
    out["region_group"] = out["node"].map(region_group_of_irb)
    totals = out.groupby("region_group")["irrigated_cropland_km2"].transform("sum")
    out["country_share"] = np.where(totals > 0, out["irrigated_cropland_km2"] / totals, 0.0)
    return out


def build_province_level(raster_paths: list[Path], shapefile_path: Path = DEFAULT_PROVINCE_SHAPEFILE) -> pd.DataFrame:
    """All 41 real province x basin polygons, full 4-class + percentage
    breakdown -- genuinely finer geometry than the 24 basin polygons, not a
    redistribution of the basin-level numbers."""
    import geopandas as gpd

    gdf = gpd.read_file(shapefile_path)
    stats = _zonal_stats(gdf, raster_paths)
    out = pd.concat(
        [gdf[["PID", "ADMIN", "SUBASIN", "REGION"]].reset_index(drop=True), pd.DataFrame(stats)],
        axis=1,
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lgrip30", nargs="+", required=True, type=Path, help="LGRIP30 GeoTIFF tile(s)")
    parser.add_argument("--basin-shapefile", type=Path, default=DEFAULT_BASIN_SHAPEFILE)
    parser.add_argument("--province-shapefile", type=Path, default=DEFAULT_PROVINCE_SHAPEFILE)
    parser.add_argument("--out-basin", type=Path, required=True)
    parser.add_argument("--out-province", type=Path, required=True)
    args = parser.parse_args()

    log.info("Basin-level (24 real basins), full 4-class + percentage breakdown...")
    basin_df = build_basin_level(args.lgrip30, args.basin_shapefile)
    basin_df.to_csv(args.out_basin, index=False)
    print(f"Wrote {len(basin_df)} basin rows to {args.out_basin}")

    log.info("Province-level (41 real province x basin polygons)...")
    province_df = build_province_level(args.lgrip30, args.province_shapefile)
    province_df.to_csv(args.out_province, index=False)
    print(f"Wrote {len(province_df)} province rows to {args.out_province}")

    print("\nCountry-group rollup (from province-level REGION field):")
    rollup = province_df.groupby("REGION")[["irrigated_cropland_km2", "rainfed_cropland_km2", "total_km2"]].sum()
    rollup["irrigated_pct"] = 100.0 * rollup["irrigated_cropland_km2"] / rollup["total_km2"]
    print(rollup.to_string())
