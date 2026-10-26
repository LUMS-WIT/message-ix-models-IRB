"""Extract real SPAM2020 (v2r2) crop harvested-area data for the Indus
basin from user-supplied global GeoTIFFs -- every one of SPAM's 46 crops,
at each of its 3 technology splits (All / Irrigated / Rainfed), per real
basin polygon (``Indus_bcu.shp``'s 24 BCUs) and per Pakistan's real country
boundary.

Source data (not committed -- supply your own; not bundled with this repo)
---------------------------------------------------------------------------
``spam2020_V2r2_global_H_<CROP>_<TECH>.tif``, one file per (crop, tech):

- 46 crop codes (SPAM's own standard list -- BANA, BARL, BEAN, CASS, CHIC,
  CITR, CNUT, COCO, COFF, COTT, COWP, GROU, LENT, MAIZ, MILL, OCER, OFIB,
  OILP, ONIO, OOIL, OPUL, ORTS, PIGE, PLNT, PMIL, POTA, RAPE, RCOF, REST,
  RICE, RUBB, SESA, SORG, SOYB, SUGB, SUGC, SUNF, SWPO, TEAS, TEMF, TOBA,
  TOMA, TROF, VEGE, WHEA, YAMS -- confirmed present, 46 x 3 = 138 files, in
  the directory this script was pointed at).
- Technology split: ``A`` (all technologies combined), ``I`` (irrigated),
  ``R`` (rainfed) -- NOTE this is SPAM's own 3-way split in this particular
  download; SPAM2020 also publishes a finer rainfed breakout (high/low
  input/subsistence) in some distributions, not present here.
- Global grid: 4320 x 2160, 5 arc-minute (1/12 degree) resolution, EPSG:4326,
  float32, nodata -3.4028235e38 (confirmed directly from the files, not
  assumed). Harvested area in hectares (SPAM's own standard unit and
  definition -- NOT physical area: a double-cropped cell's harvested area
  counts each season, so it can legitimately exceed the cell's own physical
  area; this is a real, documented SPAM convention, not a data error).

Method
------
Each basin polygon (and Pakistan's real country boundary, cached from
:mod:`..pre_processing` -- actually from the sibling NEST-Pakistan project's
``pakistan_country_boundary.geojson``, see :data:`PAKISTAN_BOUNDARY_GEOJSON`)
is rasterized ONCE onto SPAM's own native grid (not the other way around --
SPAM's grid is coarse, 5 arc-min/~9km, so this is the cheap direction), then
every one of the 138 input files is read once and summed per zone via
``numpy.bincount`` -- avoids 138 x 25 repeated polygon-mask calls.

Outputs
-------
1. ``spam2020_crop_harvested_area_basin_IRB.csv`` -- long format: node,
   crop, tech, harvested_area_ha. 24 basins x 46 crops x 3 tech = 3312 rows.
2. ``spam2020_crop_harvested_area_pakistan_country_IRB.csv`` -- same, but
   summed over Pakistan's real full country boundary (not just its
   Indus-basin portion) -- 46 crops x 3 tech = 138 rows.
3. (optional, --project-crops) ``spam2020_crop_harvested_area_project_crops_IRB.csv``
   -- a best-effort rollup onto this project's own 9 modeled crops
   (crops/common.py's CROPS), for crops where the mapping is unambiguous
   (cotton->COTT, maize->MAIZ, rice->RICE, sugarcane->SUGC, wheat->WHEA) or
   a documented many-to-one aggregate (fruit, pulses -- see
   :data:`PROJECT_CROP_CROSSWALK`'s own comments for exactly which SPAM
   codes are summed and why). This is a judgment call, not an authoritative
   crosswalk -- use output #1 (the raw 46-crop table) as the primary,
   unambiguous source; this one is a convenience view.

Usage
-----
::

    python build_spam2020_crop_harvested_area_IRB.py \\
        --spam-dir "C:/Users/IST/Documents/Faraz Data/spam2020V2r2_global_harvested_area" \\
        --out-basin spam2020_crop_harvested_area_basin_IRB.csv \\
        --out-pakistan spam2020_crop_harvested_area_pakistan_country_IRB.csv \\
        --project-crops
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pid_crosswalk import CROSSWALK  # noqa: E402

log = logging.getLogger(__name__)

_NETWORK_RAW_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "network" / "raw"
_CROPS_RAW_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "crops" / "raw"
DEFAULT_BASIN_SHAPEFILE = _NETWORK_RAW_DIR / "Indus_bcu.shp"
DEFAULT_PROVINCE_SHAPEFILE = _CROPS_RAW_DIR / "Indus_prov_bcu.shp"

#: Cached from Natural Earth (see render_lgrip30_maps.py in the sibling
#: NEST-Pakistan project, which fetched and cached it the same way) --
#: not part of this package's own committed data, supply your own path if
#: it has moved.
DEFAULT_PAKISTAN_BOUNDARY = Path(
    r"C:\Users\IST\Documents\WIT-WORK\NEST-Pakistan\NEST_IRB\data\pakistan_country_boundary.geojson"
)

#: SPAM2020's 46 standard crop codes -- confirmed against the actual
#: filenames present (46 crops x 3 tech = 138 files), not assumed from
#: SPAM's documentation alone.
CROPS_SPAM = [
    "BANA", "BARL", "BEAN", "CASS", "CHIC", "CITR", "CNUT", "COCO", "COFF",
    "COTT", "COWP", "GROU", "LENT", "MAIZ", "MILL", "OCER", "OFIB", "OILP",
    "ONIO", "OOIL", "OPUL", "ORTS", "PIGE", "PLNT", "PMIL", "POTA", "RAPE",
    "RCOF", "REST", "RICE", "RUBB", "SESA", "SORG", "SOYB", "SUGB", "SUGC",
    "SUNF", "SWPO", "TEAS", "TEMF", "TOBA", "TOMA", "TROF", "VEGE", "WHEA",
    "YAMS",
]
TECHS = ["A", "I", "R"]

#: EXHAUSTIVE rollup of all 46 SPAM crop codes onto this project's 9
#: modeled crops (crops/common.py's CROPS) -- every SPAM code appears in
#: exactly one bucket (confirmed: 2+1+5+2+2+8+6+12+8 = 46), so no real
#: Indus-basin crop is silently left unrepresented. This is a deliberate
#: choice to cover everything rather than leave ~20 real, non-trivial
#: crops (potato, rapeseed, sesame, groundnut, sorghum, barley, ...)
#: unmapped, as the first version of this crosswalk did -- it trades away
#: some precision (several groupings below are real judgment calls, not
#: SPAM's own categories) for completeness, on the explicit basis that no
#: model code changes (crops/common.py's CROPS stays at exactly 9).
#: Fit strength is noted per bucket; weakest fits are called out plainly
#: rather than presented as equally solid.
PROJECT_CROP_CROSSWALK: dict[str, list[str]] = {
    # --- strong fits: same crop or same narrow family ---
    "wheat": ["WHEA", "BARL"],                              # both temperate
    # rabi (winter) cereal grains, same growing season and water pattern.
    "rice": ["RICE"],                                        # direct match, no close relative worth merging in
    "maize": ["MAIZ", "SORG", "PMIL", "MILL", "OCER"],       # all coarse
    # kharif (summer) cereal grains -- same family as maize agronomically,
    # not a SPAM category of its own.
    "cotton": ["COTT", "OFIB"],                               # fiber crops
    "sugarcane": ["SUGC", "SUGB"],                            # sugar crops
    # --- moderate fits: same real-world "vegetable/root crop" shelf, even
    # though SPAM tracks each as its own code ---
    "vegetables": ["VEGE", "POTA", "ONIO", "TOMA", "ORTS", "SWPO", "CASS", "YAMS"],
    # closes the real gap flagged earlier: potato/onion/tomato are
    # genuine, non-trivial Indus-basin crops SPAM tracks separately from
    # its own "VEGE" aggregate; CASS/YAMS are ~0 ha in the Indus basin but
    # are the same root-crop family, included for completeness.
    "pulses": ["BEAN", "CHIC", "COWP", "LENT", "PIGE", "OPUL"],  # all SPAM
    # legumes -- unchanged from the original crosswalk.
    "fruit": ["BANA", "CITR", "PLNT", "TEMF", "TROF", "COCO", "COFF", "RCOF", "TEAS", "RUBB", "OILP", "CNUT"],
    # perennial/plantation tree crops -- BANA/CITR/PLNT/TEMF/TROF are
    # genuinely fruit; COCO/COFF/RCOF/TEAS/RUBB/OILP/CNUT are NOT fruit in
    # the normal sense (cocoa, coffee, tea, rubber, oil palm, coconut) but
    # share fruit's perennial-tree water-use pattern and have no other
    # natural home among these 9 buckets -- all are ~0 ha in the real
    # Indus basin (confirmed: among the 13 SPAM crops with zero real area
    # here), so this placement doesn't materially affect any real number.
    # --- weakest fit: the designated catch-all, called out explicitly ---
    "fodder": ["OOIL", "RAPE", "SESA", "GROU", "SUNF", "SOYB", "TOBA", "REST"],
    # NOT a real agronomic match -- oilseeds (OOIL/RAPE/SESA/GROU/SUNF/
    # SOYB), tobacco, and SPAM's own "REST" catch-all have nothing in
    # common with fodder/forage crops. They're placed here only because
    # "fodder" is the one model crop with zero SPAM equivalent of its own,
    # making it the only bucket that doesn't already have a better-fitting
    # home for them. This group is real and non-trivial in the Indus
    # basin (oilseeds alone: ~1.18M ha, comparable to cotton's ~1.6M ha) --
    # treat any "fodder" total built from this crosswalk as a stand-in for
    # "everything else," not as real fodder/forage water demand.
}


def build_zone_raster(shapefile_path: Path, grid_transform, grid_shape, id_field: str = "PID") -> tuple[np.ndarray, list[str]]:
    """Rasterize every polygon in `shapefile_path` onto SPAM's own grid.
    Returns (zone_array, labels) where zone_array holds 1..N (0 = outside
    every polygon) and labels[i] is the id_field value for zone i+1.
    """
    gdf = gpd.read_file(shapefile_path)
    labels = gdf[id_field].astype(str).tolist()
    shapes = [(geom, i + 1) for i, geom in enumerate(gdf.geometry)]
    zones = rasterize(shapes, out_shape=grid_shape, transform=grid_transform, fill=0, dtype="int32")
    return zones, labels


def zonal_sums(zones: np.ndarray, n_zones: int, spam_dir: Path) -> pd.DataFrame:
    """Sum every (crop, tech) GeoTIFF's pixel values per zone via
    bincount -- each file is read exactly once."""
    zones_flat = zones.ravel()
    rows = []
    for crop in CROPS_SPAM:
        for tech in TECHS:
            path = spam_dir / f"spam2020_V2r2_global_H_{crop}_{tech}.tif"
            if not path.exists():
                # Filenames in the wild have been seen with both
                # "spam2020_V2r2" and "spam2020V2r2" casing -- try the
                # alternate before giving up on a real file.
                alt = spam_dir / f"spam2020V2r2_global_H_{crop}_{tech}.tif"
                path = alt if alt.exists() else path
            if not path.exists():
                log.warning("Missing file for crop=%s tech=%s (%s) -- skipped", crop, tech, path.name)
                continue
            with rasterio.open(path) as src:
                band = src.read(1)
                nodata = src.nodata
            band = np.where(band == nodata, 0.0, band) if nodata is not None else band
            band = np.where(band < 0, 0.0, band)  # defensive: no real harvested area is negative
            sums = np.bincount(zones_flat, weights=band.ravel(), minlength=n_zones + 1)
            for zone_i in range(1, n_zones + 1):
                rows.append({"zone": zone_i, "crop": crop, "tech": tech, "harvested_area_ha": float(sums[zone_i])})
    return pd.DataFrame(rows)


def build_province_level(spam_dir: Path, province_shapefile: Path, grid_transform, grid_shape) -> pd.DataFrame:
    """SPAM harvested area for all 41 real province x basin polygons
    (``Indus_prov_bcu.shp``) -- genuinely finer than the 24-basin table,
    same real zonal-sum method, not a redistribution of the basin totals.
    Labeled by the province shapefile's own PID/ADMIN/SUBASIN/REGION, same
    convention as build_lgrip30_precise_IRB.py's province output -- no
    attempt to assign a model `node` (see that script's docstring for why:
    one SUBASIN label can legitimately span more than one country's basin).
    """
    gdf = gpd.read_file(province_shapefile)
    zones, labels = build_zone_raster(province_shapefile, grid_transform, grid_shape, id_field="PID")
    df = zonal_sums(zones, len(labels), spam_dir)
    meta = gdf[["PID", "ADMIN", "SUBASIN", "REGION"]].reset_index(drop=True)
    meta["zone"] = range(1, len(meta) + 1)
    df = df.merge(meta, on="zone", how="left").drop(columns="zone")
    return df[["PID", "ADMIN", "SUBASIN", "REGION", "crop", "tech", "harvested_area_ha"]]


def build(spam_dir: Path, basin_shapefile: Path, pakistan_boundary: Path, province_shapefile: Path | None = None):
    with rasterio.open(next(spam_dir.glob("spam*_H_WHEA_A.tif"))) as ref:
        grid_transform, grid_shape = ref.transform, ref.shape

    log.info("Rasterizing 24 real Indus basin polygons onto SPAM's native grid ...")
    basin_zones, basin_labels = build_zone_raster(basin_shapefile, grid_transform, grid_shape, id_field="PID")
    basin_df = zonal_sums(basin_zones, len(basin_labels), spam_dir)
    # basin_labels are the shapefile's own PID values (e.g. "PAK_1") --
    # crosswalked here to this project's real node/BCU_name convention
    # (e.g. "12|PAKISTAN"), the same mapping every other build_*_IRB.py
    # script in this directory uses, so this output joins directly onto
    # the rest of the project's node-keyed data.
    basin_df["node"] = basin_df["zone"].map(lambda z: CROSSWALK.get(basin_labels[z - 1], basin_labels[z - 1]))
    basin_df = basin_df.drop(columns="zone")[["node", "crop", "tech", "harvested_area_ha"]]

    log.info("Rasterizing Pakistan's real country boundary onto SPAM's native grid ...")
    pak_gdf = gpd.read_file(pakistan_boundary)
    pak_gdf = pak_gdf.assign(_id="PAKISTAN")
    pak_zones, _ = build_zone_raster_from_gdf(pak_gdf, grid_transform, grid_shape)
    pak_df = zonal_sums(pak_zones, 1, spam_dir)
    pak_df = pak_df.drop(columns="zone")[["crop", "tech", "harvested_area_ha"]]

    province_df = None
    if province_shapefile is not None:
        log.info("Rasterizing 41 real province x basin polygons onto SPAM's native grid ...")
        province_df = build_province_level(spam_dir, province_shapefile, grid_transform, grid_shape)

    return basin_df, pak_df, province_df


def build_zone_raster_from_gdf(gdf, grid_transform, grid_shape):
    shapes = [(geom, 1) for geom in gdf.geometry]
    zones = rasterize(shapes, out_shape=grid_shape, transform=grid_transform, fill=0, dtype="int32")
    return zones, ["PAKISTAN"]


def project_crop_rollup(basin_df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort rollup of the raw 46-crop basin table onto this
    project's own 9 modeled crops -- see PROJECT_CROP_CROSSWALK's own
    comments for exactly what's summed and why; NOT authoritative."""
    rows = []
    for project_crop, spam_codes in PROJECT_CROP_CROSSWALK.items():
        subset = basin_df[basin_df["crop"].isin(spam_codes)]
        summed = subset.groupby(["node", "tech"], as_index=False)["harvested_area_ha"].sum()
        summed["project_crop"] = project_crop
        summed["spam_codes_summed"] = ",".join(spam_codes)
        rows.append(summed)
    return pd.concat(rows, ignore_index=True)[["node", "project_crop", "tech", "harvested_area_ha", "spam_codes_summed"]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spam-dir", required=True, type=Path)
    parser.add_argument("--basin-shapefile", type=Path, default=DEFAULT_BASIN_SHAPEFILE)
    parser.add_argument("--pakistan-boundary", type=Path, default=DEFAULT_PAKISTAN_BOUNDARY)
    parser.add_argument("--out-basin", type=Path, required=True)
    parser.add_argument("--out-pakistan", type=Path, required=True)
    parser.add_argument("--project-crops", action="store_true", help="Also write the project-crop rollup CSV")
    parser.add_argument("--out-project-crops", type=Path, default=None)
    parser.add_argument("--province-shapefile", type=Path, default=DEFAULT_PROVINCE_SHAPEFILE)
    parser.add_argument("--out-province", type=Path, default=None, help="Also write the 41-province breakdown CSV")
    args = parser.parse_args()

    basin_df, pak_df, province_df = build(
        args.spam_dir, args.basin_shapefile, args.pakistan_boundary,
        province_shapefile=args.province_shapefile if args.out_province else None,
    )
    basin_df.to_csv(args.out_basin, index=False)
    print(f"Wrote {len(basin_df)} rows to {args.out_basin}")

    pak_df.to_csv(args.out_pakistan, index=False)
    print(f"Wrote {len(pak_df)} rows to {args.out_pakistan}")

    if args.out_province and province_df is not None:
        province_df.to_csv(args.out_province, index=False)
        print(f"Wrote {len(province_df)} rows to {args.out_province}")

    if args.project_crops:
        out_pc = args.out_project_crops or args.out_basin.with_name("spam2020_crop_harvested_area_project_crops_IRB.csv")
        pc_df = project_crop_rollup(basin_df)
        pc_df.to_csv(out_pc, index=False)
        print(f"Wrote {len(pc_df)} rows to {out_pc}")

    print("\nPakistan (full country) top-10 crops by harvested area, tech=A:")
    top10 = pak_df[pak_df["tech"] == "A"].sort_values("harvested_area_ha", ascending=False).head(10)
    print(top10.to_string(index=False))
