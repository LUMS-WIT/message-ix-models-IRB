"""Build ``data/water/network/basin_links_IRB.csv`` from real Indus basin data.

This is a one-off, standalone conversion script -- not part of the
importable :mod:`message_ix_models` package, run manually whenever the
source data changes. It documents exactly where the real inter-basin
network data (river routing, canals, transmission lines) that
:mod:`message_ix_models.model.water.data.network` consumes actually came
from, and how it was converted -- so the conversion is reproducible and
auditable, per this repo's own data-handling conventions (``doc/data.rst``:
"if message_ix_models relies on certain adjustments to the data, do not
commit the adjusted data [alone] -- commit code that performs the
adjustments").

Source data
-----------
Originally from the team's SharePoint ("NEST Indus/data_phase_1/Abdul
Moiz's Data" and other synced copies of the same `indus_model/input`
folder), now committed in this repo under ``data/water/network/raw/``
(small: shapefile + CSVs together are under 2.5 MB, well within a normal
git object -- no Git LFS needed)::

    data/water/network/raw/
        Indus_bcu.shp / .dbf / .shx / .prj  -- master basin shapefile:
                                                PID, DOWN (downstream
                                                basin), REGION, outlet
                                                coordinates. All 24 basins.
                                                Only .dbf (the attribute
                                                table) is read by this
                                                script; .shp/.shx/.prj are
                                                kept alongside so the file
                                                is a complete, usable
                                                shapefile for anyone who
                                                wants the geometry too.
        indus_bcu_canals.csv                -- real named canals: PID_i,
                                                PID_o, capacity_m3_per_sec,
                                                length_km.
        PID_distances_km.csv                -- real basin-to-basin
                                                distances.
        basin_transmission/
            existing_routes.csv             -- real electricity
                                                transmission routes + 2015
                                                capacity (MW).

(Committed rather than fetched or left local-only, per this project's "no
external data fetching" convention and an explicit decision to keep this
raw source in-repo for reproducibility.)

These are the exact files the legacy R+GAMS indus_ix model
(``NEST/MESSAGEix/basin_msggdx_load_inputs.r``) reads to build its own
per-arc technologies -- this script is the Python-side equivalent
conversion, targeting :mod:`network`'s CSV schema instead of one
hand-written GAMS technology block per link type.

The PID -> BCU_name crosswalk
------------------------------
The source files name basins ``AFG_1``, ``AFG_2``, ``CHN_1..3``,
``IND_1..6``, ``PAK_1..13`` -- not the ``1|AFGHAN_SOUTH``, ``12|PAKISTAN``
naming already used by ``basins_by_region_simpl_IRB.csv`` (the delineation
this project's Python model already relies on). The crosswalk was first
inferred ordinally (PID suffix order within each country) and found wrong
for 16 of 24 basins once the delineation shapefile behind the hydrology
data was located; it is now spatially verified -- see
:mod:`_pid_crosswalk`. Only ``CROSSWALK`` there needs correcting if the
delineation ever changes -- everything downstream of it is generic.

Usage
-----
Run this script directly -- no editing needed, source data is committed
in-repo. It overwrites ``data/water/network/basin_links_IRB.csv`` in
place::

    python build_basin_links_IRB.py
"""

from pathlib import Path

import pandas as pd

from _pid_crosswalk import CROSSWALK
from _shapefile import read_dbf

_NETWORK_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "network"

SOURCE_DIR = _NETWORK_DATA_DIR / "raw"
OUTPUT_PATH = _NETWORK_DATA_DIR / "basin_links_IRB.csv"


def build() -> pd.DataFrame:
    rows = []

    # --- 1. River routing, from Indus_bcu.dbf's DOWN field ---
    bcu = {r["PID"]: r for r in read_dbf(SOURCE_DIR / "Indus_bcu.dbf")}
    assert set(bcu) == set(CROSSWALK), (
        f"Indus_bcu.dbf PIDs don't match CROSSWALK: {set(bcu) ^ set(CROSSWALK)}"
    )
    for pid, r in bcu.items():
        down = r["DOWN"]
        if not down:
            continue  # PAK_4: outlet to the sea, no downstream edge
        rows.append({
            "from_basin": CROSSWALK[pid], "to_basin": CROSSWALK[down], "link_type": "river",
            "name": f"real DOWN routing: {pid}->{down}", "distance_km": "",
            "existing_capacity": "", "capacity_unit": "MCM/year", "bidirectional": "false",
            "source": "Indus_bcu.dbf DOWN field, via inferred PID crosswalk",
        })

    # --- 2. Real canals, from indus_bcu_canals.csv ---
    canals = pd.read_csv(SOURCE_DIR / "indus_bcu_canals.csv")
    for _, r in canals.iterrows():
        pid_i, pid_o = r["PID_i"], r["PID_o"]
        if pid_i not in CROSSWALK or pid_o not in CROSSWALK:
            continue
        cap_mcm_year = r["capacity_m3_per_sec"] * 31.536  # m3/s -> MCM/year
        rows.append({
            "from_basin": CROSSWALK[pid_i], "to_basin": CROSSWALK[pid_o], "link_type": "canal_conv",
            "name": f"real canal: {r['name']}", "distance_km": r["length_km"],
            "existing_capacity": round(cap_mcm_year, 1), "capacity_unit": "MCM/year",
            "bidirectional": "false", "source": "indus_bcu_canals.csv, via inferred PID crosswalk",
        })

    # --- 3. Real transmission routes, from basin_transmission/existing_routes.csv ---
    trans = pd.read_csv(SOURCE_DIR / "basin_transmission" / "existing_routes.csv")
    trans.columns = [c.strip() for c in trans.columns]
    seen_routes = set()
    for _, r in trans.iterrows():
        tec = r["tec"]
        if "|" not in tec or tec in seen_routes:
            continue
        pid_i, pid_o = tec.split("|")
        if pid_i not in CROSSWALK or pid_o not in CROSSWALK:
            continue
        seen_routes.add(tec)
        cap_mw = r["value"]
        rows.append({
            "from_basin": CROSSWALK[pid_i], "to_basin": CROSSWALK[pid_o], "link_type": "transmission",
            "name": f"real transmission route: {tec}", "distance_km": "",
            "existing_capacity": cap_mw if cap_mw and cap_mw > 0 else "",
            "capacity_unit": "MW", "bidirectional": "true",
            "source": "basin_transmission/existing_routes.csv, via inferred PID crosswalk",
        })

    # --- 4. Fill in real distances for river/transmission rows ---
    dist_df = pd.read_csv(SOURCE_DIR / "PID_distances_km.csv")
    dist_lookup = {}
    for _, r in dist_df.iterrows():
        if "|" not in str(r["tec"]):
            continue
        a, b = r["tec"].split("|")
        if a in CROSSWALK and b in CROSSWALK:
            dist_lookup[frozenset({CROSSWALK[a], CROSSWALK[b]})] = r["dist"]
    for row in rows:
        if row["distance_km"] == "":
            key = frozenset({row["from_basin"], row["to_basin"]})
            if key in dist_lookup:
                row["distance_km"] = round(dist_lookup[key], 1)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()

    # Sanity check: this script must never silently produce a technology
    # name collision -- network.py disambiguates duplicates with a "#N"
    # suffix, but that's meant for genuinely distinct real links (e.g.
    # three separate real canals between the same basin pair), not a bug
    # in this conversion.
    check = df["link_type"] + "|" + df["from_basin"] + "|" + df["to_basin"]
    dupes = check[check.duplicated(keep=False)]
    if not dupes.empty:
        print(f"NOTE: {len(dupes)} rows share a (link_type, from, to) triple -- "
              "expected for real parallel infrastructure, network.py will "
              "disambiguate with #2/#3 suffixes:")
        print(df.loc[dupes.index, ["from_basin", "to_basin", "link_type", "name"]])

    print(f"Built {len(df)} rows: {df['link_type'].value_counts().to_dict()}")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
