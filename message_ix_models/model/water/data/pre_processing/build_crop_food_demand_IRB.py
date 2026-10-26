"""Port the real, population-driven crop food-demand rows out of NEST's
master demand CSV (``indus_demands_new.csv``) into this project's node
convention -- the missing "food demand" half of the crop/irrigation port
(see :mod:`~..crops.irrigation_tech`'s and :mod:`~..crops.land`'s module
docstrings: "nothing upstream of this layer forces any of this to actually
run").

Where this data comes from
---------------------------
Traced directly in ``NEST/MESSAGEix/basin_msggdx_load_inputs.r`` (~line
349): ``indus_demands_new.csv``'s ``sector == 'crop'`` rows become
``demand.par`` at ``level='raw'``, ``commodity=<crop>_yield`` in the R
model -- a real, pre-computed target production trajectory, not something
this port derives live. The R script's own comment describes its
construction: "historical demand of 2000 (GAEZ) projected with FAO
national production levels and SSP population growth" -- i.e. this file
*is* the population-driven food-demand computation
(``per_capita_norm x population``) already done, not something to redo
from a separately-sourced per-capita consumption norm.

``food_import_export.r``'s output (considered and rejected as an
alternative source during this port's design) turned out to be a dead
end -- loaded in ``load_inputs.r`` but never actually joined into
anything downstream, unlike this file.

Input
-----
``indus_demands_new.csv`` (committed here; copied from the same NEST Indus
project folder as this repo's other raw-data ports). Columns: ``scenario,
sector, type, pid, year, month, value, units``. Filtered to ``sector ==
'crop'``, ``scenario == 'SSP2'`` (this repo's existing scenario
convention). ``type`` is the crop name (exact match to
:data:`~..crops.common.CROPS`, no remapping needed -- confirmed:
``cotton, fodder, fruit, maize, pulses, rice, sugarcane, vegetables,
wheat``). ``month`` is always 1 for these rows (annual, not monthly --
no day-weighting needed, unlike the withdrawal CSVs elsewhere in this
package). ``units`` is always ``"kton"``; normalized here to ``"kt"`` to
match :mod:`~..crops.irrigation_tech`'s/:mod:`~..crops.rainfed`'s own
``crop_yield`` output unit string exactly (same physical unit, different
label -- keeping the string identical avoids registering a second
redundant unit).

750 rows: 9 crops x 17 PIDs (Afghanistan + India + Pakistan only -- no
China, matching the rest of this food-trade-adjacent data's real
coverage) x up to 6 decadal years (2015, 2020, 2030, ..., 2060).

Output
------
``crop_food_demand_IRB.csv``: columns ``crop, node (BCU_name), year,
value (kt), unit``.

Usage
-----
::

    python build_crop_food_demand_IRB.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from _pid_crosswalk import CROSSWALK

log = logging.getLogger(__name__)

_CROPS_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "water" / "crops"
DEFAULT_INPUT = _CROPS_DATA_DIR / "indus_demands_new.csv"
DEFAULT_OUTPUT = _CROPS_DATA_DIR / "crop_food_demand_IRB.csv"

#: This repo's existing scenario convention (e.g. OECD_SSP_POP.csv filters,
#: demands.r's own filter) -- the source CSV also carries SSP1/3/4/5 in
#: principle, but only SSP2 rows are actually present for sector=='crop'.
SCENARIO = "SSP2"


def build(input_csv: Path = DEFAULT_INPUT, scenario: str = SCENARIO) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = df[(df["sector"] == "crop") & (df["scenario"] == scenario)].copy()

    df["node"] = df["pid"].map(CROSSWALK)
    unmapped = df[df["node"].isna()]
    if not unmapped.empty:
        log.warning(
            "%d row(s) have a pid not in the crosswalk (pids: %s) -- dropped",
            len(unmapped), sorted(unmapped["pid"].unique().tolist()),
        )
    df = df.dropna(subset=["node"])

    df = df.rename(columns={"type": "crop"})
    df["unit"] = "kt"  # kton -> kt, matches crop_yield output's own unit string

    result = (
        df.groupby(["crop", "node", "year", "unit"], as_index=False)["value"].sum()
    )
    return result[["crop", "node", "year", "value", "unit"]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = build()
    result.to_csv(DEFAULT_OUTPUT, index=False)
    print(
        f"Wrote {len(result)} rows covering {result['node'].nunique()} basins, "
        f"{result['crop'].nunique()} crops to {DEFAULT_OUTPUT}"
    )
