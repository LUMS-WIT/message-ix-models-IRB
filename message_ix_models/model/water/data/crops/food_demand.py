"""The missing demand-side half of the crop/rainfed/irrigation port: a real
``demand`` on ``<crop>_yield``, so crop/rainfed/irrigation technologies have
an actual economic reason to produce food, not only the water-withdrawal
floor's indirect pull (see :mod:`.irrigation_tech`'s and :mod:`.land`'s
module docstrings: "nothing upstream of this layer forces any of this to
actually run").

Data model
----------
One ``demand`` row per (crop, node, year): ``demand[node, crop_yield_commodity(crop),
"crop_yield", year] = crop_food_demand_IRB.csv``'s value (kt) -- built by
``pre_processing/build_crop_food_demand_IRB.py`` from NEST's own
``indus_demands_new.csv`` (a real, already-computed, population-driven
food-demand trajectory; see that script's docstring for the full
provenance).

Why a hard ``demand`` par, unlike the water-withdrawal floor
--------------------------------------------------------------
:mod:`.irrigation_tech`'s ``hist_irr_withdrawal_<basin>`` mechanism is a
*relation*, not a ``demand``, because ``surfacewater_basin`` is a shared,
pooled commodity -- storage, network transfers, and every extraction
technology all draw on it, so a second hard ``demand`` there would
double-book the commodity balance. ``<crop>_yield`` has no such conflict:
checked directly (a full-package search), it is produced *only* by
``rainfed_<crop>``/``irr_<method>_<crop>`` and consumed by nothing else --
the same pattern :mod:`~..demands`'s ``add_sectoral_demands`` already uses
for ``urban_mw``/``rural_mw``/``industry_mw``. A hard ``demand`` here is
the correct mechanism, not a shortcut.

``food_import_<crop>``: a backstop, not an optional extra
------------------------------------------------------------
A hard ``demand`` with nothing softening it is only ever as safe as the
model's real production capacity -- confirmed for real: a full nexus
build+solve with this demand alone produced a proven-infeasible scenario
(GAMS conflict analysis: basin ``1|AFGHAN_SOUTH``, ``rice_yield`` demand
of ~21 kt at 2030, unreachable given that basin's real water availability
even using every irrigation method at its limit). Unlike the water floor
(a *relation*, softened by ``SLACK_RELATION_BOUND_LO``), a ``demand`` par
has no built-in slack -- so this module adds one explicitly: a
``food_import_<crop>`` technology at every node with real demand data,
producing ``<crop>_yield`` from nothing (no real ``input`` -- it stands in
for food sourced from outside the modeled system, the same conceptual role
NEST's own ``food_import_export.r`` gestured at, though that specific
script's output was never actually wired into anything, see
``build_crop_food_demand_IRB.py``'s docstring) at a high but finite
placeholder cost (:data:`FOOD_IMPORT_VAR_COST`). This guarantees the
scenario always has a way to meet the demand, while the cost keeps it
firmly a last resort: real production is always cheaper wherever it is
physically possible, so `food_import_<crop>` activity is itself a useful
diagnostic -- basins where it runs are exactly the ones whose real land/
water/technology capacity cannot meet population-driven demand.
:data:`FOOD_IMPORT_VAR_COST` is a deliberately large, round constant, not
real import-price data (matches the water floor's own fixed 1e6/unit slack
penalty in spirit -- a structural backstop value, not a calibrated one).

Consistency guard
------------------
Ported directly from NEST's own anti-join (``basin_msggdx_load_inputs.r``,
~line 407-420): a demand row is only kept if its (crop, node) pair has
real supply-side ``irrigation_yield`` data in ``crop_input_data_IRB.csv``
-- otherwise nothing in the model could ever produce it, and the demand
would just be a guaranteed infeasibility rather than a real economic pull.
Matches the R model's own check exactly (only ``irrigation_yield`` is
checked, not ``rain-fed_yield`` too -- replicated as-is here, not
independently re-derived).

Year matching
--------------
The source data is decadal after 2020 (2015, 2020, 2030, ..., 2060) --
each model year is matched to the *nearest* available data year, the same
convention :mod:`.irrigation_tech` already uses for its own decadal crop
water requirement data (not interpolated).

Scope
-----
AFG/IND/PAK basins only -- the source data has no China coverage (matches
``food_import_export.r``'s and ``indus_demands_new.csv``'s own real scope).
Basins outside that scope simply get no crop demand, not a fabricated
zero.
"""

import logging

import pandas as pd
from message_ix import make_df

from message_ix_models import Context, ScenarioInfo
from message_ix_models.model.water.utils import get_vintage_and_active_years
from message_ix_models.util import broadcast, same_node, same_time

from .common import (
    CROPS,
    basins_with_crop_data,
    crop_yield_commodity,
    read_crop_food_demand,
    read_crop_input_data,
)

log = logging.getLogger(__name__)

#: Technology name for the food-import backstop, one per crop.
FOOD_IMPORT_TECH_PREFIX = "food_import_"

#: Deliberately large, round placeholder (USD/kt) -- not real import-price
#: data. See module docstring's "food_import_<crop>: a backstop, not an
#: optional extra" section for why this exists and why the exact number is
#: a structural constant, not a calibrated one.
FOOD_IMPORT_VAR_COST = 10_000.0

#: Short, arbitrary lifetime (years) -- an import decision is naturally an
#: annual one, not a multi-year capital asset; kept short so this backstop
#: never accumulates capacity it doesn't need. No historical_new_capacity
#: is ever registered for it, so none of "Multi-vintage..." in
#: irrigation_tech.py's docstring (phantom vintages from combining a short
#: lifetime with a historical seed) applies here.
FOOD_IMPORT_LIFETIME = 5


def _nearest_year(available_years: list[int], target: int) -> int:
    return min(available_years, key=lambda y: abs(y - target))


def food_import_technology_names(context: "Context") -> list[str]:
    """Return every ``food_import_<crop>`` technology name for `context`.

    Registers all 9 (one per crop) regardless of per-basin data coverage --
    matches :func:`.irrigation_tech.irrigation_technology_names`'s
    convention.

    Returns ``[]`` if ``context.crop_demand_enabled`` is explicitly set to
    :obj:`False` -- see :func:`add_crop_demand`'s own check for why this
    exists as a per-run toggle rather than a config.yaml constant.
    """
    if not getattr(context, "crop_demand_enabled", True):
        return []
    valid_basins = getattr(context, "valid_basins", None) or set()
    if not (basins_with_crop_data() & set(valid_basins)):
        return []
    return [f"{FOOD_IMPORT_TECH_PREFIX}{crop}" for crop in CROPS]


def add_crop_demand(context: "Context") -> dict[str, pd.DataFrame]:
    """Build the ``demand`` parameter data for real, population-driven crop
    food demand.

    Parameters
    ----------
    context : .Context

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        ``{"demand": ...}``, or ``{}`` if there is no real demand data (or
        none of it survives the consistency guard) for this context's
        valid basins.
    """
    # Per-run kill switch: unlike the config.yaml constants sibling modules
    # use (e.g. common.CONFIG["emission_factor_enabled"]), this is a
    # context attribute, not a committed data setting -- set
    # context.crop_demand_enabled = False on a live context/scenario script
    # to isolate mechanism 1 (irrigation_tech.py's water-withdrawal floor)
    # without touching this module's code or its tests (which don't set
    # this attribute, so default to enabled).
    if not getattr(context, "crop_demand_enabled", True):
        return {}

    # _filter_basins() treats an empty/unset valid_basins as "no filter" (its
    # documented behavior, shared by every reader in this package) -- an
    # explicit check is needed here, same as every sibling module
    # (irrigation_tech.py, land.py, rainfed.py) already does its own way.
    if not getattr(context, "valid_basins", None):
        return {}

    demand_df = read_crop_food_demand(context)
    if demand_df.empty:
        return {}

    # Consistency guard: only keep (crop, node) pairs with real supply-side
    # irrigation_yield data -- see module docstring.
    supply = read_crop_input_data(context)
    supply_pairs = set(
        zip(
            supply.loc[supply["par"] == "irrigation_yield", "crop"],
            supply.loc[supply["par"] == "irrigation_yield", "node"],
        )
    )
    keep = demand_df.apply(lambda r: (r["crop"], r["node"]) in supply_pairs, axis=1)
    if (~keep).any():
        dropped_pairs = sorted(
            set(zip(demand_df.loc[~keep, "crop"], demand_df.loc[~keep, "node"]))
        )
        log.warning(
            "Dropping %d crop food-demand row(s) with no matching supply-side "
            "irrigation_yield data (crop, node): %s",
            int((~keep).sum()),
            dropped_pairs,
        )
    demand_df = demand_df[keep]
    if demand_df.empty:
        return {}

    scen = context.get_scenario()
    info = ScenarioInfo(scen)

    # Full, natural get_vintage_and_active_years() -- same proven pattern
    # irrigation_tech.py/land.py/rainfed.py use, see this module's own
    # docstring for why a short lifetime is safe here specifically (no
    # historical_new_capacity ever registered for this technology).
    yv_ya = get_vintage_and_active_years(info, FOOD_IMPORT_LIFETIME)
    year_vtg_all = sorted(set(yv_ya["year_vtg"]))

    demand_frames = []
    out_frames = []
    tl_frames = []
    cons_frames = []
    cf_frames = []
    var_frames = []

    for (crop, node), group in demand_df.groupby(["crop", "node"]):
        if crop not in CROPS:
            continue
        available_years = sorted(group["year"].unique())
        year_lookup = group.set_index("year")["value"]
        matched_years = [_nearest_year(available_years, y) for y in info.Y]
        node_loc = "B" + node

        demand_frames.append(
            make_df(
                "demand",
                node=node_loc,
                commodity=crop_yield_commodity(crop),
                level="crop_yield",
                year=list(info.Y),
                time="year",
                value=[float(year_lookup[y]) for y in matched_years],
                unit="kt",
            )
        )

        tech = f"{FOOD_IMPORT_TECH_PREFIX}{crop}"
        out_frames.append(
            make_df(
                "output", technology=tech, node_loc=node_loc,
                year_vtg=yv_ya["year_vtg"], year_act=yv_ya["year_act"],
                value=1.0, unit="kt", level="crop_yield",
                commodity=crop_yield_commodity(crop), mode="M1",
            )
            # broadcast() must run BEFORE same_node()/same_time(): those two
            # copy the *current* value of node_loc/time into node_dest/
            # time_dest -- calling them first (as an earlier version of this
            # function did) copies NaN, since "time" doesn't exist yet at
            # that point. That NaN then reaches the JDBC backend as a Double
            # where a String is expected -- exactly the
            # "ClassCastException: Double cannot be cast to String" a real
            # build hit. Order matters; matches irrigation_tech.py's own
            # proven-correct broadcast-then-same_node-then-same_time order.
            .pipe(broadcast, time=["year"])
            .pipe(same_node)
            .pipe(same_time)
        )
        tl_frames.append(
            make_df(
                "technical_lifetime", technology=tech, node_loc=node_loc,
                value=FOOD_IMPORT_LIFETIME, unit="y",
            ).pipe(broadcast, year_vtg=year_vtg_all)
        )
        cons_frames.append(
            make_df(
                "construction_time", technology=tech, node_loc=node_loc,
                value=0, unit="y",
            ).pipe(broadcast, year_vtg=year_vtg_all)
        )
        cf_frames.append(
            make_df(
                "capacity_factor", technology=tech, node_loc=node_loc,
                value=1.0, unit="%",
            ).pipe(broadcast, yv_ya, time=["year"])
        )
        var_frames.append(
            make_df(
                "var_cost", technology=tech, node_loc=node_loc, mode="M1",
                value=FOOD_IMPORT_VAR_COST, unit="USD/kt",
            ).pipe(broadcast, yv_ya, time=["year"])
        )

    if not demand_frames:
        return {}

    def _concat(frames):
        return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()

    return {
        "demand": _concat(demand_frames),
        "output": _concat(out_frames),
        "technical_lifetime": _concat(tl_frames),
        "construction_time": _concat(cons_frames),
        "capacity_factor": _concat(cf_frames),
        "var_cost": _concat(var_frames),
    }
