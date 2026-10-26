"""Prepare data for ``irr_<method>_<crop>`` technologies -- the third and
final layer of the crop/rainfed/irrigation port of the legacy R+GAMS
indus_ix model's crop submodel
(``NEST/MESSAGEix/basin_msggdx_technologies.r``, ~line 12802-13012). This is
the layer that actually gives the model real, basin-level irrigation water
(and electricity) demand -- the whole point of this migration (see
``data/demands.py``'s now-bypassed, GLOBIOM-linked ``add_irrigation_demand``
for what this replaces on the IRB/country build path).

Data model
----------
One technology per (irrigation method, crop) pair -- 7 real methods
(``irr_flood``, ``irr_drip``, ``irr_sprinkler``, ``irr_canal_lining_flood``,
``irr_smart``, ``irr_drip_smart``, ``irr_sprinkler_smart``) x 9 crops, named
``<method>_<crop>``. Present only in basins with real, positive
``irrigation_yield`` AND real crop-water-requirement data for that crop --
the intersection, matching the R model's own node filter.

Consumes, per unit of activity (1 Mha of land put under this irrigation
method for this crop):

- ``<crop>_land`` (the shared land pool from :mod:`.land`), level ``crop``.
- **Surface water withdrawal**, scaled from the real annual crop water
  requirement by ``1 / (water_efficiency * field_efficiency_conv)`` --
  ``water_efficiency`` is real, per-method data (``irr_tech_data_IRB.csv``,
  0.40 for flood up to 0.95 for smart drip); ``field_efficiency_conv``
  (0.4675) is the fixed field/conveyance-loss constant from
  ``config.yaml``. Reuses the existing **basin-tier**
  ``surfacewater_basin`` / ``water_avail_basin`` commodity (matches
  :mod:`.network`), *not* the region-tier ``freshwater``/``water_supply``
  pair the old simplified ``irrigation_cereal`` etc. use -- see the
  crop/irrigation migration plan's naming-reconciliation section.
- **Electricity**, proportional to the withdrawal rate via the method's
  real ``electricity_intensity`` (kWh/m3, converted to GWa/MCM). Reuses
  the existing ``electr`` / ``final`` commodity (the only electricity
  commodity in the repo). Omitted entirely for methods with zero
  electricity intensity (gravity-fed flood/canal-lining methods).

Produces:

- The generic ``crop_land`` commodity (level ``area``), same shared pool
  :mod:`.rainfed` produces.
- ``<crop>_yield`` (level ``crop_yield``), simplified to an annual rate
  (see :mod:`.rainfed`'s docstring for the same simplification there).
- **Irrigation losses -> groundwater recharge**: the portion of withdrawn
  water not actually consumed by the crop
  (``withdrawal * (1/(water_efficiency*field_efficiency_conv) - 1)``)
  feeds the **existing** ``groundwater_basin`` / ``water_avail_basin``
  commodity, which the existing ``gw_recharge`` technology already
  consumes (``water_supply.py``, ``demands.py``) -- no new commodity
  needed.

Only ``irr_flood_<crop>`` gets a historical-capacity bound from real 2015
irrigated-land data (``crop_irr_land_2015``) -- matches the R model's own
``grepl('irr_flood_', ...)`` rule (``historical_capacity`` flag in
``config.yaml``), since that's the only method with real pre-2015 land
data broken out. All 7 methods can compete for *new* investment, though
-- see "Vintaging" below for why this isn't the fidelity gap it first
appeared to be.

Deliberate simplifications relative to the R model
----------------------------------------------------
- **Electricity-flexibility output dropped.** The R model gives "smart"
  irrigation methods a secondary output to a ``flexibility``/
  ``energy_secondary`` commodity (demand-response service value). Nothing
  in this model produces or consumes a matching commodity -- same
  "unbacked orphan commodity" reasoning as dropping machinery energy in
  :mod:`.land`.
- **No SW/GW source split.** The real
  ``historical_irrigation_withdrawals_IRB.csv`` records separate
  ``irrigation_sw_diversion``/``irrigation_gw_diversion`` historical
  activity, implying the R model has a *further* technology layer
  splitting irrigation by water source that isn't part of the
  ``crop_<crop>``/``rainfed_<crop>``/``irr_<method>_<crop>`` block this
  port targets (out of scope to trace through the 13,000+ line source
  file). This port draws only from ``surfacewater_basin``; the historical
  floor below (sw+gw combined) is treated as a single combined target on
  that one commodity -- a documented simplification, not a hidden gap.
- **Real vs. computed water year.** The real crop-water-requirement data
  (``crop_irrigation_water_IRB.csv``) is decadal
  (2010/2020/.../2060); each model year's ``year_act`` is matched to the
  *nearest* available data year rather than interpolated.

Demand-pull: the historical-withdrawal floor
----------------------------------------------
Nothing upstream of this layer (no food demand, see :mod:`.land`'s module
docstring) forces any of this to actually run -- resolved with the user by
adding a **relation** per basin (``hist_irr_withdrawal_<basin>``) with
``relation_lower`` set to the real, annualized historical sw+gw diversion
total for that basin (``historical_irrigation_withdrawals_IRB.csv``), and
``relation_activity`` coefficients equal to each technology's own real
withdrawal rate -- i.e. "the sum of water actually withdrawn by every
irrigation technology active in this basin must be at least the observed
historical baseline." This is what makes this layer's addition to a real
scenario solve produce a genuine, nonzero effect instead of the same
zero-effect result the network module's `river`/`canal_conv` links had
(nothing there required them to be used; here, something does).

The floor is not always fully reachable, and the model must be solved
accordingly. Checked directly against the real committed scenario data
(summing each basin's real historical capacity x its relation coefficient
against that basin's floor): **all 16 real Pakistan/India/Afghan basins
with a relation fall 10-25% short** of their floor using only
``irr_flood_<crop>``'s real historical capacity -- because
``historical_irrigation_withdrawals_IRB.csv``'s total reflects *every*
irrigation method historically used at that basin, not just flood (the
only method this port ever gives real capacity, per its own
``historical_capacity`` convention -- see "Multi-vintage..." below for
why). This was invisible before the phantom-vintage fix (below): free
phantom capacity could paper over any shortfall. Resolved with the user:
**the floor is soft, not hard.** Any real solve of a scenario carrying
this relation must enable message_ix's native
``SLACK_RELATION_BOUND_LO`` mechanism (disabled by default) via
``scenario.solve(..., gams_args=['--SLACK_RELATION_BOUND_LO=""'])`` --
without it, the real scenario is genuinely infeasible. With it,
``RELATION_CONSTRAINT_LO`` becomes ``REL + SLACK_RELATION_BOUND_LO >=
relation_lower`` (model_core.gms), and the model reports any unavoidable
shortfall via the ``SLACK_RELATION_BOUND_LO`` variable rather than
refusing to solve -- penalized at a large, fixed 1e6/unit in the
objective (a message_ix-wide constant, not a parameter this port
controls), so the model still strongly prefers real capacity wherever
available.

Vintaging -- three designs tried, in order, before landing on the one
that actually works
--------------------------------------------------------------------------
Getting this right took three attempts, each a real bug found and fixed,
not a style preference:

1. **Single long-lived historical vintage** (one real ``year_vtg <
   firstyear``, ``life_exp`` up to 100y spanning the whole horizon).
   Real ``historical_new_capacity`` works here (a genuine pre-model
   vintage), but GAMS's own data-completeness check
   (``data_load.gms``'s "Technical lifetime not defined" abort) requires
   ``technical_lifetime`` at *every* model-horizon year the technology
   is merely *active* in, not only its construction year -- confirmed:
   narrowing this aborts the real solve. Those extra years then become
   legitimate ``CAP_NEW`` vintages with no real ``input``/``output`` row
   of their own -- "phantom" capacity, free to build, drawing no real
   resource. A ``relation_lower``-forced technology exploited exactly
   this: real-scenario ACT showed activity at a ``year_vtg`` its own
   ``input`` table had no row for. Found via a commodity-balance sanity
   check (``irr_flood_<crop>`` activity vs. matching ``crop_<crop>``
   output): **all** 144 "nonzero activity" rows from an earlier
   checkpoint were mismatched -- activity with no real land behind it.
2. **Explicitly blocking the phantom loophole**
   (``bound_new_capacity_up = 0`` on the extra years) reproduced a
   *third* bug, confirmed independently via ~25 hand-built minimal GAMS
   scenarios: a technology with more than one valid vintage active in
   the same year, forced by ``relation_activity``/``relation_lower`` or
   ``bound_activity_lo``, breaks ``RELATION_EQUIVALENCE``'s presolve in
   this GAMS/CPLEX build ("... infeasible, all entries at implied
   bounds") -- confirmed to happen *even when* the second vintage is
   hard-capped to zero capacity, not only when it's a free "new-build"
   option. A follow-up (prohibitive ``inv_cost`` instead of a hard
   bound) avoided the infeasibility but still let CPLEX fund a small
   amount of phantom capacity regardless of cost -- confirming this
   isn't a rational tradeoff GAMS is making, but a presolve/formulation
   artifact of the ``SUM(vintage$(map_tec_lifetime(...)))`` construct
   whenever more than one distinct ``year_vtg`` exists for a
   relation-forced technology.
3. **same_year_only everywhere, re-seeding the same real value every
   period.** Sidestepped #2's phantom-vintage problem entirely (every
   active year is its own independent vintage; no second one ever
   exists to create a loophole) -- but rested on a wrong premise.
   ``historical_new_capacity`` registered at a model-horizon
   ``year_vtg`` is *silently ignored* by GAMS: confirmed directly
   against ``model_core.gms``, ``CAPACITY_MAINTENANCE_HIST`` -- the
   only equation that reads ``historical_new_capacity`` -- is gated on
   ``historical(vintage)``, true only for a genuine pre-model year.
   Every model-horizon vintage instead gets its capacity *solely* from
   ``CAP_NEW`` via ``CAPACITY_MAINTENANCE_NEW``. The "144 nonzero
   activity" result this design produced was real, but came entirely
   from unconstrained new investment, not from the real 2015 land data
   believed at the time. Tried adding ``bound_new_capacity_up = 0``
   here too (to block that investment) and confirmed it makes capacity
   **zero, always** -- with no ``historical_new_capacity`` contribution
   possible at a model-horizon vintage, blocking ``CAP_NEW`` leaves
   nothing else to supply capacity at all.

**The actual fix**: the full, natural, *unrestricted*
``get_vintage_and_active_years()`` result for every method -- exactly
the pattern any ordinary message_ix investment technology uses, not
filtered to one vintage and not forced ``same_year_only``. Every
model-horizon year gets a legitimate "build new here" vintage candidate
with its own real, matching ``input``/``output``/cost data (not just
``technical_lifetime``, which is what made attempt #1's phantom
vintages exploitable) -- so any new capacity GAMS chooses to build is
genuine and cost-real, never free, and there is no artificial second,
data-less vintage for ``RELATION_EQUIVALENCE`` to trip over.
``historical_new_capacity`` is registered only at the one real pre-model
seed year (``irr_flood`` only, the only method with real pre-2015
irrigated-land data), where it actually applies.

**Real, accepted tradeoff, decided with the user**: new investment
beyond the real 2015 baseline remains possible at any method wherever
the real cost data favors it -- this was tested and found to happen for
real (basins investing hundreds of millions to billions of USD to close
the gap to their historical-withdrawal floor, rather than pay the
relation's soft-bound slack penalty). Structurally forbidding this
(``bound_new_capacity_up = 0`` everywhere) was tried and rejected: it
made capacity zero *everywhere* (see #3 above), which is worse than
allowing economically-justified new investment. This also *removes* an
earlier fidelity gap noted in this module: since new capacity genuinely
*can* be built from a zero base given a strong enough incentive
(contrary to an earlier, incompletely-tested claim to the contrary), all
7 irrigation methods can now compete for real activity here, not only
``irr_flood`` -- closer to the R model's own apparent intent.

Land/irrigation capacity under this design no longer represents a fresh
annual choice (the R model's ``lft = 1``) or a single fixed 2015 seed
the model only ever draws down -- it is a real historical baseline that
the model can extend via genuine new investment where the economics
justify it. That is a real fidelity choice, made deliberately and
knowingly, not a hidden side effect.
"""

import logging

import pandas as pd
from message_ix import make_df

from message_ix_models import Context, ScenarioInfo
from message_ix_models.model.water.utils import (
    get_vintage_and_active_years,
    kWh_m3_TO_GWa_MCM,
)
from message_ix_models.util import broadcast, same_node, same_time

from .common import (
    CONFIG,
    CROPS,
    IRRIGATION_METHODS,
    basins_with_crop_data,
    country_group_of,
    crop_land_commodity,
    crop_yield_commodity,
    irr_tech_name,
    read_crop_input_data,
    read_crop_irrigation_water_annual,
    read_historical_irrigation_withdrawals,
    read_irr_tech_data,
    read_irrigation_demand_growth,
)
from .irrigation_source import IRRIGATION_WATER, irrigation_source_enabled
from .rainfed import GENERIC_LAND_COMMODITY

log = logging.getLogger(__name__)

#: USD/ha -> USD/Mha. See land.py module docstring for the same conversion.
_HA_TO_MHA_COST = 1e6


def _sanitize(basin: str) -> str:
    """Turn a BCU_name like "12|PAKISTAN" into a GAMS-identifier-safe
    string, matching the ``mode = "M" + BCU_name.replace("|", "_")``
    convention already used in irrigation.py."""
    return basin.replace("|", "_")


def _nearest_year(available_years: list[int], target: int) -> int:
    return min(available_years, key=lambda y: abs(y - target))


def irrigation_technology_names(context: "Context") -> list[str]:
    """Return every ``<method>_<crop>`` technology name for ``context``.

    Registers all 63 (7 methods x 9 crops) regardless of per-basin data
    coverage -- matches :func:`.land.crop_technology_names`'s convention.
    """
    valid_basins = getattr(context, "valid_basins", None) or set()
    if not (basins_with_crop_data() & set(valid_basins)):
        return []
    return [irr_tech_name(m, c) for m in IRRIGATION_METHODS for c in CROPS]


def irrigation_commodity_names() -> list[str]:
    """Every commodity this module produces that :mod:`.rainfed` doesn't
    already register (both produce the same two: the generic
    ``crop_land`` and every ``<crop>_yield``) -- kept as its own function
    for symmetry with the other layers, even though the set is identical."""
    return [GENERIC_LAND_COMMODITY] + [crop_yield_commodity(c) for c in CROPS]


def irrigation_relation_names(context: "Context") -> list[str]:
    """One ``hist_irr_withdrawal_<basin>`` relation per valid basin --
    see module docstring's "Demand-pull" section."""
    valid_basins = set(getattr(context, "valid_basins", None) or set()) & basins_with_crop_data()
    return [f"hist_irr_withdrawal_{_sanitize(b)}" for b in sorted(valid_basins)]


def add_irrigation_techs(context: "Context") -> dict[str, pd.DataFrame]:
    """Build ``irr_<method>_<crop>`` technologies and the historical-
    withdrawal demand-pull relation.

    Parameters
    ----------
    context : .Context

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        Keys are MESSAGE parameter names such as 'input', 'output',
        'relation_activity'.
    """
    valid_basins = set(getattr(context, "valid_basins", None) or set()) & basins_with_crop_data()
    if not valid_basins:
        return {}

    scen = context.get_scenario()
    scenario_info = ScenarioInfo(scen)
    sub_time = pd.Series(context.time)
    firstyear = scen.firstmodelyear
    field_eff = CONFIG["field_efficiency_conv"]

    df_input = read_crop_input_data(context)
    df_irr_yield = df_input[(df_input["par"] == "irrigation_yield") & (df_input["value"] > 0)]
    df_cap = df_input[df_input["par"] == "crop_irr_land_2015"]
    df_water_all = read_crop_irrigation_water_annual(context)
    df_tech = read_irr_tech_data()

    # With the irrigation source layer on (irrigation_source.py), water comes
    # from the diversion technologies at the farm gate instead of straight
    # from the river, and the floor is carried by those diversions.
    source_layer = irrigation_source_enabled(context)
    water_in = IRRIGATION_WATER if source_layer else ("surfacewater_basin", "water_avail_basin")

    inp_frames, out_frames = [], []
    tl_frames, cons_frames, cf_frames = [], [], []
    inv_frames, fix_frames, var_frames, hist_cap_frames = [], [], [], []
    withdrawal_rows = []  # accumulated for the demand-pull relation

    for method, method_spec in IRRIGATION_METHODS.items():
        params = df_tech[df_tech["irr_tech"] == method].set_index("par")["value"]
        water_eff = float(params.get("water_efficiency", float("nan")))
        life_exp = params.get("life_exp")
        if pd.isna(water_eff) or pd.isna(life_exp) or water_eff <= 0:
            log.warning(
                "Skipping irrigation method %s: missing/invalid water_efficiency or life_exp",
                method,
            )
            continue
        elec_intensity = float(params.get("electricity_intensity", 0.0) or 0.0)
        cap_factor = float(params.get("cap_factor", 1.0) or 1.0)
        lifetime = float(life_exp)
        conveyance = water_eff * field_eff

        def cost(par: str) -> float:
            return float(params.get(par, 0.0) or 0.0) * _HA_TO_MHA_COST

        # Full, natural get_vintage_and_active_years() for every method --
        # see "Phantom vintages" in the module docstring for the full
        # story of why a single filtered-down vintage (historical-only or
        # same_year_only) doesn't work well here, and why this design
        # (matching how any ordinary message_ix investment technology
        # behaves) is the one that keeps real historical data meaningful
        # while still allowing real new investment.
        #
        # historical_capacity methods (irr_flood only, matching the R
        # model): real capacity comes from historical_new_capacity
        # registered at the one genuine pre-model seed year -- the only
        # year it actually applies (confirmed against model_core.gms:
        # historical_new_capacity feeds CAP only for a genuine historical
        # vintage).
        #
        # Everyone else: no historical_new_capacity (no real pre-2015
        # irrigated-land data broken out by method) -- structurally
        # present with real cost/water data, capacity-bearing only via
        # new investment where the real economics favor it. This also
        # removes an earlier fidelity gap: new capacity genuinely *can*
        # be built from a zero base given a strong enough incentive
        # (confirmed for real against the Pakistan scenario -- the
        # earlier claim that it categorically cannot was based on an
        # incomplete experiment), so all 7 methods can compete here now,
        # not only irr_flood.
        want_historical = method_spec.get("historical_capacity", False)
        yv_ya = get_vintage_and_active_years(scenario_info, lifetime)
        year_vtg_all = sorted(set(yv_ya["year_vtg"]))
        hist_years = [y for y in year_vtg_all if y < firstyear]
        hist_year_vtg = max(hist_years) if hist_years else None

        for crop in CROPS:
            crop_yield = df_irr_yield[df_irr_yield["crop"] == crop]
            crop_water = df_water_all[df_water_all["crop"] == crop]
            if crop_yield.empty or crop_water.empty:
                continue
            nodes = sorted(set(crop_yield["node"]) & set(crop_water["node"]) & valid_basins)
            if not nodes:
                continue

            tech = irr_tech_name(method, crop)
            node_loc_flat = pd.Series(["B" + n for n in nodes])

            base = pd.merge(pd.DataFrame({"node": nodes}), yv_ya, how="cross")
            available_years = sorted(crop_water["year"].unique())
            base["water_year"] = base["year_act"].map(
                lambda y: _nearest_year(available_years, y)
            )
            cw = crop_water[["node", "year", "value"]].rename(
                columns={"year": "water_year", "value": "water_per_mha"}
            )
            base = base.merge(cw, on=["node", "water_year"], how="left")
            base["water_per_mha"] = base["water_per_mha"].fillna(0.0)
            base["withdrawal_mcm_per_mha"] = base["water_per_mha"] / conveyance
            base["elec_gwa_per_mha"] = (
                base["withdrawal_mcm_per_mha"] * elec_intensity * kWh_m3_TO_GWa_MCM
            )
            # Recharge scales off the RAW crop water requirement, not the
            # already-scaled withdrawal -- matches the R model exactly
            # (`(1/(water_efficiency*field_efficiency_conv) - 1) * value`,
            # where `value` is crop_water.df's raw value). Using withdrawal
            # here instead would double-scale: at conveyance=0.5 e.g., it
            # would claim 100% of withdrawn water becomes recharge (0%
            # reaching the crop), instead of the correct 50% loss fraction.
            # recharge = withdrawal - raw_requirement, i.e. exactly the
            # loss between what's diverted and what the crop actually uses.
            base["gw_recharge_mcm_per_mha"] = base["water_per_mha"] * (
                1.0 / conveyance - 1.0
            )
            base["node_loc"] = "B" + base["node"]

            yield_lookup = crop_yield.set_index("node")["value"]
            base["yield_value"] = base["node"].map(yield_lookup).fillna(0.0)

            # --- input: crop land, freshwater, (optional) electricity
            inp_frames.append(
                make_df(
                    "input", technology=tech, node_loc=base["node_loc"],
                    year_vtg=base["year_vtg"], year_act=base["year_act"],
                    value=1.0, unit="Mha", level="crop",
                    commodity=crop_land_commodity(crop), mode="M1",
                )
                .pipe(broadcast, time=sub_time)
                .pipe(same_node)
                .pipe(same_time)
            )
            inp_frames.append(
                make_df(
                    "input", technology=tech, node_loc=base["node_loc"],
                    year_vtg=base["year_vtg"], year_act=base["year_act"],
                    value=base["withdrawal_mcm_per_mha"], unit="MCM",
                    commodity=water_in[0], level=water_in[1], mode="M1",
                )
                .pipe(broadcast, time=sub_time)
                .pipe(same_node)
                .pipe(same_time)
            )
            if elec_intensity > 0:
                inp_frames.append(
                    make_df(
                        "input", technology=tech, node_loc=base["node_loc"],
                        year_vtg=base["year_vtg"], year_act=base["year_act"],
                        value=base["elec_gwa_per_mha"], unit="GWa",
                        level="final", commodity="electr", mode="M1",
                    )
                    .pipe(broadcast, time=sub_time)
                    .pipe(same_node)
                    .pipe(same_time)
                )

            # --- output: generic crop_land, <crop>_yield, groundwater recharge
            out_frames.append(
                make_df(
                    "output", technology=tech, node_loc=base["node_loc"],
                    year_vtg=base["year_vtg"], year_act=base["year_act"],
                    value=1.0, unit="Mha", level="area",
                    commodity=GENERIC_LAND_COMMODITY, mode="M1",
                )
                .pipe(broadcast, time=sub_time)
                .pipe(same_node)
                .pipe(same_time)
            )
            out_frames.append(
                make_df(
                    "output", technology=tech, node_loc=base["node_loc"],
                    year_vtg=base["year_vtg"], year_act=base["year_act"],
                    value=base["yield_value"], unit="kt", level="crop_yield",
                    commodity=crop_yield_commodity(crop), mode="M1",
                )
                .pipe(broadcast, time=sub_time)
                .pipe(same_node)
                .pipe(same_time)
            )
            out_frames.append(
                make_df(
                    "output", technology=tech, node_loc=base["node_loc"],
                    year_vtg=base["year_vtg"], year_act=base["year_act"],
                    value=base["gw_recharge_mcm_per_mha"], unit="MCM",
                    level="water_avail_basin", commodity="groundwater_basin", mode="M1",
                )
                .pipe(broadcast, time=sub_time)
                .pipe(same_node)
                .pipe(same_time)
            )

            # --- technical lifetime / construction time / capacity factor
            tl_frames.append(
                make_df(
                    "technical_lifetime", technology=tech, value=lifetime,
                    unit="y", node_loc=node_loc_flat,
                ).pipe(broadcast, year_vtg=year_vtg_all)
            )
            cons_frames.append(
                make_df(
                    "construction_time", technology=tech, value=0,
                    unit="y", node_loc=node_loc_flat,
                ).pipe(broadcast, year_vtg=year_vtg_all)
            )
            cf_frames.append(
                make_df(
                    "capacity_factor", technology=tech, value=cap_factor,
                    unit="%", node_loc=node_loc_flat,
                ).pipe(broadcast, yv_ya, time=sub_time)
            )

            # --- costs (method-level, real irr_tech_data.csv data)
            inv_frames.append(
                make_df(
                    "inv_cost", technology=tech, value=cost("inv_cost"),
                    unit="USD/Mha", node_loc=node_loc_flat,
                ).pipe(broadcast, year_vtg=year_vtg_all)
            )
            fix_frames.append(
                make_df(
                    "fix_cost", technology=tech, value=cost("fix_cost"),
                    unit="USD/Mha", node_loc=node_loc_flat,
                ).pipe(broadcast, yv_ya)
            )
            var_frames.append(
                make_df(
                    "var_cost", technology=tech, value=cost("var_cost"),
                    unit="USD/Mha", mode="M1", node_loc=node_loc_flat,
                ).pipe(broadcast, yv_ya, time=sub_time)
            )

            # --- historical capacity: only the historically-seeded method
            # (irr_flood), only at the one genuine pre-model seed year --
            # the only year historical_new_capacity actually applies.
            if want_historical and hist_year_vtg is not None:
                crop_cap = df_cap[
                    (df_cap["crop"] == crop) & (df_cap["value"] > 0) & (df_cap["node"].isin(nodes))
                ]
                if not crop_cap.empty:
                    hist_cap_frames.append(
                        make_df(
                            "historical_new_capacity", technology=tech,
                            node_loc="B" + crop_cap["node"], year_vtg=hist_year_vtg,
                            value=crop_cap["value"], unit="Mha",
                        )
                    )

            # --- accumulate real withdrawal rate for the demand-pull relation
            withdrawal_rows.append(
                base.loc[base["withdrawal_mcm_per_mha"] > 0, ["node", "year_act", "withdrawal_mcm_per_mha"]]
                .assign(technology=tech)
            )

    def _concat(frames):
        return (
            pd.concat(frames).drop_duplicates().reset_index(drop=True)
            if frames
            else pd.DataFrame()
        )

    results = {
        "input": _concat(inp_frames),
        "output": _concat(out_frames),
        "technical_lifetime": _concat(tl_frames),
        "construction_time": _concat(cons_frames),
        "capacity_factor": _concat(cf_frames),
        "inv_cost": _concat(inv_frames),
        "fix_cost": _concat(fix_frames),
        "var_cost": _concat(var_frames),
    }
    if hist_cap_frames:
        results["historical_new_capacity"] = _concat(hist_cap_frames)

    # --- demand-pull: relation_activity / relation_lower per basin
    if withdrawal_rows:
        all_withdrawal = pd.concat(withdrawal_rows, ignore_index=True)
        # relation_activity has no year_vtg dimension -- multiple vintages
        # active in the same year_act share the same withdrawal rate in
        # this design, so collapse to one row per (node, technology,
        # year_act) before writing, or add_par would see colliding index
        # keys.
        all_withdrawal = all_withdrawal.drop_duplicates(subset=["node", "technology", "year_act"])

        hist = read_historical_irrigation_withdrawals(context)
        hist_total = hist.groupby("node", as_index=False)["value"].sum()
        hist_lookup = hist_total.set_index("node")["value"]

        # Demand-growth index (population/GDP, GCAM-AgLU-style), per IRB
        # country group/year -- see build_irrigation_demand_growth_IRB.py.
        # Scales the historical floor up over the horizon instead of
        # leaving it pinned at the ~2015 historical value for every future
        # year. 1.0 at the growth data's own base year (2015) and wherever
        # a group/year has no data (missing growth CSV, or a year outside
        # its range), so the floor degrades to today's flat behavior rather
        # than blocking the build.
        growth_df = read_irrigation_demand_growth()
        if growth_df.empty:
            growth_lookup: pd.Series = pd.Series(dtype=float)
            growth_years_by_region: dict = {}
        else:
            growth_lookup = growth_df.set_index(["region", "year"])["demand_index"]
            growth_years_by_region = growth_df.groupby("region")["year"].apply(list).to_dict()

        def _growth_index(node: str, year: int) -> float:
            region = country_group_of(node)
            years = growth_years_by_region.get(region)
            if not years:
                return 1.0
            match = year if year in years else _nearest_year(years, year)
            return float(growth_lookup.get((region, match), 1.0))

        rel_act_frames, rel_lo_frames = [], []
        for node in sorted(set(all_withdrawal["node"]) & set(hist_lookup.index)):
            # Full historical sw+gw diversion at the base year (the earlier
            # DEBUG 0.6 relaxation is removed); the growth index then
            # scales it per year below.
            base_floor = float(hist_lookup[node])
            if base_floor <= 0:
                continue
            node_rows = all_withdrawal[all_withdrawal["node"] == node]
            if node_rows.empty:
                continue
            relation = f"hist_irr_withdrawal_{_sanitize(node)}"
            rel_act_frames.append(
                make_df(
                    "relation_activity",
                    relation=relation,
                    node_rel="B" + node,
                    year_rel=node_rows["year_act"],
                    node_loc="B" + node,
                    technology=node_rows["technology"],
                    year_act=node_rows["year_act"],
                    mode="M1",
                    value=node_rows["withdrawal_mcm_per_mha"],
                    unit="MCM",
                )
            )
            for year in sorted(node_rows["year_act"].unique()):
                floor = base_floor * _growth_index(node, int(year))
                rel_lo_frames.append(
                    make_df(
                        "relation_lower",
                        relation=relation,
                        node_rel="B" + node,
                        year_rel=int(year),
                        value=floor,
                        unit="MCM",
                    )
                )

        # Source layer on: the floor's coefficients come from the diversion
        # technologies (irrigation_source.py), not from the crop techs.
        if rel_act_frames and not source_layer:
            results["relation_activity"] = _concat(rel_act_frames)
        if rel_lo_frames:
            results["relation_lower"] = _concat(rel_lo_frames)

    return {k: v for k, v in results.items() if not v.empty}
