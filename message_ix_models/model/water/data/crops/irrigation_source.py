"""Irrigation water-source layer -- Python port of the legacy R+GAMS
indus_ix model's irrigation diversion technologies
(``NEST/MESSAGEix/basin_msggdx_technologies.r``):

======================================  ===========  ===============================
technology                              R lines      water path
======================================  ===========  ===============================
``irrigation_sw_diversion``             10368-10474  river -> farm gate (canal)
``smart_irrigation_sw_diversion``       10476-10578  river -> farm gate (improved)
``irrigation_gw_diversion``             10580-10670  aquifer -> farm gate (tubewell)
``smart_irrigation_gw_diversion``       10672-10770  aquifer -> farm gate (improved)
``urban_wastewater_irrigation``         11483-11570  urban wastewater -> farm gate
``irrigation_desal_seawater``           11662-11738  seawater -> farm gate
======================================  ===========  ===============================

ON by default (``context.irrigation_source_layer``, default True): every
irrigation water source is available. Set it to False to build the older
river-only configuration, in which nothing here is registered and
:mod:`.irrigation_tech` draws ``surfacewater_basin`` directly.

Why this layer exists
---------------------
In the R model, crop irrigation technologies take water at the farm gate
(``freshwater`` / ``irrigation_final``) and the diversion technologies above
supply it from the river or the aquifer. :mod:`.irrigation_tech` collapsed
that into a direct ``surfacewater_basin`` draw, which (a) skips the 25%
canal loss, and (b) gives irrigation no access to groundwater at all --
although roughly a third of the historical-withdrawal floor it is held to
is groundwater pumping. With the layer on:

- ``irr_<method>_<crop>`` takes ``irrigation_water`` / ``irrigation_final``
  instead, at ``crop water / water_efficiency``: the diversions here already
  carry the conveyance loss (25% canal seepage), and ``water_efficiency`` is
  the method's own field application efficiency, so the R model's extra
  ``field_efficiency_conv`` (0.85 field x 0.55 watercourse, "assuming all
  irrigation is basically flood") would count both losses a second time and
  send ~80% of farm-gate water back to the aquifer for flood irrigation.
- The ``hist_irr_withdrawal_<basin>`` floor is applied to the diversion
  technologies (coefficient 1 per MCM withdrawn at the source), which is
  what its source data (``irrigation_sw_diversion`` +
  ``irrigation_gw_diversion`` historical activity) actually measures.

Commodity/level mapping (R -> this model)
-----------------------------------------
- R ``freshwater``/``river_div``  -> ``surfacewater_basin``/``water_avail_basin``
- R ``freshwater``/``aquifer``    -> ``groundwater_basin``/``water_avail_basin``
  (renewable, mode ``M1``) or no water input (non-renewable, mode ``M2``,
  the R ``gw_extract`` path; tagged by :mod:`..fossil_groundwater`)
- R ``renewable_gw``/``aquifer``  -> ``groundwater_basin``/``water_avail_basin``
- R ``freshwater``/``irrigation_final`` -> ``irrigation_water``/``irrigation_final``
- R ``wastewater``/``urban_secondary`` -> ``urban_collected_wst``/``water_treat``
- R seawater (no input)           -> ``salinewater_basin``/``water_avail_basin``,
  matching this model's own ``distillation``/``membrane`` convention, so
  only basins with saline-water supply can use it.

Units
-----
The R model uses flows in MCM/day, electricity in MW and costs in million
USD2010; this model uses MCM/year, GWa and million USD (its cost unit:
power plants are USD/kW x GW):

- costs: ``R value / 365`` -> million USD per (MCM/year) of capacity
- electricity: ``R MW per (MCM/day) / 365 / 1e3`` -> GWa per MCM

Deliberate simplifications
--------------------------
- The smart variants' ``flexibility`` output is dropped, as in
  :mod:`.irrigation_tech` (nothing produces/consumes that commodity here).
- ``min_utilization_factor`` and ``irrigation_desal_seawater``'s
  ``growth_new_capacity_up`` are not ported: with no matching
  ``initial_new_capacity_up`` the growth bound would pin new desalination
  capacity at zero.
- Historical capacity is seeded only for the two conventional diversions,
  from the same historical sw/gw withdrawals as the floor (annual value /
  capacity factor), at the one genuine pre-model vintage.
- The fossil mode (``M2``) carries a small ``var_cost``
  (:data:`FOSSIL_DEPLETION_COST`). In R both groundwater paths cost nothing,
  so without a fossil limit the split between renewable and fossil pumping
  is arbitrary; this makes the model use renewable recharge first.
"""

import logging

import pandas as pd
from message_ix import make_df

from message_ix_models import Context, ScenarioInfo
from message_ix_models.model.water.utils import get_vintage_and_active_years
from message_ix_models.util import broadcast, package_data_path, same_node, same_time

from .common import basins_with_crop_data, read_historical_irrigation_withdrawals

log = logging.getLogger(__name__)

#: Farm-gate irrigation water produced by every technology in this module
#: and consumed by ``irr_<method>_<crop>`` when the layer is enabled.
IRRIGATION_WATER = ("irrigation_water", "irrigation_final")

SURFACE = ("surfacewater_basin", "water_avail_basin")
GROUND = ("groundwater_basin", "water_avail_basin")

#: Mode of a groundwater diversion that pumps non-renewable (fossil) water.
FOSSIL_MODE = "M2"

#: var_cost of the fossil mode, million USD per MCM (0.001 USD/m3): a
#: tie-break so renewable groundwater is pumped before the aquifer is mined.
FOSSIL_DEPLETION_COST = 0.001

#: R ``basin_msggdx_technologies.r`` parameters, in R units (MCM/day,
#: MW per MCM/day, million USD2010 per MCM/day of capacity).
#: ``source``: input water (commodity, level) for mode M1; ``fossil``: also
#: offers a no-input non-renewable mode M2; ``seepage``: share returned to
#: the aquifer; ``hist_tec``: historical-withdrawal row seeding capacity.
TECH_DATA = {
    "irrigation_sw_diversion": dict(
        source=SURFACE, efficiency=0.75, seepage=0.25, electricity_mw=6,
        lifetime=50, cap_factor=0.9, construction_time=1, inv_cost=57, fix_cost=3,
        hist_tec="irrigation_sw_diversion",
    ),
    "smart_irrigation_sw_diversion": dict(
        source=SURFACE, efficiency=0.75 + 0.03, seepage=0.25 - 0.03, electricity_mw=6,
        lifetime=50, cap_factor=0.9, construction_time=1, inv_cost=57 * 1.1, fix_cost=3 * 1.1,
    ),
    "irrigation_gw_diversion": dict(
        source=GROUND, fossil=True, efficiency=0.95, electricity_mw=16,
        lifetime=20, cap_factor=0.9, construction_time=1, inv_cost=8.5, fix_cost=1,
        hist_tec="irrigation_gw_diversion",
    ),
    "smart_irrigation_gw_diversion": dict(
        source=GROUND, fossil=True, efficiency=1.0, electricity_mw=16,
        lifetime=20, cap_factor=0.9, construction_time=1, inv_cost=8.5 * 1.1, fix_cost=1 * 1.1,
    ),
    "urban_wastewater_irrigation": dict(
        source=("urban_collected_wst", "water_treat"), efficiency=0.8, electricity_mw=80,
        lifetime=30, cap_factor=0.9, construction_time=1, inv_cost=1350, fix_cost=99,
    ),
    "irrigation_desal_seawater": dict(
        source=("salinewater_basin", "water_avail_basin"), efficiency=1.0, electricity_mw=167,
        lifetime=30, cap_factor=0.9, construction_time=1, inv_cost=1650, fix_cost=165,
    ),
}

#: Diversions whose withdrawal counts toward the historical-withdrawal floor
#: (the R model's historical activity covers exactly these two layers).
FLOOR_TECHS = [
    "irrigation_sw_diversion",
    "smart_irrigation_sw_diversion",
    "irrigation_gw_diversion",
    "smart_irrigation_gw_diversion",
]


def musd_per_mcm_day_to_musd_per_mcm_year(value: float) -> float:
    """R million USD per (MCM/day) -> million USD per (MCM/year).

    The model's cost unit is million USD (power plants: USD/kW x GW; IIASA's
    own water techs, e.g. extract_surfacewater ~0.16 per MCM/year), so the
    R value is only rescaled from a daily to an annual flow.
    """
    return value / 365


def mw_per_mcm_day_to_gwa_per_mcm(value: float) -> float:
    return value / 365 / 1e3


def irrigation_source_enabled(context: "Context") -> bool:
    return bool(getattr(context, "irrigation_source_layer", True))


def _valid_basins(context: "Context") -> set[str]:
    return set(getattr(context, "valid_basins", None) or set()) & basins_with_crop_data()


def _sanitize(basin: str) -> str:
    return basin.replace("|", "_")


def irrigation_source_technology_names(context: "Context") -> list[str]:
    """Every technology in this module, or ``[]`` if the layer is off."""
    if not irrigation_source_enabled(context) or not _valid_basins(context):
        return []
    return list(TECH_DATA)


def irrigation_source_commodity_names() -> list[str]:
    return [IRRIGATION_WATER[0]]


def irrigation_source_level_names() -> list[str]:
    return [IRRIGATION_WATER[1]]


def _modes(spec: dict) -> list[str]:
    return ["M1", FOSSIL_MODE] if spec.get("fossil") else ["M1"]


#: Same raw-REGION -> node mapping infrastructure.py uses for its electricity
#: inputs (basins draw power from their region's node, not the basin node,
#: which has no electricity supply of its own).
REGION_NODE_MAP = {"PAKISTAN": "R12_PAK"}


def electricity_origin(context: "Context", basins) -> dict[str, str]:
    """Basin node -> node electricity is drawn from (the basin node itself
    when there is no delineation file for this region, e.g. in tests)."""
    path = package_data_path("water", "delineation", f"basins_by_region_simpl_{context.regions}.csv")
    origin = {"B" + b: "B" + b for b in basins}
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["BCU_name"].isin(basins)]
        origin.update(zip("B" + df["BCU_name"].astype(str), df["REGION"].replace(REGION_NODE_MAP)))
    return origin


def add_irrigation_source_techs(context: "Context") -> dict[str, pd.DataFrame]:
    """Build the irrigation diversion technologies and their share of the
    ``hist_irr_withdrawal_<basin>`` floor. Returns ``{}`` when disabled."""
    basins = _valid_basins(context)
    if not irrigation_source_enabled(context) or not basins:
        return {}
    if getattr(context, "bottom_up_water_demand_enabled", False):
        log.warning(
            "irrigation_source_layer moves the irrigation floor's coefficients to the "
            "diversion technologies; the bottom-up relation reuses irrigation_tech's "
            "relation_activity, which is empty in this configuration."
        )

    scen = context.get_scenario()
    info = ScenarioInfo(scen)
    firstyear = scen.firstmodelyear
    sub_time = pd.Series(context.time)
    nodes = sorted(basins)
    node_loc = pd.Series(["B" + n for n in nodes])
    elec_origin = electricity_origin(context, basins)

    hist = read_historical_irrigation_withdrawals(context)
    hist = hist[hist["node"].isin(basins)].groupby(["node", "tec"], as_index=False)["value"].sum()

    frames: dict[str, list[pd.DataFrame]] = {}

    def add(par: str, df: pd.DataFrame) -> None:
        frames.setdefault(par, []).append(df)

    for tech, spec in TECH_DATA.items():
        yv_ya = get_vintage_and_active_years(info, spec["lifetime"])
        year_vtg = sorted(set(yv_ya["year_vtg"]))
        base = pd.merge(pd.DataFrame({"node_loc": node_loc}), yv_ya, how="cross")
        common = dict(technology=tech, node_loc=base["node_loc"],
                      year_vtg=base["year_vtg"], year_act=base["year_act"])
        elec = mw_per_mcm_day_to_gwa_per_mcm(spec["electricity_mw"])

        def io(par, mode, commodity, level, value, unit):
            add(par, make_df(par, **common, mode=mode, commodity=commodity, level=level,
                             value=value, unit=unit)
                .pipe(broadcast, time=sub_time).pipe(same_node).pipe(same_time))

        for mode in _modes(spec):
            if mode == "M1":
                io("input", mode, *spec["source"], 1.0, "MCM")
            add("input", make_df("input", **common, mode=mode, commodity="electr", level="final",
                                 node_origin=base["node_loc"].map(elec_origin), value=elec,
                                 unit="GWa")
                .pipe(broadcast, time=sub_time).pipe(same_time))
            io("output", mode, *IRRIGATION_WATER, spec["efficiency"], "MCM")
            if spec.get("seepage"):
                io("output", mode, *GROUND, spec["seepage"], "MCM")
            if mode == FOSSIL_MODE:
                add("var_cost", make_df("var_cost", **common, mode=mode,
                                        value=FOSSIL_DEPLETION_COST, unit="USD/MCM")
                    .pipe(broadcast, time=sub_time))

        flat = dict(technology=tech, node_loc=node_loc)
        add("technical_lifetime", make_df("technical_lifetime", **flat, value=spec["lifetime"],
                                          unit="y").pipe(broadcast, year_vtg=year_vtg))
        add("construction_time", make_df("construction_time", **flat,
                                         value=spec["construction_time"], unit="y")
            .pipe(broadcast, year_vtg=year_vtg))
        add("capacity_factor", make_df("capacity_factor", **flat, value=spec["cap_factor"],
                                       unit="%").pipe(broadcast, yv_ya, time=sub_time))
        add("inv_cost", make_df("inv_cost", **flat,
                                value=musd_per_mcm_day_to_musd_per_mcm_year(spec["inv_cost"]),
                                unit="USD/MCM").pipe(broadcast, year_vtg=year_vtg))
        add("fix_cost", make_df("fix_cost", **flat,
                                value=musd_per_mcm_day_to_musd_per_mcm_year(spec["fix_cost"]),
                                unit="USD/MCM").pipe(broadcast, yv_ya))

        hist_years = [y for y in year_vtg if y < firstyear]
        if spec.get("hist_tec") and hist_years:
            h = hist[(hist["tec"] == spec["hist_tec"]) & (hist["value"] > 0)]
            if not h.empty:
                add("historical_new_capacity", make_df(
                    "historical_new_capacity", technology=tech, node_loc="B" + h["node"],
                    year_vtg=max(hist_years), value=h["value"] / spec["cap_factor"], unit="MCM",
                ))

        # The historical-withdrawal floor: 1 per MCM withdrawn at the source,
        # every mode (renewable or fossil groundwater both count).
        if tech in FLOOR_TECHS:
            ya = sorted(y for y in set(yv_ya["year_act"]) if y >= firstyear)
            for mode in _modes(spec):
                add("relation_activity", make_df(
                    "relation_activity", technology=tech, mode=mode, value=1.0, unit="MCM",
                    relation=[f"hist_irr_withdrawal_{_sanitize(n)}" for n in nodes],
                    node_rel=node_loc, node_loc=node_loc,
                ).pipe(broadcast, year_act=ya).assign(year_rel=lambda d: d["year_act"]))

    return {
        par: pd.concat(dfs).drop_duplicates().reset_index(drop=True)
        for par, dfs in frames.items()
    }
