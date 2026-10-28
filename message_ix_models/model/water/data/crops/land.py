"""Prepare data for ``crop_<crop>`` technologies -- the root layer of the
three-tier crop/rainfed/irrigation port of the legacy R+GAMS indus_ix
model's crop submodel (``NEST/MESSAGEix/basin_msggdx_technologies.r``,
~line 12558-12690).

Data model
----------
One technology per crop (``crop_<crop>``, 9 crops -- see
:data:`.common.CROPS`), present in every valid basin. It represents land
*committed* to growing a crop -- bounded by real 2015 land-area data
(``crop_irr_land_2015`` + ``crop_rainfed_land_2015`` from
``crop_input_data_IRB.csv``), not yet split into irrigated vs. rainfed
cultivation. Its only output is the intermediate commodity
``<crop>_land`` (Mha, level ``crop``) -- consumed downstream by
:mod:`.rainfed` (``rainfed_<crop>``) and :mod:`.irrigation_tech`
(``irr_<method>_<crop>``), which each draw from this shared land pool and
apply their own irrigated/rainfed-specific costs, water/electricity
consumption, and yield.

This matches the R model's own structure exactly: ``crop_<crop>``'s
``output`` block builds only ``<crop>_land`` -- no yield, no residue. Yield
is a ``rainfed_<crop>``/``irr_<method>_<crop>``-layer output, not a
``crop_<crop>``-layer one.

Deliberate simplifications relative to the R model
----------------------------------------------------
- **On-farm machinery energy input dropped.** The R model gives
  ``crop_<crop>`` a small generic ``energy`` input (diesel-equivalent
  proxy, Rao et al. 2018 average intensity). Nothing in this Python model
  produces or consumes a matching commodity, and inventing an unbacked one
  risks the same "orphaned, never-solved" problem as the GLOBIOM-linked
  ``add_irrigation_demand()`` this port replaces (see module docstring in
  ``data/demands.py``). Explicit user decision -- see the crop/irrigation
  migration plan. ``crop_<crop>`` technologies here have **no input at
  all**, capacity-bounded purely by ``historical_new_capacity`` (a
  legitimate MESSAGE pattern, e.g. resource-extraction technologies).
- **Fertilizer emission_factor** (Phase 4). The R model computes this via
  a province->basin area-weighted GIS raster extraction, not available
  here (no geopandas/shapely/rasterio) -- resolved instead by hand-parsing
  the real province/basin shapefile directly
  (``pre_processing/build_fertilizer_weights_IRB.py`` ->
  ``emission_factor_IRB.csv``). Uses the real irrigated rate where
  available, falling back to rainfed (matching the R model's own priority
  rule) -- basins with no fertilizer data in any overlapping province
  (purely Afghan/Chinese basins) simply have no ``emission_factor`` row,
  not a fabricated zero. Registered under a new ``CO2eq`` emission
  category (``config.yaml``'s ``emission_category``/``type_emission``),
  not one of the base model's individual-species codes, since the source
  data is already a single combined GWP-weighted figure -- same pattern
  the water module already uses for its own ``fresh_return`` emission
  category. Gated by ``config.yaml``'s ``emission_factor_enabled`` flag;
  when disabled, omitted entirely (honest -- zero emissions tracked --
  rather than a fabricated placeholder value).
- **Cost unit conversion.** ``crop_tech_data_IRB.csv`` labels
  ``inv_cost``/``fix_cost``/``var_cost`` "USD per ha", while this
  technology's capacity/activity is in Mha (1 Mha = 1e6 ha). The model's
  cost unit is million USD (power plants: USD/kW x GW), and USD/ha equals
  million USD/Mha numerically, so the per-ha figure is used unchanged, as
  the R model does. (An earlier version multiplied by 1e6, which made crop
  costs a million times too large relative to the energy system.)
- **Annual (lft=1) vintaging dropped -- three attempts, real bugs found
  and fixed at each step, not a style choice.** The R model re-decides
  land allocation fresh every year (``lft = 1``). That's not used here.
  The full story, in order:

  1. **Same-year-only, seeded at a pre-model year (first try).** With a
     1-year lifetime, this technology can only ever draw capacity from
     ``historical_new_capacity`` registered at *that same pre-model
     year* -- structurally unreachable, since a same-year-only
     technology's own vintages are always ``year_vtg == year_act >=
     firstmodelyear``. Confirmed for real: every ``crop_<crop>``
     technology showed exactly 0 activity in every real-scenario check
     (216 of 216 (node, year) combinations).
  2. **A single long-lived historical vintage (second try).** Traded #1
     for a subtler bug: GAMS's own ``technical_lifetime`` data-
     completeness check forces extra, data-less "vintage" entries at
     every year the technology is merely *active*, and downstream demand
     pull exploited those instead of ever running the real vintage. See
     :mod:`.irrigation_tech`'s module docstring ("Phantom vintages").
  3. **Same-year-only everywhere, re-seeding the same real value every
     period (third try).** Sidestepped #2's phantom-vintage problem, but
     turned out to rest on a wrong premise: ``historical_new_capacity``
     registered at a model-horizon ``year_vtg`` is *silently ignored* by
     GAMS (confirmed directly against ``model_core.gms``:
     ``CAPACITY_MAINTENANCE_HIST`` -- the only equation that reads
     ``historical_new_capacity`` -- is gated on ``historical(vintage)``,
     true only for a genuine pre-model year). The capacity this design
     showed working was entirely from *unconstrained new investment*,
     not from real 2015 land data as believed at the time -- see
     :mod:`.irrigation_tech`'s module docstring for the full
     correction.

  **The actual fix**: the full, natural, *unrestricted*
  ``get_vintage_and_active_years()`` result (:func:`.utils
  .get_vintage_and_active_years`, no ``same_year_only``, no filtering to
  one vintage) -- exactly the pattern any ordinary message_ix investment
  technology uses. This gives every model-horizon year a legitimate
  "build new here" vintage candidate, each with its own real, matching
  ``output``/cost data (not just ``technical_lifetime``, which is what
  made attempt #2's phantom vintages exploitable) -- so any new capacity
  GAMS chooses to build is genuine and cost-real, never free.
  ``historical_new_capacity`` is registered only at the one genuine
  pre-model seed year, where it actually applies. Real, accepted
  tradeoff: new investment beyond the real 2015 baseline remains
  possible wherever the real cost data favors it -- attempts to
  structurally forbid this (``bound_new_capacity_up = 0`` everywhere)
  were tried and made capacity zero *everywhere* instead (see
  :mod:`.irrigation_tech`'s docstring), so this is not a design choice
  free of consequence, but the one that keeps real historical data
  meaningful.
"""

import logging

import pandas as pd
from message_ix import make_df

from message_ix_models import Context, ScenarioInfo
from message_ix_models.model.water.utils import get_vintage_and_active_years
from message_ix_models.util import broadcast, same_node, same_time

from .common import (
    CONFIG,
    CROPS,
    basins_with_crop_data,
    crop_land_commodity,
    crop_tech_name,
    read_crop_input_data,
    read_crop_tech_data,
    read_emission_factor,
)

log = logging.getLogger(__name__)

#: Land-area parameters (crop_input_data_IRB.csv's `par` values) that sum to
#: this technology's historical capacity.
_LAND_2015_PARS = ["crop_irr_land_2015", "crop_rainfed_land_2015"]

#: USD/ha -> the model's cost unit per Mha. The model counts costs in million
#: USD (power plants: USD/kW x GW), and USD/ha x 1e6 ha/Mha / 1e6 = million
#: USD/Mha, so the per-ha figure is used unchanged -- as the R model does.
_HA_TO_MHA_COST = 1.0

#: kgCO2eq/ha -> tCO2eq/Mha (x1e6 ha->Mha, /1e3 kg->t -- net x1e3).
_KG_HA_TO_T_MHA = 1e3


def crop_technology_names(context: "Context") -> list[str]:
    """Return every ``crop_<crop>`` technology name for ``context``.

    Called from :func:`message_ix_models.model.water.build.get_spec` to
    register these (data-driven crop list, not statically enumerable in
    ``technology.yaml``) names ahead of :func:`add_crop_land_techs` adding
    parameter data that references them. Empty if there are no valid
    basins for this region set.
    """
    valid_basins = getattr(context, "valid_basins", None) or set()
    if not (basins_with_crop_data() & set(valid_basins)):
        return []
    return [crop_tech_name(c) for c in CROPS]


def crop_commodity_names() -> list[str]:
    """Return every ``<crop>_land`` commodity name this module produces."""
    return [crop_land_commodity(c) for c in CROPS]


def add_crop_land_techs(context: "Context") -> dict[str, pd.DataFrame]:
    """Build ``crop_<crop>`` land-commitment technologies.

    Parameters
    ----------
    context : .Context

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        Keys are MESSAGE parameter names such as 'output', 'inv_cost'.
    """
    valid_basins = sorted(
        (getattr(context, "valid_basins", None) or set()) & basins_with_crop_data()
    )
    if not valid_basins:
        return {}

    scen = context.get_scenario()
    scenario_info = ScenarioInfo(scen)
    lifetime = CONFIG["crop_technical_lifetime"]
    firstyear = scen.firstmodelyear
    sub_time = pd.Series(context.time)

    # Full, natural get_vintage_and_active_years() -- the same pattern any
    # ordinary message_ix investment technology uses, deliberately NOT
    # restricted to a single vintage. See irrigation_tech.py's module
    # docstring ("Phantom vintages" / the historical-capacity-vs-no-new-
    # investment tension) for the full story of why: real
    # historical_new_capacity only ever applies at a genuine pre-model
    # year_vtg, so getting it to matter requires a real historical vintage
    # to exist in yv_ya at all -- but GAMS's own technical_lifetime
    # completeness check independently requires every model-horizon year
    # the technology could ever be active in to also be a valid *vintage*
    # candidate. Rather than fight that (which created an exploitable
    # "phantom", data-less vintage when tried), this gives every one of
    # those candidate vintages real, matching input/output/cost data too
    # -- exactly like normal investment technologies -- so any new
    # capacity GAMS chooses to build is genuine, resource- and cost-real,
    # never free. Real historical capacity persists from the one true
    # seed; new investment remains possible where the real economics
    # favor it (accepted tradeoff -- see module docstring).
    yv_ya = get_vintage_and_active_years(scenario_info, lifetime)
    year_vtg_all = sorted(set(yv_ya["year_vtg"]))
    hist_years = [y for y in year_vtg_all if y < firstyear]
    hist_year_vtg = max(hist_years) if hist_years else None

    node_loc = pd.Series(["B" + b for b in valid_basins])

    df_land = read_crop_input_data(context)
    df_land = df_land[df_land["par"].isin(_LAND_2015_PARS)]
    cap_by_crop_node = df_land.groupby(["crop", "node"], as_index=False)["value"].sum()

    df_costs = read_crop_tech_data()
    df_costs = df_costs[df_costs["par"].isin(["inv_cost", "fix_cost", "var_cost"])]
    costs_by_crop = df_costs.set_index(["crop", "par"])["value"]

    def cost(crop: str, par: str) -> float:
        try:
            return float(costs_by_crop.loc[(crop, par)]) * _HA_TO_MHA_COST
        except KeyError:
            return 0.0

    emission_factor_enabled = CONFIG.get("emission_factor_enabled", False)
    df_emission = read_emission_factor(context) if emission_factor_enabled else pd.DataFrame()

    out_frames, tl_frames, cons_frames, cf_frames = [], [], [], []
    inv_frames, fix_frames, var_frames, hist_cap_frames = [], [], [], []
    ef_frames = []

    for crop in CROPS:
        tech = crop_tech_name(crop)
        commodity = crop_land_commodity(crop)

        out_frames.append(
            make_df(
                "output",
                technology=tech,
                value=1.0,
                unit="Mha",
                level="crop",
                commodity=commodity,
                mode="M1",
                node_loc=node_loc,
            )
            .pipe(broadcast, yv_ya, time=sub_time)
            .pipe(same_node)
            .pipe(same_time)
        )

        tl_frames.append(
            make_df(
                "technical_lifetime", technology=tech, value=lifetime, unit="y", node_loc=node_loc
            ).pipe(broadcast, year_vtg=year_vtg_all)
        )
        cons_frames.append(
            make_df(
                "construction_time", technology=tech, value=1, unit="y", node_loc=node_loc
            ).pipe(broadcast, year_vtg=year_vtg_all)
        )
        cf_frames.append(
            make_df(
                "capacity_factor", technology=tech, value=1.0, unit="%", node_loc=node_loc
            ).pipe(broadcast, yv_ya, time=sub_time)
        )

        inv_frames.append(
            make_df(
                "inv_cost", technology=tech, value=cost(crop, "inv_cost"), unit="USD/Mha", node_loc=node_loc
            ).pipe(broadcast, year_vtg=year_vtg_all)
        )
        fix_frames.append(
            make_df(
                "fix_cost", technology=tech, value=cost(crop, "fix_cost"), unit="USD/Mha", node_loc=node_loc
            ).pipe(broadcast, yv_ya)
        )
        var_frames.append(
            make_df(
                "var_cost",
                technology=tech,
                value=cost(crop, "var_cost"),
                unit="USD/Mha",
                mode="M1",
                node_loc=node_loc,
            ).pipe(broadcast, yv_ya, time=sub_time)
        )

        crop_cap = cap_by_crop_node[cap_by_crop_node["crop"] == crop]
        crop_cap = crop_cap[crop_cap["value"] > 0]
        if hist_year_vtg is not None and not crop_cap.empty:
            hist_cap_frames.append(
                make_df(
                    "historical_new_capacity",
                    technology=tech,
                    node_loc="B" + crop_cap["node"],
                    year_vtg=hist_year_vtg,
                    value=crop_cap["value"],
                    unit="Mha",
                )
            )

        if emission_factor_enabled and not df_emission.empty:
            crop_ef = df_emission[df_emission["crop"] == crop]
            # Prefer the irrigated rate where real data has one, else fall
            # back to rainfed -- matches the R model's own priority rule
            # (basin_msggdx_technologies.r: irrigated unless no irrigated
            # entry exists for that basin/crop).
            crop_ef = (
                crop_ef.sort_values("irrigation", ascending=False)  # "rainfed" < "irrigated"
                .drop_duplicates(subset="node", keep="last")
            )
            crop_ef = crop_ef[crop_ef["node"].isin(valid_basins)]
            if not crop_ef.empty:
                ef_node_loc = "B" + crop_ef["node"]
                ef_frames.append(
                    make_df(
                        "emission_factor",
                        technology=tech,
                        node_loc=ef_node_loc,
                        mode="M1",
                        emission=CONFIG["emission_category"],
                        value=crop_ef["value"] * _KG_HA_TO_T_MHA,
                        unit="tCO2eq",
                    ).pipe(broadcast, yv_ya)
                )

    def _concat(frames):
        return (
            pd.concat(frames).drop_duplicates().reset_index(drop=True)
            if frames
            else pd.DataFrame()
        )

    results = {
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
    if ef_frames:
        results["emission_factor"] = _concat(ef_frames)

    return {k: v for k, v in results.items() if not v.empty}
