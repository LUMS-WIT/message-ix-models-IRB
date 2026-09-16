"""Diagnostic plots for a solved MESSAGEix-Nexus scenario.

A Python port of NEST's ``basin_msggdx_diagnostics.r`` (the legacy R+GAMS
``indus_ix`` model's post-solve diagnostics/plotting script), rewritten
against data actually present in this repo's Pakistan/R12 build rather than
the old Indus-model's own level/commodity naming, which does not match --
see ``PORTING_NOTES`` below.

Mirrors the R script's core recipe for every report: pull raw scenario
tables (``ACT``, ``CAP``, ``input``, ``output``, costs), join activity
against per-unit values to get real quantities, classify technology names
into human-readable categories via regex, aggregate, and plot.

PORTING_NOTES
-------------
Three of the original R script's eight reports are deliberately NOT ported
here, because the data they need doesn't exist in this repo (not because
they were hard):

- ``storage.pdf`` needs the ``STORAGE`` GAMS variable, which requires a
  storage/reservoir module this repo doesn't have yet.
- ``water_electricity_flow_maps.pdf`` needs real basin/ocean GIS shapefiles
  (the R script reads them from a ``P:/is-wel/...`` network drive not
  present here) -- the same missing-external-data problem as
  ``storage_capacity.r``'s dam data.
- The R model's "final energy by use" split (``urban_final``/
  ``rural_final``/``industry_final``/``agriculture_final`` levels) has no
  equivalent here: this build's ``urban_mw``/``rural_mw``/``industry_mw``
  commodities are *water* demand (unit ``MCM/year``), not electricity
  end-use categories, so there is nothing structurally equivalent to port.

Everything else here was verified against a real solved scenario
(``MESSAGEix-Pakistan 1`` / ``LTS_nexus_debug`` v7) before being written,
not translated blind from the R model's own (different) naming.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

# Must be set before pyplot is imported. This module only ever saves PDFs to
# disk -- it never needs a GUI window -- and running with the default
# interactive backend (Tk) in the same process as ixmp's embedded JVM caused
# a hard JVM crash (Tcl_AsyncDelete / topLevelExceptionFilter) at interpreter
# shutdown, confirmed while testing this module against a real scenario.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402

if TYPE_CHECKING:
    from message_ix import Scenario

log = logging.getLogger(__name__)

#: Technology-name regex -> human category, verified against the real
#: technology set of MESSAGEix-Pakistan 1 / LTS_nexus_debug (819 techs).
#: Order matters: more specific patterns are listed before general ones.
TECH_CATEGORIES: dict[str, str] = {
    # NB technology names are "irrigation_cereal" etc (the "irr_*" spelling
    # belongs to the *level* set, e.g. output level "irr_cereal") -- both
    # patterns are kept in case a build uses the shorter tech-name form.
    "irrigation": r"^irr_|^irrigation_",
    "crop land": r"^crop_|^rainfed_",
    "network transfer": r"^canal_conv\||^river\||^trs_",
    "desalination": r"^membrane$|^distillation$|^desal_",
    "water extraction": r"^extract_",
    # NOT storage -- verified against their own definition
    # (water_supply.py, ~line 360: "input dataframe for slack technology
    # balancing equality with demands"). ``gw_recharge``/``return_flow``
    # draw 1:1 from groundwater_basin/surfacewater_basin purely to let the
    # commodity balance close as an equality; there is no reservoir/dam or
    # other real storage technology anywhere in this repo (see the module
    # docstring's PORTING_NOTES). Whatever these two carry is water that
    # went unused by every real technology in the basin, not water banked
    # for later use.
    "balancing slack (unallocated water)": r"^gw_recharge$|^return_flow$",
    "hydro": r"^hydro_",
    "coal": r"^coal_",
    "gas": r"^gas_|^LNG_",
    "nuclear": r"^nuc_",
    "solar": r"^solar_",
    "wind": r"^wind_",
    "geothermal": r"^geo_",
    "biomass": r"^bio_",
}

#: Technologies that draw electricity to move or treat water -- the
#: "energy for water" nexus link. Verified present in the real scenario.
WATER_ELECTRICITY_TECHS = (
    "extract_surfacewater",
    "extract_groundwater",
    "extract_gw_fossil",
    "extract_salinewater_basin",
    "extract_salinewater_cool",
    "membrane",
    "distillation",
)

#: Hydropower technologies -- the "water for energy" nexus link.
HYDRO_TECHS = ("hydro_hc", "hydro_lc")

#: ``demand`` parameter commodity -> human label, verified against every
#: ``commodity=`` literal in :mod:`~message_ix_models.model.water.data.demands`.
#: A *positive* ``demand`` value is a withdrawal (a sink -- water leaving the
#: modelled node to serve an end use); a *negative* value is exogenous supply
#: entering the node (natural water availability, or wastewater handed back
#: to the system) -- see :func:`node_demand_flows`. This dict only supplies
#: the label; sink/source is always read off the sign of ``value``, not
#: guessed from the commodity name, so an unexpected sign is still shown
#: correctly (just with a generic label if the commodity is unrecognized).
DEMAND_LABELS: dict[str, str] = {
    "urban_mw": "urban connected demand",
    "rural_mw": "rural connected demand",
    "industry_mw": "industry demand",
    "urban_disconnected": "urban disconnected demand",
    "rural_disconnected": "rural disconnected demand",
    "urban_collected_wst": "urban wastewater (collected)",
    "rural_collected_wst": "rural wastewater (collected)",
    "urban_uncollected_wst": "urban wastewater (uncollected)",
    "rural_uncollected_wst": "rural wastewater (uncollected)",
    "industry_uncollected_wst": "industry wastewater (uncollected)",
    "surfacewater_basin": "natural surface water availability",
    "groundwater_basin": "natural groundwater availability",
}


def classify_technology(
    tech: pd.Series, categories: dict[str, str] = TECH_CATEGORIES
) -> pd.Series:
    """Map technology names to a human category via regex.

    Unmatched technologies are labelled ``"other"`` rather than dropped, so
    totals computed from the categorized frame still reconcile with the raw
    data.
    """
    result = pd.Series("other", index=tech.index)
    for name, pattern in categories.items():
        mask = tech.str.contains(pattern, regex=True, na=False) & (result == "other")
        result = result.where(~mask, name)
    return result


def _load(scen: "Scenario") -> dict[str, pd.DataFrame]:
    """Pull the raw scenario tables every report below is built from."""
    return {
        "ACT": scen.var("ACT"),
        "CAP": scen.var("CAP"),
        "CAP_NEW": scen.var("CAP_NEW"),
        "input": scen.par("input"),
        "output": scen.par("output"),
        "demand": scen.par("demand"),
        "inv_cost": scen.par("inv_cost"),
        "fix_cost": scen.par("fix_cost"),
        "var_cost": scen.par("var_cost"),
        "PRICE_COMMODITY": scen.var("PRICE_COMMODITY"),
    }


#: Table -> the column identifying "what quantity is this" for a unit check.
#: Cost tables are keyed by technology (their unit should be currency/time,
#: not physical, so checking commodity would be meaningless); par tables
#: that carry a commodity are keyed by commodity instead.
_UNIT_CHECK_KEY: dict[str, str] = {
    "demand": "commodity",
    "input": "commodity",
    "output": "commodity",
    "inv_cost": "technology",
    "fix_cost": "technology",
    "var_cost": "technology",
}


def check_unit_consistency(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flag every commodity/technology recorded with more than one distinct
    ``unit`` string, within a table and across tables.

    Motivation: this build's own source uses more than one literal spelling
    for the same physical unit across files -- e.g. ``demands.py`` writes
    water-availability ``demand`` rows as ``"MCM/year"`` while
    ``report.py``'s fast reporting path (:func:`~.report._build_water_report_df`)
    hardcodes ``"MCM/yr"`` for every flow it collects, regardless of what
    unit the underlying ``input``/``output`` par row actually carries.
    pandas will happily sum/plot values that carry different unit strings
    without complaint, so nothing else in this module would ever surface
    that mismatch -- this is the check for it.

    Starts from water availability (``demand`` rows at level
    ``water_avail_basin`` for ``surfacewater_basin``/``groundwater_basin``)
    because that is the root supply of the whole node balance everything
    else in :func:`node_balance_flows`/:func:`node_demand_flows` is built
    from -- a unit error there propagates through every downstream flow.

    Parameters
    ----------
    data :
        As returned by :func:`_load` (or any dict with the same keys/shape).

    Returns
    -------
    pd.DataFrame
        Columns ``scope`` ("water availability" / "<table>" / "cross-table"),
        ``key`` (commodity or technology), ``units`` (the distinct unit
        strings found, joined for display), ``n_rows``. Empty if every table
        is internally consistent, water availability included.
    """
    rows: list[dict] = []

    def _within_table(scope: str, df: pd.DataFrame, key_col: str) -> None:
        if df is None or df.empty or "unit" not in df.columns or key_col not in df.columns:
            return
        for key, sub in df.groupby(key_col):
            units = sorted(sub["unit"].dropna().unique())
            if len(units) > 1:
                rows.append({
                    "scope": scope, "key": key,
                    "units": ", ".join(units), "n_rows": len(sub),
                })

    dmd = data.get("demand", pd.DataFrame())
    if dmd is not None and not dmd.empty and "level" in dmd.columns:
        avail = dmd[dmd["level"] == "water_avail_basin"]
        _within_table("water availability", avail, "commodity")
        # surfacewater_basin and groundwater_basin are different commodities,
        # so the per-commodity check above won't catch the two disagreeing
        # with each other -- check the whole water_avail_basin group too.
        avail_units = sorted(avail["unit"].dropna().unique())
        if len(avail_units) > 1:
            rows.append({
                "scope": "water availability",
                "key": "(surfacewater_basin vs groundwater_basin)",
                "units": ", ".join(avail_units),
                "n_rows": len(avail),
            })

    for table, key_col in _UNIT_CHECK_KEY.items():
        _within_table(table, data.get(table, pd.DataFrame()), key_col)

    # Cross-table: same commodity, different unit in input vs. output vs. demand.
    commodity_units: dict[str, dict[str, list[str]]] = {}
    for table in ("input", "output", "demand"):
        df = data.get(table)
        if df is None or df.empty or "unit" not in df.columns:
            continue
        for commodity, units in df.groupby("commodity")["unit"]:
            commodity_units.setdefault(commodity, {})[table] = sorted(units.dropna().unique())
    for commodity, per_table in commodity_units.items():
        all_units = {u for units in per_table.values() for u in units}
        if len(all_units) > 1:
            detail = "; ".join(f"{t}={'/'.join(u)}" for t, u in per_table.items())
            rows.append({
                "scope": "cross-table", "key": commodity,
                "units": detail, "n_rows": None,
            })

    if not rows:
        return pd.DataFrame(columns=["scope", "key", "units", "n_rows"])
    result = pd.DataFrame(rows)
    scope_order = {"water availability": 0, "cross-table": 1}
    result["_sort"] = result["scope"].map(lambda s: scope_order.get(s, 2))
    return result.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def check_scenario_unit_consistency(scen: "Scenario") -> pd.DataFrame:
    """One-call wrapper: load `scen` and run :func:`check_unit_consistency`."""
    return check_unit_consistency(_load(scen))


def cost_by_technology(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Investment + fixed + variable cost per technology per year.

    Port of ``basin_msggdx_diagnostics.r``'s "COSTS" section (lines 69-163):
    ``CAP_NEW``/``CAP``/``ACT`` joined against ``inv_cost``/``fix_cost``/
    ``var_cost``, summed per (node, technology, year), classified into a
    human-readable category.
    """
    cap_new = data["CAP_NEW"].rename(columns={"lvl": "cap_new"})
    inv = data["inv_cost"].rename(columns={"value": "inv_unit"})[
        ["node_loc", "technology", "year_vtg", "inv_unit"]
    ]
    ic = cap_new.merge(inv, on=["node_loc", "technology", "year_vtg"], how="left")
    ic["invc"] = ic["cap_new"] * ic["inv_unit"].fillna(0)
    ic = (
        ic.groupby(["node_loc", "technology", "year_vtg"], as_index=False)["invc"]
        .sum()
        .rename(columns={"year_vtg": "year"})
    )

    cap = data["CAP"].rename(columns={"lvl": "cap"})
    fix = data["fix_cost"].rename(columns={"value": "fix_unit"})[
        ["node_loc", "technology", "year_vtg", "year_act", "fix_unit"]
    ]
    fc = cap.merge(
        fix, on=["node_loc", "technology", "year_vtg", "year_act"], how="left"
    )
    fc["fixc"] = fc["cap"] * fc["fix_unit"].fillna(0)
    fc = (
        fc.groupby(["node_loc", "technology", "year_act"], as_index=False)["fixc"]
        .sum()
        .rename(columns={"year_act": "year"})
    )

    act = data["ACT"].rename(columns={"lvl": "act"})
    var = data["var_cost"].rename(columns={"value": "var_unit"})[
        ["node_loc", "technology", "year_vtg", "year_act", "mode", "time", "var_unit"]
    ]
    vc = act.merge(
        var,
        on=["node_loc", "technology", "year_vtg", "year_act", "mode", "time"],
        how="left",
    )
    vc["varc"] = vc["act"] * vc["var_unit"].fillna(0)
    vc = (
        vc.groupby(["node_loc", "technology", "year_act"], as_index=False)["varc"]
        .sum()
        .rename(columns={"year_act": "year"})
    )

    result = ic.merge(fc, on=["node_loc", "technology", "year"], how="outer")
    result = result.merge(vc, on=["node_loc", "technology", "year"], how="outer")
    result[["invc", "fixc", "varc"]] = result[["invc", "fixc", "varc"]].fillna(0)
    result["total_cost"] = result["invc"] + result["fixc"] + result["varc"]
    result["category"] = classify_technology(result["technology"])
    return result[result["total_cost"] > 0].reset_index(drop=True)


def _flow(
    data: dict[str, pd.DataFrame],
    par: str,
    filt: dict,
    value_col: str = "flow",
) -> pd.DataFrame:
    """Join ``ACT`` against a filtered slice of ``input``/``output`` to get
    a real physical quantity (activity x per-unit value)."""
    df = data[par]
    for col, values in filt.items():
        df = df[df[col].isin(values)] if isinstance(values, (list, tuple)) else df[df[col] == values]
    if df.empty:
        return pd.DataFrame(columns=["node_loc", "technology", "year", value_col])

    merged = df.merge(
        data["ACT"],
        on=["node_loc", "technology", "year_vtg", "year_act", "mode", "time"],
        how="inner",
        suffixes=("_rate", "_act"),
    )
    merged[value_col] = merged["value"] * merged["lvl"]
    return (
        merged.groupby(["node_loc", "technology", "year_act"], as_index=False)[
            value_col
        ]
        .sum()
        .rename(columns={"year_act": "year"})
    )


def nexus_interactions(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The three headline nexus numbers, ported from
    ``basin_msggdx_diagnostics.r``'s "Nexus interactions" section
    (lines 246-334), re-targeted at this build's real technology set.

    Returns
    -------
    dict with keys ``energy_for_water``, ``water_for_energy``,
    ``water_for_irrigation`` -- each a long DataFrame of
    (node_loc, technology, year, value).

    Note on ``water_for_energy``: confirmed empty in this build, not a bug.
    ``hydro_hc``/``hydro_lc`` have zero ``input`` rows at all -- hydropower
    here is resource/capacity-constrained, not represented as consuming a
    water commodity, unlike the old R model. Kept in the return value (as an
    empty frame) so callers can see this explicitly rather than the key
    silently disappearing.
    """
    energy_for_water = _flow(
        data,
        "input",
        {"commodity": "electr", "technology": list(WATER_ELECTRICITY_TECHS)},
        value_col="electr_MWa",
    )

    water_for_energy = _flow(
        data,
        "input",
        {
            "commodity": ["freshwater", "freshwater_basin", "surfacewater_basin"],
            "technology": list(HYDRO_TECHS),
        },
        value_col="water_MCM",
    )

    water_for_irrigation = _flow(
        data,
        "input",
        {"commodity": "surfacewater_basin", "level": "water_avail_basin"},
        value_col="water_MCM",
    )
    water_for_irrigation = water_for_irrigation[
        water_for_irrigation["technology"].str.startswith("irr_")
    ]

    return {
        "energy_for_water": energy_for_water,
        "water_for_energy": water_for_energy,
        "water_for_irrigation": water_for_irrigation,
    }


def node_balance_flows(
    data: dict[str, pd.DataFrame],
    node: str,
    year: int,
    include_electricity: bool = False,
    min_flow: float = 1e-3,
) -> pd.DataFrame:
    """Input/output flows through a single node for one year.

    A technology's own location (``node_loc``) is not always the node whose
    balance we want: inter-basin transfer technologies (:mod:`network`) sit
    at their *origin* basin (``node_loc == from_node``) but deliver to a
    different ``node_dest``. So "what happens at `node`" is defined as:
    input rows where ``node_origin == node`` (`node` supplies the commodity
    consumed there) unioned with output rows where ``node_dest == node``
    (`node` receives the commodity produced there) -- which for the common
    same-node technologies (``node_loc == node_origin == node_dest``, via
    :func:`~message_ix_models.util.same_node`) is equivalent to filtering on
    ``node_loc``, and additionally picks up cross-basin transfers correctly.

    Returns a long DataFrame with columns ``direction`` ("in"/"out"),
    ``technology``, ``commodity``, ``level``, ``flow`` (activity x per-unit
    value, in the parameter's native unit -- MCM/yr for water, GWa/yr for
    electricity).
    """
    act = data["ACT"]
    act = act[act["year_act"] == year]
    merge_cols = ["node_loc", "technology", "year_vtg", "year_act", "mode", "time"]

    inp = data["input"]
    inp = inp[(inp["node_origin"] == node) & (inp["year_act"] == year)]
    inflow = inp.merge(act, on=merge_cols, how="inner")
    inflow["flow"] = inflow["value"] * inflow["lvl"]
    inflow["direction"] = "in"

    out = data["output"]
    out = out[(out["node_dest"] == node) & (out["year_act"] == year)]
    outflow = out.merge(act, on=merge_cols, how="inner")
    outflow["flow"] = outflow["value"] * outflow["lvl"]
    outflow["direction"] = "out"

    cols = ["direction", "technology", "commodity", "level", "flow"]
    result = pd.concat([inflow[cols], outflow[cols]], ignore_index=True)
    result = result.groupby(
        ["direction", "technology", "commodity", "level"], as_index=False
    )["flow"].sum()

    if not include_electricity:
        result = result[result["commodity"] != "electr"]
    result = result[result["flow"].abs() >= min_flow]
    return result.reset_index(drop=True)


def node_demand_flows(
    data: dict[str, pd.DataFrame],
    node: str,
    year: int,
    min_flow: float = 1e-3,
) -> pd.DataFrame:
    """Exogenous ``demand`` parameter rows touching a single node for one
    year -- the piece :func:`node_balance_flows` cannot see, because a
    ``demand`` entry is not tied to any technology/``ACT``.

    This is where the actual end-use split lives: urban/rural/industry
    withdrawal (positive ``value``, a sink -- summed over sub-annual
    ``time`` slices, same as the technology flows), plus two kinds of
    exogenous *supply* (negative ``value``, entering the node rather than
    leaving it): wastewater handed back by a sector, and natural water
    availability (``surfacewater_basin``/``groundwater_basin`` at level
    ``water_avail_basin`` -- rainfall/river/aquifer recharge, i.e. "input
    from where" for the whole node balance).

    Irrigation demand is deliberately NOT read from here, for a different
    reason than it looks: this repo has *two* irrigation implementations.
    The old one (``data/demands.py``'s ``add_irrigation_demand``) tied crop
    water requirement to the GLOBIOM land emulator via ``land_input``/
    ``land_output`` -- but per :mod:`~.data.crops.irrigation_tech`'s own
    module docstring, that path is "now-bypassed" on the current IRB build.
    The technology set actually in play is ``irr_<method>_<crop>`` (7
    methods x 9 crops, :mod:`~.data.crops.irrigation_tech`), which takes
    ``surfacewater_basin`` @ ``water_avail_basin`` directly as ``input`` --
    so, like the old module's withdrawal, this is a normal technology edge
    already visible via :func:`node_balance_flows`, nothing exogenous to
    add here. What's genuinely different about it: nothing forces this
    withdrawal to happen via a per-commodity ``demand`` value the way
    urban/rural/industry works. Instead each basin carries a *relation*
    (``hist_irr_withdrawal_<basin>``) with ``relation_lower`` set to real
    historical sw+gw withdrawal and ``relation_activity`` coefficients on
    every irrigation technology active there -- a cross-technology
    inequality ("total irrigation withdrawal in this basin >= historical
    baseline"), not a single commodity/level balance term, so it cannot be
    drawn as one Sankey edge the way ``demand`` can. The floor is soft
    (``SLACK_RELATION_BOUND_LO``): real basins fall 10-25% short of it.

    There is no reservoir/dam storage anywhere in this repo (see the module
    docstring's ``PORTING_NOTES``) -- ``gw_recharge``/``return_flow`` are
    NOT a storage proxy despite the name: verified against their own
    definition (``water_supply.py``, ~line 360) as "slack technology[ies]
    balancing equality with demands" -- i.e. whatever resource goes
    unclaimed by every real technology in the basin, not water banked for
    later use. Categorized "balancing slack (unallocated water)" by
    :data:`TECH_CATEGORIES`, appearing via :func:`node_balance_flows` like
    any other technology.

    Returns a long DataFrame with columns ``kind`` ("demand" for a sink,
    "supply" for exogenous inflow), ``label`` (human name from
    :data:`DEMAND_LABELS`, falling back to the raw commodity), ``commodity``,
    ``level``, ``flow`` (always >= 0 -- the sign is carried by ``kind``,
    not by the value).
    """
    dmd = data["demand"]
    dmd = dmd[(dmd["node"] == node) & (dmd["year"] == year)]
    if dmd.empty:
        return pd.DataFrame(columns=["kind", "label", "commodity", "level", "flow"])

    dmd = dmd.copy()
    dmd["kind"] = dmd["value"].apply(lambda v: "demand" if v > 0 else "supply")
    dmd["label"] = dmd["commodity"].map(DEMAND_LABELS).fillna(dmd["commodity"])
    dmd["flow"] = dmd["value"].abs()

    result = dmd.groupby(
        ["kind", "label", "commodity", "level"], as_index=False
    )["flow"].sum()
    result = result[result["flow"] >= min_flow]
    return result.reset_index(drop=True)


def _sankey_node_columns(
    labels: list[str], links: list[tuple[str, str, float]]
) -> list[float]:
    """Assign each node an x position by longest-path distance from a
    source, so the diagram reads left-to-right as a narrative (availability
    -> extraction -> distribution -> ... -> demand) instead of whatever
    order plotly's own auto-layout happens to pick -- which, left alone,
    can put unrelated source-only nodes (e.g. natural groundwater and
    surface water availability) at visually arbitrary positions relative to
    each other and to the withdrawals/demands they ultimately balance
    against, even though they're both already leftmost by construction.

    ``Supply: ...`` nodes are pinned to column 0 and ``Demand: ...`` nodes
    to the last column regardless of their computed distance, since they
    are sources/sinks by definition; everything else is however many hops
    it actually is from a source along real flow edges (a technology
    chained three deep, e.g. extraction -> distribution -> irrigation, ends
    up visibly further right than a technology fed directly from a source).
    """
    from collections import deque

    graph: dict[str, list[str]] = {label: [] for label in labels}
    indeg: dict[str, int] = dict.fromkeys(labels, 0)
    for s, t, _ in links:
        graph[s].append(t)
        indeg[t] += 1

    layer: dict[str, int] = dict.fromkeys(labels, 0)
    indeg_work = dict(indeg)
    queue = deque(label for label in labels if indeg_work[label] == 0)
    while queue:
        n = queue.popleft()
        for m in graph[n]:
            layer[m] = max(layer[m], layer[n] + 1)
            indeg_work[m] -= 1
            if indeg_work[m] == 0:
                queue.append(m)
    # Any node still with indeg_work > 0 sits on a cycle (return flows,
    # inter-basin loops); it keeps whatever layer it reached before its
    # last unresolved predecessor -- an approximation, fine for layout.

    for label in labels:
        if label.startswith("Supply: "):
            layer[label] = 0
    pre_demand_max = max(layer.values(), default=0)
    for label in labels:
        if label.startswith("Demand: "):
            layer[label] = pre_demand_max + 1

    max_layer = max(layer.values(), default=0) or 1
    return [0.001 + 0.998 * (layer[label] / max_layer) for label in labels]


def plot_node_balance_sankey(
    flows: pd.DataFrame,
    node: str,
    year: int,
    demand: pd.DataFrame | None = None,
    out_dir: Path | None = None,
):
    """Sankey diagram of a node's water balance for one year.

    ``flows`` is the output of :func:`node_balance_flows`: input commodities
    feed into technology nodes on the left, technology nodes feed output
    commodities on the right -- e.g. ``surfacewater (water_supply) ->
    extract_surfacewater -> freshwater (water_supply) -> urban_t_d ->
    freshwater (final)`` reads as extraction, then distribution.

    ``demand``, from :func:`node_demand_flows`, supplies the two things
    ``flows`` cannot show because they are not technology activity: end-use
    withdrawal (``kind == "demand"``, drawn as commodity -> ``"Demand: ..."``
    sink) and exogenous supply (``kind == "supply"``, drawn as
    ``"Supply: ..."`` source -> commodity) -- natural water availability and
    wastewater handed back both enter this way. Pass ``None`` to draw the
    technology-only diagram (the original behaviour).

    Uses plotly rather than matplotlib: unlike the rest of this module (see
    the module docstring), a Sankey needs more than a bar/line axes, and
    plotly avoids the Tk/JVM crash this module works around because it never
    touches Tkinter -- it renders to a standalone HTML file, not a GUI
    window in the same process as ixmp's embedded JVM.

    Returns the :class:`plotly.graph_objects.Figure`; also writes
    ``water_balance_<node>_<year>.html`` under ``out_dir`` if given.
    """
    import plotly.graph_objects as go

    has_demand = demand is not None and not demand.empty
    if flows.empty and not has_demand:
        raise ValueError(f"No input/output flows found for node={node!r}, year={year!r}")

    def _commodity_node(commodity: str, level: str) -> str:
        return f"{commodity} ({level})"

    links: list[tuple[str, str, float]] = []
    for _, r in flows[flows["direction"] == "in"].iterrows():
        links.append((_commodity_node(r["commodity"], r["level"]), r["technology"], r["flow"]))
    for _, r in flows[flows["direction"] == "out"].iterrows():
        links.append((r["technology"], _commodity_node(r["commodity"], r["level"]), r["flow"]))

    if has_demand:
        for _, r in demand[demand["kind"] == "demand"].iterrows():
            links.append((
                _commodity_node(r["commodity"], r["level"]),
                f"Demand: {r['label']}",
                r["flow"],
            ))
        for _, r in demand[demand["kind"] == "supply"].iterrows():
            links.append((
                f"Supply: {r['label']}",
                _commodity_node(r["commodity"], r["level"]),
                r["flow"],
            ))

    labels = sorted({s for s, _, _ in links} | {t for _, t, _ in links})
    index = {label: i for i, label in enumerate(labels)}

    technologies = sorted(set(flows["technology"])) if not flows.empty else []
    category_by_tech = dict(
        zip(technologies, classify_technology(pd.Series(technologies)))
    ) if technologies else {}
    palette = {
        "irrigation": "#2a9d8f", "crop land": "#8ab17d", "network transfer": "#e9c46a",
        "desalination": "#e76f51", "water extraction": "#457b9d",
        "balancing slack (unallocated water)": "#577590",
        "hydro": "#1d3557",
        "coal": "#6d6875", "gas": "#b5838d", "nuclear": "#6a4c93", "solar": "#f4a261",
        "wind": "#90be6d", "geothermal": "#f9844a", "biomass": "#43aa8b", "other": "#adb5bd",
    }

    def _node_color(label: str) -> str:
        if label.startswith("Demand: "):
            return "#e63946"  # sink: red
        if label.startswith("Supply: "):
            return "#06d6a0"  # exogenous source: green
        if label in category_by_tech:
            return palette.get(category_by_tech[label], "#adb5bd")
        return "#4c9be8"  # commodity (level) node

    node_colors = [_node_color(label) for label in labels]
    node_x = _sankey_node_columns(labels, links)

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=14,
                line=dict(color="black", width=0.4),
                label=labels,
                color=node_colors,
                x=node_x,
                hovertemplate="%{label}: %{value:.3g}<extra></extra>",
            ),
            link=dict(
                source=[index[s] for s, _, _ in links],
                target=[index[t] for _, t, _ in links],
                value=[v for _, _, v in links],
                hovertemplate="%{source.label} -> %{target.label}: %{value:.3g}<extra></extra>",
            ),
        )
    )
    fig.update_layout(title_text=f"Water balance at {node}, {year}", font_size=11)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_node = str(node).replace("|", "_").replace(" ", "_")
        path = out_dir / f"water_balance_{safe_node}_{year}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        log.info("Sankey diagram written to %s", path)

    return fig


def energy_mix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Installed capacity by source category, per node and year.

    Port of ``basin_msggdx_diagnostics.r``'s "COMMODITIES" / capacity
    section (lines 495-731), simplified to installed capacity (``CAP``)
    classified by source -- the R script's separate "new capacity" and
    generation-by-source variants follow the identical pattern with
    ``CAP_NEW``/``ACT`` in place of ``CAP``.
    """
    cap = data["CAP"].copy()
    cap["category"] = classify_technology(cap["technology"])
    fuel_categories = {"coal", "gas", "nuclear", "hydro", "solar", "wind", "geothermal", "biomass"}
    cap = cap[cap["category"].isin(fuel_categories)]
    return (
        cap.groupby(["node_loc", "category", "year_act"], as_index=False)["lvl"]
        .sum()
        .rename(columns={"year_act": "year", "lvl": "capacity_GW"})
    )


def crop_land(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Land area committed per crop, split rainfed vs. irrigated.

    Port of ``basin_msggdx_diagnostics.r``'s "CROP AREA and WATER" section
    (lines 830-877). Confirmed directly against real data: land-committing
    technologies output commodity ``crop_land`` at level ``area``.
    """
    out = data["output"]
    land = out[(out["commodity"] == "crop_land") & (out["level"] == "area")]
    merged = land.merge(
        data["ACT"],
        on=["node_loc", "technology", "year_vtg", "year_act", "mode", "time"],
        how="inner",
    )
    merged["area_Mha"] = merged["value"] * merged["lvl"]
    merged["method"] = merged["technology"].apply(
        lambda t: "rainfed" if t.startswith("rainfed_") else "irrigated"
    )
    merged["crop"] = merged["technology"].str.replace(
        r"^(rainfed_|irr_[a-z_]+_)", "", regex=True
    )
    return (
        merged.groupby(["node_loc", "crop", "method", "year_act"], as_index=False)[
            "area_Mha"
        ]
        .sum()
        .rename(columns={"year_act": "year"})
    )


def commodity_shadow_price(
    data: dict[str, pd.DataFrame], commodities: tuple[str, ...] = ("freshwater", "crop_land")
) -> pd.DataFrame:
    """Marginal (shadow) price of selected commodities over time.

    Port of ``basin_msggdx_diagnostics.r``'s "commodity_costs.pdf" section
    (lines 1660-1732), which reads the ``PRICE_COMMODITY`` dual-value
    variable -- confirmed present and populated in this build's solution.
    """
    pc = data["PRICE_COMMODITY"]
    pc = pc[pc["commodity"].isin(commodities)]
    return pc.rename(columns={"lvl": "shadow_price"})[
        ["node", "commodity", "level", "year", "time", "shadow_price"]
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _stacked_bar(
    df: pd.DataFrame, x: str, y: str, hue: str, title: str, ylabel: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if df.empty:
        ax.text(0.5, 0.5, "no data in this build", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    else:
        pivot = df.groupby([x, hue])[y].sum().unstack(fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cost_by_technology(df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "cost_by_technology.pdf"
    agg = df.groupby(["year", "category"], as_index=False)["total_cost"].sum()
    _stacked_bar(
        agg, "year", "total_cost", "category",
        "Total cost by technology category", "USD", path,
    )
    return path


def plot_nexus_interactions(flows: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    path = out_dir / "nexus_interactions.pdf"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = {
        "energy_for_water": ("Energy for water", "MWa"),
        "water_for_energy": ("Water for energy (hydro)", "MCM"),
        "water_for_irrigation": ("Water for irrigation", "MCM"),
    }
    for ax, (key, (title, unit)) in zip(axes, labels.items()):
        df = flows[key]
        if df.empty:
            ax.text(0.5, 0.5, "no data in this build", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        value_col = [c for c in df.columns if c not in ("node_loc", "technology", "year")][0]
        agg = df.groupby("year", as_index=False)[value_col].sum()
        ax.bar(agg["year"].astype(str), agg[value_col], color="#1f7a8c")
        ax.set_title(title)
        ax.set_ylabel(unit)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_energy_mix(df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "energy_mix.pdf"
    _stacked_bar(
        df, "year", "capacity_GW", "category",
        "Installed capacity by source", "GW", path,
    )
    return path


def plot_crop_land(df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "crop_land.pdf"
    _stacked_bar(
        df, "year", "area_Mha", "crop",
        "Crop area by crop type", "Mha", path,
    )
    return path


def plot_commodity_shadow_price(df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "commodity_shadow_price.pdf"
    fig, ax = plt.subplots(figsize=(8, 5))
    for commodity, sub in df.groupby("commodity"):
        agg = sub.groupby("year", as_index=False)["shadow_price"].mean()
        ax.plot(agg["year"], agg["shadow_price"], marker="o", label=commodity)
    ax.set_title("Average commodity shadow price")
    ax.set_ylabel("USD per unit")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def node_water_balance_sankey(
    scen: "Scenario",
    node: str,
    year: int,
    include_electricity: bool = False,
    include_demand: bool = True,
    min_flow: float = 1e-3,
    out_dir: str | Path | None = None,
):
    """One-call entry point: load `scen`, compute, and plot a node's water
    balance for `year` as a Sankey diagram -- technology activity
    (:func:`node_balance_flows`) plus, by default, exogenous demand and
    supply (:func:`node_demand_flows`: end-use withdrawal, wastewater
    return, natural water availability). Set ``include_demand=False`` for
    the technology-only diagram.

    Example
    -------
    >>> from message_ix_models.model.water.report_diagnostics import (
    ...     node_water_balance_sankey,
    ... )
    >>> fig = node_water_balance_sankey(scen, node="R12_PAK", year=2030)
    >>> fig.show()  # opens in the notebook / browser
    """
    data = _load(scen)
    flows = node_balance_flows(
        data, node=node, year=year,
        include_electricity=include_electricity, min_flow=min_flow,
    )
    demand = (
        node_demand_flows(data, node=node, year=year, min_flow=min_flow)
        if include_demand else None
    )
    return plot_node_balance_sankey(
        flows, node=node, year=year, demand=demand, out_dir=out_dir
    )


# ---------------------------------------------------------------------------
# Time series (from the report.py nexus CSV)
# ---------------------------------------------------------------------------
#
# Unlike the Sankey helpers above, these read the *already-reported* CSV
# that report.report()/report_full() writes to reporting_output/ -- the
# IAMC-named variables built by _build_water_report_df + _apply_iamc_mapping
# (e.g. "Water Extraction|Groundwater", "Generation|Electricity|Hydro") --
# rather than re-deriving flows from raw ACT/input/output. That way a time
# series plot always reflects report.py's own variable definitions; if a
# variable is missing, it belongs in report.py's map_agg, not reinvented
# here (see the "Generation|Electricity|Hydro" / "Capacity|Electricity|Hydro"
# / "Water Withdrawal|Electricity|Hydro*" variables added there for this).

#: Column names in the CSV report.py writes (already lower-case there).
_NEXUS_CSV_COLUMNS = ["model", "scenario", "region", "variable", "unit", "year", "value"]


def find_latest_nexus_csv(scen: "Scenario") -> Path:
    """Locate the most recently written ``reporting_output/<model>_<scenario>_nexus*.csv``
    for `scen`, matching the versioned-filename scheme ``report()`` writes
    (``..._nexus.csv``, then ``_v1``, ``_v2``, ... on repeated runs so no
    previous result is overwritten).
    """
    from message_ix_models.util import package_data_path

    out_dir = package_data_path().parents[0] / "reporting_output"
    base = f"{scen.model}_{scen.scenario}_nexus"
    candidates = sorted(
        out_dir.glob(f"{base}*.csv"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {base}*.csv found under {out_dir} -- run report.report() or "
            "report_full() first"
        )
    return candidates[-1]


def load_nexus_report(source: "Scenario | str | Path") -> pd.DataFrame:
    """Load a report.py nexus CSV as a long DataFrame.

    `source` is either a solved :class:`~message_ix.Scenario` (the latest
    matching CSV under ``reporting_output/`` is found and loaded, via
    :func:`find_latest_nexus_csv`) or a path to a specific CSV.
    """
    path = find_latest_nexus_csv(source) if not isinstance(source, (str, Path)) else Path(source)
    df = pd.read_csv(path)
    missing = set(_NEXUS_CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected column(s) {missing}")
    return df


def timeseries_by_variable(
    report_df: pd.DataFrame,
    variables: str | list[str],
    regions: list[str] | None = None,
    combine_regions: bool = False,
    national_label: str = "Pakistan (all nodes)",
) -> pd.DataFrame:
    """Slice a nexus report DataFrame down to a time series for one or more
    IAMC variables.

    Parameters
    ----------
    report_df :
        Long DataFrame as returned by :func:`load_nexus_report` (columns
        ``model, scenario, region, variable, unit, year, value``).
    variables :
        One variable name, or a list (e.g. every per-technology breakdown
        of a category) -- rows for all of them are returned, still split
        by variable, so pass one at a time for a single line.
    regions :
        Restrict to these region/node values (e.g. specific basins). ``None``
        keeps every region present in `report_df`.
    combine_regions :
        If ``True``, sum across all remaining regions into one national
        series per (variable, year) -- this is "combine all Pakistan [nodes]
        into one line" -- labelled `national_label` in the returned
        ``region`` column. If ``False`` (default), each region is kept as
        its own series (one row per region/variable/year), so a caller can
        plot per-node detail instead.

    Returns
    -------
    pd.DataFrame
        Columns ``region, variable, unit, year, value``, one row per
        (region, variable, year).
    """
    if isinstance(variables, str):
        variables = [variables]
    df = report_df[report_df["variable"].isin(variables)]
    if regions is not None:
        df = df[df["region"].isin(regions)]
    if df.empty:
        return pd.DataFrame(columns=["region", "variable", "unit", "year", "value"])

    if combine_regions:
        result = df.groupby(["variable", "unit", "year"], as_index=False)["value"].sum()
        result["region"] = national_label
    else:
        result = df.groupby(
            ["region", "variable", "unit", "year"], as_index=False
        )["value"].sum()
    return result[["region", "variable", "unit", "year", "value"]]


def plot_timeseries(
    report_df: pd.DataFrame,
    variables: str | list[str],
    regions: list[str] | None = None,
    combine_regions: bool = False,
    national_label: str = "Pakistan (all nodes)",
    title: str | None = None,
    out_dir: Path | None = None,
    filename: str | None = None,
):
    """Simple line plot of one or more IAMC variables over time, optionally
    combining every region into one national total (see
    :func:`timeseries_by_variable` for what `combine_regions` does).

    One line per (region, variable) pair actually present in the data --
    e.g. `variables=["Generation|Electricity|Hydro"], combine_regions=True`
    draws a single national hydro generation line; leaving `combine_regions`
    False instead draws one line per basin/region.

    Returns the :class:`matplotlib.figure.Figure`; also saves a PNG under
    `out_dir` if given (`filename` defaults to a slug of the variable(s)).
    """
    data = timeseries_by_variable(
        report_df, variables, regions=regions,
        combine_regions=combine_regions, national_label=national_label,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    if data.empty:
        ax.text(0.5, 0.5, "no data for these variables/regions",
                ha="center", va="center", transform=ax.transAxes)
    else:
        for (region, variable), sub in data.groupby(["region", "variable"]):
            sub = sub.sort_values("year")
            label = variable if region == sub["region"].iloc[0] and data["region"].nunique() == 1 else f"{variable} - {region}"
            ax.plot(sub["year"], sub["value"], marker="o", label=label)
        unit = data["unit"].iloc[0]
        ax.set_ylabel(unit)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    ax.set_title(title or (variables if isinstance(variables, str) else " / ".join(variables)))
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            slug = re.sub(r"[^\w]+", "_", variables if isinstance(variables, str) else "_".join(variables))
            filename = f"timeseries_{slug}.png"
        path = out_dir / filename
        plt.savefig(path, dpi=150)
        log.info("Time series plot written to %s", path)
    return fig


def plot_hydro_timeseries(
    report_df: pd.DataFrame,
    regions: list[str] | None = None,
    combine_regions: bool = True,
    national_label: str = "Pakistan (all nodes)",
    out_dir: Path | None = None,
):
    """"Proper hydro" time series: generation and installed capacity side by
    side, national totals by default (``combine_regions=True`` -- pass
    ``False`` to see per-basin/region lines instead).

    Sources ``"Generation|Electricity|Hydro"`` (GWa/yr) and
    ``"Capacity|Electricity|Hydro"`` (GW) -- both added to report.py's
    ``_apply_iamc_mapping`` alongside this function, since neither existed
    in the fast (actually-used) reporting path before.

    Returns the :class:`matplotlib.figure.Figure`; also saves
    ``hydro_timeseries.png`` under `out_dir` if given.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    specs = [
        ("Generation|Electricity|Hydro", "Hydro generation"),
        ("Capacity|Electricity|Hydro", "Hydro installed capacity"),
    ]
    for ax, (variable, subtitle) in zip(axes, specs):
        data = timeseries_by_variable(
            report_df, variable, regions=regions,
            combine_regions=combine_regions, national_label=national_label,
        )
        if data.empty:
            ax.text(0.5, 0.5, "no data in this build", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            for region, sub in data.groupby("region"):
                sub = sub.sort_values("year")
                ax.plot(sub["year"], sub["value"], marker="o", label=region)
            ax.set_ylabel(data["unit"].iloc[0])
            if data["region"].nunique() > 1:
                ax.legend(fontsize=8)
        ax.set_title(subtitle)
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "hydro_timeseries.png"
        plt.savefig(path, dpi=150)
        log.info("Hydro time series plot written to %s", path)
    return fig


def run_diagnostics(scen: "Scenario", out_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Run every ported report against a solved scenario, saving plots to
    ``out_dir`` and returning the underlying data for inspection/testing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading scenario data for diagnostics")
    data = _load(scen)

    results: dict[str, pd.DataFrame] = {}

    results["cost_by_technology"] = cost_by_technology(data)
    plot_cost_by_technology(results["cost_by_technology"], out_dir)

    flows = nexus_interactions(data)
    results.update(flows)
    plot_nexus_interactions(flows, out_dir)

    results["energy_mix"] = energy_mix(data)
    plot_energy_mix(results["energy_mix"], out_dir)

    results["crop_land"] = crop_land(data)
    plot_crop_land(results["crop_land"], out_dir)

    results["commodity_shadow_price"] = commodity_shadow_price(data)
    plot_commodity_shadow_price(results["commodity_shadow_price"], out_dir)

    log.info("Diagnostics written to %s", out_dir)
    return results
