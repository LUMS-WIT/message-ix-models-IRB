"""Non-renewable (fossil) groundwater accounting and limit -- Python port of
the legacy R+GAMS indus_ix model's groundwater split and its
"Nonrenewable Groundwater extraction" policy:

- ``basin_msggdx_technologies.r`` 8234-8381: ``gw_extract`` (pumps aquifer
  water with no renewable input and emits ``groundwater``) vs.
  ``renew_gw_extract`` (draws the renewable ``renewable_gw`` pool, emits 0).
- ``basin_msggdx_policies.r`` 547-576: bound the ``groundwater`` emission per
  node and year to a share of a baseline run's fossil extraction.

This model already has both extraction paths -- ``extract_groundwater``
(input ``groundwater_basin``, i.e. renewable) and ``extract_gw_fossil`` (no
water input) -- but nothing tracked or limited the fossil one, so it was an
unlimited free source (it supplied 3.8-12.6 km3/yr of household/industry
water in ``LTS_nexus_debug`` v4). This module adds the R accounting:

- emission ``fossil_groundwater`` (MCM) = activity of ``extract_gw_fossil``
  and of the non-renewable mode of the irrigation groundwater diversions
  (:mod:`.crops.irrigation_source`, mode ``M2``) when that layer is on;
- optionally, ``bound_emission`` on it per basin node and year.

Accounting is ON by default (``context.fossil_gw_accounting``, default
True; set False to drop it). ``context.fossil_gw_limit`` (a DataFrame with
columns ``node``, ``year``, ``value`` in MCM/year, e.g. from
:func:`limit_from_baseline`) also adds the bound; it defaults to None (no
limit), since it needs a solved baseline.

The R policy's baseline shares (:data:`R_BASELINE_SHARE`) are decadal
(2020-2060); each model year uses the nearest R year, and any year after
2060 the 2060 share (0 -- no fossil groundwater). R's rule that a node with
zero baseline extraction gets that year's largest node bound is kept, since
it is what the R model did (it avoids a zero bound on basins whose baseline
happened not to pump).
"""

import logging

import pandas as pd
from message_ix import make_df

from message_ix_models import Context, ScenarioInfo

from .crops.irrigation_source import (
    FOSSIL_MODE,
    TECH_DATA,
    irrigation_source_technology_names,
)

log = logging.getLogger(__name__)

EMISSION = "fossil_groundwater"
TYPE_EMISSION = "fossil_groundwater"
TYPE_TEC = "fossil_groundwater"

#: The existing fossil extraction technology (water_supply.py), mode M1.
FOSSIL_EXTRACT = "extract_gw_fossil"

#: ``pct_bsl`` in basin_msggdx_policies.r: share of the baseline's fossil
#: groundwater extraction allowed in each year.
R_BASELINE_SHARE = {2020: 0.5, 2030: 0.1, 2040: 0.1, 2050: 0.1, 2060: 0.0}


def fossil_gw_enabled(context: "Context") -> bool:
    return bool(getattr(context, "fossil_gw_accounting", True)) or (
        getattr(context, "fossil_gw_limit", None) is not None
    )


def _fossil_sources(context: "Context") -> list[tuple[str, str]]:
    """(technology, mode) pairs that pump non-renewable groundwater."""
    sources = [(FOSSIL_EXTRACT, "M1")]
    sources += [
        (t, FOSSIL_MODE)
        for t in irrigation_source_technology_names(context)
        if TECH_DATA[t].get("fossil")
    ]
    return sources


def fossil_gw_technology_names(context: "Context") -> list[str]:
    """Technologies mapped to the ``fossil_groundwater`` type_tec."""
    if not fossil_gw_enabled(context):
        return []
    return sorted({t for t, _ in _fossil_sources(context)})


def limit_from_baseline(
    baseline: pd.DataFrame, years, shares: dict | None = None, beta: float = 1.0
) -> pd.DataFrame:
    """R's limit: ``share(year) * baseline(node, year) * beta`` per node.

    ``baseline`` has columns ``node``, ``year``, ``value`` (fossil
    groundwater extraction of a solved baseline, MCM/year -- e.g. from
    :func:`baseline_from_scenario`). Returns ``node``, ``year``, ``value``.
    """
    shares = shares or R_BASELINE_SHARE
    r_years = sorted(shares)
    base = baseline.groupby(["node", "year"], as_index=False)["value"].sum()
    rows = []
    for year in years:
        r_year = r_years[-1] if year > r_years[-1] else min(r_years, key=lambda y: abs(y - year))
        b = base[base["year"] == year]
        if b.empty:
            continue
        v = b.assign(value=b["value"] * shares[r_year] * beta)
        # R: a node with a zero bound takes that year's largest node bound.
        v.loc[v["value"] == 0, "value"] = v["value"].max()
        rows.append(v)
    if not rows:
        return pd.DataFrame(columns=["node", "year", "value"])
    return pd.concat(rows, ignore_index=True)[["node", "year", "value"]]


def baseline_from_scenario(scen) -> pd.DataFrame:
    """Fossil groundwater extraction per node and year in a solved scenario
    built with accounting on (``EMISS`` of :data:`EMISSION`)."""
    emiss = scen.var("EMISS", filters={"emission": [EMISSION], "type_tec": [TYPE_TEC]})
    # EMISS is also reported at aggregate (region/country) nodes; keep the
    # basin nodes ("B<basin>") only, where the limit is applied.
    emiss = emiss[emiss["node"].astype(str).str.startswith("B")]
    return emiss.rename(columns={"lvl": "value"})[["node", "year", "value"]]


def add_fossil_groundwater(context: "Context") -> dict[str, pd.DataFrame]:
    """``emission_factor`` (1 MCM per MCM pumped) on every fossil source,
    plus ``bound_emission`` when ``context.fossil_gw_limit`` is set."""
    if not fossil_gw_enabled(context):
        return {}

    info = ScenarioInfo(context.get_scenario())
    pairs = info.yv_ya
    pairs = pairs[pairs["year_act"] >= pairs["year_vtg"]]
    basins = sorted(getattr(context, "valid_basins", None) or [])
    if not basins:
        return {}
    node_loc = pd.Series(["B" + b for b in basins])

    ef = []
    for tech, mode in _fossil_sources(context):
        base = pd.merge(pd.DataFrame({"node_loc": node_loc}), pairs, how="cross")
        ef.append(make_df(
            "emission_factor", technology=tech, mode=mode, emission=EMISSION,
            node_loc=base["node_loc"], year_vtg=base["year_vtg"], year_act=base["year_act"],
            value=1.0, unit="MCM",
        ))
    results = {"emission_factor": pd.concat(ef, ignore_index=True)}

    limit = getattr(context, "fossil_gw_limit", None)
    if limit is not None and len(limit):
        lim = pd.DataFrame(limit)
        lim = lim[lim["year"].isin(info.Y)]
        results["bound_emission"] = make_df(
            "bound_emission", node=lim["node"].values, type_emission=TYPE_EMISSION,
            type_tec=TYPE_TEC, type_year=lim["year"].astype(str).values,
            value=lim["value"].values, unit="MCM",
        )
    return results


def fossil_gw_type_years(context: "Context") -> list[str]:
    """``type_year`` elements the bound needs (one per bounded year)."""
    limit = getattr(context, "fossil_gw_limit", None)
    if limit is None or not len(limit):
        return []
    return sorted({str(y) for y in pd.DataFrame(limit)["year"]})
