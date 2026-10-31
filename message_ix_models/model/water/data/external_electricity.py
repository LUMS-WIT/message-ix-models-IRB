"""Electricity for water technologies in basins outside Pakistan.

Water technologies draw electricity from their basin's region node
(:func:`.crops.irrigation_source.electricity_origin`). Pakistani basins draw
from ``R12_PAK``, the national energy model, so pumping, treatment and
pressurised irrigation there compete for real generation capacity. The other
regions of the Indus basin (Afghanistan, China, India) have no energy model
here, so their region nodes get one supply technology instead:

- ``electr_supply_water_ext`` outputs ``electr`` / ``final`` at each of those
  nodes with no limit, so every water technology can run;
- its ``var_cost`` (:data:`EXTERNAL_ELECTRICITY_COST`) makes the model weigh
  electricity there as it would in Pakistan, rather than treating it as free.

Its activity is the electricity those regions' water sector would need,
i.e. the extra supply their own energy systems would have to provide.

ON by default (``context.external_electricity``, default True).

Prices
------
``context.external_electricity_cost`` sets the price. It is one of:

- ``None``: the supply is free;
- a number: one price for every external node and year;
- a mapping ``{region: price}``, where a price is a number or a path
  ``{year: number}`` (linear between the years given, flat before the first
  and after the last). A ``default`` key covers regions not listed. A region
  with neither raises an error, so no node is left free by accident.

Prices are entered in ``context.external_electricity_cost_unit`` (default
``M USD/GWa``, the model's cost unit; ``USD/kWh`` and ``USD/MWh`` are
converted, 1 GWa = 8.76e9 kWh) and in US dollars of
``context.external_electricity_currency_year``. They are converted to
dollars of ``context.model_currency_year`` with the US GDP deflator
(:data:`GDP_DEFLATOR`, or ``context.gdp_deflator``). With either year unset,
no deflation is applied.
"""

import logging

import pandas as pd
from message_ix import make_df

from message_ix_models import Context, ScenarioInfo
from message_ix_models.util import broadcast, same_time

from ..utils import region_node_map
from .crops.irrigation_source import electricity_origin

log = logging.getLogger(__name__)

#: The supply technology (same name as the zero-cost supply the NEST-Pakistan
#: build script adds, which this replaces with a costed one).
EXTERNAL_TECH = "electr_supply_water_ext"

#: Node of the Indus national energy model (default of ``region_node_map``).
NATIONAL_NODE = "R12_PAK"

#: var_cost, million USD per GWa (= 0.08 USD/kWh): close to the shadow price
#: of final electricity at R12_PAK in LTS_nexus_debug v1 (690 in 2030).
EXTERNAL_ELECTRICITY_COST = 700.0

#: kWh in one GWa (1e6 kW x 8760 h).
KWH_PER_GWA = 8.76e9

#: Factor from a price unit to the model's cost unit, million USD per GWa.
PRICE_UNITS = {
    "M USD/GWa": 1.0,
    "USD/kWh": KWH_PER_GWA / 1e6,  # 0.08 USD/kWh -> 700.8
    "USD/MWh": KWH_PER_GWA / 1e9,  # 80 USD/MWh -> 700.8
}

#: US GDP implicit price deflator, 2017 = 100 (BEA NIPA table 1.1.9, annual
#: averages, rounded; replace via ``context.gdp_deflator`` for exact values).
GDP_DEFLATOR = {
    2005: 81.5, 2010: 90.0, 2015: 98.2, 2017: 100.0, 2018: 102.3, 2019: 104.0,
    2020: 105.4, 2021: 110.2, 2022: 118.0, 2023: 122.3, 2024: 125.2,
}


def to_model_cost(value: float, unit: str = "M USD/GWa", currency_year: int | None = None,
                  model_currency_year: int | None = None, deflator: dict | None = None) -> float:
    """Convert one electricity price to million USD (of ``model_currency_year``)
    per GWa."""
    if unit not in PRICE_UNITS:
        raise ValueError(f"electricity price unit {unit!r}; use one of {sorted(PRICE_UNITS)}")
    value = float(value) * PRICE_UNITS[unit]
    if currency_year is None or model_currency_year is None or currency_year == model_currency_year:
        return value
    d = deflator or GDP_DEFLATOR
    missing = {currency_year, model_currency_year} - set(d)
    if missing:
        raise ValueError(f"no GDP deflator for {sorted(missing)}; add it to context.gdp_deflator")
    return value * d[model_currency_year] / d[currency_year]


def _path(price, years: list[int]) -> pd.Series:
    """A number or {year: number} as a value per model year."""
    if not isinstance(price, dict):
        return pd.Series(float(price), index=years)
    given = pd.Series({int(y): float(v) for y, v in price.items()}).sort_index()
    idx = sorted(set(years) | set(given.index))
    s = given.reindex(idx).interpolate(method="index", limit_area="inside")
    return s.bfill().ffill().reindex(years)


def external_electricity_prices(context: "Context", nodes: list[str], years: list[int]) -> pd.DataFrame | None:
    """Price of external electricity per node and year, in the model's cost
    unit (million USD per GWa). None when the supply is free."""
    cost = getattr(context, "external_electricity_cost", EXTERNAL_ELECTRICITY_COST)
    if cost is None:
        return None
    if not isinstance(cost, dict):
        cost = {"default": cost}
    unknown = set(cost) - set(nodes) - {"default"}
    if unknown:
        log.warning(f"external electricity prices for unknown region(s) ignored: {sorted(unknown)}")
    missing = [n for n in nodes if n not in cost and "default" not in cost]
    if missing:
        raise ValueError(f"no external electricity price for {missing}: add them to "
                         "external_electricity_cost or give a 'default'")
    conv = dict(unit=getattr(context, "external_electricity_cost_unit", None) or "M USD/GWa",
                currency_year=getattr(context, "external_electricity_currency_year", None),
                model_currency_year=getattr(context, "model_currency_year", None),
                deflator=getattr(context, "gdp_deflator", None))
    rows = []
    for n in nodes:
        for y, v in _path(cost.get(n, cost.get("default")), years).items():
            rows.append((n, y, to_model_cost(v, **conv)))
    return pd.DataFrame(rows, columns=["node_loc", "year_act", "value"])


def external_electricity_enabled(context: "Context") -> bool:
    return bool(getattr(context, "external_electricity", True))


def external_electricity_nodes(context: "Context") -> list[str]:
    """Region nodes without an energy model that water technologies draw
    power from: every region not mapped to a national energy node by
    :func:`.utils.region_node_map` (for the Indus: all but Pakistan)."""
    mapping = region_node_map(context)
    basins = sorted(getattr(context, "valid_basins", None) or [])
    origin = electricity_origin(context, basins)
    modelled = set(mapping.values())
    return sorted({o for b, o in origin.items()
                   if o not in modelled and not any(b.endswith("|" + r) for r in mapping)})


def external_electricity_technology_names(context: "Context") -> list[str]:
    if not external_electricity_enabled(context) or not external_electricity_nodes(context):
        return []
    return [EXTERNAL_TECH]


def add_external_electricity(context: "Context") -> dict[str, pd.DataFrame]:
    """``output`` of electricity at each non-Pakistani region node, and its
    ``var_cost``. Returns ``{}`` when disabled."""
    nodes = external_electricity_nodes(context) if external_electricity_enabled(context) else []
    if not nodes:
        return {}

    info = ScenarioInfo(context.get_scenario())
    years = [y for y in info.Y]
    base = pd.merge(pd.DataFrame({"node_loc": nodes}), pd.DataFrame({"year_act": years}), how="cross")
    common = dict(technology=EXTERNAL_TECH, node_loc=base["node_loc"], year_vtg=base["year_act"],
                  year_act=base["year_act"], mode="M1")
    sub_time = pd.Series(context.time)

    results = {
        "output": make_df("output", **common, node_dest=base["node_loc"], commodity="electr",
                          level="final", value=1.0, unit="GWa")
        .pipe(broadcast, time=sub_time).pipe(same_time),
    }
    prices = external_electricity_prices(context, nodes, years)
    if prices is not None:
        value = base.merge(prices, how="left")["value"]
        results["var_cost"] = make_df("var_cost", **common, value=value, unit="USD/GWa").pipe(
            broadcast, time=sub_time)
        log.info(f"{EXTERNAL_TECH}: {len(nodes)} external nodes, var_cost (M USD/GWa, first model year) "
                 + ", ".join(f"{n} {v:.0f}" for n, v in
                             prices[prices["year_act"] == info.y0].set_index("node_loc")["value"].items()))
    else:
        log.info(f"{EXTERNAL_TECH}: {len(nodes)} external nodes, free")
    return results
