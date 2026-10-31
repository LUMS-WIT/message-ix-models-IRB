"""Extract every reported result from a solved scenario into one tidy table.

:func:`extract` returns columns ``scenario, node, year, theme, indicator,
unit, value``, one row per basin (or region) node. :func:`aggregate` sums it
to the region groups of ``config.yaml`` (Pakistan, India, China, Afghanistan,
upper basins, Indus). The PDF reports, CSV exports and scenario comparison
all read this one table, so every number in them comes from the same place.

Themes: ``water``, ``energy-for-water``, ``water-for-energy``, ``energy``,
``land``, ``investment``, ``costs``, ``emissions``.
"""

import logging
import re

import numpy as np
import pandas as pd

from ..config import national_node

log = logging.getLogger(__name__)

ACT_KEYS = ["node_loc", "technology", "year_vtg", "year_act", "mode", "time"]
COLUMNS = ["scenario", "node", "year", "theme", "indicator", "unit", "value"]
CROPS = "cotton|fodder|fruit|maize|pulses|rice|sugarcane|vegetables|wheat"

# ------------------------------------------------------------ classification
IRRIGATION_SUPPLY = {"irrigation_sw_diversion", "smart_irrigation_sw_diversion", "irrigation_gw_diversion",
                     "smart_irrigation_gw_diversion", "urban_wastewater_irrigation", "irrigation_desal_seawater"}
CITY_SUPPLY = {"extract_surfacewater", "extract_groundwater", "extract_gw_fossil", "extract_salinewater_basin",
               "distillation", "membrane", "urban_recycle", "rural_recycle"}
DISTRIBUTION = {"urban_t_d", "rural_t_d", "urban_unconnected", "rural_unconnected", "industry_unconnected"}
WASTEWATER = {"urban_sewerage", "rural_sewerage", "urban_discharge", "rural_discharge", "urban_untreated",
              "rural_untreated", "industry_untreated"}
WATER_LINKS = {"basin_to_reg_core", "basin_to_reg_plus", "reg_to_basin", "return_flow", "gw_recharge",
               "extract_salinewater_cool"}
EXTERNAL_ELECTRICITY = "electr_supply_water_ext"

#: Water withdrawn at the source: technology -> (source, use, measured on).
WITHDRAWALS = {
    "irrigation_sw_diversion": ("River", "Irrigation", "input"),
    "smart_irrigation_sw_diversion": ("River", "Irrigation", "input"),
    "irrigation_gw_diversion": ("Groundwater", "Irrigation", "act"),
    "smart_irrigation_gw_diversion": ("Groundwater", "Irrigation", "act"),
    "irrigation_desal_seawater": ("Desalination", "Irrigation", "output"),
    "urban_wastewater_irrigation": ("Recycled wastewater", "Irrigation", "output"),
    "extract_surfacewater": ("River", "Cities & industry", "act"),
    "extract_groundwater": ("Groundwater", "Cities & industry", "act"),
    "extract_gw_fossil": ("Fossil groundwater", "Cities & industry", "act"),
    "distillation": ("Desalination", "Cities & industry", "output"),
    "membrane": ("Desalination", "Cities & industry", "output"),
    "urban_recycle": ("Recycled wastewater", "Cities & industry", "output"),
    "rural_recycle": ("Recycled wastewater", "Cities & industry", "output"),
    "basin_to_reg_core": ("River", "Energy sector", "act"),
}
CITY_DEMANDS = {"urban_mw": "Cities: piped", "urban_disconnected": "Cities: not connected",
                "industry_mw": "Industry", "rural_mw": "Villages: piped", "rural_disconnected": "Villages: not connected"}
PRICES = {"surfacewater_basin": "river water", "groundwater_basin": "groundwater",
          "freshwater_basin": "city & industry water", "irrigation_water": "farm-gate water"}
COOLING = {"cl_fresh": "closed loop, fresh", "ot_fresh": "once-through, fresh", "ot_saline": "once-through, saline",
           "air": "air"}
FUELS = [("coal", "Coal"), ("lignite", "Coal"), ("igcc", "Coal"), ("gas", "Gas"), ("foil", "Oil"), ("loil", "Oil"),
         ("oil", "Oil"), ("nuc", "Nuclear"), ("hydro", "Hydro"), ("solar", "Solar"), ("csp", "Solar"),
         ("wind", "Wind"), ("bio", "Biomass"), ("geo", "Geothermal")]

#: Indicators that are not sums over nodes (left out of :func:`aggregate`,
#: except prices, which are averaged).
NOT_ADDITIVE = {"Inflow from upstream basins", "Outflow to downstream basins"}


def sector(tech: str) -> str:
    t = str(tech)
    if t.startswith(("irr_", "rainfed_", "crop_")):
        return "Crops & land"
    if t in IRRIGATION_SUPPLY:
        return "Irrigation supply"
    if t in CITY_SUPPLY:
        return "Water supply (cities & industry)"
    if t in DISTRIBUTION:
        return "Water distribution"
    if t in WASTEWATER:
        return "Wastewater"
    if "__" in t and t.split("__")[1] in COOLING:
        return "Power plant cooling"
    if t.startswith(("river|", "canal_conv|")) or t in WATER_LINKS:
        return "Water transfers"
    if t.startswith("reservoir_"):
        return "Water storage"
    if t == EXTERNAL_ELECTRICITY:
        return "External electricity"
    return "Energy system"


def fuel(tech: str) -> str:
    t = str(tech).lower()
    for prefix, name in FUELS:
        if t.startswith(prefix):
            return name
    return "Other"


def irrigation_method(tech: str) -> str | None:
    m = re.match(rf"^(irr_.+|rainfed)_({CROPS})$", str(tech))
    if not m:
        return None
    return m.group(1).replace("irr_", "").replace("_", " ") if m.group(1) != "rainfed" else "rainfed"


# ------------------------------------------------------------------ helpers
def _var(scen, name, filters=None) -> pd.DataFrame:
    try:
        df = scen.var(name, filters=filters) if filters else scen.var(name)
    except Exception:  # noqa: BLE001  (variable missing from this solution)
        return pd.DataFrame()
    if isinstance(df, dict):
        return pd.DataFrame([df])
    if "lvl" in df:
        df = df.assign(lvl=pd.to_numeric(df["lvl"], errors="coerce").fillna(0.0))
    return df


def _rows(df: pd.DataFrame, node, year, theme, indicator, unit, value="value") -> pd.DataFrame:
    """Tidy rows from ``df``; ``node``/``year``/``indicator`` are column names
    or constants."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS[1:])
    get = lambda x: df[x] if isinstance(x, str) and x in df else x  # noqa: E731
    out = pd.DataFrame({"node": get(node), "year": get(year), "theme": theme, "indicator": get(indicator),
                        "unit": unit, "value": df[value].astype(float)})
    return out.groupby(["node", "year", "theme", "indicator", "unit"], as_index=False)["value"].sum()


class Flows:
    """ACT x input/output coefficients for the reporting years, computed once."""

    def __init__(self, scen, years):
        self.scen, self.years = scen, list(years)
        self.act = _var(scen, "ACT", {"year_act": self.years})
        self._cache = {}

    def get(self, par: str) -> pd.DataFrame:
        if par not in self._cache:
            df = self.scen.par(par, filters={"year_act": self.years})
            m = df.merge(self.act[ACT_KEYS + ["lvl"]], on=ACT_KEYS, how="inner")
            self._cache[par] = m.assign(flow=m["value"] * m["lvl"])
        return self._cache[par]


# --------------------------------------------------------------------- themes
def water(scen, f: Flows) -> pd.DataFrame:
    years, parts = f.years, []
    dem = scen.par("demand", filters={"year": years})
    nat = dem[dem["commodity"].isin(["surfacewater_basin", "groundwater_basin"])].copy()
    nat["indicator"] = nat["commodity"].map({"surfacewater_basin": "Natural runoff",
                                             "groundwater_basin": "Natural recharge"})
    parts.append(_rows(nat.assign(value=-nat["value"]), "node", "year", "water", "indicator", "MCM/yr"))
    city = dem[dem["commodity"].isin(CITY_DEMANDS)].copy()
    city["indicator"] = "Demand: " + city["commodity"].map(CITY_DEMANDS)
    parts.append(_rows(city, "node", "year", "water", "indicator", "MCM/yr"))

    out, inp, act = f.get("output"), f.get("input"), f.act
    river = out[out["technology"].str.startswith("river|") & (out["commodity"] == "surfacewater_basin")]
    parts.append(_rows(river, "node_dest", "year_act", "water", "Inflow from upstream basins", "MCM/yr", "flow"))
    rin = inp[inp["technology"].str.startswith("river|") & (inp["commodity"] == "surfacewater_basin")]
    parts.append(_rows(rin, "node_origin", "year_act", "water", "Outflow to downstream basins", "MCM/yr", "flow"))

    # Withdrawals by source and by use.
    rows = []
    for tech, (source, use, on) in WITHDRAWALS.items():
        if on == "act":
            d = act[act["technology"] == tech].assign(flow=lambda x: x["lvl"])
            if tech.endswith("gw_diversion"):
                d = d.assign(source=np.where(d["mode"] == "M2", "Fossil groundwater", "Groundwater"))
        else:
            src = out if on == "output" else inp
            d = src[(src["technology"] == tech) & src["commodity"].isin(
                ["surfacewater_basin", "freshwater_basin", "irrigation_water"])]
        if d.empty:
            continue
        d = d.assign(source=d.get("source", source), use=use, year=d["year_act"])
        rows.append(d[["node_loc", "year", "source", "use", "flow"]])
    if rows:
        w = pd.concat(rows)
        w["source"] = w["source"].replace({"Groundwater": "Renewable groundwater"})
        parts.append(_rows(w.assign(indicator="Withdrawal: " + w["source"]), "node_loc", "year", "water",
                           "indicator", "MCM/yr", "flow"))
        parts.append(_rows(w.assign(indicator="Withdrawal for: " + w["use"]), "node_loc", "year", "water",
                           "indicator", "MCM/yr", "flow"))

    for tech, label in (("return_flow", "Unused river water leaving"), ("gw_recharge", "Unused groundwater leaving")):
        parts.append(_rows(act[act["technology"] == tech], "node_loc", "year_act", "water", label, "MCM/yr", "lvl"))

    floor = irrigation_floor(scen, f)
    if not floor.empty:
        for col, label in (("floor", "Irrigation demand (historical withdrawals)"),
                           ("delivered", "Irrigation demand met"), ("unmet", "Irrigation demand unmet")):
            parts.append(_rows(floor, "node", "year", "water", label, "MCM/yr", col))

    price = _var(scen, "PRICE_COMMODITY", {"commodity": list(PRICES), "year": years})
    if not price.empty:
        price = price.assign(indicator="Shadow price: " + price["commodity"].map(PRICES))
        parts.append(_rows(price, "node", "year", "water", "indicator", "M USD/MCM", "lvl").assign(
            value=lambda d: d["value"]))
    return pd.concat(parts, ignore_index=True)


def irrigation_floor(scen, f: Flows) -> pd.DataFrame:
    """Per basin and year: historical-withdrawal floor, water counted toward
    it, and the shortfall (soft bound slack)."""
    rels = [r for r in scen.set("relation") if str(r).startswith("hist_irr_withdrawal_")]
    if not rels:
        return pd.DataFrame()
    lower = scen.par("relation_lower", filters={"relation": rels, "year_rel": f.years})
    ra = scen.par("relation_activity", filters={"relation": rels, "year_rel": f.years})
    act = f.act.groupby(["node_loc", "technology", "year_act", "mode"], as_index=False)["lvl"].sum()
    d = ra.merge(act, on=["node_loc", "technology", "year_act", "mode"])
    d = d[d["year_rel"] == d["year_act"]].assign(delivered=lambda x: x["value"] * x["lvl"])
    d = d.groupby(["relation", "node_rel", "year_rel"], as_index=False)["delivered"].sum()
    out = lower.rename(columns={"value": "floor"})[["relation", "node_rel", "year_rel", "floor"]].merge(
        d, how="left").fillna({"delivered": 0.0})
    out["unmet"] = (out["floor"] - out["delivered"]).clip(lower=0.0)
    return out.rename(columns={"node_rel": "node", "year_rel": "year"})


def energy_for_water(scen, f: Flows, national: str) -> pd.DataFrame:
    inp = f.get("input")
    e = inp[(inp["commodity"] == "electr")].copy()
    e["sector"] = e["technology"].map(sector)
    e = e[~e["sector"].isin(["Energy system", "External electricity"])]
    parts = [_rows(e.assign(indicator="Electricity for " + e["sector"]), "node_loc", "year_act",
                   "energy-for-water", "indicator", "GWa", "flow")]
    # The origin split is the water sector's total; plant cooling is the
    # energy system's own use and stays a separate indicator.
    e = e[e["sector"] != "Power plant cooling"].copy()
    e["origin"] = np.where(e["node_origin"] == national, f"national energy system ({national})",
                           "external supply (no energy model)")
    parts.append(_rows(e.assign(indicator="Electricity for water from " + e["origin"]), "node_loc", "year_act",
                       "energy-for-water", "indicator", "GWa", "flow"))
    ext = f.act[f.act["technology"] == EXTERNAL_ELECTRICITY]
    parts.append(_rows(ext, "node_loc", "year_act", "energy-for-water",
                       "Additional electricity needed (regions without energy model)", "GWa", "lvl"))
    vc = scen.par("var_cost", filters={"technology": [EXTERNAL_ELECTRICITY], "year_act": f.years})
    if not vc.empty:
        vc = vc.groupby(["node_loc", "year_act"], as_index=False)["value"].mean()
        parts.append(_rows(vc, "node_loc", "year_act", "energy-for-water",
                           "External electricity price (regions without energy model)", "M USD/GWa"))
        bill = ext.groupby(["node_loc", "year_act"], as_index=False)["lvl"].sum().merge(vc)
        bill["value"] = bill["lvl"] * bill["value"]
        parts.append(_rows(bill, "node_loc", "year_act", "energy-for-water",
                           "Cost of additional electricity (regions without energy model)", "M USD/yr"))
    return pd.concat(parts, ignore_index=True)


def water_for_energy(scen, f: Flows, national: str) -> pd.DataFrame:
    inp, out, parts = f.get("input"), f.get("output"), []
    b2r = inp[inp["technology"].str.startswith("basin_to_reg") & inp["commodity"].isin(
        ["surfacewater_basin", "freshwater_basin"])]
    parts.append(_rows(b2r, "node_loc", "year_act", "water-for-energy", "Basin water sent to power plants",
                       "MCM/yr", "flow"))
    w = inp[inp["commodity"].isin(["surfacewater", "saline_ppl"]) & (inp["level"].isin(["water_supply", "saline_supply"]))]
    cool = w[w["technology"].str.contains("__")].copy()
    cool["indicator"] = "Cooling water: " + cool["technology"].str.split("__").str[1].map(COOLING).fillna("other")
    parts.append(_rows(cool, "node_loc", "year_act", "water-for-energy", "indicator", "MCM/yr", "flow"))
    other = w[~w["technology"].str.contains("__")]
    parts.append(_rows(other.assign(indicator="Process water: " + other["technology"].map(fuel)), "node_loc",
                       "year_act", "water-for-energy", "indicator", "MCM/yr", "flow"))
    ret = out[(out["technology"] == "reg_to_basin") & (out["commodity"] == "surfacewater_basin")]
    parts.append(_rows(ret, "node_dest", "year_act", "water-for-energy", "Power plant water returned to rivers",
                       "MCM/yr", "flow"))
    return pd.concat(parts, ignore_index=True)


def energy(scen, f: Flows, national: str) -> pd.DataFrame:
    out, parts = f.get("output"), []
    gen = out[(out["commodity"] == "electr") & (out["level"] == "secondary") & (out["node_loc"] == national)]
    gen = gen[gen["technology"].map(sector) == "Energy system"]
    parts.append(_rows(gen.assign(indicator="Electricity generation: " + gen["technology"].map(fuel)),
                       "node_loc", "year_act", "energy", "indicator", "GWa", "flow"))
    cap = _var(scen, "CAP", {"node_loc": [national], "year_act": f.years})
    if not cap.empty:
        pp = cap[cap["technology"].isin(set(gen["technology"]))]
        parts.append(_rows(pp.assign(indicator="Power capacity: " + pp["technology"].map(fuel)), "node_loc",
                           "year_act", "energy", "indicator", "GW", "lvl"))
    dem = scen.par("demand", filters={"node": [national], "year": f.years})
    parts.append(_rows(dem.assign(indicator="Useful energy demand: " + dem["commodity"]), "node", "year",
                       "energy", "indicator", "GWa"))
    price = _var(scen, "PRICE_COMMODITY", {"node": [national], "commodity": ["electr"], "level": ["final"],
                                            "year": f.years})
    parts.append(_rows(price, "node", "year", "energy", "Electricity price (final)", "M USD/GWa", "lvl"))
    return pd.concat(parts, ignore_index=True)


def land(scen, f: Flows) -> pd.DataFrame:
    a = f.act.copy()
    a["method"] = a["technology"].map(irrigation_method)
    a = a.dropna(subset=["method"])
    a["crop"] = a["technology"].str.extract(rf"_({CROPS})$")[0]
    return pd.concat([
        _rows(a.assign(indicator="Crop area: " + a["method"]), "node_loc", "year_act", "land", "indicator",
              "Mha", "lvl"),
        _rows(a.assign(indicator="Crop area by crop: " + a["crop"]), "node_loc", "year_act", "land",
              "indicator", "Mha", "lvl"),
    ], ignore_index=True)


def investment_and_costs(scen, f: Flows) -> pd.DataFrame:
    parts = []
    capn = _var(scen, "CAP_NEW", {"year_vtg": f.years})
    inv = scen.par("inv_cost", filters={"year_vtg": f.years})
    if not capn.empty and not inv.empty:
        m = capn.merge(inv, on=["node_loc", "technology", "year_vtg"])
        m = m.assign(cost=m["lvl"] * m["value"], indicator="Investment: " + m["technology"].map(sector))
        parts.append(_rows(m, "node_loc", "year_vtg", "investment", "indicator", "M USD/yr", "cost"))
        pp = capn[capn["technology"].map(sector) == "Energy system"]
        parts.append(_rows(pp.assign(indicator="New capacity: " + pp["technology"].map(fuel)), "node_loc",
                           "year_vtg", "investment", "indicator", "GW/yr", "lvl"))
    cap = _var(scen, "CAP", {"year_act": f.years})
    fix = scen.par("fix_cost", filters={"year_act": f.years})
    if not cap.empty and not fix.empty:
        m = cap.merge(fix, on=["node_loc", "technology", "year_vtg", "year_act"])
        m = m.assign(cost=m["lvl"] * m["value"], indicator="Fixed O&M: " + m["technology"].map(sector))
        parts.append(_rows(m, "node_loc", "year_act", "costs", "indicator", "M USD/yr", "cost"))
    var = scen.par("var_cost", filters={"year_act": f.years})
    if not var.empty:
        m = var.merge(f.act[ACT_KEYS + ["lvl"]], on=ACT_KEYS)
        m = m.assign(cost=m["lvl"] * m["value"], indicator="Variable cost: " + m["technology"].map(sector))
        parts.append(_rows(m, "node_loc", "year_act", "costs", "indicator", "M USD/yr", "cost"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS[1:])


def emissions(scen, f: Flows) -> pd.DataFrame:
    em = _var(scen, "EMISS", {"year": f.years})
    if em.empty:
        return pd.DataFrame(columns=COLUMNS[1:])
    em = em[em["type_tec"] == "all"] if "all" in set(em["type_tec"]) else em
    return _rows(em.assign(indicator="Emissions: " + em["emission"]), "node", "year", "emissions", "indicator",
                 "model units", "lvl")


def extract(scen, cfg: dict) -> pd.DataFrame:
    """Every reported indicator for ``scen`` (one row per node and year)."""
    years = cfg["report"]["years"]
    national = national_node(cfg)
    f = Flows(scen, years)
    parts = [water(scen, f), energy_for_water(scen, f, national), water_for_energy(scen, f, national),
             energy(scen, f, national), land(scen, f), investment_and_costs(scen, f), emissions(scen, f)]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df = df[df["value"].abs() > 1e-9]
    df.insert(0, "scenario", scen.scenario)
    df["year"] = df["year"].astype(int)
    return df[COLUMNS].reset_index(drop=True)


def objective(scen) -> float:
    obj = _var(scen, "OBJ")
    return float(obj["lvl"].iloc[0]) if not obj.empty else float("nan")


# ---------------------------------------------------------------- aggregation
def node_group(node: str, groups: dict) -> str | None:
    for name, keys in groups.items():
        if any(k in str(node) for k in keys):
            return name
    return None


def aggregate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Sum ``df`` (from :func:`extract`) to the region groups of
    ``report.groups``, plus "Upper basins" (every group but the first, the
    national one) and the whole basin (``basin.name``). Prices are averaged;
    basin-to-basin flows are left out (they cancel within groups)."""
    groups = cfg["report"]["groups"]
    d = df[~df["indicator"].isin(NOT_ADDITIVE)].copy()
    d["region"] = d["node"].map(lambda n: node_group(n, groups))
    d = d.dropna(subset=["region"]).reset_index(drop=True)
    keys = ["scenario", "region", "year", "theme", "indicator", "unit"]
    d["is_price"] = d["indicator"].str.contains("price")

    def agg(sub, region=None):
        s = sub if region is None else sub.assign(region=region)
        a = s[~s["is_price"]].groupby(keys, as_index=False)["value"].sum()
        p = s[s["is_price"]].groupby(keys, as_index=False)["value"].mean()
        return pd.concat([a, p], ignore_index=True)

    home = next(iter(groups))
    out = [agg(d), agg(d[d["region"] != home], "Upper basins"), agg(d, whole_basin(cfg))]
    return pd.concat(out, ignore_index=True)


def whole_basin(cfg: dict) -> str:
    return cfg["basin"].get("name", "Whole basin")


def region_order(cfg: dict) -> list[str]:
    """Regions in reporting order: groups, upper basins, whole basin."""
    return list(cfg["report"]["groups"]) + ["Upper basins", whole_basin(cfg)]


def river_balance(scen, year: int) -> pd.DataFrame:
    """Whole-system river water balance in ``year`` (MCM): natural runoff
    plus everything flowing into river pools vs. everything flowing out.
    River links appear on both sides, so a gap would mean water is created
    or lost. This is the double-counting check."""
    f = Flows(scen, [year])
    dem = scen.par("demand", filters={"commodity": ["surfacewater_basin"], "year": [year]})
    rows = [("in", "Natural runoff (all basins)", -dem["value"].sum())]
    for side, par in (("in", "output"), ("out", "input")):
        d = f.get(par)
        d = d[d["commodity"] == "surfacewater_basin"]
        g = d.groupby(d["technology"].map(lambda t: "River links" if str(t).startswith("river|") else t))["flow"].sum()
        rows += [(side, k, v) for k, v in g.items() if abs(v) > 1e-6]
    return pd.DataFrame(rows, columns=["side", "item", "MCM"])
