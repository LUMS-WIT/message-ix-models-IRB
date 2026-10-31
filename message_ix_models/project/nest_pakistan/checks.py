"""Input and assumption checks on a built (not necessarily solved) scenario.

Each check returns a :class:`Check`: a status (PASS, WARN, FAIL, INFO), a
one-line finding and an optional table. ``mix-models nest-pak check`` runs
them all and writes a PDF and CSVs. The checks encode problems found while
building the Indus model, so they are caught automatically next time:

- inputs drawn from a node where nothing supplies that commodity (the drip
  and sprinkler methods once took electricity at basin nodes);
- growth limits on technologies with no starting activity (they can never
  run: ``extract_surfacewater``);
- outputs nothing uses (crop yields with no demand);
- river links that create or lose water.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from .basin_data import status as data_status
from .calibration import consistency_check

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    status: str
    finding: str
    table: pd.DataFrame = field(default_factory=pd.DataFrame)


def check_data_files(cfg) -> Check:
    s = data_status(cfg["basin"]["code"], cfg["nexus"]["rcp"], cfg["nexus"]["rel"])
    missing = s[~s["present"]]
    req = missing[missing["need"] == "required"]
    st = "FAIL" if len(req) else ("WARN" if len(missing) else "PASS")
    return Check("Input data files", st, f"{len(s) - len(missing)} of {len(s)} input files present for basin "
                 f"{cfg['basin']['code']}; {len(req)} required file(s) missing.",
                 s[["step", "file", "need", "present", "global data"]])


def check_consistency(scen) -> Check:
    issues = consistency_check(scen)
    # Only model years are optimised: bounds in historical years cannot make it infeasible.
    first = scen.firstmodelyear
    unreachable = [r for r in issues["activity_lo_infeasible"] if r["year_act"] >= first]
    t = pd.DataFrame([{"check": k, "issues": len(v)} for k, v in issues.items()])
    t["note"] = t["check"].map({
        "bound_conflicts": "lower bound above upper bound: infeasible",
        "activity_lo_infeasible": f"activity lower bound above capacity; {len(unreachable)} in model years "
                                  f"(>= {first}), the rest historical and not optimised",
        "orphan_inputs": "heuristic, see 'Every input has a supplier'"}).fillna("")
    blocking = len(issues["bound_conflicts"]) + len(unreachable)
    st = "FAIL" if blocking else ("WARN" if t["issues"].sum() else "PASS")
    return Check("Pre-solve consistency", st,
                 f"{len(issues['bound_conflicts'])} bound conflict(s) and {len(unreachable)} unreachable activity "
                 f"bound(s) in model years (these make the model infeasible); {t['issues'].sum()} issue(s) in total.", t)


def check_supplied_inputs(scen) -> Check:
    """Every (node, commodity, level) a technology takes from must have a
    producer there, a balance it can draw from, or be a resource level."""
    inp = scen.par("input")[["technology", "node_origin", "commodity", "level"]].drop_duplicates()
    out = scen.par("output")[["node_dest", "commodity", "level"]].drop_duplicates()
    out = out.rename(columns={"node_dest": "node_origin"}).assign(supplied=True)
    dem = scen.par("demand")
    # Fixed supplies are negative demand; zero rows mean "none here", which is
    # data too (e.g. no wastewater in a basin), not a missing link.
    neg = dem[dem["value"] <= 0][["node", "commodity", "level"]].drop_duplicates().rename(columns={"node": "node_origin"})
    # Resource, renewable and stock levels are supplied by resource parameters
    # (resource_volume, renewable potentials), not by a technology's output.
    exempt = (set(scen.set("level_resource")) | set(scen.set("level_renewable")) | set(scen.set("level_storage"))
              | {"resource", "renewable", "stocks"})
    m = inp[~inp["level"].isin(exempt)].merge(out, how="left").merge(neg.assign(data=True), how="left")
    bad = m[m["supplied"].isna() & m["data"].isna()]
    by = bad.groupby(["node_origin", "commodity", "level"])["technology"].agg(
        lambda t: f"{len(set(t))}: " + ", ".join(sorted(set(t))[:4]) + (" ..." if len(set(t)) > 4 else ""))
    t = by.reset_index().rename(columns={"technology": "technologies that cannot run"})
    # Technologies with nothing to draw on just stay at zero: the model still
    # solves, so this is a warning to read, not a failure.
    return Check("Every input has a supplier", "WARN" if len(t) else "PASS",
                 f"{len(t)} (node, commodity, level) input(s) with no producer: those technologies stay at zero. "
                 "Check that none of them should run.", t)


def check_growth_without_start(scen) -> Check:
    g = scen.par("growth_activity_up")[["node_loc", "technology"]].drop_duplicates()
    if g.empty:
        return Check("Growth limits with a starting point", "PASS", "No growth_activity_up constraints.")
    start = pd.concat([scen.par(p)[["node_loc", "technology"]] for p in
                       ("initial_activity_up", "historical_activity", "historical_new_capacity")]).drop_duplicates()
    m = g.merge(start.assign(has=True), how="left")
    bad = m[m["has"].isna()].groupby("technology")["node_loc"].agg(lambda n: f"{len(n)} node(s)").reset_index()
    return Check("Growth limits with a starting point", "WARN" if len(bad) else "PASS",
                 f"{len(bad)} technolog(ies) have a growth limit but no initial activity or history, "
                 "so their activity can never leave zero.", bad)


def check_dead_ends(scen) -> Check:
    out = scen.par("output")[["technology", "commodity", "level"]].drop_duplicates()
    used = scen.par("input")[["commodity", "level"]].drop_duplicates().assign(used=True)
    dem = scen.par("demand")[["commodity", "level"]].drop_duplicates().assign(demand=True)
    eq = pd.DataFrame(scen.set("balance_equality"))
    m = out.merge(used, how="left").merge(dem, how="left")
    dead = m[m["used"].isna() & m["demand"].isna()]
    dead = dead[~dead["level"].isin(["share", "dummy_supply", "dummy_end", "dummy_emission", "dummy"])]
    t = dead.groupby(["commodity", "level"])["technology"].agg(lambda x: len(set(x))).reset_index(
        name="producing technologies")
    t["note"] = "nothing uses it; it carries no value in the model"
    st = "WARN" if len(t) else "PASS"
    return Check("Outputs that something uses", st,
                 f"{len(t)} commodity/level pair(s) are produced but never used or demanded "
                 f"({len(eq)} pools must balance exactly).", t)


def check_river_links(scen) -> Check:
    y = scen.firstmodelyear
    inp = scen.par("input", filters={"year_act": [y]})
    out = scen.par("output", filters={"year_act": [y]})
    i = inp[inp["technology"].str.startswith("river|")]["value"]
    o = out[out["technology"].str.startswith("river|")]["value"]
    n = inp[inp["technology"].str.startswith("river|")]["technology"].nunique()
    ok = len(i) and (i == 1).all() and (o <= 1 + 1e-9).all()
    return Check("River links conserve water", "PASS" if ok else "FAIL",
                 f"{n} river links; input coefficient {i.min() if len(i) else '-'}-{i.max() if len(i) else '-'}, "
                 f"output {o.min() if len(o) else '-'}-{o.max() if len(o) else '-'} (1 = no water created or lost).")


def check_natural_water(scen, cfg) -> Check:
    years = cfg["report"]["years"]
    d = scen.par("demand", filters={"commodity": ["surfacewater_basin", "groundwater_basin"], "year": years})
    t = (-d.pivot_table(index="node", columns=["commodity"], values="value", aggfunc="mean")).round(1)
    t.columns = [f"mean {c} (MCM/yr)" for c in t.columns]
    floor = scen.par("relation_lower", filters={"year_rel": years})
    floor = floor[floor["relation"].str.startswith("hist_irr_withdrawal_")]
    if not floor.empty:
        t = t.join(floor.groupby("node_rel")["value"].mean().rename("irrigation floor (MCM/yr)"))
    neg = (t.filter(like="mean") < 0).any(axis=1).sum()
    return Check("Natural water per basin", "FAIL" if neg else "INFO",
                 f"{len(t)} basin nodes with runoff/recharge data; {neg} with negative values. Compare the floor with "
                 "local water: inflow from upstream comes on top.", t.reset_index())


def external_price_rows(ctx: dict) -> list[tuple]:
    """Each external electricity price as entered and in the model's unit,
    so the unit conversion can be read in the checks report."""
    from message_ix_models.model.water.data.external_electricity import to_model_cost

    if not ctx.get("external_electricity", True):
        return [("external electricity", "off")]
    cost = ctx.get("external_electricity_cost")
    if cost is None:
        return [("external electricity price", "free")]
    unit = ctx.get("external_electricity_cost_unit") or "M USD/GWa"
    cy, my = ctx.get("external_electricity_currency_year"), ctx.get("model_currency_year")
    rows = []
    for region, p in (cost if isinstance(cost, dict) else {"all regions": cost}).items():
        path = p if isinstance(p, dict) else {"all years": p}
        txt = "; ".join(f"{y}: {v} {unit} ({cy or ''}) = {to_model_cost(v, unit, cy, my):.0f} M USD"
                        f"{f'({my})' if my else ''}/GWa" for y, v in path.items())
        rows.append((f"external electricity price: {region}", txt))
    return rows


def check_assumptions(cfg) -> Check:
    rows = [("basin", f"{cfg['basin']['name']} ({cfg['basin']['code']})"),
            ("national energy node(s)", str(cfg["basin"].get("region_node_map")))]
    n = cfg["nexus"]
    rows += [(k, n[k]) for k in ("ssp", "rcp", "rel", "sdg", "time")]
    ctx = n.get("context") or {}
    rows += [(f"water switch: {k}", v) for k, v in ctx.items() if k != "external_electricity_cost"]
    rows += external_price_rows(ctx)
    rows += [("energy relaxations", sum(len(v) for v in cfg["energy"].get("relax", {}).values())),
             ("calibration fixes from config", len(cfg["calibration"].get("bound_activity_lo", []))),
             ("solver", str(cfg["solve"].get("gams_args")))]
    return Check("Assumptions of this run", "INFO", "Settings from config.yaml and --set overrides.",
                 pd.DataFrame(rows, columns=["setting", "value"]).astype(str))


def run_checks(scen, cfg) -> list[Check]:
    checks = [check_assumptions(cfg), check_data_files(cfg)]
    if scen is not None:
        for f in (check_consistency, check_supplied_inputs, check_growth_without_start, check_dead_ends,
                  check_river_links):
            try:
                checks.append(f(scen))
            except Exception as e:  # noqa: BLE001
                checks.append(Check(f.__name__, "WARN", f"check could not run: {e}"))
        checks.append(check_natural_water(scen, cfg))
    return checks


def summary(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame([{"check": c.name, "status": c.status, "finding": c.finding} for c in checks])
