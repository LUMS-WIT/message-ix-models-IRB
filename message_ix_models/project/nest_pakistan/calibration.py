"""Pre-solve consistency check and the calibration fixes it drives.

Ported from NEST-Pakistan's ``water_integration1_model.py`` (which stays as
the legacy notebook path). The evidence-derived values (bound conflicts in
the LTS energy data) now come from ``config.yaml`` ``calibration`` instead of
code, and every change is recorded in a log returned to the caller and
written next to the scenario's reports.
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

ACT_BOUND_KEYS = ["node_loc", "technology", "year_act", "mode", "time"]


def consistency_check(scen) -> dict[str, list[dict]]:
    """Pure data analysis, no GAMS call. Finds what makes a scenario
    infeasible by construction or leaves technologies unusable."""
    issues = {k: [] for k in ("bound_conflicts", "activity_lo_infeasible", "missing_lifetime",
                              "missing_cap_factor", "orphan_inputs")}

    def check_bounds(lo_par, up_par, keys):
        lo, up = scen.par(lo_par), scen.par(up_par)
        if lo.empty or up.empty:
            return
        m = lo.merge(up, on=keys, suffixes=("_lo", "_up"))
        for _, r in m[m["value_lo"] > m["value_up"] + 1e-9].iterrows():
            issues["bound_conflicts"].append({"pair": (lo_par, up_par), **{k: r[k] for k in keys},
                                              "lo": r["value_lo"], "up": r["value_up"]})

    check_bounds("bound_activity_lo", "bound_activity_up", ACT_BOUND_KEYS)
    check_bounds("bound_new_capacity_lo", "bound_new_capacity_up", ["node_loc", "technology", "year_vtg"])
    check_bounds("bound_total_capacity_lo", "bound_total_capacity_up", ["node_loc", "technology", "year_act"])

    # bound_activity_lo larger than what available capacity can deliver.
    lifetime = scen.par("technical_lifetime")
    hist_cap = scen.par("historical_new_capacity")
    new_cap_up = scen.par("bound_new_capacity_up")
    tot_cap_up = scen.par("bound_total_capacity_up")
    capfac = scen.par("capacity_factor")
    bal = scen.par("bound_activity_lo")
    for _, row in bal[bal["value"] > 1e-9].iterrows():
        node, tec, ya = row["node_loc"], row["technology"], row["year_act"]
        lt = lifetime[(lifetime["node_loc"] == node) & (lifetime["technology"] == tec)]
        if lt.empty:
            continue
        alive = lt[(lt["year_vtg"] <= ya) & (lt["year_vtg"] + lt["value"] > ya)]["year_vtg"].tolist()
        cap = 0.0
        if alive:
            sel = lambda df: df[(df["node_loc"] == node) & (df["technology"] == tec)  # noqa: E731
                                & df["year_vtg"].isin(alive)]["value"].sum()
            cap = sel(hist_cap) + sel(new_cap_up)
        ceiling = tot_cap_up[(tot_cap_up["node_loc"] == node) & (tot_cap_up["technology"] == tec)
                             & (tot_cap_up["year_act"] == ya)]
        if not ceiling.empty:
            cap = min(cap, ceiling["value"].sum())
        cf = capfac[(capfac["node_loc"] == node) & (capfac["technology"] == tec) & (capfac["year_act"] == ya)]
        cf_val = cf["value"].iloc[0] if not cf.empty else 1.0
        if cap * cf_val < row["value"] - 1e-6:
            issues["activity_lo_infeasible"].append({
                "node_loc": node, "technology": tec, "year_act": ya, "bound_activity_lo": row["value"],
                "max_reachable_activity": cap * cf_val, "available_capacity": cap, "capacity_factor_used": cf_val})

    # Investment technologies with output but no technical_lifetime.
    inv_tec = set(scen.par("inv_cost")["technology"].unique())
    out = scen.par("output")[["node_loc", "technology", "year_vtg", "year_act"]].drop_duplicates()
    m = out[out["technology"].isin(inv_tec)][["node_loc", "technology", "year_vtg"]].drop_duplicates().merge(
        lifetime[["node_loc", "technology", "year_vtg"]].drop_duplicates(), how="left", indicator=True)
    for r in m[m["_merge"] == "left_only"].itertuples():
        issues["missing_lifetime"].append({"node_loc": r.node_loc, "technology": r.technology, "year_vtg": r.year_vtg})

    # Output-bearing technologies with no capacity_factor.
    m = out.merge(capfac[["node_loc", "technology", "year_vtg", "year_act"]].drop_duplicates(),
                  how="left", indicator=True)
    for r in m[m["_merge"] == "left_only"].itertuples():
        issues["missing_cap_factor"].append({"node_loc": r.node_loc, "technology": r.technology,
                                             "year_vtg": r.year_vtg, "year_act": r.year_act})

    # Inputs nothing produces (heuristic: resource/renewable/storage levels exempt).
    exempt = set(scen.set("level_resource")) | set(scen.set("level_renewable")) | set(scen.set("level_storage"))
    inp = scen.par("input")[["node_origin", "commodity", "level"]].drop_duplicates().rename(columns={"node_origin": "node"})
    outp = scen.par("output")[["node_dest", "commodity", "level"]].drop_duplicates().rename(columns={"node_dest": "node"})
    m = inp[~inp["level"].isin(exempt)].merge(outp.assign(has=True), how="left")
    issues["orphan_inputs"] = m[m["has"].isna()][["node", "commodity", "level"]].to_dict("records")
    return issues


def summarise(issues: dict) -> pd.DataFrame:
    return pd.DataFrame([{"check": k, "issues": len(v)} for k, v in issues.items()])


def calibrate(scen, cfg: dict) -> pd.DataFrame:
    """Run the consistency check, apply the configured fixes, check again.
    Returns the calibration log (one row per change)."""
    c = cfg["calibration"]
    log_rows: list[dict] = []
    record = lambda **kw: log_rows.append(kw)  # noqa: E731

    issues = consistency_check(scen)
    log.info("consistency check before calibration:\n%s", summarise(issues).to_string(index=False))

    with scen.transact("NEST calibration: fixes flagged by the consistency check"):
        if c.get("collapse_bound_conflicts", True):
            for row in issues["bound_conflicts"]:
                if row["pair"] != ("bound_activity_lo", "bound_activity_up"):
                    continue
                df = scen.par("bound_activity_lo", filters={k: [row[k]] for k in ACT_BOUND_KEYS})
                if not df.empty:
                    scen.add_par("bound_activity_lo", df.assign(value=row["up"]))
                    record(parameter="bound_activity_lo", node=row["node_loc"], technology=row["technology"],
                           year=row["year_act"], old=row["lo"], new=row["up"],
                           reason="lower bound above upper bound: collapsed to the upper bound")

        if c.get("cap_unreachable_activity_lo", True):
            for row in issues["activity_lo_infeasible"]:
                df = scen.par("bound_activity_lo", filters={"node_loc": [row["node_loc"]],
                                                            "technology": [row["technology"]],
                                                            "year_act": [row["year_act"]]})
                if not df.empty:
                    scen.add_par("bound_activity_lo", df.assign(value=row["max_reachable_activity"]))
                    record(parameter="bound_activity_lo", node=row["node_loc"], technology=row["technology"],
                           year=row["year_act"], old=row["bound_activity_lo"], new=row["max_reachable_activity"],
                           reason=f"available capacity {row['available_capacity']:.4g} GW x capacity factor "
                                  f"{row['capacity_factor_used']:.3g} cannot reach the lower bound")

        for node, tec, year, value, mode, reason in c.get("bound_activity_lo", []):
            df = scen.par("bound_activity_lo", filters={"node_loc": [node], "technology": [tec], "year_act": [year]})
            if df.empty:
                continue
            old = float(df["value"].iloc[0])
            scen.add_par("bound_activity_lo", df.assign(value=value))
            record(parameter="bound_activity_lo", node=node, technology=tec, year=year, old=old, new=value,
                   reason=reason)

        if c.get("fill_missing_lifetime", True) and issues["missing_lifetime"]:
            lifetime = scen.par("technical_lifetime")
            miss = pd.DataFrame(issues["missing_lifetime"])
            for tec, rows in miss.groupby("technology"):
                donor = lifetime[lifetime["technology"] == tec]
                if donor.empty:
                    log.warning(f"no technical_lifetime anywhere for {tec}; left missing")
                    continue
                value = float(donor["value"].median())
                scen.add_par("technical_lifetime", rows.assign(value=value, unit=donor["unit"].iloc[0]))
                record(parameter="technical_lifetime", node="(several)", technology=tec, year=None, old=None,
                       new=value, reason=f"missing at {len(rows)} node/vintage(s); median of other nodes")

        if c.get("fill_missing_capacity_factor", True):
            miss = pd.DataFrame(issues["missing_cap_factor"])
            if not miss.empty and c.get("capacity_factor_scope", "all") == "water":
                miss = miss[miss["technology"].isin(set(c.get("water_capacity_factor_techs", [])))]
            if len(miss):
                scen.add_par("capacity_factor", miss.assign(time="year", value=1.0, unit="-"))
                for tec, rows in miss.groupby("technology"):
                    record(parameter="capacity_factor", node="(several)", technology=tec, year=None, old=None,
                           new=1.0, reason=f"missing for {len(rows)} node/year(s); water technology, full availability")

    after = consistency_check(scen)
    log.info("consistency check after calibration:\n%s", summarise(after).to_string(index=False))
    return pd.DataFrame(log_rows, columns=["parameter", "node", "technology", "year", "old", "new", "reason"])
