"""Reporting for NEST-Pakistan: tidy result tables, CSV exports and PDF reports.

- :func:`report` -- one solved scenario: CSVs at node and region level plus a
  PDF covering every theme (or the ones asked for).
- :func:`compare` -- several solved scenarios in one PDF and CSV.
"""

import logging
from pathlib import Path

import pandas as pd

from ..config import output_dir
from . import extract as ex
from . import pdf as pdf_

log = logging.getLogger(__name__)

__all__ = ["report", "compare", "results"]


def results(scen, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(node-level, region-level) result tables of a solved scenario."""
    if not scen.has_solution():
        raise ValueError(f"{scen.scenario} v{scen.version} has no solution: solve it first")
    nodes = ex.extract(scen, cfg)
    return nodes, ex.aggregate(nodes, cfg)


def report(scen, cfg: dict, themes=None, levels=("regions", "basins"), node: str | None = None,
           out: Path | None = None) -> dict[str, Path]:
    """Write ``results_nodes.csv``, ``results_regions.csv`` and the PDF
    report(s) for ``scen`` to ``out`` (default: the configured report folder
    / scenario name). Returns the written paths."""
    out = Path(out) if out else output_dir(cfg, scen.scenario)
    out.mkdir(parents=True, exist_ok=True)
    nodes, reg = results(scen, cfg)
    paths = {"nodes_csv": out / "results_nodes.csv", "regions_csv": out / "results_regions.csv"}
    nodes.to_csv(paths["nodes_csv"], index=False)
    reg.to_csv(paths["regions_csv"], index=False)
    meta = {"scenario": scen.scenario, "version": scen.version, "objective": ex.objective(scen)}

    years = cfg["report"]["years"]
    balance = ex.river_balance(scen, 2030 if 2030 in years else years[0])
    balance.to_csv(out / "river_balance_check.csv", index=False)
    cal = out / "calibration_log.csv"
    calibration = pd.read_csv(cal) if cal.exists() else None
    suffix = "" if not themes else "_" + "-".join(themes)
    paths["pdf"] = pdf_.scenario_report(out / f"report_{scen.scenario}{suffix}.pdf", nodes, reg, cfg, meta,
                                        themes=themes, levels=levels, balance=balance, calibration=calibration)
    if node:
        paths["node_pdf"] = pdf_.node_report(out / f"node_{node.replace('|', '_')}_{scen.scenario}.pdf", nodes,
                                             cfg, meta, node)
    return paths


def compare(scens: list, cfg: dict, region: str | None = None, themes=None, out: Path | None = None) -> dict:
    """One PDF and CSV comparing ``scens`` (solved scenarios) for ``region``."""
    names = [s.scenario if not names_dup(scens) else f"{s.scenario} v{s.version}" for s in scens]
    regs, objectives = [], {}
    for s, name in zip(scens, names):
        _, reg = results(s, cfg)
        regs.append(reg.assign(scenario=name))
        objectives[name] = ex.objective(s)
    reg = pd.concat(regs, ignore_index=True)
    out = Path(out) if out else output_dir(cfg, "comparison_" + "_vs_".join(n.replace(" ", "") for n in names))
    out.mkdir(parents=True, exist_ok=True)
    reg.to_csv(out / "comparison_regions.csv", index=False)
    pdf = pdf_.comparison_report(out / "comparison.pdf", reg, cfg, names, objectives, region=region, themes=themes)
    return {"csv": out / "comparison_regions.csv", "pdf": pdf}


def names_dup(scens) -> bool:
    names = [s.scenario for s in scens]
    return len(set(names)) < len(names)
