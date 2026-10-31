"""The NEST-Pakistan pipeline, one function per step.

1. :func:`build_energy` -- the national energy model from its Excel workbook.
2. :func:`relax_energy` -- drop constraints that bind once water enters.
3. :func:`build_nexus` -- clone, add the NEST water/land module.
4. :func:`calibrate_nexus` -- consistency check + documented fixes.
5. :func:`solve` -- solve, with the irrigation floor as a soft bound.

:func:`run` chains 2-5. Each step takes the open platform and the loaded
configuration (:mod:`.config`) and returns the scenario it worked on, so the
steps can also be called from a notebook.
"""

import logging
import time

import pandas as pd

from .calibration import calibrate
from .config import expand, output_dir

log = logging.getLogger(__name__)


def load(mp, cfg: dict, scenario: str, version: int | None = None, retries: int = 4):
    """A scenario of ``cfg["model"]``. Retries briefly: right after the
    platform opens, HSQLDB can still be replaying its log."""
    import message_ix

    for attempt in range(retries):
        try:
            kw = {"version": version} if version else {}
            return message_ix.Scenario(mp, model=cfg["model"], scenario=scenario, **kw)
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(1.5)


def solve(scen, cfg: dict):
    s = cfg["solve"]
    case = f"{scen.model}__{scen.scenario}__v{scen.version}"
    log.info(f"solving {case}")
    scen.solve(solve_options=s.get("solve_options", {}), case=case, gams_args=s.get("gams_args", []))
    log.info(f"solved {case}: OBJ = {scen.var('OBJ')['lvl']:.6g}")
    return scen


def build_energy(mp, cfg: dict, solve_it: bool = True):
    """The national energy model as a new version of ``energy.scenario``."""
    import message_ix

    e = cfg["energy"]
    path = expand(e["excel"], cfg.get("nest_repo"))
    if not path.exists():
        raise FileNotFoundError(f"energy workbook not found: {path} (set energy.excel or nest_repo)")
    scen = message_ix.Scenario(mp, model=cfg["model"], scenario=e["scenario"], version="new")
    log.info(f"reading {path}")
    scen.read_excel(path, add_units=True, commit_steps=False, init_items=True)
    scen.set_as_default()
    if solve_it:
        solve(scen, cfg)
    return scen


def relax_energy(mp, cfg: dict):
    """Remove the ``energy.relax`` constraints from the energy baseline
    (idempotent). The removed rows are saved as CSV for reference."""
    scen = load(mp, cfg, cfg["energy"]["scenario"])
    if scen.has_solution():
        scen.remove_solution()
    out = output_dir(cfg, "energy_relaxed")
    removed = []
    with scen.transact("NEST: relax energy constraints that bind with water costs"):
        for par, techs in cfg["energy"].get("relax", {}).items():
            for tec in techs:
                df = scen.par(par, filters={"technology": [tec]})
                if df.empty:
                    continue
                df.to_csv(out / f"removed_{par}_{tec}.csv", index=False)
                scen.remove_par(par, df)
                removed.append(f"{par}[{tec}]")
    log.info("removed: " + (", ".join(removed) or "nothing (already relaxed)"))
    return scen


def nexus_context(mp, cfg: dict, scenario: str | None = None):
    """The message_ix_models Context the water module builds from."""
    from message_ix_models import Context
    from message_ix_models.model.water.cli import water_ini

    n, code = cfg["nexus"], cfg["basin"]["code"]
    c = Context()
    c.handle_cli_args(model_name=cfg["model"], scenario_name=cfg["energy"]["scenario"])
    c.core._mp = mp
    c.update(regions=code)
    c.type_reg = "basin"
    c.map_ISO_c = {code: None}
    c.region_node_map = dict(cfg["basin"].get("region_node_map") or {})
    c.ssp, c.RCP, c.SDG, c.REL = n["ssp"], n["rcp"], n["sdg"], n["rel"]
    c.time, c.nexus_set = n["time"], n["nexus_set"]
    c.output_model = cfg["model"]
    c.output_scenario = scenario or n["scenario"]
    for k, v in (n.get("context") or {}).items():
        setattr(c, k, v)
    water_ini(c, code, None)
    return c


def build_nexus(mp, cfg: dict, scenario: str | None = None):
    """Clone the (relaxed) energy baseline and add the NEST water/land module."""
    from message_ix_models.model.water.build import main as water_build

    name = scenario or cfg["nexus"]["scenario"]
    context = nexus_context(mp, cfg, name)
    base = load(mp, cfg, cfg["energy"]["scenario"])
    scen = base.clone(model=cfg["model"], scenario=name, keep_solution=False)
    log.info(f"cloned {base.scenario} v{base.version} -> {scen.scenario} v{scen.version}")
    water_build(context, scen)
    return scen


def calibrate_nexus(scen, cfg: dict) -> pd.DataFrame:
    changes = calibrate(scen, cfg)
    path = output_dir(cfg, scen.scenario) / "calibration_log.csv"
    changes.to_csv(path, index=False)
    log.info(f"{len(changes)} calibration change(s); log: {path}")
    return changes


def run(mp, cfg: dict, scenario: str | None = None, relax: bool = True):
    """Relax the energy baseline, build, calibrate and solve the nexus."""
    if relax:
        relax_energy(mp, cfg)
    scen = build_nexus(mp, cfg, scenario)
    calibrate_nexus(scen, cfg)
    return solve(scen, cfg)
