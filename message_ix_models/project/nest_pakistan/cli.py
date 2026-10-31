"""Command line for the NEST-Pakistan workflow: ``mix-models nest-pak ...``.

Typical sequence::

    mix-models nest-pak data                      # which input files exist for the basin
    mix-models nest-pak build-energy              # national energy model from Excel (once)
    mix-models nest-pak run -s LTS_nexus          # relax, build, calibrate, solve, report
    mix-models nest-pak check -s LTS_nexus        # input/assumption checks (PDF)
    mix-models nest-pak report -s LTS_nexus --theme water --theme energy-for-water
    mix-models nest-pak compare -s LTS_nexus -s LTS_nexus_riveronly

Every command takes ``--config FILE`` and ``--set key=value`` (YAML value,
dotted key) before the command name, e.g.
``mix-models nest-pak --set nexus.rcp=7p0 run -s LTS_nexus_rcp70``.
"""

import logging

import click

log = logging.getLogger(__name__)

THEME_CHOICES = ["water", "energy-for-water", "water-for-energy", "energy", "land", "investment", "costs",
                 "emissions"]


def _cfg():
    return click.get_current_context().meta["nest_pak_cfg"]


def _platform():
    from .config import open_platform

    try:
        return open_platform(_cfg())
    except Exception as e:  # noqa: BLE001
        if "lock" in str(e).lower():
            raise click.ClickException(
                "The database is in use by another process (a notebook kernel or script). Close it there with "
                "mp.close_db() or restart that kernel, then run this command again.")
        raise


def _scenario(mp, name, version):
    from .workflow import load

    scen = load(mp, _cfg(), name, version)
    click.echo(f"{scen.model} / {scen.scenario} v{scen.version}")
    return scen


def _spec(text: str) -> tuple[str, int | None]:
    """``NAME`` or ``NAME:VERSION``."""
    name, _, v = text.partition(":")
    return name, int(v) if v else None


@click.group("nest-pak")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              help="YAML file overriding the default config.yaml.")
@click.option("--set", "overrides", multiple=True, metavar="KEY=VALUE",
              help="Override one setting, e.g. --set nexus.rcp=7p0 (repeatable).")
@click.pass_context
def cli(ctx, config_path, overrides):
    """NEST-Pakistan: national energy model + NEST water/land module of a river basin."""
    from .config import load_config

    ctx.meta["nest_pak_cfg"] = load_config(config_path, overrides)


@cli.command("config")
def show_config():
    """Print the configuration in effect (after --config and --set)."""
    import yaml

    click.echo(yaml.safe_dump(_cfg(), sort_keys=False))


@cli.command("data")
@click.option("--basin", help="Basin code (default: basin.code of the config).")
def data(basin):
    """List the input files a basin needs, whether they exist, and their global data source."""
    from .basin_data import ENERGY_MODEL_NOTE, status
    from .config import output_dir

    cfg = _cfg()
    code = basin or cfg["basin"]["code"]
    s = status(code, cfg["nexus"]["rcp"], cfg["nexus"]["rel"])
    for step, g in s.groupby("step"):
        click.echo(f"\n{step}")
        for _, r in g.iterrows():
            mark = "ok" if r["present"] else ("MISSING" if r["need"] == "required" else "absent")
            click.echo(f"  [{mark:7}] {r['file']}  ({r['need']}; global data: {r['global data']})")
    path = output_dir(cfg, f"data_{code}") / f"data_requirements_{code}.csv"
    s.to_csv(path, index=False)
    click.echo(f"\n{ENERGY_MODEL_NOTE}\n\nFull table with sources and scripts: {path}")


@cli.command("monthly-data")
@click.option("--rcp", help="Climate scenario to draw (default: nexus.rcp).")
@click.option("--year", type=int, default=2030, show_default=True, help="Year to draw.")
@click.option("--no-build", is_flag=True, help="Draw the existing files in data/water/monthly/ without rebuilding.")
def monthly_data(rcp, year, no_build):
    """Build the unit-consistent monthly input files and draw them in a PDF."""
    from .monthly_data import report

    path, tab = report(_cfg(), rcp=rcp, year=year, build=not no_build)
    click.echo(tab[tab["region"] == "Whole basin"].round(0).to_string(index=False))
    click.echo(f"\nwrote {path}")


@cli.command("list")
def list_scenarios():
    """Scenarios of the configured model in the database."""
    mp = _platform()
    try:
        df = mp.scenario_list(model=_cfg()["model"], default=False)
        cols = [c for c in ("scenario", "version", "is_default", "cre_date") if c in df]
        click.echo(df[cols].sort_values(["scenario", "version"]).to_string(index=False))
    finally:
        mp.close_db()


@cli.command("build-energy")
@click.option("--no-solve", is_flag=True, help="Only read the Excel workbook.")
def build_energy(no_solve):
    """Build the national energy model from its Excel workbook (energy.excel)."""
    from .workflow import build_energy as _build

    mp = _platform()
    try:
        scen = _build(mp, _cfg(), solve_it=not no_solve)
        click.echo(f"built {scen.scenario} v{scen.version}")
    finally:
        mp.close_db()


@cli.command("build")
@click.option("-s", "--scenario", help="Name of the nexus scenario (default: nexus.scenario).")
@click.option("--no-relax", is_flag=True, help="Skip relaxing the energy baseline.")
@click.option("--solve/--no-solve", default=False, help="Also solve after calibration.")
def build(scenario, no_relax, solve):
    """Relax the energy baseline, add the NEST water/land module and calibrate."""
    from . import workflow as wf

    cfg = _cfg()
    mp = _platform()
    try:
        if not no_relax:
            wf.relax_energy(mp, cfg)
        scen = wf.build_nexus(mp, cfg, scenario)
        wf.calibrate_nexus(scen, cfg)
        if solve:
            wf.solve(scen, cfg)
        click.echo(f"built {scen.scenario} v{scen.version}" + (" and solved" if solve else ""))
    finally:
        mp.close_db()


@cli.command("solve")
@click.option("-s", "--scenario", required=True, help="NAME or NAME:VERSION.")
def solve(scenario):
    """Solve an already built scenario."""
    from .workflow import solve as _solve

    mp = _platform()
    try:
        scen = _scenario(mp, *_spec(scenario))
        if scen.has_solution():
            scen.remove_solution()
        _solve(scen, _cfg())
    finally:
        mp.close_db()


@cli.command("run")
@click.option("-s", "--scenario", help="Name of the nexus scenario (default: nexus.scenario).")
@click.option("--no-relax", is_flag=True)
@click.option("--report/--no-report", default=True, help="Write the full report after solving.")
def run(scenario, no_relax, report):
    """Build, calibrate, solve and report one nexus scenario."""
    from . import workflow as wf
    from .report import report as _report

    cfg = _cfg()
    mp = _platform()
    try:
        scen = wf.run(mp, cfg, scenario, relax=not no_relax)
        if report:
            for k, p in _report(scen, cfg).items():
                click.echo(f"{k}: {p}")
    finally:
        mp.close_db()


@cli.command("check")
@click.option("-s", "--scenario", help="NAME or NAME:VERSION of a built scenario (omit: data files only).")
def check(scenario):
    """Check input data and model structure before solving; writes a PDF."""
    from .checks import run_checks, summary
    from .config import output_dir
    from .report.pdf import checks_report

    cfg = _cfg()
    scen, mp = None, None
    if scenario:
        mp = _platform()
        scen = _scenario(mp, *_spec(scenario))
    try:
        checks = run_checks(scen, cfg)
        out = output_dir(cfg, scen.scenario if scen else f"data_{cfg['basin']['code']}")
        summary(checks).to_csv(out / "checks.csv", index=False)
        for c in checks:
            if not c.table.empty:
                c.table.to_csv(out / f"check_{c.name.lower().replace(' ', '_')}.csv", index=False)
        click.echo(summary(checks).to_string(index=False))
        click.echo(f"\nPDF: {checks_report(out / 'checks.pdf', checks, cfg, scenario)}")
    finally:
        if mp:
            mp.close_db()


@cli.command("report")
@click.option("-s", "--scenario", required=True, help="NAME or NAME:VERSION of a solved scenario.")
@click.option("--theme", "themes", multiple=True, type=click.Choice(THEME_CHOICES),
              help="Only these themes (repeatable; default: all).")
@click.option("--level", "levels", multiple=True, type=click.Choice(["regions", "basins"]),
              help="Region-group pages, basin pages, or both (default).")
@click.option("--node", help="Also a one-basin report, e.g. '16|PAKISTAN' or 'B16|PAKISTAN'.")
def report(scenario, themes, levels, node):
    """PDF + CSV report of a solved scenario: basin, national, upper-basin and whole-basin results."""
    from .report import report as _report

    if node and not node.startswith("B"):
        node = "B" + node
    mp = _platform()
    try:
        scen = _scenario(mp, *_spec(scenario))
        paths = _report(scen, _cfg(), themes=list(themes) or None, levels=tuple(levels) or ("regions", "basins"),
                        node=node)
        for k, p in paths.items():
            click.echo(f"{k}: {p}")
    finally:
        mp.close_db()


@cli.command("compare")
@click.option("-s", "--scenario", "scenarios", multiple=True, required=True,
              help="NAME or NAME:VERSION; give two or more.")
@click.option("--region", help="Region to compare (default: the national group).")
@click.option("--theme", "themes", multiple=True, type=click.Choice(THEME_CHOICES))
def compare(scenarios, region, themes):
    """Compare solved scenarios in one PDF + CSV."""
    from .report import compare as _compare

    if len(scenarios) < 2:
        raise click.UsageError("give at least two scenarios with -s")
    mp = _platform()
    try:
        scens = [_scenario(mp, *_spec(s)) for s in scenarios]
        for k, p in _compare(scens, _cfg(), region=region, themes=list(themes) or None).items():
            click.echo(f"{k}: {p}")
    finally:
        mp.close_db()
