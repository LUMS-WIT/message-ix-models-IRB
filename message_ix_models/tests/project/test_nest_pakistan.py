"""Tests for message_ix_models.project.nest_pakistan (no database needed)."""

import pandas as pd
import pytest
from click.testing import CliRunner

from message_ix_models.project.nest_pakistan import basin_data
from message_ix_models.project.nest_pakistan.cli import cli
from message_ix_models.project.nest_pakistan.config import apply_overrides, load_config, national_node
from message_ix_models.project.nest_pakistan.report import extract as ex
from message_ix_models.project.nest_pakistan.report import pdf as pdf_


def test_config_defaults_and_overrides(tmp_path):
    cfg = load_config()
    assert cfg["basin"]["code"] == "IRB"
    assert national_node(cfg) == "R12_PAK"
    assert cfg["nexus"]["context"]["external_electricity"] is True

    cfg = load_config(overrides=["nexus.rcp=7p0", "nexus.context.external_electricity_cost=null",
                                 "report.years=[2030, 2050]"])
    assert cfg["nexus"]["rcp"] == "7p0"
    assert cfg["nexus"]["context"]["external_electricity_cost"] is None
    assert cfg["report"]["years"] == [2030, 2050]

    f = tmp_path / "my.yaml"
    f.write_text("basin:\n  code: NIL\n  name: Nile\n  region_node_map: {EGYPT: R12_EGY}\n")
    cfg = load_config(f)
    assert cfg["basin"]["code"] == "NIL" and national_node(cfg) == "R12_EGY"
    assert cfg["nexus"]["ssp"] == "SSP1"  # keys the file does not set keep their defaults

    with pytest.raises(ValueError):
        apply_overrides(cfg, ["no_equals_sign"])


def test_classification():
    assert ex.sector("irrigation_gw_diversion") == "Irrigation supply"
    assert ex.sector("irr_drip_wheat") == "Crops & land"
    assert ex.sector("coal_ppl__cl_fresh") == "Power plant cooling"
    assert ex.sector("river|9|INDIA_EAST|16|PAKISTAN") == "Water transfers"
    assert ex.sector("coal_ppl") == "Energy system"
    assert ex.fuel("gas_cc_ccs") == "Gas" and ex.fuel("solar_res3") == "Solar"
    assert ex.irrigation_method("irr_drip_smart_wheat") == "drip smart"
    assert ex.irrigation_method("rainfed_rice") == "rainfed"
    assert ex.irrigation_method("irrigation_sw_diversion") is None
    assert pdf_.family("Withdrawal: River") == ("Withdrawal", "River")
    assert pdf_.family("Irrigation demand unmet") == ("Irrigation demand unmet", "")


def _sample():
    rows = [
        ("B12|PAKISTAN", "Withdrawal: River", "MCM/yr", 100.0),
        ("B16|PAKISTAN", "Withdrawal: River", "MCM/yr", 50.0),
        ("B9|INDIA_EAST", "Withdrawal: River", "MCM/yr", 30.0),
        ("B2|AFGHAN_NORTH", "Withdrawal: Renewable groundwater", "MCM/yr", 5.0),
        ("B12|PAKISTAN", "Shadow price: river water", "M USD/MCM", 2.0),
        ("B16|PAKISTAN", "Shadow price: river water", "M USD/MCM", 4.0),
        ("B16|PAKISTAN", "Inflow from upstream basins", "MCM/yr", 999.0),
        ("R12_PAK", "Electricity for water from national energy system (R12_PAK)", "GWa", 3.0),
    ]
    df = pd.DataFrame(rows, columns=["node", "indicator", "unit", "value"])
    df["theme"] = df["indicator"].map(lambda i: "energy-for-water" if i.startswith("Electricity") else "water")
    return df.assign(scenario="S", year=2030)[ex.COLUMNS]


def test_aggregate_groups_upper_basins_and_whole_basin():
    cfg = load_config()
    reg = ex.aggregate(_sample(), cfg).set_index(["region", "indicator"])["value"]
    assert reg[("Pakistan", "Withdrawal: River")] == 150
    assert reg[("India", "Withdrawal: River")] == 30
    assert reg[("Upper basins", "Withdrawal: River")] == 30
    assert reg[("Indus River Basin", "Withdrawal: River")] == 180
    assert reg[("Pakistan", "Shadow price: river water")] == 3  # prices are averaged
    assert reg[("Pakistan", "Electricity for water from national energy system (R12_PAK)")] == 3
    # basin-to-basin flows cancel within groups and are not aggregated
    assert "Inflow from upstream basins" not in reg.index.get_level_values("indicator")
    assert ex.region_order(cfg)[-2:] == ["Upper basins", "Indus River Basin"]


def test_pdf_reports_from_tables(tmp_path):
    cfg = load_config()
    nodes = pd.concat([_sample(), _sample().assign(year=2050, value=lambda d: d["value"] * 2)])
    reg = ex.aggregate(nodes, cfg)
    meta = {"scenario": "S", "version": 1, "objective": 1.0}
    p = pdf_.scenario_report(tmp_path / "r.pdf", nodes, reg, cfg, meta)
    assert p.stat().st_size > 1000
    head = pdf_.headline_table(reg, ex.region_order(cfg), [2030, 2050])
    assert head.loc[(head["region"] == "Pakistan") & (head["year"] == 2030), "Withdrawn km3"].item() == 0.15
    reg2 = pd.concat([reg, reg.assign(scenario="T")])
    assert pdf_.comparison_report(tmp_path / "c.pdf", reg2, cfg, ["S", "T"], {"S": 1, "T": 2}).exists()


def test_basin_data_status():
    s = basin_data.status("IRB")
    assert s[s["need"] == "required"]["present"].all()
    new = basin_data.status("ZZZ")
    assert not new[new["need"] == "required"]["present"].any()
    assert set(new["global data"]) <= {"yes", "partly", "no"}


def test_cli_config_and_data(tmp_path):
    runner = CliRunner()
    r = runner.invoke(cli, ["--set", f"report.output_dir={tmp_path.as_posix()}", "--set", "nexus.rcp=7p0", "config"])
    assert r.exit_code == 0, r.output
    assert "rcp: 7p0" in r.output
    r = runner.invoke(cli, ["--set", f"report.output_dir={tmp_path.as_posix()}", "data"])
    assert r.exit_code == 0, r.output
    assert "1 Spatial structure" in r.output
    assert (tmp_path / "data_IRB" / "data_requirements_IRB.csv").exists()


def test_monthly_data_units():
    """Monthly hydrology is a rate (km3/yr per month): as volumes, the months
    add up to the annual mean, and month groupings keep the total."""
    from message_ix_models.model.water.data.pre_processing import build_monthly_data_IRB as bm

    h = bm.hydrology()
    q = h[(h.variable == "qtot") & (h.rcp == "no_climate") & (h.rel == "mean") & (h.year == 2030)]
    assert q["value"].sum() == pytest.approx(217.0, abs=1.0)  # = annual no_climate file, 216.5
    for scheme in bm.SCHEMES.values():
        assert sorted(m for ms in scheme.values() for m in ms) == list(range(1, 13))
        assert bm.group_months(q.drop(columns="unit"), scheme)["value"].sum() == pytest.approx(q["value"].sum())
    lvl = h[(h.variable == "qtot") & (h.rcp == "2p6") & (h.year == 2050)].groupby("rel")["value"].sum()
    assert lvl["high"] < lvl["med"] < lvl["low"]
    i = bm.irrigation()
    assert i.groupby(["source", "node", "year"], dropna=False)["share"].sum().round(6).isin([0, 1]).all()
