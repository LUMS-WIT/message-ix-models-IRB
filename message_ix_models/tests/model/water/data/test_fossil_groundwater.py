"""Tests for fossil_groundwater.py -- the R model's non-renewable
groundwater accounting and its "Nonrenewable Groundwater extraction" limit."""

import pandas as pd
import pytest

from message_ix_models.model.water.data import fossil_groundwater as fgw
from message_ix_models.model.water.data.crops import irrigation_source as src
from message_ix_models.tests.model.water.conftest import water_params

IRB = [{**water_params("R12"), "regions": "IRB"}]


def test_limit_from_baseline_r_rule():
    """share(nearest R year) x baseline x beta, per node; a zero-baseline
    node takes that year's largest node bound (R's own rule)."""
    base = pd.DataFrame([
        {"node": "B1|X", "year": 2020, "value": 100.0},
        {"node": "B2|X", "year": 2020, "value": 0.0},
        {"node": "B1|X", "year": 2025, "value": 80.0},
        {"node": "B1|X", "year": 2070, "value": 50.0},
    ])
    lim = fgw.limit_from_baseline(base, years=[2020, 2025, 2070]).set_index(["node", "year"])["value"]
    assert lim[("B1|X", 2020)] == pytest.approx(50.0)  # 0.5 x 100
    assert lim[("B2|X", 2020)] == pytest.approx(50.0)  # zero -> year max
    assert lim[("B1|X", 2025)] == pytest.approx(40.0)  # 2025 nearest R year 2020 (tie -> 2020): 0.5
    assert lim[("B1|X", 2070)] == pytest.approx(0.0)   # after 2060: the 2060 share, 0
    beta = fgw.limit_from_baseline(base, years=[2020], beta=2.0)
    assert beta.loc[beta["node"] == "B1|X", "value"].item() == pytest.approx(100.0)


@pytest.mark.parametrize("water_context", IRB, indirect=True)
def test_accounting_on_by_default_and_can_be_switched_off(water_context, water_scenario):
    assert fgw.FOSSIL_EXTRACT in fgw.fossil_gw_technology_names(water_context)
    assert fgw.fossil_gw_type_years(water_context) == []  # no limit by default
    water_context.fossil_gw_accounting = False
    assert fgw.fossil_gw_technology_names(water_context) == []
    assert fgw.add_fossil_groundwater(water_context) == {}
    assert fgw.fossil_gw_type_years(water_context) == []


@pytest.mark.parametrize("water_context", IRB, indirect=True)
def test_accounting_tags_every_fossil_source(water_context, water_scenario):
    water_context.fossil_gw_accounting = True
    water_context.irrigation_source_layer = False
    assert fgw.fossil_gw_technology_names(water_context) == [fgw.FOSSIL_EXTRACT]
    ef = fgw.add_fossil_groundwater(water_context)["emission_factor"]
    assert set(ef["technology"]) == {fgw.FOSSIL_EXTRACT}
    assert set(ef["emission"]) == {fgw.EMISSION} and set(ef["value"]) == {1.0}
    assert (ef["year_act"] >= ef["year_vtg"]).all()
    assert set(ef["node_loc"]) == {"B" + b for b in water_context.valid_basins}

    # With the irrigation source layer on, its fossil mode is tagged too.
    water_context.irrigation_source_layer = True
    ef = fgw.add_fossil_groundwater(water_context)["emission_factor"]
    tagged = set(zip(ef["technology"], ef["mode"]))
    assert tagged == {(fgw.FOSSIL_EXTRACT, "M1"), ("irrigation_gw_diversion", src.FOSSIL_MODE),
                      ("smart_irrigation_gw_diversion", src.FOSSIL_MODE)}


@pytest.mark.parametrize("water_context", IRB, indirect=True)
def test_limit_becomes_bound_emission(water_context, water_scenario):
    node = "B" + sorted(water_context.valid_basins)[0]
    water_context.fossil_gw_limit = pd.DataFrame([
        {"node": node, "year": 2020, "value": 5.0},
        {"node": node, "year": 1990, "value": 9.0},  # outside the horizon: dropped
    ])
    result = fgw.add_fossil_groundwater(water_context)
    assert "emission_factor" in result  # a limit implies accounting
    be = result["bound_emission"]
    assert be.to_dict("records") == [{
        "node": node, "type_emission": fgw.TYPE_EMISSION, "type_tec": fgw.TYPE_TEC,
        "type_year": "2020", "value": 5.0, "unit": "MCM"}]
    assert fgw.fossil_gw_type_years(water_context) == ["1990", "2020"]


def _solve(test_context, monkeypatch, limit, model):
    """One basin, floor 8 MCM, no river water: groundwater must supply all
    of it. Fossil pumping is cheaper than the renewable aquifer source."""
    from message_ix import Scenario

    from message_ix_models import ScenarioInfo

    basin, y = "1|GWTEST", 2020
    node = f"B{basin}"
    monkeypatch.setattr(src, "basins_with_crop_data", lambda: {basin})
    monkeypatch.setattr(src, "read_historical_irrigation_withdrawals", lambda context: pd.DataFrame(
        [{"node": basin, "tec": "irrigation_gw_diversion", "value": 8.0}]))

    mp = test_context.get_platform()
    scen = Scenario(mp, model=model, scenario="gw", version="new")
    scen.add_horizon(year=[2010, 2015, y], firstmodelyear=y)
    scen.add_set("node", ["World", node])
    scen.add_set("commodity", ["surfacewater_basin", "groundwater_basin", "irrigation_water",
                               "electr", "urban_collected_wst", "salinewater_basin",
                               "freshwater_basin"])
    scen.add_set("level", ["water_avail_basin", "irrigation_final", "final", "water_treat",
                           "water_supply_basin"])
    scen.add_set("mode", ["M1", "M2"])
    scen.add_par("interestrate", [y], value=0.05, unit="-")
    for unit in ["MCM", "GWa", "USD/MCM", "-", "y", "%"]:
        if unit not in mp.units():
            mp.add_unit(unit)

    test_context.time = "year"
    test_context.valid_basins = {basin}
    test_context.irrigation_source_layer = True
    test_context.fossil_gw_accounting = True
    test_context.fossil_gw_limit = limit
    test_context.set_scenario(scen)
    test_context["water build info"] = ScenarioInfo(scen)
    monkeypatch.setattr(type(test_context), "get_scenario", lambda self: scen)

    relation = "hist_irr_withdrawal_1_GWTEST"
    scen.add_set("technology", [*src.irrigation_source_technology_names(test_context),
                                fgw.FOSSIL_EXTRACT, "aquifer_source", "power_source"])
    scen.add_set("relation", [relation])
    scen.add_set("emission", [fgw.EMISSION])
    scen.add_set("type_emission", [fgw.TYPE_EMISSION])
    scen.add_set("cat_emission", [[fgw.TYPE_EMISSION, fgw.EMISSION]])
    scen.add_set("type_tec", [fgw.TYPE_TEC])
    scen.add_set("cat_tec", [[fgw.TYPE_TEC, t] for t in fgw.fossil_gw_technology_names(test_context)])
    for ty in fgw.fossil_gw_type_years(test_context):
        scen.add_set("type_year", [ty])
        scen.add_set("cat_year", [[ty, int(ty)]])

    def source(tech, commodity, level, var_cost):
        key = dict(node_loc=node, technology=tech, year_vtg=y, year_act=y)
        scen.add_par("output", pd.DataFrame([{**key, "mode": "M1", "node_dest": node,
                                              "commodity": commodity, "level": level, "time": "year",
                                              "time_dest": "year", "value": 1.0, "unit": "MCM"}]))
        scen.add_par("technical_lifetime", pd.DataFrame([{"node_loc": node, "technology": tech,
                                                          "year_vtg": y, "value": 10, "unit": "y"}]))
        scen.add_par("capacity_factor", pd.DataFrame([{**key, "time": "year", "value": 1.0, "unit": "%"}]))
        scen.add_par("var_cost", pd.DataFrame([{**key, "mode": "M1", "time": "year",
                                                "value": var_cost, "unit": "USD/MCM"}]))

    source("aquifer_source", "groundwater_basin", "water_avail_basin", 1000.0)  # renewable, dearer
    source("power_source", "electr", "final", 0.0)
    source(fgw.FOSSIL_EXTRACT, "freshwater_basin", "water_supply_basin", 0.0)  # present, idle

    for par, df in src.add_irrigation_source_techs(test_context).items():
        scen.add_par(par, df)
    for par, df in fgw.add_fossil_groundwater(test_context).items():
        scen.add_par(par, df)
    scen.add_par("relation_lower", pd.DataFrame([{"relation": relation, "node_rel": node,
                                                  "year_rel": y, "value": 8.0, "unit": "MCM"}]))
    scen.commit("pytest fossil groundwater")
    scen.solve(model="MESSAGE", solve_options={"lpmethod": "1"})
    return scen, node, y


def _fossil_and_renewable(scen, y):
    act = scen.var("ACT", filters={"technology": ["irrigation_gw_diversion",
                                                  "smart_irrigation_gw_diversion"], "year_act": [y]})
    return (act.loc[act["mode"] == src.FOSSIL_MODE, "lvl"].sum(),
            act.loc[act["mode"] == "M1", "lvl"].sum())


def test_without_limit_fossil_is_used_and_counted(test_context, monkeypatch):
    scen, node, y = _solve(test_context, monkeypatch, limit=None, model="fossil-gw-nolimit")
    fossil, renewable = _fossil_and_renewable(scen, y)
    assert fossil == pytest.approx(8.0, abs=1e-4)  # cheaper, unconstrained
    assert renewable == pytest.approx(0.0, abs=1e-4)
    emiss = scen.var("EMISS", filters={"emission": [fgw.EMISSION], "type_tec": [fgw.TYPE_TEC],
                                       "node": [node], "year": [y]})
    assert emiss["lvl"].sum() == pytest.approx(8.0, abs=1e-4)


def test_limit_forces_renewable_groundwater(test_context, monkeypatch):
    """bound 1 MCM fossil: fossil pumping drops to 1, the renewable aquifer
    supplies the other 7, and EMISS equals the fossil pumping."""
    limit = pd.DataFrame([{"node": "B1|GWTEST", "year": 2020, "value": 1.0}])
    scen, node, y = _solve(test_context, monkeypatch, limit=limit, model="fossil-gw-limit")
    fossil, renewable = _fossil_and_renewable(scen, y)
    assert fossil == pytest.approx(1.0, abs=1e-4)
    assert renewable == pytest.approx(7.0, abs=1e-4)
    emiss = scen.var("EMISS", filters={"emission": [fgw.EMISSION], "type_tec": [fgw.TYPE_TEC],
                                       "node": [node], "year": [y]})
    assert emiss["lvl"].sum() == pytest.approx(1.0, abs=1e-4)
    assert fgw.baseline_from_scenario(scen).query("node == @node and year == @y")["value"].sum() \
        == pytest.approx(1.0, abs=1e-4)
