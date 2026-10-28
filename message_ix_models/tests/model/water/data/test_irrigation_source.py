"""Tests for crops/irrigation_source.py -- the R model's irrigation sw/gw
diversion layer -- and for irrigation_tech.py's behaviour with it on."""

import pandas as pd
import pytest

from message_ix_models.model.water.data.crops import irrigation_source as src
from message_ix_models.model.water.data.crops.common import read_historical_irrigation_withdrawals
from message_ix_models.model.water.data.crops.irrigation_tech import (
    add_irrigation_techs,
    irrigation_relation_names,
)
from message_ix_models.tests.model.water.conftest import water_params

IRB = [{**water_params("R12"), "regions": "IRB"}]


def test_unit_conversions():
    # R: million USD per (MCM/day); here: million USD per (MCM/year), the model's cost unit
    assert src.musd_per_mcm_day_to_musd_per_mcm_year(57) == pytest.approx(57 / 365)
    # R: MW per (MCM/day) == 0.024 * value kWh/m3; here: GWa per MCM
    assert src.mw_per_mcm_day_to_gwa_per_mcm(6) == pytest.approx(6 / 365 / 1e3)
    # 6 MW per MCM/day is 0.144 kWh/m3; 1 kWh/m3 = 1/8760 GWa/MCM
    assert src.mw_per_mcm_day_to_gwa_per_mcm(6) == pytest.approx(0.144 / 8760, rel=1e-9)


def test_tech_data_matches_r_model():
    """Spot-check the parameters copied from basin_msggdx_technologies.r."""
    d = src.TECH_DATA
    assert d["irrigation_sw_diversion"]["efficiency"] + d["irrigation_sw_diversion"]["seepage"] == 1.0
    assert d["smart_irrigation_sw_diversion"]["efficiency"] == pytest.approx(0.78)
    assert d["irrigation_gw_diversion"]["efficiency"] == 0.95
    assert d["irrigation_gw_diversion"]["electricity_mw"] == 16
    assert d["irrigation_sw_diversion"]["lifetime"] == 50
    assert d["irrigation_gw_diversion"]["lifetime"] == 20
    assert {t for t, s in d.items() if s.get("fossil")} == {
        "irrigation_gw_diversion", "smart_irrigation_gw_diversion",
    }


@pytest.mark.parametrize("water_context", IRB, indirect=True)
def test_enabled_by_default_and_can_be_switched_off(water_context, water_scenario):
    assert set(src.irrigation_source_technology_names(water_context)) == set(src.TECH_DATA)
    water_context.irrigation_source_layer = False
    assert src.irrigation_source_technology_names(water_context) == []
    assert src.add_irrigation_source_techs(water_context) == {}


@pytest.mark.parametrize("water_context", [{**water_params("R12"), "regions": "R12"}], indirect=True)
def test_unrelated_region_is_empty(water_context):
    water_context.irrigation_source_layer = True
    assert src.irrigation_source_technology_names(water_context) == []


@pytest.mark.parametrize("water_context", IRB, indirect=True)
def test_add_irrigation_source_techs(water_context, water_scenario, assert_message_params):
    water_context.irrigation_source_layer = True
    assert set(src.irrigation_source_technology_names(water_context)) == set(src.TECH_DATA)

    result = src.add_irrigation_source_techs(water_context)
    assert_message_params(
        result,
        expected_keys=["input", "output", "technical_lifetime", "construction_time",
                       "capacity_factor", "inv_cost", "fix_cost", "var_cost", "relation_activity"],
    )
    inp, out = result["input"], result["output"]

    # Every technology delivers farm-gate irrigation water at its R efficiency.
    iw = out[(out["commodity"] == "irrigation_water") & (out["level"] == "irrigation_final")]
    for tech, spec in src.TECH_DATA.items():
        assert iw.loc[iw["technology"] == tech, "value"].unique().tolist() == pytest.approx([spec["efficiency"]])

    # Canal seepage returns to the aquifer.
    seep = out[(out["technology"] == "irrigation_sw_diversion") & (out["commodity"] == "groundwater_basin")]
    assert set(seep["value"]) == {0.25}

    # Groundwater diversions: M1 draws the renewable pool, M2 (fossil) has no water input.
    gw = inp[inp["technology"] == "irrigation_gw_diversion"]
    assert set(gw.loc[gw["mode"] == "M1", "commodity"]) == {"groundwater_basin", "electr"}
    assert set(gw.loc[gw["mode"] == src.FOSSIL_MODE, "commodity"]) == {"electr"}
    # Electricity comes from the basin's region node (as for every other water
    # technology) -- basin nodes have no electricity supply of their own.
    e = inp[inp["commodity"] == "electr"]
    origin = dict(zip(e["node_loc"], e["node_origin"]))
    assert all(origin[n] == "R12_PAK" for n in origin if n.endswith("|PAKISTAN"))
    assert all(origin[n] == n.split("|")[1] for n in origin if not n.endswith("|PAKISTAN"))

    elec = gw.loc[gw["commodity"] == "electr", "value"]
    assert elec.unique().tolist() == pytest.approx([16 / 365 / 1e3])

    # Only the fossil mode carries the depletion tie-break cost.
    vc = result["var_cost"]
    assert set(vc["mode"]) == {src.FOSSIL_MODE}
    assert set(vc["technology"]) == {"irrigation_gw_diversion", "smart_irrigation_gw_diversion"}
    assert set(vc["value"]) == {src.FOSSIL_DEPLETION_COST}

    # Costs converted from R units (million USD per MCM/day -> per MCM/year).
    inv = result["inv_cost"]
    assert inv.loc[inv["technology"] == "irrigation_sw_diversion", "value"].unique().tolist() == \
        pytest.approx([57 / 365])

    # This fixture's horizon has no pre-model year, so no historical
    # capacity (checked in the solve test, whose horizon has one).
    assert "historical_new_capacity" not in result
    assert not read_historical_irrigation_withdrawals(water_context).empty

    # The floor counts every sw/gw diversion (both modes) at 1 per MCM, on the
    # same per-basin relations irrigation_tech registers -- and nothing else.
    rel = result["relation_activity"]
    assert set(rel["technology"]) == set(src.FLOOR_TECHS)
    assert set(rel["value"]) == {1.0}
    assert set(rel["relation"]) <= set(irrigation_relation_names(water_context))
    assert (rel["year_rel"] == rel["year_act"]).all()
    assert set(rel.loc[rel["technology"] == "irrigation_gw_diversion", "mode"]) == {"M1", "M2"}


@pytest.mark.parametrize("water_context", IRB, indirect=True)
def test_irrigation_tech_with_source_layer(water_context, water_scenario):
    """With the layer on, crop techs draw farm-gate water at crop water /
    water_efficiency -- without R's field_efficiency_conv, since the
    diversions carry the conveyance loss -- and leave the floor's
    coefficients to the diversions; the floor itself (relation_lower) is
    unchanged."""
    from message_ix_models.model.water.data.crops.common import CONFIG

    water_context.irrigation_source_layer = False
    off = add_irrigation_techs(water_context)
    water_context.irrigation_source_layer = True
    on = add_irrigation_techs(water_context)

    def water_rows(r, commodity):
        i = r["input"]
        return i[i["commodity"] == commodity].drop(columns=["commodity", "level"])

    assert water_rows(on, "surfacewater_basin").empty
    off_w = water_rows(off, "surfacewater_basin").reset_index(drop=True)
    on_w = water_rows(on, "irrigation_water").reset_index(drop=True)
    pd.testing.assert_frame_equal(off_w.drop(columns="value"), on_w.drop(columns="value"))
    assert on_w["value"].to_numpy() == pytest.approx(
        off_w["value"].to_numpy() * CONFIG["field_efficiency_conv"])
    assert "relation_activity" in off and "relation_activity" not in on
    pd.testing.assert_frame_equal(off["relation_lower"], on["relation_lower"])


def _solve_scenario(test_context, monkeypatch, *, river_mcm, hist_sw, hist_gw, model):
    """One basin, firstyear 2020, the source layer on, a floor of
    hist_sw + hist_gw MCM, and free river water capped at ``river_mcm``."""
    from message_ix import Scenario

    from message_ix_models import ScenarioInfo
    from message_ix_models.model.water.data.crops import common as common_mod

    regions, basin, firstyear = "SRCTEST", "1|SRCTEST", 2020
    node = f"B{basin}"
    hist = pd.DataFrame([
        {"node": basin, "tec": "irrigation_sw_diversion", "value": hist_sw},
        {"node": basin, "tec": "irrigation_gw_diversion", "value": hist_gw},
    ])
    monkeypatch.setattr(src, "basins_with_crop_data", lambda: {basin})
    monkeypatch.setattr(src, "read_historical_irrigation_withdrawals", lambda context: hist)
    monkeypatch.setattr(common_mod, "basins_with_crop_data", lambda: {basin})

    mp = test_context.get_platform()
    scen = Scenario(mp, model=model, scenario="src", version="new")
    scen.add_horizon(year=[2010, 2015, firstyear], firstmodelyear=firstyear)
    scen.add_set("node", ["World", node])
    scen.add_set("commodity", ["surfacewater_basin", "groundwater_basin", "irrigation_water",
                               "electr", "urban_collected_wst", "salinewater_basin"])
    scen.add_set("level", ["water_avail_basin", "irrigation_final", "final", "water_treat"])
    scen.add_set("mode", ["M1", "M2"])
    scen.add_par("interestrate", [firstyear], value=0.05, unit="-")
    for unit in ["MCM", "GWa", "USD/MCM", "-", "y", "%"]:
        if unit not in mp.units():
            mp.add_unit(unit)

    test_context.regions = regions
    test_context.time = "year"
    test_context.valid_basins = {basin}
    test_context.irrigation_source_layer = True
    test_context.set_scenario(scen)
    test_context["water build info"] = ScenarioInfo(scen)
    monkeypatch.setattr(type(test_context), "get_scenario", lambda self: scen)

    relation = "hist_irr_withdrawal_1_SRCTEST"
    sources = ["river_source", "aquifer_source", "power_source"]
    scen.add_set("technology", [*src.irrigation_source_technology_names(test_context), *sources])
    scen.add_set("relation", [relation])

    def source(tech, commodity, level, var_cost):
        common = dict(node_loc=node, technology=tech, year_vtg=firstyear, year_act=firstyear,
                      mode="M1", time="year")
        scen.add_par("output", pd.DataFrame([{**common, "node_dest": node, "commodity": commodity,
                                              "level": level, "time_dest": "year", "value": 1.0,
                                              "unit": "MCM"}]))
        scen.add_par("technical_lifetime", pd.DataFrame([{"node_loc": node, "technology": tech,
                                                          "year_vtg": firstyear, "value": 10, "unit": "y"}]))
        scen.add_par("capacity_factor", pd.DataFrame([{**{k: common[k] for k in
                                                          ("node_loc", "technology", "year_vtg", "year_act", "time")},
                                                       "value": 1.0, "unit": "%"}]))
        scen.add_par("var_cost", pd.DataFrame([{**common, "value": var_cost, "unit": "USD/MCM"}]))

    source("river_source", "surfacewater_basin", "water_avail_basin", 1.0)
    source("aquifer_source", "groundwater_basin", "water_avail_basin", 1.0)
    source("power_source", "electr", "final", 0.0)
    scen.add_par("bound_activity_up", pd.DataFrame([{
        "node_loc": node, "technology": "river_source", "year_act": firstyear, "mode": "M1",
        "time": "year", "value": river_mcm, "unit": "MCM"}]))

    for par, df in src.add_irrigation_source_techs(test_context).items():
        scen.add_par(par, df)
    scen.add_par("relation_lower", pd.DataFrame([{
        "relation": relation, "node_rel": node, "year_rel": firstyear,
        "value": hist_sw + hist_gw, "unit": "MCM"}]))
    return scen, firstyear


def _price_groundwater(scen, y, value):
    """var_cost on every groundwater-diversion vintage/mode active in ``y``."""
    scen.add_par("var_cost", pd.DataFrame([{
        "node_loc": "B1|SRCTEST", "technology": t, "year_vtg": v, "year_act": y, "mode": m,
        "time": "year", "value": value, "unit": "USD/MCM"}
        for t in ["irrigation_gw_diversion", "smart_irrigation_gw_diversion"]
        for v in [2015, 2020] for m in ["M1", "M2"]]))


def _act(scen, techs, year, mode=None):
    a = scen.var("ACT", filters={"technology": techs, "year_act": [year]})
    if mode:
        a = a[a["mode"] == mode]
    return a["lvl"].sum()


def test_source_layer_solves_and_uses_groundwater(test_context, monkeypatch):
    """Floor 8 MCM (6 sw + 2 gw historical), river capped at 2 MCM: the river
    diversions take all 2 MCM and groundwater diversions supply the other 6
    -- groundwater irrigation the old direct-draw design could not offer."""
    scen, y = _solve_scenario(test_context, monkeypatch, river_mcm=2.0, hist_sw=6.0, hist_gw=2.0,
                              model="irr-source-solve-test")
    _price_groundwater(scen, y, 1e4)  # river water is the cheaper source
    scen.commit("pytest irrigation source layer")
    scen.solve(model="MESSAGE", solve_options={"lpmethod": "1"})

    # Historical capacity = historical annual withdrawal / capacity factor,
    # at the genuine pre-model vintage, for the two conventional diversions only.
    hnc = scen.par("historical_new_capacity").set_index("technology")
    assert hnc.loc["irrigation_sw_diversion", "value"] == pytest.approx(6.0 / 0.9)
    assert hnc.loc["irrigation_gw_diversion", "value"] == pytest.approx(2.0 / 0.9)
    assert set(hnc["year_vtg"]) == {2015}

    sw = _act(scen, ["irrigation_sw_diversion", "smart_irrigation_sw_diversion"], y)
    gw = _act(scen, ["irrigation_gw_diversion", "smart_irrigation_gw_diversion"], y)
    assert sw == pytest.approx(2.0, abs=1e-4)
    assert gw == pytest.approx(6.0, abs=1e-4)
    rel = scen.var("REL", filters={"relation": ["hist_irr_withdrawal_1_SRCTEST"]})
    assert rel["lvl"].sum() == pytest.approx(8.0, abs=1e-4)


def test_source_layer_canal_losses(test_context, monkeypatch):
    """Floor met from the river only (enough river water): farm-gate water =
    0.75 x diverted and 0.25 x diverted seeps to the aquifer."""
    scen, y = _solve_scenario(test_context, monkeypatch, river_mcm=100.0, hist_sw=8.0, hist_gw=0.0,
                              model="irr-source-loss-test")
    _price_groundwater(scen, y, 1e6)  # floor met from the river
    scen.commit("pytest irrigation source losses")
    scen.solve(model="MESSAGE", solve_options={"lpmethod": "1"})

    act = scen.var("ACT", filters={"year_act": [y]})
    act = act[act["lvl"] > 1e-9]
    out = scen.par("output", filters={"year_act": [y]})
    flows = out.merge(act, on=["node_loc", "technology", "year_vtg", "year_act", "mode", "time"])
    flows["flow"] = flows["value"] * flows["lvl"]
    diverted = _act(scen, ["irrigation_sw_diversion", "smart_irrigation_sw_diversion"], y)
    farm = flows.loc[flows["commodity"] == "irrigation_water", "flow"].sum()
    seepage = flows.loc[(flows["commodity"] == "groundwater_basin")
                        & flows["technology"].str.contains("sw_diversion"), "flow"].sum()
    assert diverted == pytest.approx(8.0, abs=1e-4)
    assert farm + seepage == pytest.approx(diverted, abs=1e-4)
    assert 0.75 * diverted <= farm + 1e-6 <= 0.78 * diverted + 1e-4
