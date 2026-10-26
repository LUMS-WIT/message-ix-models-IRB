"""End-to-end: does a scenario carrying add_crop_demand()'s output actually
solve under GAMS -- not just a DataFrame-shape check. Mirrors
test_irrigation_tech.py::test_add_irrigation_techs_solves's pattern:
isolate the one new mechanism (a hard `demand` on <crop>_yield) with a
minimal stub producing technology, rather than re-exercising the full
real land/rainfed/irrigation chain (already separately solve-tested).

Why this is worth checking, unlike the water-withdrawal floor: that floor
is a *relation* with SLACK_RELATION_BOUND_LO as an escape valve -- it can
never make a scenario infeasible outright. A `demand` par has no such
built-in slack; if nothing in the scenario can produce enough of a
commodity to meet it, the scenario is genuinely infeasible, full stop.
"""

import pandas as pd
import pytest


def test_add_crop_demand_solves(test_context, monkeypatch):
    from message_ix import Scenario

    from message_ix_models import ScenarioInfo
    from message_ix_models.model.water.data.crops import common as common_mod
    from message_ix_models.model.water.data.crops import food_demand as fd_mod

    basin = "1|TESTCROPDEM"
    node = f"B{basin}"
    firstyear = 2020

    monkeypatch.setattr(fd_mod, "CROPS", ["testcrop"])

    demand_data = pd.DataFrame(
        [
            {"crop": "testcrop", "node": basin, "year": 2015, "value": 100.0, "unit": "kt"},
            {"crop": "testcrop", "node": basin, "year": 2060, "value": 100.0, "unit": "kt"},
        ]
    )
    supply_data = pd.DataFrame(
        [{"crop": "testcrop", "par": "irrigation_yield", "node": basin, "value": 5.0}]
    )
    monkeypatch.setattr(common_mod, "read_crop_food_demand", lambda context: demand_data)
    monkeypatch.setattr(common_mod, "read_crop_input_data", lambda context: supply_data)
    monkeypatch.setattr(fd_mod, "read_crop_food_demand", common_mod.read_crop_food_demand)
    monkeypatch.setattr(fd_mod, "read_crop_input_data", common_mod.read_crop_input_data)

    mp = test_context.get_platform()
    scen = Scenario(mp, model="crop-demand-solve-test", scenario="testcrop", version="new")
    scen.add_horizon(year=[2015, firstyear], firstmodelyear=firstyear)
    scen.add_set("node", ["World", node])
    scen.add_set("commodity", ["testcrop_yield"])
    scen.add_set("level", ["crop_yield"])
    scen.add_set("mode", ["M1"])
    scen.add_set("technology", ["crop_producer", "food_import_testcrop"])
    scen.add_par("interestrate", [firstyear], value=0.05, unit="-")

    for unit in ["kt", "USD/kt", "-", "y", "%"]:
        if unit not in mp.units():
            mp.add_unit(unit)

    test_context.regions = "TESTCROPDEM"
    test_context.time = "year"
    test_context.type_reg = "global"
    test_context.valid_basins = {basin}
    test_context.set_scenario(scen)

    monkeypatch.setattr(type(test_context), "get_scenario", lambda self: scen)

    # A simple stub technology able to produce as much testcrop_yield as
    # needed, at a small positive cost -- isolates "does the demand par
    # itself solve/bind" from the real (already-tested) production chain.
    scen.add_par(
        "output",
        pd.DataFrame(
            {
                "node_loc": [node], "technology": ["crop_producer"],
                "year_vtg": [firstyear], "year_act": [firstyear], "mode": ["M1"],
                "node_dest": [node], "commodity": ["testcrop_yield"], "level": ["crop_yield"],
                "time": ["year"], "time_dest": ["year"], "value": [1.0], "unit": ["kt"],
            }
        ),
    )
    scen.add_par(
        "technical_lifetime",
        pd.DataFrame(
            {"node_loc": [node], "technology": ["crop_producer"], "year_vtg": [firstyear], "value": [10], "unit": ["y"]}
        ),
    )
    scen.add_par(
        "capacity_factor",
        pd.DataFrame(
            {
                "node_loc": [node], "technology": ["crop_producer"], "year_vtg": [firstyear],
                "year_act": [firstyear], "time": ["year"], "value": [1.0], "unit": ["%"],
            }
        ),
    )
    scen.add_par(
        "var_cost",
        pd.DataFrame(
            {
                "node_loc": [node], "technology": ["crop_producer"], "year_vtg": [firstyear],
                "year_act": [firstyear], "mode": ["M1"], "time": ["year"], "value": [1e-6], "unit": ["USD/kt"],
            }
        ),
    )

    result = fd_mod.add_crop_demand(test_context)
    assert "demand" in result, "add_crop_demand() returned no demand data for this synthetic setup"
    # Commit EVERY par add_crop_demand() returns (demand, output,
    # technical_lifetime, construction_time, capacity_factor, var_cost) --
    # not just "demand". A real build commits all of them via add_par_data();
    # committing only "demand" here previously let a real bug in the
    # food_import_<crop> `output` par (wrong same_node()/same_time() pipe
    # order -> NaN time_dest -> JDBC ClassCastException) pass this test
    # while still breaking a real build.
    for par_name, df in result.items():
        scen.add_par(par_name, df)

    scen.commit("pytest crop-demand solve check")
    scen.solve(model="MESSAGE", solve_options={"lpmethod": "1"})

    act = scen.var("ACT", filters={"technology": ["crop_producer"], "year_act": [firstyear]})
    print(act)
    assert not act.empty, "crop_producer never activated -- demand did not bind"
    assert act["lvl"].sum() == pytest.approx(100.0, abs=1e-4), (
        "expected exactly enough activity to meet the 100 kt demand"
    )

    # The backstop is present (committed above) but should be unused: real
    # production (var_cost 1e-6) is vastly cheaper than the import backstop
    # (FOOD_IMPORT_VAR_COST, 10,000 USD/kt) and uncapacitated here, so the
    # solver should never touch it.
    import_act = scen.var(
        "ACT", filters={"technology": ["food_import_testcrop"], "year_act": [firstyear]}
    )
    print(import_act)
    assert import_act.empty or import_act["lvl"].sum() == pytest.approx(0.0, abs=1e-6), (
        "food_import backstop should be unused when real production can fully meet demand"
    )
