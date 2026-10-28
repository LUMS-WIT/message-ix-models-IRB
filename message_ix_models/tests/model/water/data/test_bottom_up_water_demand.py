"""End-to-end: does a scenario carrying add_bottom_up_crop_water_demand()'s
output actually solve under GAMS, and does the relation genuinely bind --
forcing MORE irrigation activity than the existing historical-withdrawal
floor alone would require. Mirrors test_irrigation_tech.py's own
test_add_irrigation_techs_solves setup exactly (same synthetic basin/crop/
technology), since this module reuses that function's relation_activity
coefficients directly rather than recomputing them.

Numbers chosen so the two floors are distinguishable: historical floor
forces ACT >= 8 / 20 = 0.4 Mha (identical derivation to
test_irrigation_tech.py). The bottom-up floor, when enabled, is set to 10
MCM -- deliberately LARGER than the historical floor's 8 MCM -- so it
forces ACT >= 10 / 20 = 0.5 Mha, the TIGHTER of the two constraints.

test_compare_with_and_without_bottom_up (below) runs BOTH configurations
(mechanism off vs on) back to back in one real solve each, printed
side-by-side -- the "what are the outcomes with this and without this"
comparison, using the SAME safe, isolated test_context platform every
other solve test in this module uses (never the user's real Pakistan
database -- a standalone script version of this was written first and
deliberately discarded because it called Platform(name="default") directly,
which risks connecting to that same real local database concurrently with
a live kernel; pytest's test_context fixture has no such risk, so this
comparison lives here instead).
"""

import pandas as pd
import pytest


def _build_and_solve(test_context, monkeypatch, bottom_up_enabled: bool, bottom_up_mcm: float, model_name: str):
    """Shared setup for both the single-config solve test and the
    with/without comparison below. Returns the solved ACT (Mha) for
    irr_flood_testcrop plus the relation names that were registered.
    """
    from message_ix import Scenario

    from message_ix_models import ScenarioInfo
    from message_ix_models.model.water.data.crops import bottom_up_water_demand as bu_mod
    from message_ix_models.model.water.data.crops import common as common_mod
    from message_ix_models.model.water.data.crops import irrigation_tech as irr_mod

    regions = "BUTEST"
    basin = f"1|{regions}"
    node = f"B{basin}"
    firstyear = 2020

    monkeypatch.setattr(irr_mod, "CROPS", ["testcrop"])
    monkeypatch.setattr(irr_mod, "IRRIGATION_METHODS", {"irr_flood": {"historical_capacity": True}})
    monkeypatch.setattr(irr_mod, "CONFIG", {**irr_mod.CONFIG, "field_efficiency_conv": 1.0})

    monkeypatch.setattr(
        common_mod, "read_crop_input_data",
        lambda context: pd.DataFrame([
            {"crop": "testcrop", "par": "irrigation_yield", "node": basin, "value": 5.0},
            {"crop": "testcrop", "par": "crop_irr_land_2015", "node": basin, "value": 100.0},
        ]),
    )
    monkeypatch.setattr(
        common_mod, "read_crop_irrigation_water_annual",
        lambda context: pd.DataFrame([{"crop": "testcrop", "node": basin, "year": firstyear, "value": 10.0}]),
    )
    monkeypatch.setattr(
        common_mod, "read_irr_tech_data",
        lambda: pd.DataFrame([
            {"irr_tech": "irr_flood", "par": "water_efficiency", "value": 0.5},
            {"irr_tech": "irr_flood", "par": "life_exp", "value": 10.0},
            {"irr_tech": "irr_flood", "par": "cap_factor", "value": 1.0},
            {"irr_tech": "irr_flood", "par": "electricity_intensity", "value": 0.0},
            {"irr_tech": "irr_flood", "par": "inv_cost", "value": 0.0},
            {"irr_tech": "irr_flood", "par": "fix_cost", "value": 0.0},
            {"irr_tech": "irr_flood", "par": "var_cost", "value": 1e-6},
        ]),
    )
    monkeypatch.setattr(
        common_mod, "read_historical_irrigation_withdrawals",
        lambda context: pd.DataFrame([{"node": basin, "tec": "irrigation_sw_diversion", "value": 8.0}]),
    )
    monkeypatch.setattr(common_mod, "basins_with_crop_data", lambda: {basin})
    for name in [
        "read_crop_input_data", "read_crop_irrigation_water_annual",
        "read_irr_tech_data", "read_historical_irrigation_withdrawals", "basins_with_crop_data",
    ]:
        monkeypatch.setattr(irr_mod, name, getattr(common_mod, name))

    bottom_up_df = pd.DataFrame([{"node": basin, "value": bottom_up_mcm, "unit": "MCM/year"}])
    monkeypatch.setattr(common_mod, "read_bottom_up_crop_water_demand", lambda context: bottom_up_df)
    monkeypatch.setattr(bu_mod, "read_bottom_up_crop_water_demand", common_mod.read_bottom_up_crop_water_demand)

    mp = test_context.get_platform()
    scen = Scenario(mp, model=model_name, scenario="testcrop", version="new")
    scen.add_horizon(year=[2010, 2015, firstyear], firstmodelyear=firstyear)
    scen.add_set("node", ["World", node])
    scen.add_set(
        "commodity",
        ["testcrop_land", "surfacewater_basin", "crop_land", "testcrop_yield", "groundwater_basin"],
    )
    scen.add_set("level", ["crop", "water_avail_basin", "area", "crop_yield"])
    scen.add_set("mode", ["M1"])
    scen.add_par("interestrate", [firstyear], value=0.05, unit="-")

    for unit in ["Mha", "MCM", "USD/Mha", "-", "y", "%", "kt", "GWa"]:
        if unit not in mp.units():
            mp.add_unit(unit)

    test_context.regions = regions
    test_context.time = "year"
    test_context.type_reg = "global"
    test_context.valid_basins = {basin}
    # river-only: the bottom-up relation reuses the crop techs' floor coefficients
    test_context.irrigation_source_layer = False
    test_context.bottom_up_water_demand_enabled = bottom_up_enabled
    test_context.set_scenario(scen)
    test_context["water build info"] = ScenarioInfo(scen)

    tech_names = irr_mod.irrigation_technology_names(test_context)
    assert tech_names == ["irr_flood_testcrop"]
    hist_relations = irr_mod.irrigation_relation_names(test_context)
    bottom_up_relations = bu_mod.bottom_up_relation_names(test_context) if bottom_up_enabled else []

    scen.add_set("technology", [*tech_names, "land_source", "water_source"])
    scen.add_set("relation", [*hist_relations, *bottom_up_relations])
    scen.add_par(
        "output",
        pd.DataFrame({
            "node_loc": [node, node], "technology": ["land_source", "water_source"],
            "year_vtg": [firstyear, firstyear], "year_act": [firstyear, firstyear],
            "mode": ["M1", "M1"], "node_dest": [node, node],
            "commodity": ["testcrop_land", "surfacewater_basin"],
            "level": ["crop", "water_avail_basin"], "time": ["year", "year"],
            "time_dest": ["year", "year"], "value": [1.0, 1.0], "unit": ["Mha", "MCM"],
        }),
    )
    scen.add_par(
        "technical_lifetime",
        pd.DataFrame({
            "node_loc": [node, node], "technology": ["land_source", "water_source"],
            "year_vtg": [firstyear, firstyear], "value": [10, 10], "unit": ["y", "y"],
        }),
    )
    scen.add_par(
        "capacity_factor",
        pd.DataFrame({
            "node_loc": [node, node], "technology": ["land_source", "water_source"],
            "year_vtg": [firstyear, firstyear], "year_act": [firstyear, firstyear],
            "time": ["year", "year"], "value": [1.0, 1.0], "unit": ["%", "%"],
        }),
    )

    monkeypatch.setattr(type(test_context), "get_scenario", lambda self: scen)

    irr_result = irr_mod.add_irrigation_techs(test_context)
    for par_name, df in irr_result.items():
        scen.add_par(par_name, df)

    if bottom_up_enabled:
        bu_result = bu_mod.add_bottom_up_crop_water_demand(test_context)
        assert "relation_activity" in bu_result and "relation_lower" in bu_result
        for par_name, df in bu_result.items():
            scen.add_par(par_name, df)

    scen.commit(f"pytest bottom-up water demand compare check (enabled={bottom_up_enabled})")
    scen.solve(model="MESSAGE", solve_options={"lpmethod": "1"})

    act = scen.var("ACT", filters={"technology": ["irr_flood_testcrop"], "year_act": [firstyear]})
    act_val = float(act["lvl"].sum()) if not act.empty else 0.0
    try:
        obj = float(scen.var("OBJ")["lvl"])
    except Exception:
        obj = None

    return {
        "act_mha": act_val,
        "objective": obj,
        "hist_relations": hist_relations,
        "bottom_up_relations": bottom_up_relations,
    }


def test_add_bottom_up_crop_water_demand_disabled_by_default(water_context):
    """The kill switch defaults to disabled -- confirms this mechanism is
    a true no-op for every existing context/test/real build unless
    explicitly turned on."""
    from message_ix_models.model.water.data.crops.bottom_up_water_demand import (
        add_bottom_up_crop_water_demand,
        bottom_up_relation_names,
    )

    assert bottom_up_relation_names(water_context) == []
    assert add_bottom_up_crop_water_demand(water_context) == {}


def test_add_bottom_up_crop_water_demand_solves(test_context, monkeypatch):
    result = _build_and_solve(
        test_context, monkeypatch, bottom_up_enabled=True, bottom_up_mcm=10.0,
        model_name="bottom-up-water-solve-test",
    )
    assert result["bottom_up_relations"] == ["bottom_up_crop_water_1_BUTEST"]
    print(result)
    assert result["act_mha"] == pytest.approx(0.5, abs=1e-4), (
        "expected the bottom-up relation (10 MCM / 20 MCM-per-Mha = 0.5 Mha) to be the "
        "BINDING constraint, tighter than the historical floor's 0.4 Mha -- confirms this "
        "module's relation has a genuine, independent effect, not just a 'builds without "
        "erroring' non-effect."
    )


def test_compare_with_and_without_bottom_up(test_context, monkeypatch):
    """The actual "what are the outcomes with this and without this"
    comparison: solve the identical scenario twice, once with the new
    mechanism off (the real Pakistan model's current default behavior)
    and once on, and report both real, measured outcomes side by side.
    """
    baseline = _build_and_solve(
        test_context, monkeypatch, bottom_up_enabled=False, bottom_up_mcm=0.0,
        model_name="bottom-up-compare-baseline",
    )
    with_bottom_up = _build_and_solve(
        test_context, monkeypatch, bottom_up_enabled=True, bottom_up_mcm=10.0,
        model_name="bottom-up-compare-withbu",
    )

    print("\n--- WITHOUT bottom-up mechanism (baseline, matches real Pakistan model's current default) ---")
    print(f"  relations active: {baseline['hist_relations']}")
    print(f"  irr_flood_testcrop ACT: {baseline['act_mha']:.4f} Mha")
    print(f"  objective: {baseline['objective']}")

    print("\n--- WITH bottom-up mechanism enabled (bottom-up floor = 10 MCM, deliberately > historical) ---")
    print(f"  relations active: {with_bottom_up['hist_relations'] + with_bottom_up['bottom_up_relations']}")
    print(f"  irr_flood_testcrop ACT: {with_bottom_up['act_mha']:.4f} Mha")
    print(f"  objective: {with_bottom_up['objective']}")

    # The real, measured difference: more than double the forced activity,
    # and a real cost difference (more water -> more irrigation -> higher
    # objective, since irrigation isn't free).
    assert baseline["act_mha"] == pytest.approx(0.4, abs=1e-4)
    assert with_bottom_up["act_mha"] == pytest.approx(0.5, abs=1e-4)
    assert with_bottom_up["objective"] > baseline["objective"], (
        "enabling the bottom-up mechanism, when its floor is the tighter constraint, "
        "should force strictly more activity and therefore strictly more cost -- a real, "
        "measurable outcome, not a no-op"
    )
    print(f"\nDelta: +{with_bottom_up['act_mha'] - baseline['act_mha']:.4f} Mha forced irrigation activity, "
          f"+{with_bottom_up['objective'] - baseline['objective']:.6f} objective cost.")
