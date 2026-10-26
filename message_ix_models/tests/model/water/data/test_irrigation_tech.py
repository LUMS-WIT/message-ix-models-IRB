import pandas as pd
import pytest

from message_ix_models.model.water.data.crops.common import CROPS, IRRIGATION_METHODS, crop_yield_commodity
from message_ix_models.model.water.data.crops.irrigation_tech import (
    add_irrigation_techs,
    irrigation_commodity_names,
    irrigation_relation_names,
    irrigation_technology_names,
)
from message_ix_models.model.water.data.crops.rainfed import GENERIC_LAND_COMMODITY
from message_ix_models.tests.model.water.conftest import water_params


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_irrigation_technology_names(water_context):
    names = irrigation_technology_names(water_context)
    assert len(names) == len(IRRIGATION_METHODS) * len(CROPS)
    assert "irr_flood_wheat" in names
    assert "irr_drip_rice" in names


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "R12"}],
    indirect=True,
)
def test_irrigation_technology_names_unrelated_region_is_empty(water_context):
    assert irrigation_technology_names(water_context) == []
    assert irrigation_relation_names(water_context) == []


def test_irrigation_commodity_names():
    names = irrigation_commodity_names()
    assert GENERIC_LAND_COMMODITY in names
    assert crop_yield_commodity("wheat") in names


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_irrigation_relation_names(water_context):
    names = irrigation_relation_names(water_context)
    assert names, "expected at least one basin relation for the real IRB context"
    assert all(n.startswith("hist_irr_withdrawal_") for n in names)
    assert len(names) == len(set(names)), "relation names must be unique per basin"


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_add_irrigation_techs(water_context, water_scenario, assert_message_params):
    """add_irrigation_techs() returns well-formed MESSAGE parameter data,
    using the real, committed crop/irrigation data -- including the
    demand-pull relation."""
    result = add_irrigation_techs(context=water_context)

    assert_message_params(
        result,
        expected_keys=[
            "input",
            "output",
            "technical_lifetime",
            "construction_time",
            "capacity_factor",
            "inv_cost",
            "fix_cost",
            "var_cost",
            "relation_activity",
            "relation_lower",
        ],
    )

    inp = result["input"]
    out = result["output"]

    # Every present technology draws crop land + freshwater at minimum;
    # electricity only for methods with nonzero electricity_intensity
    # (irr_flood/irr_canal_lining_flood are gravity-fed, zero electricity).
    present = sorted(inp["technology"].unique())
    assert present, "expected at least one irr_<method>_<crop> technology with real data"
    for tech in present:
        rows = inp[inp["technology"] == tech]
        commodities = set(rows["commodity"])
        assert commodities >= {"surfacewater_basin"}
        assert any(c.endswith("_land") for c in commodities)

    # Freshwater input reuses the existing basin-tier commodity/level --
    # not the region-tier freshwater/water_supply pair the old simplified
    # irrigation technologies use (the naming-reconciliation requirement).
    water_rows = inp[inp["commodity"] == "surfacewater_basin"]
    assert set(water_rows["level"]) == {"water_avail_basin"}
    assert (water_rows["value"] > 0).all()

    # Electricity, where present, reuses the existing electr/final pair.
    elec_rows = inp[inp["commodity"] == "electr"]
    if not elec_rows.empty:
        assert set(elec_rows["level"]) == {"final"}
        assert (elec_rows["value"] > 0).all()

    # Groundwater recharge output reuses the existing groundwater_basin
    # commodity (feeding the existing gw_recharge technology) -- no new
    # commodity invented for irrigation losses.
    gw_rows = out[out["commodity"] == "groundwater_basin"]
    assert not gw_rows.empty
    assert set(gw_rows["level"]) == {"water_avail_basin"}
    assert (gw_rows["value"] >= 0).all()

    # Basin-local: node_loc == node_origin (input) / node_dest (output).
    assert (inp["node_loc"] == inp["node_origin"]).all()
    assert (out["node_loc"] == out["node_dest"]).all()

    # Only irr_flood_<crop> gets a historical capacity bound (config's
    # historical_capacity flag) -- irr_drip_<crop> etc. never do (see
    # module docstring: only the historically-seeded method can ever have
    # real capacity in this port).
    if "historical_new_capacity" in result:
        hist_techs = set(result["historical_new_capacity"]["technology"].unique())
        assert all(t.startswith("irr_flood_") for t in hist_techs)

    # relation_activity coefficients equal the real per-technology
    # withdrawal rate (same value already used in the water input row for
    # that technology/node/year) -- not an arbitrary placeholder.
    rel_act = result["relation_activity"]
    assert (rel_act["value"] > 0).all()
    assert set(rel_act["mode"]) == {"M1"}

    # relation_lower is a real, positive historical floor that now grows
    # over the horizon -- scaled per year by a population/GDP demand-growth
    # index (see build_irrigation_demand_growth_IRB.py) instead of staying
    # flat at the historical baseline for every future year. Non-decreasing
    # across years for a given basin (SSP2 population/GDP are both
    # increasing over this horizon), not merely a single repeated value.
    rel_lo = result["relation_lower"]
    assert (rel_lo["value"] > 0).all()
    for relation, group in rel_lo.groupby("relation"):
        ordered = group.sort_values("year_rel")["value"]
        assert (ordered.diff().dropna() >= 0).all(), (
            f"{relation}: floor should not decrease across years"
        )


def test_add_irrigation_techs_no_valid_basins(water_context, water_scenario):
    water_context.valid_basins = set()
    assert add_irrigation_techs(water_context) == {}


def test_add_irrigation_techs_solves(test_context, monkeypatch):
    """End-to-end: the generated data (including the demand-pull relation)
    actually solves under GAMS, and the relation genuinely forces nonzero
    irrigation activity -- not just a DataFrame-shape check.

    This is the test that matters most for this module: the whole point of
    the demand-pull relation is to avoid the network module's zero-effect
    outcome (real data, real technologies, but nothing ever uses them). A
    real GAMS solve is the only way to confirm the relation actually binds.

    Numbers chosen so the answer is exact and hand-verifiable: water
    requirement 10 MCM/Mha, water_efficiency 0.5, field_efficiency_conv
    monkeypatched to 1.0 (no extra field loss) -> conveyance 0.5 ->
    withdrawal rate 20 MCM/Mha. Historical floor 8 MCM (no relaxation
    factor) -> relation forces ACT >= 8/20 = 0.4 Mha. This basin's
    ``region_group`` ("IRRTEST") has no entry in the demand-growth-index
    CSV, so the growth-index multiplier correctly falls back to 1.0 (no
    additional scaling) -- confirmed by the result landing on exactly
    8/20 = 0.4, not some other value. A small positive var_cost makes the
    LP minimize ACT to exactly that floor (zero cost would leave it
    degenerate -- any ACT >= 0.4 optimal).
    """
    from message_ix import Scenario

    from message_ix_models import ScenarioInfo
    from message_ix_models.model.water.data.crops import common as common_mod
    from message_ix_models.model.water.data.crops import irrigation_tech as irr_mod

    regions = "IRRTEST"
    basin = f"1|{regions}"
    node = f"B{basin}"
    firstyear = 2020

    monkeypatch.setattr(irr_mod, "CROPS", ["testcrop"])
    monkeypatch.setattr(irr_mod, "IRRIGATION_METHODS", {"irr_flood": {"historical_capacity": True}})
    monkeypatch.setattr(irr_mod, "CONFIG", {**irr_mod.CONFIG, "field_efficiency_conv": 1.0})

    monkeypatch.setattr(
        common_mod,
        "read_crop_input_data",
        lambda context: pd.DataFrame(
            [
                {"crop": "testcrop", "par": "irrigation_yield", "node": basin, "value": 5.0},
                {"crop": "testcrop", "par": "crop_irr_land_2015", "node": basin, "value": 100.0},
            ]
        ),
    )
    monkeypatch.setattr(
        common_mod,
        "read_crop_irrigation_water_annual",
        lambda context: pd.DataFrame(
            [{"crop": "testcrop", "node": basin, "year": firstyear, "value": 10.0}]
        ),
    )
    monkeypatch.setattr(
        common_mod,
        "read_irr_tech_data",
        lambda: pd.DataFrame(
            [
                {"irr_tech": "irr_flood", "par": "water_efficiency", "value": 0.5},
                {"irr_tech": "irr_flood", "par": "life_exp", "value": 10.0},
                {"irr_tech": "irr_flood", "par": "cap_factor", "value": 1.0},
                {"irr_tech": "irr_flood", "par": "electricity_intensity", "value": 0.0},
                {"irr_tech": "irr_flood", "par": "inv_cost", "value": 0.0},
                {"irr_tech": "irr_flood", "par": "fix_cost", "value": 0.0},
                {"irr_tech": "irr_flood", "par": "var_cost", "value": 1e-6},  # USD/ha -> tiny after x1e6
            ]
        ),
    )
    monkeypatch.setattr(
        common_mod,
        "read_historical_irrigation_withdrawals",
        lambda context: pd.DataFrame([{"node": basin, "tec": "irrigation_sw_diversion", "value": 8.0}]),
    )
    monkeypatch.setattr(common_mod, "basins_with_crop_data", lambda: {basin})
    # irrigation_tech.py imported these read_* functions directly into its
    # own namespace -- patching common_mod alone doesn't redirect calls
    # already bound to irr_mod's names.
    for name in [
        "read_crop_input_data",
        "read_crop_irrigation_water_annual",
        "read_irr_tech_data",
        "read_historical_irrigation_withdrawals",
        "basins_with_crop_data",
    ]:
        monkeypatch.setattr(irr_mod, name, getattr(common_mod, name))

    mp = test_context.get_platform()
    scen = Scenario(mp, model="irr-tech-solve-test", scenario="testcrop", version="new")
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
    test_context.set_scenario(scen)
    test_context["water build info"] = ScenarioInfo(scen)

    tech_names = irr_mod.irrigation_technology_names(test_context)
    assert tech_names == ["irr_flood_testcrop"]
    relation_names = irr_mod.irrigation_relation_names(test_context)
    assert relation_names == [f"hist_irr_withdrawal_1_{regions}"]

    # Free (zero-cost, uncapacitated) technologies supplying testcrop_land
    # and surfacewater_basin -- this test is isolating irr_flood_testcrop's
    # own behavior (does the relation force real activity backed by real
    # capacity?), not exercising land.py or a real water-supply chain. Both
    # sources are required: irr_flood_testcrop's own real historical
    # capacity (from crop_irr_land_2015) only ever covers *its own*
    # capacity -- it still needs something else to supply the
    # testcrop_land and surfacewater_basin commodities it consumes as
    # input, exactly as it would in the real scenario (there, land.py's
    # crop_<crop> and the basin's real surface-water balance play this
    # role; see irrigation_tech.py's module docstring for how this test
    # setup's earlier lack of a real water source was accidentally masked
    # by the "phantom vintage" bug -- a phantom vintage needed no input at
    # all, so the missing water source went unnoticed until that bug was
    # fixed).
    scen.add_set("technology", [*tech_names, "land_source", "water_source"])
    scen.add_set("relation", relation_names)
    scen.add_par(
        "output",
        pd.DataFrame(
            {
                "node_loc": [node, node], "technology": ["land_source", "water_source"],
                "year_vtg": [firstyear, firstyear], "year_act": [firstyear, firstyear],
                "mode": ["M1", "M1"], "node_dest": [node, node],
                "commodity": ["testcrop_land", "surfacewater_basin"],
                "level": ["crop", "water_avail_basin"], "time": ["year", "year"],
                "time_dest": ["year", "year"], "value": [1.0, 1.0], "unit": ["Mha", "MCM"],
            }
        ),
    )
    # A technology needs technical_lifetime + capacity_factor for its CAP
    # to actually be buildable/usable -- without these, ACT stays pinned
    # at 0 regardless of cost or output data (caught via a manual
    # diagnostic script while developing this test: omitting them produces
    # an "ACTIVITY_BOUND_LO: 0 >= X" infeasibility that looks like a
    # relation/data bug but is actually just this).
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

    result = irr_mod.add_irrigation_techs(test_context)
    for par_name, df in result.items():
        scen.add_par(par_name, df)

    scen.commit("pytest irrigation-tech solve check")
    scen.solve(model="MESSAGE", solve_options={"lpmethod": "1"})

    act = scen.var(
        "ACT", filters={"technology": ["irr_flood_testcrop"], "year_act": [firstyear]}
    )
    assert act["lvl"].sum() == pytest.approx(0.4, abs=1e-4)


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_withdrawal_recharge_arithmetic_invariant(water_context, water_scenario):
    """withdrawal == raw crop water requirement + groundwater recharge, for
    every (technology, node, year) -- a real physical invariant, not just a
    DataFrame-shape check.

    This is a regression test for a real bug caught during development:
    recharge was initially computed by scaling the already-efficiency-
    scaled *withdrawal* value a second time, instead of the raw crop water
    requirement -- silently claiming e.g. 100% of withdrawn water becomes
    recharge (0% reaching the crop) at conveyance efficiency 0.5, instead
    of the correct 50% loss fraction. Caught only by checking this
    arithmetic relationship directly, not by shape/NaN/duplicate checks.
    """
    from message_ix_models.model.water.data.crops.common import CONFIG, read_irr_tech_data

    result = add_irrigation_techs(context=water_context)
    inp = result["input"]
    out = result["output"]

    water_in = inp[inp["commodity"] == "surfacewater_basin"][
        ["technology", "node_loc", "year_act", "value"]
    ].rename(columns={"value": "withdrawal"})
    gw_out = out[out["commodity"] == "groundwater_basin"][
        ["technology", "node_loc", "year_act", "value"]
    ].rename(columns={"value": "recharge"})

    merged = water_in.merge(gw_out, on=["technology", "node_loc", "year_act"], how="inner")
    assert not merged.empty

    # For every row, recharge / withdrawal == 1 - conveyance_efficiency
    # (the loss fraction) -- derive conveyance per technology from the
    # real irr_tech_data.csv rather than hardcoding a method's efficiency,
    # so this stays correct as the underlying data changes.
    irr_tech_df = read_irr_tech_data().set_index(["irr_tech", "par"])["value"]
    field_eff = CONFIG["field_efficiency_conv"]

    def method_of(tech: str) -> str:
        # technology name is "<method>_<crop>"; strip the trailing "_<crop>"
        for crop in CROPS:
            if tech.endswith(f"_{crop}"):
                return tech[: -(len(crop) + 1)]
        raise ValueError(tech)

    for row in merged.itertuples():
        method = method_of(row.technology)
        water_eff = irr_tech_df[(method, "water_efficiency")]
        conveyance = water_eff * field_eff
        # raw_requirement = withdrawal * conveyance;
        # recharge = raw_requirement * (1/conveyance - 1) = withdrawal * (1 - conveyance).
        assert row.recharge == pytest.approx(row.withdrawal * (1 - conveyance), rel=1e-6)
