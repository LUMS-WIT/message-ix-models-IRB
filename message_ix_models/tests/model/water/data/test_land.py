import pandas as pd
import pytest

from message_ix_models.model.water.data.crops.common import CROPS, crop_land_commodity, crop_tech_name
from message_ix_models.model.water.data.crops.land import (
    add_crop_land_techs,
    crop_commodity_names,
    crop_technology_names,
)
from message_ix_models.tests.model.water.conftest import water_params


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_crop_technology_names(water_context):
    """All 9 crop_<crop> technologies are registered for the real IRB
    context (whose valid_basins genuinely overlap the real crop data)."""
    names = crop_technology_names(water_context)
    assert set(names) == {crop_tech_name(c) for c in CROPS}


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "R12"}],
    indirect=True,
)
def test_crop_technology_names_unrelated_region_is_empty(water_context):
    """A region whose basin naming doesn't overlap the real (IRB-specific)
    crop data gets zero technologies, not 9 technologies with no real data
    behind them -- the crop CSVs aren't region-suffixed like
    basin_links_<regions>.csv, so this guard is what keeps the module a
    safe no-op outside IRB."""
    assert crop_technology_names(water_context) == []
    assert add_crop_land_techs(water_context) == {}


def test_crop_commodity_names():
    assert set(crop_commodity_names()) == {crop_land_commodity(c) for c in CROPS}


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_add_crop_land_techs(water_context, water_scenario, assert_message_params):
    """add_crop_land_techs() returns well-formed MESSAGE parameter data,
    using the real, committed crop data."""
    result = add_crop_land_techs(context=water_context)

    assert_message_params(
        result,
        expected_keys=[
            "output",
            "technical_lifetime",
            "construction_time",
            "capacity_factor",
            "inv_cost",
            "fix_cost",
            "var_cost",
        ],
    )

    # No "input" at all -- machinery energy deliberately dropped from scope
    # (see land.py module docstring); this technology is capacity-bounded
    # purely by historical_new_capacity.
    assert "input" not in result

    # Every crop's technology produces exactly its own <crop>_land
    # commodity at level 'crop', basin-local (node_loc == node_dest).
    out = result["output"]
    assert set(out["level"].unique()) == {"crop"}
    for crop in CROPS:
        tech = crop_tech_name(crop)
        rows = out[out["technology"] == tech]
        assert not rows.empty, f"no output rows for {tech}"
        assert set(rows["commodity"]) == {crop_land_commodity(crop)}
    assert (out["node_loc"] == out["node_dest"]).all()

    # historical_new_capacity, where present, is strictly positive (zero/
    # missing capacity rows are dropped, not written as explicit zeros).
    if "historical_new_capacity" in result:
        assert (result["historical_new_capacity"]["value"] > 0).all()

    # Phase 4: real fertilizer emission_factor, using the real committed
    # data (emission_factor_enabled defaults to True in config.yaml).
    assert "emission_factor" in result
    ef = result["emission_factor"]
    assert (ef["value"] > 0).all()
    assert set(ef["emission"]) == {"CO2eq"}
    assert set(ef["mode"]) == {"M1"}
    # At least one real Pakistan basin has an emission_factor for wheat --
    # a concrete, non-vacuous coverage check, not just "some rows exist".
    wheat_pak = ef[
        (ef["technology"] == crop_tech_name("wheat"))
        & ef["node_loc"].str.endswith("|PAKISTAN")
    ]
    assert not wheat_pak.empty


def test_add_crop_land_techs_no_valid_basins(water_context, water_scenario):
    """No valid_basins at all (e.g. basin filtering excluded everything) ->
    empty dict, not an error."""
    water_context.valid_basins = set()
    assert add_crop_land_techs(water_context) == {}


def test_add_crop_land_techs_solves(test_context, monkeypatch):
    """End-to-end: the generated data actually solves under GAMS, and CAP
    (from historical_new_capacity) genuinely bounds ACT -- not just a
    DataFrame-shape check.

    Mirrors test_network.py's test_add_network_techs_solves, which caught a
    real bug (broadcast(**kwargs) string-explosion) the shape-only tests
    did not. crop.py has *more* per-crop loop iterations than network.py's
    per-link-type loop, so this is the top implementation risk here too
    (see land.py module docstring).

    Real design (see land.py's module docstring, "Annual (lft=1) vintaging
    dropped"): the full, natural get_vintage_and_active_years() result --
    a real historical vintage (2015, pre-model in this test's horizon)
    plus a legitimate new-build vintage at the model-horizon year (2020).
    The model should prefer the real, already-paid-for historical
    capacity over paying to build new capacity when the historical
    capacity alone already meets demand.
    """
    from message_ix import Scenario

    from message_ix_models.model.water.data.crops import land as land_mod
    from message_ix_models.model.water.data.crops import common as common_mod

    regions = "CROPTEST"
    basin = f"1|{regions}"
    node = f"B{basin}"
    firstyear = 2020

    monkeypatch.setattr(land_mod, "CROPS", ["testcrop"])
    monkeypatch.setattr(
        land_mod,
        "CONFIG",
        {**land_mod.CONFIG, "crop_technical_lifetime": 10},
    )

    crop_input = pd.DataFrame(
        [
            {"crop": "testcrop", "par": "crop_irr_land_2015", "node": basin, "time": "year", "unit": "Mha", "value": 6.0},
            {"crop": "testcrop", "par": "crop_rainfed_land_2015", "node": basin, "time": "year", "unit": "Mha", "value": 4.0},
        ]
    )
    crop_tech = pd.DataFrame(
        [
            {"crop": "testcrop", "par": "inv_cost", "time": "year", "unit": "USD per ha", "value": 10.0},
            {"crop": "testcrop", "par": "fix_cost", "time": "year", "unit": "USD per ha", "value": 1.0},
            {"crop": "testcrop", "par": "var_cost", "time": "year", "unit": "USD per ha", "value": 0.5},
        ]
    )
    real_package_data_path = common_mod.package_data_path

    def fake_package_data_path(*parts):
        if parts == ("water", "crops", "crop_input_data_IRB.csv"):
            path = tmp_path_file(crop_input)
            return path
        if parts == ("water", "crops", "crop_tech_data_IRB.csv"):
            return tmp_path_file(crop_tech)
        return real_package_data_path(*parts)

    # Small helper: write a DataFrame once per call, cached by object id so
    # repeated reads within one test don't rewrite the file every time.
    import tempfile
    from pathlib import Path

    _written: dict[int, Path] = {}

    def tmp_path_file(df: pd.DataFrame) -> Path:
        key = id(df)
        if key not in _written:
            p = Path(tempfile.mkstemp(suffix=".csv")[1])
            df.to_csv(p, index=False)
            _written[key] = p
        return _written[key]

    monkeypatch.setattr(common_mod, "package_data_path", fake_package_data_path)

    mp = test_context.get_platform()
    scen = Scenario(mp, model="crop-land-solve-test", scenario="testcrop", version="new")
    scen.add_horizon(year=[2010, 2015, firstyear], firstmodelyear=firstyear)
    scen.add_set("node", ["World", node])
    scen.add_set("commodity", ["testcrop_land"])
    scen.add_set("level", ["crop"])
    scen.add_set("mode", ["M1"])
    scen.add_par("interestrate", [firstyear], value=0.05, unit="-")

    for unit in ["Mha", "USD/Mha", "-", "y", "%"]:
        if unit not in mp.units():
            mp.add_unit(unit)

    test_context.regions = regions
    test_context.time = "year"
    test_context.type_reg = "global"
    test_context.valid_basins = {basin}
    test_context.set_scenario(scen)
    from message_ix_models import ScenarioInfo

    test_context["water build info"] = ScenarioInfo(scen)

    tech_names = land_mod.crop_technology_names(test_context)
    assert tech_names == ["crop_testcrop"]
    scen.add_set("technology", tech_names)

    scen.add_par(
        "demand",
        pd.DataFrame(
            {
                "node": [node],
                "commodity": ["testcrop_land"],
                "level": ["crop"],
                "year": [firstyear],
                "time": ["year"],
                "value": [10.0],
                "unit": ["Mha"],
            }
        ),
    )

    monkeypatch.setattr(type(test_context), "get_scenario", lambda self: scen)

    result = land_mod.add_crop_land_techs(test_context)
    for par_name, df in result.items():
        scen.add_par(par_name, df)

    scen.commit("pytest crop-land solve check")
    scen.solve(model="MESSAGE", solve_options={"lpmethod": "1"})

    act = scen.var("ACT", filters={"technology": ["crop_testcrop"], "year_act": [firstyear]})
    # Total historical land (6 + 4 = 10 Mha) exactly meets the 10 Mha
    # demand -- CAP is fully utilized, ACT sums to 10.0 exactly. GAMS
    # enumerates one ACT row per valid (year_vtg, year_act) pair -- both
    # the real 2015 historical vintage and the 2020 new-build option --
    # even though only one is actually used. The real, already-funded
    # 2015 vintage supplies the full 10.0 Mha; the 2020 new-build option
    # sits at 0 (confirms the model prefers real historical capacity over
    # paying inv_cost to build unnecessary new capacity -- the cost data
    # is wired in and actually influences the solution).
    assert act["lvl"].sum() == pytest.approx(10.0, abs=1e-4)
    used = act[act["lvl"] > 1e-6]
    assert len(used) == 1
    assert used["year_vtg"].iloc[0] == 2015


def test_cost_unit_conversion():
    """Costs stay numerically equal to the source's USD/ha: in the model's
    cost unit (million USD) that is million USD/Mha -- see land.py module
    docstring. Sanity-checks against the real committed
    crop_tech_data_IRB.csv rather than a synthetic fixture, so a change to
    the source data or the conversion constant surfaces here."""
    from message_ix_models.model.water.data.crops.common import read_crop_tech_data
    from message_ix_models.model.water.data.crops.irrigation_tech import _HA_TO_MHA_COST as IRR
    from message_ix_models.model.water.data.crops.land import _HA_TO_MHA_COST

    assert _HA_TO_MHA_COST == IRR == pytest.approx(1.0)
    df = read_crop_tech_data()
    wheat_inv = df[(df["crop"] == "wheat") & (df["par"] == "inv_cost")]["value"].iloc[0]
    assert wheat_inv > 0
