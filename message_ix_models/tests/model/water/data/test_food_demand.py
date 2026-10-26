import pandas as pd
import pytest

from message_ix_models.model.water.data.crops import food_demand as fd_mod
from message_ix_models.model.water.data.crops.food_demand import add_crop_demand
from message_ix_models.tests.model.water.conftest import water_params


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_add_crop_demand(water_context, water_scenario, assert_message_params):
    """add_crop_demand() returns well-formed demand data, using the real,
    committed crop food-demand data (indus_demands_new.csv port)."""
    result = add_crop_demand(context=water_context)

    assert_message_params(result, expected_keys=["demand"])

    dmd = result["demand"]
    assert not dmd.empty, "expected at least one crop food-demand row for the real IRB context"
    # >= 0, not > 0: the real source data legitimately includes exact zeros
    # (e.g. cotton demand phasing out to 0 in some Afghan node by 2050) --
    # a real projection, not missing data.
    assert (dmd["value"] >= 0).all()
    assert set(dmd["level"]) == {"crop_yield"}
    assert all(c.endswith("_yield") for c in dmd["commodity"].unique())

    # Node ids carry the "B" prefix, matching the rest of this port's convention.
    assert all(n.startswith("B") for n in dmd["node"].unique())

    # No China coverage in the source data (indus_demands_new.csv is
    # AFG/IND/PAK only) -- confirm no Chinese basin ever got a row.
    assert not any("CHINA" in n for n in dmd["node"].unique())

    # Every year in the model horizon gets a row (nearest-year matched).
    info = water_context["water build info"]
    assert set(dmd["year"]) == set(info.Y)


def test_add_crop_demand_no_valid_basins(water_context):
    water_context.valid_basins = set()
    assert add_crop_demand(water_context) == {}


@pytest.mark.parametrize(
    "water_context",
    [{**water_params("R12"), "regions": "IRB"}],
    indirect=True,
)
def test_add_crop_demand_disabled(water_context):
    """context.crop_demand_enabled = False fully disables the mechanism --
    the per-run kill switch used to isolate the water-withdrawal floor
    (irrigation_tech.py) without touching this module's code."""
    water_context.crop_demand_enabled = False
    assert add_crop_demand(water_context) == {}
    assert fd_mod.food_import_technology_names(water_context) == []


def test_add_crop_demand_drops_unmatched_rows(monkeypatch):
    """Consistency guard: a demand row with no matching supply-side
    irrigation_yield data must be dropped, not silently kept as a
    guaranteed infeasibility (mirrors NEST's own anti-join,
    basin_msggdx_load_inputs.r ~line 407-420)."""

    class FakeInfo:
        Y = [2020]
        # get_vintage_and_active_years() reads this directly off `info` --
        # a single (year_vtg, year_act) combination is enough for this test.
        yv_ya = pd.DataFrame({"year_vtg": [2020], "year_act": [2020]})

    class FakeContext:
        valid_basins = {"1|TEST"}

        def get_scenario(self):
            return None  # unused -- ScenarioInfo is monkeypatched below

    demand_data = pd.DataFrame(
        [
            {"crop": "wheat", "node": "1|TEST", "year": 2020, "value": 100.0, "unit": "kt"},
            # rice has no matching supply-side yield data below -- must be dropped.
            {"crop": "rice", "node": "1|TEST", "year": 2020, "value": 50.0, "unit": "kt"},
        ]
    )
    supply_data = pd.DataFrame(
        [{"crop": "wheat", "par": "irrigation_yield", "node": "1|TEST", "value": 1000.0}]
    )

    monkeypatch.setattr(fd_mod, "read_crop_food_demand", lambda context: demand_data)
    monkeypatch.setattr(fd_mod, "read_crop_input_data", lambda context: supply_data)
    monkeypatch.setattr(fd_mod, "CROPS", ["wheat", "rice"])
    monkeypatch.setattr(fd_mod, "ScenarioInfo", lambda scen: FakeInfo())

    result = add_crop_demand(FakeContext())
    dmd = result["demand"]
    assert set(dmd["commodity"]) == {"wheat_yield"}
    assert dmd["node"].iloc[0] == "B1|TEST"
    assert dmd["value"].iloc[0] == 100.0


def test_add_crop_demand_empty_when_no_data(monkeypatch):
    class FakeContext:
        def get_scenario(self):
            raise AssertionError("should short-circuit before ever needing a scenario")

    monkeypatch.setattr(fd_mod, "read_crop_food_demand", lambda context: pd.DataFrame())
    assert add_crop_demand(FakeContext()) == {}
