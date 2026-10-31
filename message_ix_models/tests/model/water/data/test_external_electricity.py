"""Tests for external_electricity.py -- costed electricity for water
technologies at the non-Pakistani region nodes."""

import pytest

from message_ix_models.model.water.data import external_electricity as ext
from message_ix_models.tests.model.water.conftest import water_params

IRB = [{**water_params("R12"), "regions": "IRB"}]


@pytest.mark.parametrize("water_context", IRB, indirect=True)
def test_supply_at_every_external_node_only(water_context, water_scenario):
    nodes = ext.external_electricity_nodes(water_context)
    assert nodes, "the IRB delineation has basins outside Pakistan"
    assert ext.NATIONAL_NODE not in nodes
    assert not any(n.endswith("|PAKISTAN") for n in nodes)

    result = ext.add_external_electricity(water_context)
    out = result["output"]
    assert set(out["technology"]) == {ext.EXTERNAL_TECH}
    assert set(out["node_loc"]) == set(nodes)
    assert (out["node_dest"] == out["node_loc"]).all()
    assert set(zip(out["commodity"], out["level"])) == {("electr", "final")}
    assert set(out["value"]) == {1.0}
    assert (out["year_vtg"] == out["year_act"]).all()

    vc = result["var_cost"]
    assert set(vc["value"]) == {ext.EXTERNAL_ELECTRICITY_COST}
    assert len(vc) == len(out)


@pytest.mark.parametrize("water_context", IRB, indirect=True)
def test_cost_can_be_changed_dropped_or_switched_off(water_context, water_scenario):
    water_context.external_electricity_cost = 350.0
    assert set(ext.add_external_electricity(water_context)["var_cost"]["value"]) == {350.0}

    water_context.external_electricity_cost = None  # free supply
    assert "var_cost" not in ext.add_external_electricity(water_context)

    water_context.external_electricity = False
    assert ext.add_external_electricity(water_context) == {}
    assert ext.external_electricity_technology_names(water_context) == []


def test_price_units_and_currency():
    # 0.08 USD/kWh = 80 USD/MWh = 700.8 million USD per GWa (8.76e9 kWh)
    assert ext.to_model_cost(0.08, "USD/kWh") == pytest.approx(700.8)
    assert ext.to_model_cost(80, "USD/MWh") == pytest.approx(700.8)
    assert ext.to_model_cost(700.8) == 700.8
    d = ext.GDP_DEFLATOR
    assert ext.to_model_cost(0.08, "USD/kWh", 2023, 2010) == pytest.approx(700.8 * d[2010] / d[2023])
    assert ext.to_model_cost(0.08, "USD/kWh", 2023, None) == pytest.approx(700.8)
    with pytest.raises(ValueError):
        ext.to_model_cost(1, "EUR/kWh")
    with pytest.raises(ValueError):
        ext.to_model_cost(1, "USD/kWh", 1990, 2010)


def test_price_per_region_and_year(test_context):
    nodes, years = ["CHINA", "INDIA_EAST", "INDIA_WEST"], [2020, 2030, 2040, 2050, 2060]
    test_context.external_electricity_cost_unit = "USD/kWh"
    test_context.external_electricity_cost = {
        "CHINA": 0.06, "INDIA_EAST": {2030: 0.10, 2050: 0.06}, "default": 0.08}
    p = ext.external_electricity_prices(test_context, nodes, years).set_index(["node_loc", "year_act"])["value"]
    assert p["CHINA"].round(1).unique().tolist() == [525.6]
    # flat before the first year given, linear between, flat after the last
    assert p["INDIA_EAST"].round(1).tolist() == [876.0, 876.0, 700.8, 525.6, 525.6]
    assert p["INDIA_WEST"].round(1).unique().tolist() == [700.8]  # default

    test_context.external_electricity_cost = {"CHINA": 0.06}
    with pytest.raises(ValueError, match="INDIA_EAST"):
        ext.external_electricity_prices(test_context, nodes, years)
    test_context.external_electricity_cost = None
    assert ext.external_electricity_prices(test_context, nodes, years) is None


@pytest.mark.parametrize("water_context", IRB, indirect=True)
def test_var_cost_per_region(water_context, water_scenario):
    nodes = ext.external_electricity_nodes(water_context)
    water_context.external_electricity_cost_unit = "USD/kWh"
    water_context.external_electricity_cost = {nodes[0]: 0.1, "default": 0.05}
    vc = ext.add_external_electricity(water_context)["var_cost"]
    by_node = vc.groupby("node_loc")["value"].unique()
    assert by_node[nodes[0]].round(1).tolist() == [876.0]
    assert all(v.round(1).tolist() == [438.0] for n, v in by_node.items() if n != nodes[0])
