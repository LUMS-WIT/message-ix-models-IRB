from __future__ import annotations

import logging
import re
from typing import TypedDict, cast

import numpy as np
import pandas as pd
import pyam
from ixmp import Platform
from message_ix import Reporter, Scenario

from message_ix_models.model.water.utils import USD_KM3_TO_USD_MCM, m3_GJ_TO_MCM_GWa
from message_ix_models.util import package_data_path

log = logging.getLogger(__name__)


class ScenarioMetadata(TypedDict):
    """Metadata dictionary for scenario reporting.

    Attributes
    ----------
    model : str
        Model name from scenario
    scenario : str
        Scenario name
    unit : str
        Unit of measurement (e.g., "million" for population)
    """

    model: str
    scenario: str
    unit: str


# Global variable to track cooling-only mode (without the nexus module)
_cooling_only = False


# def run_old_reporting(sc: Scenario | None = None):
#     try:
#         from message_data.tools.post_processing.iamc_report_hackathon import (
#             report as legacy_report,
#         )
#     except ImportError:
#         log.warning(
#             "Missing message_data.tools.post_processing; fall back to "
#             "message_ix_models.report.legacy. This may not work."
#         )
#         from message_ix_models.report.legacy.iamc_report_hackathon import (
#             report as legacy_report,
#         )

#     if sc is None:
#         raise ValueError("Must provide a Scenario object!")
#     mp2 = sc.platform

#     log.info(
#         " Start reporting of the global energy system (old reporting scheme)"
#         f"for the scenario {sc.model}.{sc.scenario}"
#     )
#     legacy_report(
#         mp=mp2,
#         scen=sc,
#         merge_hist=True,
#         merge_ts=False,
#         run_config="default_run_config.yaml",
#     )


def reg_index(region):
    temp = []
    for i, c in enumerate(region):
        if c == "|":
            temp.append(i)
    return temp


def remove_duplicate(data):
    final_list = []
    indexes = data[data["Variable"].str.contains("basin_to_reg")].index
    for i in data["Region"].index:
        strr = data["Region"][i]
        oprlist = reg_index(strr)
        # cover the case of repeated region name (e.g. Zambia|Zambia)
        if ("|" in strr) and (strr.split("|")[0] == strr.split("|")[1]):
            final_list.append(strr.split("|")[0])
        else:
            if i in indexes:
                if len(oprlist) > 1:
                    final_list.append(strr[oprlist[0] + 1 :])
                elif len(oprlist) == 1 and oprlist[0] > 6:
                    final_list.append(strr[: oprlist[0]])
                else:
                    final_list.append(strr)
            else:
                if len(oprlist) > 1:
                    final_list.append(strr[: oprlist[1]])
                elif len(oprlist) == 1 and oprlist[0] > 6:
                    final_list.append(strr[: oprlist[0]])
                else:
                    final_list.append(strr)
    return final_list


def report_iam_definition(
    sc: Scenario,
    rep: Reporter,
    df_dmd: pd.DataFrame,
    rep_dm: Reporter,
    report_df: pd.DataFrame,
    suban: bool,
) -> pyam.IamDataFrame:
    """Function to define the report iam dataframe

    Parameters
    ----------
    sc : ixmp.Scenario
        Scenario to report
    rep : .Reporter
        Reporter object
    df_dmd : pd.DataFrame
        Dataframe with demands
    rep_dm : .Reporter
        Reporter object for demands
    report_df : pd.DataFrame
        Dataframe with report
    suban : bool
        True if subannual, False if annual

    Returns
    -------
    report_iam : pyam.IamDataFrame
        Report in pyam format
    """

    if not suban:

        def collapse_callback(df):
            """Callback function to populate the IAMC 'variable' column."""
            df["variable"] = "Water Resource|" + df["c"]
            return df.drop(["c"], axis=1)

        # Mapping from dimension IDs to column names
        rename = dict(n="region", y="year")

        rep_dm.require_compat("pyam")
        key = rep_dm.add("demand", "as_pyam", rename=rename, collapse=collapse_callback)

        # Making a dataframe for demands
        df_dmd = rep_dm.get(key).as_pandas()
        # old code left, to be revised
        # df_dmd = df_dmd.drop(columns=["exclude"])
    else:
        df_dmd["model"] = sc.model
        df_dmd["scenario"] = sc.scenario
        df_dmd["variable"] = "Water Resource|" + df_dmd["c"]
        df_dmd.rename(
            columns={"n": "region", "y": "year", "demand": "value", "h": "subannual"},
            inplace=True,
        )
        df_dmd = df_dmd[
            ["model", "scenario", "region", "variable", "subannual", "year", "value"]
        ]

    df_dmd["value"] = df_dmd["value"].abs()
    df_dmd["variable"] = df_dmd["variable"].replace(
        "Water Resource|groundwater_basin", "Water Resource|Groundwater"
    )
    df_dmd["variable"] = df_dmd["variable"].replace(
        "Water Resource|surfacewater_basin", "Water Resource|Surface Water"
    )
    df_dmd["unit"] = "MCM"
    df_dmd1 = pyam.IamDataFrame(df_dmd)

    if not suban:
        # remove_duplicate() may map different region strings (e.g. "IRB|Pakistan")
        # to the same simplified name, producing duplicate rows that pyam rejects.
        # Collapse them by summing across year columns before building the IamDataFrame.
        id_cols = ["Model", "Scenario", "Region", "Variable", "Unit"]
        year_cols = [c for c in report_df.columns if c not in id_cols]
        report_df = report_df.groupby(id_cols, as_index=False)[year_cols].sum()
        report_iam = pyam.IamDataFrame(report_df)
    else:
        # Convert to pyam dataframe
        # if subannual, get and subsittute variables
        vars_dic = ["in:nl-t-ya-m-h-no-c-l", "out:nl-t-ya-m-h-nd-c-l"]

        # other variables do not ahve sub-annual dimension, we just take
        # annual values from report_df
        vars_from_annual = ["CAP_NEW", "inv cost", "total om cost"]
        # get annual variables
        report_df1 = report_df[
            report_df["Variable"].str.contains("|".join(vars_from_annual))
        ]
        report_df1["subannual"] = "year"
        # Convert to pyam dataframe
        report_iam = pyam.IamDataFrame(report_df1)

        report_df2 = pd.DataFrame()
        for vs in vars_dic:
            qty = rep.get(vs)
            df = qty.to_dataframe()
            df.reset_index(inplace=True)
            df["model"] = sc.model
            df["scenario"] = sc.scenario
            df["variable"] = (
                vs.split(":")[0]
                + "|"
                + df["l"]
                + "|"
                + df["c"]
                + "|"
                + df["t"]
                + "|"
                + df["m"]
            )

            df.rename(
                columns={
                    "no": "reg2",  # needed to avoid dulicates
                    "nd": "reg2",
                    "nl": "reg1",
                    "ya": "year",
                    "h": "subannual",
                },
                inplace=True,
            )
            # take the right node column in case nl and no/nd are different
            df = (
                df.groupby(["model", "scenario", "variable", "subannual", "year"])
                .apply(
                    lambda x: x.assign(
                        region=(
                            x["reg2"]
                            if len(x["reg2"].unique()) > len(x["reg1"].unique())
                            else x["reg1"]
                        )
                    )
                )
                .reset_index(drop=True)
            )
            # case of
            exeption = "in|water_supply_basin|freshwater_basin|basin_to_reg"
            df["region"] = df.apply(
                lambda row: (
                    row["reg2"] if exeption in row["variable"] else row["region"]
                ),
                axis=1,
            )
            df = df[
                [
                    "model",
                    "scenario",
                    "region",
                    "variable",
                    "subannual",
                    "year",
                    "value",
                ]
            ]
            report_df2 = pd.concat([report_df2, df])

        report_df2["unit"] = ""
        report_df2.columns = report_df2.columns.astype(str)
        report_df2.columns = report_df2.columns.str.title()
        report_df2.reset_index(drop=True, inplace=True)
        report_df2["Region"] = remove_duplicate(report_df2)
        report_df2.columns = report_df2.columns.str.lower()
        # make iamc dataframe
        report_iam2 = pyam.IamDataFrame(report_df2)
        report_iam = report_iam.append(report_iam2)
    # endif

    # Merge both dataframes in pyam
    output = report_iam.append(df_dmd1)
    return output


def multiply_electricity_output_of_hydro(
    elec_hydro_var: list, report_iam: pyam.IamDataFrame
) -> pyam.IamDataFrame:
    """Function to multiply electricity output of hydro to get withdrawals

    Parameters
    ----------
    elec_hydro_var : list
        List of variables with electricity output of hydro
    report_iam : pyam.IamDataFrame
        Report in pyam format

    Returns
    -------
    report_iam : pyam.IamDataFrame
        Report in pyam format
    """
    perf_data = pd.read_csv(
        package_data_path(
            "water", "ppl_cooling_tech", "tech_water_performance_ssp_msg.csv"
        )
    )

    for var in elec_hydro_var:
        # Extract hydro technology name using regex
        match = re.search(r"hydro[^|]*", var)
        if not match:
            log.error(f"Could not extract hydro technology name from variable: {var}")
            continue
        tech_suffix = match.group(0)

        tech = perf_data[perf_data["technology_name"] == tech_suffix]
        if tech.empty:
            log.error(f"No performance data found for technology: {tech_suffix}")
            continue

        water_withdrawal_ratio = (
            tech["water_withdrawal_mid_m3_per_output"].iloc[0] * m3_GJ_TO_MCM_GWa
        )
        report_iam = report_iam.append(
            report_iam.multiply(
                f"{var}",
                water_withdrawal_ratio,
                f"Water Withdrawal|Electricity|Hydro|{tech_suffix}",
            )
        )
    return report_iam


def get_population_data(sc: Scenario, reg: str) -> pd.DataFrame:
    """Retrieve population data from scenario and map to R12 region codes

    Parameters
    ----------
    sc : Scenario
        Scenario to retrieve population data from
    reg : str
        Region specification for mapping

    Returns
    -------
    pd.DataFrame
        Population data with R12_XX region codes
    """
    population_data = pd.DataFrame()

    # Get region mapping
    mp2 = sc.platform
    reg_map = mp2.regions()

    for ur in ("urban", "rural"):
        try:
            # Get population timeseries for urban/rural
            pop_data = sc.timeseries(variable=f"Population|{ur.capitalize()}")

            # Exclude global aggregate regions
            pop_data = pop_data[
                ~pop_data.region.str.contains(
                    r"GLB\s+region|^World$|^Global$|Total",
                    case=False,
                    regex=True,
                    na=False,
                )
            ]

            if pop_data.empty:
                log.warning(f"No Population|{ur.capitalize()} data found in scenario")
                continue

            # Get unique regions from population data
            pop_reg = np.unique(pop_data["region"])

            # Map natural language region names to R12_XX codes using reg_map
            pop_mappings = reg_map[reg_map.mapped_to.isin(pop_reg)].drop(
                columns=["parent", "hierarchy"]
            )
            population_to_r12_map = dict(
                zip(pop_mappings.mapped_to, pop_mappings.region)
            )

            # Apply region mapping
            pop_data["region"] = (
                pop_data["region"].map(population_to_r12_map).fillna(pop_data["region"])
            )

            # Add urban/rural identifier
            pop_data["variable"] = f"Population|{ur.capitalize()}"

            # sc.timeseries() returns wide format (years as integer column names).
            # Convert to long format so downstream code can use pop_data["year"].
            if "year" not in pop_data.columns:
                id_vars = [
                    c for c in ["model", "scenario", "region", "variable", "unit"]
                    if c in pop_data.columns
                ]
                year_cols = [
                    c for c in pop_data.columns
                    if c not in id_vars and (isinstance(c, int) or (isinstance(c, str) and c.isdigit()))
                ]
                if year_cols:
                    pop_data = pop_data.melt(
                        id_vars=id_vars,
                        value_vars=year_cols,
                        var_name="year",
                        value_name="value",
                    )
                    pop_data["year"] = pop_data["year"].astype(int)
                    pop_data = pop_data.dropna(subset=["value"])

            population_data = pd.concat([population_data, pop_data])

        except (KeyError, ValueError) as e:
            log.warning(f"Failed to retrieve Population|{ur.capitalize()} data: {e}")
            continue

    # Check if we have future data (post-2020)
    if population_data.empty or "year" not in population_data.columns:
        log.warning("No population data found in scenario timeseries — skipping SDG6 population indicators")
        return population_data
    future_pop = population_data[population_data.year >= 2020]
    if future_pop.empty:
        log.warning("No population data with future values (>=2020) found")
    else:
        log.info(f"Population data available through year {population_data.year.max()}")

    return population_data


def get_rates_data(reg: str, ssp: str, sdgs: bool = False) -> pd.DataFrame:
    """Load and clean water access rates data from CSV

    Parameters
    ----------
    reg : str
        Region specification (R11, R12, R17, etc.)
    ssp : str
        SSP scenario (e.g., "SSP1", "SSP2", "SSP3")
    sdgs : bool
        Whether to use SDG scenario rates

    Returns
    -------
    pd.DataFrame
        Cleaned rates data with region codes matching reg parameter
    """
    # Load rates data
    load_path = package_data_path("water", "demands", "harmonized", reg)
    all_rates = pd.read_csv(load_path / f"all_rates_{ssp}.csv")

    # Filter for scenario type
    scenario_type = "SDG" if sdgs else "baseline"
    df_rate = all_rates[all_rates.variable.str.contains(scenario_type)]

    if df_rate.empty:
        log.warning(f"No rates data found for {scenario_type}")
        return pd.DataFrame()

    # Extract region from node (e.g., "105|CHN" -> "CHN")
    df_rate = df_rate.copy()
    df_rate["region_short"] = [x.split("|")[1] for x in df_rate.node]

    # Get region mapping for basin codes
    mp = Platform()
    reg_map = mp.regions()

    # Create basin to region mapping based on reg parameter
    target_regions = reg_map[reg_map.region.str.startswith(f"{reg}_")]
    basin_to_reg_map = {}
    for _, row in target_regions.iterrows():
        region_code = row.region.split("_")[1]  # R12_CHN -> CHN
        basin_to_reg_map[region_code] = row.region  # CHN -> R12_CHN

    df_rate["region"] = df_rate["region_short"].map(basin_to_reg_map)

    # Remove unmapped regions
    df_rate = df_rate.dropna(subset=["region"])
    df_rate = df_rate.drop(columns=["node", "region_short"])

    # Average rates across basins for each region/year/variable
    rates_data = (
        df_rate.groupby(["year", "variable", "region"])["value"].mean().reset_index()
    )

    return rates_data


def get_population_values(
    pop_data: pd.DataFrame, region: str, year: int
) -> tuple[float, float]:
    """Get urban and rural population values for a specific region and year.

    Parameters
    ----------
    pop_data : pd.DataFrame
        Population data
    region : str
        Region code
    year : int
        Year

    Returns
    -------
    tuple[float, float]
        Urban population value, rural population value (may be NaN)
    """
    pop_urban = pop_data[
        (pop_data.region == region)
        & (pop_data.year == year)
        & (pop_data.variable == "Population|Urban")
    ]
    pop_rural = pop_data[
        (pop_data.region == region)
        & (pop_data.year == year)
        & (pop_data.variable == "Population|Rural")
    ]

    urban_val = pop_urban.iloc[0]["value"] if not pop_urban.empty else np.nan
    rural_val = pop_rural.iloc[0]["value"] if not pop_rural.empty else np.nan

    return urban_val, rural_val


def process_rates(
    population_type: str,
    population_value: float,
    rates_data: pd.DataFrame,
    region: str,
    year: int,
    metadata: ScenarioMetadata,
) -> list[dict]:
    """Process base population, connection rates, and treatment rates.

    Parameters
    ----------
    population_type : str
        Type of population ("urban" or "rural")
    population_value : float
        Population value for the region/year
    rates_data : pd.DataFrame
        DataFrame containing connection and treatment rates
    region : str
        Region code
    year : int
        Year
    metadata : ScenarioMetadata
        Scenario metadata (model, scenario, unit)

    Returns
    -------
    list[dict]
        List of result dictionaries with population, rates, and access calculations
    """
    pt_cap = population_type.capitalize()
    base = {
        "region": region,
        "year": year,
        "variable": f"Population|{pt_cap}",
        "value": population_value,
        **metadata,
    }

    out = [base]

    # map pattern -> (variable_template, access_variable_template)
    patterns = {
        r"connection": (
            f"Connection Rate|Drinking Water|{pt_cap}",
            f"Population|Drinking Water Access|{pt_cap}",
        ),
        r"treatment": (
            f"Treatment Rate|Sanitation|{pt_cap}",
            f"Population|Sanitation Access|{pt_cap}",
        ),
    }

    # search and append if found
    for pat, (rate_var, access_var) in patterns.items():
        found = rates_data[
            rates_data.variable.str.contains(
                f"{population_type}.*{pat}", case=False, regex=True
            )
        ]
        if found.empty:
            continue
        rate = found.iloc[0]["value"]
        out.append({**base, "variable": rate_var, "value": rate})
        access = (
            (population_value * rate)
            if pd.notna(population_value) and pd.notna(rate)
            else np.nan
        )
        out.append({**base, "variable": access_var, "value": access})

    return out


def aggregate_totals(result_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Aggregate regional totals for:
      - Population|Drinking Water Access (sum)
      - Population|Sanitation Access (sum)
      - Population (Urban/Rural -> sum -> Population)
    Returns list of DataFrames"""
    totals: list[pd.DataFrame] = []

    def _group_sum(df: pd.DataFrame, new_var: str) -> pd.DataFrame:
        g = (
            df.groupby(["region", "year", "model", "scenario", "unit"], dropna=False)[
                "value"
            ]
            .sum()
            .reset_index()
        )
        g["variable"] = new_var
        return g

    # drinking water access
    drink = result_df[
        result_df.variable.str.contains(
            r"Population\|Drinking Water Access", regex=True
        )
    ]
    if not drink.empty:
        totals.append(_group_sum(drink, "Population|Drinking Water Access"))

    # sanitation access
    sani = result_df[
        result_df.variable.str.contains(r"Population\|Sanitation Access", regex=True)
    ]
    if not sani.empty:
        totals.append(_group_sum(sani, "Population|Sanitation Access"))

    # base population (Urban|Rural)
    base_pop = result_df[
        result_df.variable.str.match(r"Population\|(Urban|Rural)$", na=False)
    ]
    if not base_pop.empty:
        pop_sum = (
            base_pop.groupby(
                ["region", "year", "model", "scenario", "unit"], dropna=False
            )["value"]
            .sum()
            .reset_index()
        )
        pop_sum["variable"] = "Population"
        totals.append(pop_sum)

    return totals


def aggregate_world_totals(result_df: pd.DataFrame) -> list[pd.DataFrame]:
    """World-level aggregations:
      - sum of non-rate variables (populations/access)
      - mean of rate variables
    Preserves column set ['variable','year','model','scenario','unit','value'] and
    adds region='World'.
    """
    world: list[pd.DataFrame] = []

    pop_vars = result_df[~result_df.variable.str.contains("Rate", na=False)]
    rate_vars = result_df[result_df.variable.str.contains("Rate", na=False)]

    if not pop_vars.empty:
        wp = (
            pop_vars.groupby(
                ["variable", "year", "model", "scenario", "unit"], dropna=False
            )["value"]
            .sum()
            .reset_index()
        )
        wp["region"] = "World"
        world.append(wp)

    if not rate_vars.empty:
        wr = (
            rate_vars.groupby(
                ["variable", "year", "model", "scenario", "unit"], dropna=False
            )["value"]
            .mean()
            .reset_index()
        )
        wr["region"] = "World"
        world.append(wr)

    return world


def pop_water_access(
    sc: Scenario, reg: str, ssp: str, sdgs: bool = False
) -> pd.DataFrame:
    """Calculate population with access to water and sanitation

    Parameters
    ----------
    sc : Scenario
        Scenario to calculate access for
    reg : str
        Region specification
    ssp : str
        SSP scenario (e.g., "SSP1", "SSP2", "SSP3")
    sdgs : bool
        Whether to use SDG scenario rates

    Returns
    -------
    pd.DataFrame
        Population access data with all variables
    """
    # Get clean population and rates data
    pop_data = get_population_data(sc, reg)
    rates_data = get_rates_data(reg, ssp, sdgs)

    if pop_data.empty:
        log.warning("No population data found. Skipping calculations.")
        return pd.DataFrame()

    if rates_data.empty:
        log.warning("No rates data found. Skipping calculations.")
        return pd.DataFrame()

    # Get unique combinations of region/year from both datasets
    all_years = sorted(set(rates_data.year.unique()) | set(pop_data.year.unique()))
    all_regions = sorted(
        set(rates_data.region.unique()) | set(pop_data.region.unique())
    )

    # Get metadata from first pop row
    metadata = {}
    if not pop_data.empty:
        first_row = pop_data.iloc[0]
        metadata = {
            "model": first_row["model"],
            "scenario": first_row["scenario"],
            "unit": first_row["unit"],
        }

    results = []

    # Process each region/year combination
    for region in all_regions:
        for year in all_years:
            # Get population values for this region/year
            urban_val, rural_val = get_population_values(pop_data, region, year)

            # Get rates for this region/year
            region_year_rates = rates_data[
                (rates_data.region == region) & (rates_data.year == year)
            ]

            # Process urban and rural using the same helper function
            results.extend(
                process_rates(
                    "urban",
                    urban_val,
                    region_year_rates,
                    region,
                    year,
                    cast(ScenarioMetadata, metadata),
                )
            )
            results.extend(
                process_rates(
                    "rural",
                    rural_val,
                    region_year_rates,
                    region,
                    year,
                    cast(ScenarioMetadata, metadata),
                )
            )

    result_df = pd.DataFrame(results)

    # Add aggregated totals
    if not result_df.empty:
        # Get regional totals
        totals = aggregate_totals(result_df)
        if totals:
            result_df = pd.concat([result_df] + totals, ignore_index=True)

        # Get world totals
        world_totals = aggregate_world_totals(result_df)
        if world_totals:
            result_df = pd.concat([result_df] + world_totals, ignore_index=True)

    log.info("Population water access calculations completed")
    return result_df


def prepare_ww(ww_input: pd.DataFrame, suban: bool) -> pd.DataFrame:
    ww = ww_input[
        ww_input.variable.isin(
            ["out|final|rural_mw|rural_t_d|M1", "out|final|urban_mw|urban_t_d|M1"]
        )
    ]
    ww["commodity"] = np.where(
        ww.variable.str.contains("urban_mw"), "urban_mw", "rural_mw"
    )
    ww["wdr"] = ww["value"]
    if not suban:
        ww = ww[["region", "year", "commodity", "wdr"]]
    else:
        ww = ww[["region", "year", "subannual", "commodity", "wdr"]]
        ww = pd.concat(
            [
                ww,
                (
                    ww.groupby(["region", "year", "commodity"])["wdr"]
                    .sum()
                    .reset_index()
                    .assign(subannual="year")
                    .loc[:, ["region", "year", "subannual", "commodity", "wdr"]]
                ),
            ]
        ).reset_index(drop=True)

    return ww


def detect_scenario_mode(df_dmd: pd.DataFrame) -> tuple[bool, bool]:
    """Detect if scenario is cooling-only and if it uses subannual time slices.

    Cooling-only mode is detected when df_dmd is empty, indicating the nexus
    water module is absent from the scenario.

    Parameters
    ----------
    df_dmd : pd.DataFrame
        Demand dataframe filtered for water_avail_basin

    Returns
    -------
    suban : bool
        True if scenario uses subannual time slices
    cooling_only : bool
        True if nexus module is absent (cooling-only mode)
    """
    global _cooling_only
    if df_dmd.empty:
        _cooling_only = True
        suban = False
    else:
        _cooling_only = False
        h_values = np.unique(df_dmd["h"])
        suban = False if "year" in h_values else True

    return suban, _cooling_only


def compute_cooling_technologies(
    report_iam: pyam.IamDataFrame,
    sc: Scenario,
) -> tuple[pyam.IamDataFrame, list]:
    """Compute cooling technology metrics and return mapping rows.

    Parameters
    ----------
    report_iam : pyam.IamDataFrame
        Report in pyam format
    sc : Scenario
        Scenario to extract technology data from

    Returns
    -------
    report_iam : pyam.IamDataFrame
        Updated report with cooling technology calculations
    cooling_rows : list
        List of [name, variables, unit] rows for cooling metrics
    """
    # Freshwater cooling technologies water usage
    cooling_fresh_water = report_iam.filter(
        variable="in|water_supply|surfacewater|*fresh|*"
    ).variable
    cooling_ot_fresh_water = report_iam.filter(
        variable="in|water_supply|surfacewater|*__ot_fresh|*"
    ).variable
    cooling_cl_fresh_water = report_iam.filter(
        variable="in|water_supply|surfacewater|*__cl_fresh|*"
    ).variable
    cooling_ot_saline_water = report_iam.filter(
        variable="in|saline_supply|saline_ppl|*__ot_saline|*"
    ).variable
    # Non-cooling technologies freshwater usage
    # Read CSV to get correct list of non-cooling technologies
    FILE = "tech_water_performance_ssp_msg.csv"
    path = package_data_path("water", "ppl_cooling_tech", FILE)
    df = pd.read_csv(path)

    non_cool_df = df[
        (df["technology_group"] != "cooling")
        & (df["water_supply_type"] == "freshwater_supply")
    ]

    # Filter for technologies that exist in scenario
    all_techs = list(sc.set("technology"))
    tech_non_cool_csv = list(non_cool_df["technology_name"])
    techs_in_scenario = [tec for tec in tech_non_cool_csv if tec in all_techs]

    # Extract water input variables for non-cooling technologies
    # Non-cooling techs use surfacewater commodity at water_supply level
    non_cooling_water = [
        f"in|water_supply|surfacewater|{tech}|M1" for tech in techs_in_scenario
    ]

    # Fresh water return flow emissions
    fresh_return_emissions = report_iam.filter(variable="emis|fresh_return|*").variable

    # Cooling investments
    cooling_saline_inv = report_iam.filter(variable="inv cost|*saline").variable
    cooling_air_inv = report_iam.filter(variable="inv cost|*air").variable
    cooling_ot_fresh = report_iam.filter(variable="inv cost|*ot_fresh").variable
    cooling_cl_fresh = report_iam.filter(variable="inv cost|*cl_fresh").variable

    # Hydro electricity calculations
    elec_hydro_var = report_iam.filter(variable="out|secondary|electr|hydro*").variable
    report_iam = multiply_electricity_output_of_hydro(elec_hydro_var, report_iam)
    water_hydro_var = report_iam.filter(
        variable="Water Withdrawal|Electricity|Hydro|*"
    ).variable

    # Build cooling-specific mapping rows
    cooling_rows = [
        ["Water Withdrawal|Electricity|Hydro", water_hydro_var, "MCM/yr"],
        [
            "Water Withdrawal|Electricity|Cooling|Fresh Water",
            cooling_fresh_water,
            "MCM/yr",
        ],
        [
            "Water Withdrawal|Electricity|Cooling|Once Through|Fresh Water",
            cooling_ot_fresh_water,
            "MCM/yr",
        ],
        [
            "Water Withdrawal|Electricity|Cooling|Closed Loop|Fresh Water",
            cooling_cl_fresh_water,
            "MCM/yr",
        ],
        [
            "Water Withdrawal|Electricity|Cooling|Once Through|Saline Water",
            cooling_ot_saline_water,
            "MCM/yr",
        ],
        ["Water Withdrawal|Energy|Non-Cooling", non_cooling_water, "MCM/yr"],
        ["Water Return|Electricity|Cooling", fresh_return_emissions, "MCM/yr"],
        [
            "Investment|Infrastructure|Water|Cooling",
            cooling_ot_fresh + cooling_cl_fresh + cooling_saline_inv + cooling_air_inv,
            "million US$2010/yr",
        ],
        [
            "Investment|Infrastructure|Water|Cooling|Once through freshwater",
            cooling_ot_fresh,
            "million US$2010/yr",
        ],
        [
            "Investment|Infrastructure|Water|Cooling|Closed loop freshwater",
            cooling_cl_fresh,
            "million US$2010/yr",
        ],
        [
            "Investment|Infrastructure|Water|Cooling|Once through saline",
            cooling_saline_inv,
            "million US$2010/yr",
        ],
        [
            "Investment|Infrastructure|Water|Cooling|Air cooled",
            cooling_air_inv,
            "million US$2010/yr",
        ],
    ]

    # Store water_hydro_var for later filtering
    report_iam.metadata = getattr(report_iam, "metadata", {})
    report_iam.metadata["water_hydro_var"] = water_hydro_var
    report_iam.metadata["cooling_inv_vars"] = (
        cooling_ot_fresh + cooling_cl_fresh + cooling_saline_inv + cooling_air_inv
    )

    return report_iam, cooling_rows


def _build_water_report_df(sc: Scenario) -> pd.DataFrame:
    """Build report_df for water technologies without using message::default.

    message::default fails on large models because:
    1. It processes all 500+ technologies (timeout), and
    2. It raises pint.DimensionalityError on USD_2005/MCM water cost units.

    This function queries sc.var() / sc.par() directly for water technologies
    only and returns a wide-format DataFrame that pyam.IamDataFrame accepts.
    Variable naming matches message::default convention exactly so the rest
    of report() works unchanged:
      - "in|{level}|{commodity}|{technology}|{mode}"   for input flows
      - "out|{level}|{commodity}|{technology}|{mode}"  for output flows
      - "CAP_NEW|new capacity|{technology}"            for new capacity
      - "inv cost|{technology}"                         for investment costs
      - "total om cost|{technology}"                    for O&M costs

    Returns
    -------
    pd.DataFrame
        Wide-format with columns: Model, Scenario, Region, Variable, Unit,
        then one column per model year (as strings, e.g. "2020", "2025").
    """
    MODEL = sc.model
    SCENARIO = sc.scenario

    # Core water technologies
    WATER_TECS = [
        "extract_surfacewater", "extract_groundwater", "extract_gw_fossil",
        "membrane", "distillation",
        "return_flow", "gw_recharge",
        "urban_t_d", "rural_t_d",
        "urban_sewerage", "rural_sewerage",
        "urban_recycle", "rural_recycle",
        "urban_treatment", "rural_treatment",
        "urban_unconnected", "rural_unconnected",
        "urban_untreated", "rural_untreated",
        "industry_unconnected", "industry_untreated",
        "extract_salinewater_cool", "extract_salinewater_basin",
        "basin_to_reg", "basin_to_reg_plus",
        "irrigation_cereal", "irrigation_oilcrops", "irrigation_sugarcrops",
    ]
    # Add hydro and cooling technologies present in this scenario, plus the
    # crops/land/irrigation port (crop_<crop>, rainfed_<crop>,
    # irr_<method>_<crop> -- message_ix_models.model.water.data.crops) and
    # network.py's inter-basin transfer techs (river|.../canal_conv|.../
    # trs_|...) -- none of these are in the hardcoded WATER_TECS list above
    # because they postdate it. Without this, none of their ACT/input/
    # output ever reaches _apply_iamc_mapping, so no amount of map_agg
    # entries downstream can make e.g. irrigation withdrawal appear in the
    # reported CSV -- confirmed missing this way, not a mapping gap.
    all_tecs = sc.set("technology").tolist()
    WATER_TECS += [
        t for t in all_tecs
        if any(kw in t for kw in
               ["hydro", "__ot_fresh", "__cl_fresh", "__ot_saline", "__air"])
        or t.startswith(("irr_", "crop_", "rainfed_", "river|", "canal_conv|", "trs_|"))
    ]
    WATER_TECS = list(set(WATER_TECS))
    log.info(f"_build_water_report_df: {len(WATER_TECS)} technologies")

    ID_COLS = ["Model", "Scenario", "Region", "Variable", "Unit"]
    dfs: list[pd.DataFrame] = []

    def _collect(m: pd.DataFrame, var_col: str, unit: str,
                 year_col: str, val_col: str) -> None:
        """Vectorised: build IAMC rows from a merged DataFrame and append to dfs."""
        sub = m[[var_col, "node_loc", year_col, val_col]].copy()
        sub = sub[sub[val_col].notna() & (sub[val_col] != 0)]
        if sub.empty:
            return
        sub["Model"] = MODEL
        sub["Scenario"] = SCENARIO
        sub["Unit"] = unit
        sub = sub.rename(columns={
            "node_loc": "Region",
            var_col: "Variable",
            year_col: "year",
            val_col: "value",
        })
        dfs.append(sub[ID_COLS + ["year", "value"]])

    # ── Activity ─────────────────────────────────────────────────────────
    act = sc.var("ACT", {"technology": WATER_TECS})
    ACT_MERGE = ["node_loc", "technology", "year_vtg", "year_act", "mode", "time"]

    # ── Input flows ───────────────────────────────────────────────────────
    if not act.empty:
        inp = sc.par("input", {"technology": WATER_TECS})
        if not inp.empty:
            m = act.merge(inp[ACT_MERGE + ["commodity", "level", "value"]],
                          on=ACT_MERGE, how="inner")
            m["flow"] = m["lvl"] * m["value"]
            m["Variable"] = ("in|" + m["level"] + "|" + m["commodity"]
                             + "|" + m["technology"] + "|" + m["mode"])
            _collect(m, "Variable", "MCM/yr", "year_act", "flow")

    # ── Output flows ──────────────────────────────────────────────────────
    if not act.empty:
        out_par = sc.par("output", {"technology": WATER_TECS})
        if not out_par.empty:
            m = act.merge(out_par[ACT_MERGE + ["commodity", "level", "value"]],
                          on=ACT_MERGE, how="inner")
            m["flow"] = m["lvl"] * m["value"]
            m["Variable"] = ("out|" + m["level"] + "|" + m["commodity"]
                             + "|" + m["technology"] + "|" + m["mode"])
            _collect(m, "Variable", "MCM/yr", "year_act", "flow")

    # ── New capacity ──────────────────────────────────────────────────────
    cap_new = sc.var("CAP_NEW", {"technology": WATER_TECS})
    cap_new_nz = cap_new[cap_new["lvl"] != 0].copy() if not cap_new.empty else cap_new
    if not cap_new_nz.empty:
        cap_new_nz["Variable"] = "CAP_NEW|new capacity|" + cap_new_nz["technology"]
        _collect(cap_new_nz, "Variable", "MCM/yr", "year_vtg", "lvl")

    # ── Investment costs ──────────────────────────────────────────────────
    if not cap_new_nz.empty:
        inv = sc.par("inv_cost", {"technology": WATER_TECS})
        if not inv.empty:
            m = cap_new_nz.merge(
                inv[["node_loc", "technology", "year_vtg", "value"]],
                on=["node_loc", "technology", "year_vtg"], how="left")
            m["cost"] = m["lvl"] * m["value"].fillna(0)
            m["Variable"] = "inv cost|" + m["technology"]
            _collect(m, "Variable", "MUSD/yr", "year_vtg", "cost")

    # ── Installed capacity ───────────────────────────────────────────────
    # Not previously collected here (only CAP_NEW was) -- needed for time
    # series of standing capacity, e.g. hydro. CAP is indexed by
    # (node_loc, technology, year_vtg, year_act); _collect's groupby sums
    # over year_vtg for a fixed year_act, giving total capacity surviving
    # into that year, same as message_ix's own "Capacity|*" convention.
    # Water infrastructure capacity is in MCM (matching CAP_NEW's existing
    # convention below); hydro capacity is in GW -- these are genuinely
    # different quantities, so split by technology rather than mislabel one.
    cap = sc.var("CAP", {"technology": WATER_TECS})
    cap_nz = cap[cap["lvl"] != 0].copy() if not cap.empty else cap
    if not cap_nz.empty:
        cap_nz["Variable"] = "CAP|capacity|" + cap_nz["technology"]
        is_hydro = cap_nz["technology"].str.contains("hydro")
        if (~is_hydro).any():
            _collect(cap_nz[~is_hydro], "Variable", "MCM/yr", "year_act", "lvl")
        if is_hydro.any():
            _collect(cap_nz[is_hydro], "Variable", "GW", "year_act", "lvl")

    # ── Total O&M costs ───────────────────────────────────────────────────
    fix_c = sc.par("fix_cost", {"technology": WATER_TECS})
    if not cap.empty and not fix_c.empty:
        m = cap.merge(
            fix_c[["node_loc", "technology", "year_vtg", "year_act", "value"]],
            on=["node_loc", "technology", "year_vtg", "year_act"], how="left")
        m["cost"] = m["lvl"] * m["value"].fillna(0)
        m["Variable"] = "total om cost|" + m["technology"]
        _collect(m, "Variable", "MUSD/yr", "year_act", "cost")

    var_c = sc.par("var_cost", {"technology": WATER_TECS})
    if not act.empty and not var_c.empty:
        m = act.merge(var_c[ACT_MERGE + ["value"]], on=ACT_MERGE, how="left")
        m["cost"] = m["lvl"] * m["value"].fillna(0)
        m["Variable"] = "total om cost|" + m["technology"]
        _collect(m, "Variable", "MUSD/yr", "year_act", "cost")

    # ── Pivot to wide format ──────────────────────────────────────────────
    if not dfs:
        log.warning("_build_water_report_df: no data produced")
        return pd.DataFrame(columns=ID_COLS)

    long_df = pd.concat(dfs, ignore_index=True)
    long_df["year"] = long_df["year"].astype(int)
    long_df = long_df.groupby(ID_COLS + ["year"], as_index=False)["value"].sum()

    wide_df = long_df.pivot_table(
        index=ID_COLS, columns="year", values="value", aggfunc="sum"
    ).reset_index()
    wide_df.columns.name = None
    wide_df.columns = [str(c) for c in wide_df.columns]
    log.info(f"_build_water_report_df: {len(wide_df)} variable rows built")
    return wide_df


def _apply_iamc_mapping(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw MESSAGE variable names to IAMC-style aggregated names.

    Applies the same variable groupings as the ``map_agg_pd`` block inside
    :func:`report` but uses pure-pandas groupby instead of the slow pyam
    ``aggregate`` / ``aggregate_region`` calls.  Called by the fast path so
    the output CSV matches file 1 (IAMC format) instead of file 2 (raw names).

    Parameters
    ----------
    raw_df : pd.DataFrame
        Long-format DataFrame produced by ``_build_water_report_df`` after
        melting.  Expected columns: model, scenario, region, variable, unit,
        year, value (all lower-case).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame containing only IAMC-named rows (raw variables
        are excluded from the output).
    """
    raw_vars = raw_df["variable"].unique().tolist()

    # ── Capacity addition variable lists ─────────────────────────────────────
    urban_infrastructure = [
        "CAP_NEW|new capacity|urban_recycle",
        "CAP_NEW|new capacity|urban_sewerage",
        "CAP_NEW|new capacity|urban_t_d",
        "CAP_NEW|new capacity|urban_treatment",
        "CAP_NEW|new capacity|urban_unconnected",
        "CAP_NEW|new capacity|urban_untreated",
    ]
    rural_infrastructure = [
        "CAP_NEW|new capacity|rural_recycle",
        "CAP_NEW|new capacity|rural_sewerage",
        "CAP_NEW|new capacity|rural_t_d",
        "CAP_NEW|new capacity|rural_treatment",
        "CAP_NEW|new capacity|rural_unconnected",
        "CAP_NEW|new capacity|rural_untreated",
    ]
    urban_treatment_recycling = [
        "CAP_NEW|new capacity|urban_recycle",
        "CAP_NEW|new capacity|urban_sewerage",
        "CAP_NEW|new capacity|urban_treatment",
    ]
    rural_treatment_recycling = [
        "CAP_NEW|new capacity|rural_recycle",
        "CAP_NEW|new capacity|rural_sewerage",
        "CAP_NEW|new capacity|rural_treatment",
    ]
    rural_dist = ["CAP_NEW|new capacity|rural_t_d"]
    urban_dist = ["CAP_NEW|new capacity|urban_t_d"]
    rural_unconnected = [
        "CAP_NEW|new capacity|rural_unconnected",
        "CAP_NEW|new capacity|rural_untreated",
    ]
    urban_unconnected = [
        "CAP_NEW|new capacity|urban_unconnected",
        "CAP_NEW|new capacity|urban_untreated",
    ]
    industry_unconnected = [
        "CAP_NEW|new capacity|industry_unconnected",
        "CAP_NEW|new capacity|industry_untreated",
    ]
    extrt_sw_cap = ["CAP_NEW|new capacity|extract_surfacewater"]
    extrt_gw_cap = ["CAP_NEW|new capacity|extract_groundwater"]
    extrt_fgw_cap = ["CAP_NEW|new capacity|extract_gw_fossil"]

    # ── Investment cost variable lists ────────────────────────────────────────
    extrt_sw_inv = ["inv cost|extract_surfacewater"]
    extrt_gw_inv = ["inv cost|extract_groundwater"]
    extrt_fgw_inv = ["inv cost|extract_gw_fossil"]
    rural_infrastructure_inv = [
        "inv cost|rural_recycle", "inv cost|rural_sewerage", "inv cost|rural_t_d",
        "inv cost|rural_treatment", "inv cost|rural_unconnected", "inv cost|rural_untreated",
    ]
    urban_infrastructure_inv = [
        "inv cost|urban_recycle", "inv cost|urban_sewerage", "inv cost|urban_t_d",
        "inv cost|urban_treatment", "inv cost|urban_unconnected", "inv cost|urban_untreated",
    ]
    urban_treatment_recycling_inv = [
        "inv cost|urban_recycle", "inv cost|urban_sewerage", "inv cost|urban_treatment",
    ]
    rural_treatment_recycling_inv = [
        "inv cost|rural_recycle", "inv cost|rural_sewerage", "inv cost|rural_treatment",
    ]
    rural_dist_inv = ["inv cost|rural_t_d"]
    urban_dist_inv = ["inv cost|urban_t_d"]
    rural_unconnected_inv = ["inv cost|rural_unconnected", "inv cost|rural_untreated"]
    urban_unconnected_inv = ["inv cost|urban_unconnected", "inv cost|urban_untreated"]
    industry_unconnected_inv = [
        "inv cost|industry_unconnected", "inv cost|industry_untreated",
    ]
    saline_inv = ["inv cost|membrane", "inv cost|distillation"]

    # ── O&M cost variable lists ───────────────────────────────────────────────
    saline_totalom = ["total om cost|membrane", "total om cost|distillation"]
    extrt_sw_om = ["total om cost|extract_surfacewater"]
    extrt_gw_om = ["total om cost|extract_groundwater"]
    extrt_fgw_om = ["total om cost|extract_gw_fossil"]
    urban_infrastructure_totalom = [
        "total om cost|urban_recycle", "total om cost|urban_sewerage",
        "total om cost|urban_t_d", "total om cost|urban_treatment",
        "total om cost|urban_unconnected", "total om cost|urban_untreated",
    ]
    rural_infrastructure_totalom = [
        "total om cost|rural_recycle", "total om cost|rural_sewerage",
        "total om cost|rural_t_d", "total om cost|rural_treatment",
        "total om cost|rural_unconnected", "total om cost|rural_untreated",
    ]
    rural_treatment_recycling_totalom = [
        "total om cost|rural_recycle", "total om cost|rural_sewerage",
        "total om cost|rural_treatment",
    ]
    urban_treatment_recycling_totalom = [
        "total om cost|urban_recycle", "total om cost|urban_sewerage",
        "total om cost|urban_treatment",
    ]
    rural_dist_totalom = ["total om cost|rural_t_d"]
    urban_dist_totalom = ["total om cost|urban_t_d"]
    rural_unconnected_totalom = [
        "total om cost|rural_unconnected", "total om cost|rural_untreated",
    ]
    urban_unconnected_totalom = [
        "total om cost|urban_unconnected", "total om cost|urban_untreated",
    ]
    industry_unconnected_totalom = [
        "total om cost|industry_unconnected", "total om cost|industry_untreated",
    ]

    # ── Water flow variable lists ─────────────────────────────────────────────
    extract_sw = ["in|water_avail_basin|surfacewater_basin|extract_surfacewater|M1"]
    extract_gw = ["in|water_avail_basin|groundwater_basin|extract_groundwater|M1"]
    extract_fgw = ["out|water_supply_basin|freshwater_basin|extract_gw_fossil|M1"]
    desal_membrane = ["out|water_supply_basin|freshwater_basin|membrane|M1"]
    desal_distill = ["out|water_supply_basin|freshwater_basin|distillation|M1"]
    env_flow = ["in|water_avail_basin|surfacewater_basin|return_flow|M1"]
    gw_recharge = ["in|water_avail_basin|groundwater_basin|gw_recharge|M1"]
    rural_mwdem_unconnected = ["out|final|rural_disconnected|rural_unconnected|M1"]
    rural_mwdem_unconnected_eff = ["out|final|rural_disconnected|rural_unconnected|Mf"]
    rural_mwdem_connected = ["out|final|rural_mw|rural_t_d|M1"]
    rural_mwdem_connected_eff = ["out|final|rural_mw|rural_t_d|Mf"]
    urban_mwdem_unconnected = ["out|final|urban_disconnected|urban_unconnected|M1"]
    urban_mwdem_unconnected_eff = ["out|final|urban_disconnected|urban_unconnected|Mf"]
    urban_mwdem_connected = ["out|final|urban_mw|urban_t_d|M1"]
    urban_mwdem_connected_eff = ["out|final|urban_mw|urban_t_d|Mf"]
    industry_mwdem_unconnected = ["out|final|industry_mw|industry_unconnected|M1"]
    industry_mwdem_unconnected_eff = ["out|final|industry_mw|industry_unconnected|Mf"]
    electr_gw = ["in|final|electr|extract_groundwater|M1"]
    electr_fgw = ["in|final|electr|extract_gw_fossil|M1"]
    electr_sw = ["in|final|electr|extract_surfacewater|M1"]
    extract_saline_region = ["out|saline_supply|saline_ppl|extract_salinewater_cool|M1"]
    extract_saline_basin = [
        "out|water_avail_basin|salinewater_basin|extract_salinewater_basin|M1",
    ]
    electr_rural_trt = ["in|final|electr|rural_sewerage|M1"]
    electr_urban_trt = ["in|final|electr|urban_sewerage|M1"]
    electr_urban_recycle = ["in|final|electr|urban_recycle|M1"]
    electr_rural_recycle = ["in|final|electr|rural_recycle|M1"]
    electr_saline = ["in|final|electr|distillation|M1", "in|final|electr|membrane|M1"]
    electr_urban_t_d = ["in|final|electr|urban_t_d|M1"]
    electr_urban_t_d_eff = ["in|final|electr|urban_t_d|Mf"]
    electr_rural_t_d = ["in|final|electr|rural_t_d|M1"]
    electr_rural_t_d_eff = ["in|final|electr|rural_t_d|Mf"]
    electr_irr = [
        "in|final|electr|irrigation_cereal|M1",
        "in|final|electr|irrigation_oilcrops|M1",
        "in|final|electr|irrigation_sugarcrops|M1",
    ]
    urban_collctd_wstwtr = ["in|final|urban_collected_wst|urban_sewerage|M1"]
    rural_collctd_wstwtr = ["in|final|rural_collected_wst|rural_sewerage|M1"]
    urban_treated_wstwtr = ["in|water_treat|urban_collected_wst|urban_recycle|M1"]
    rural_treated_wstwtr = ["in|water_treat|rural_collected_wst|rural_recycle|M1"]
    urban_wstwtr_recycle = ["out|water_supply_basin|freshwater_basin|urban_recycle|M1"]
    rural_wstwtr_recycle = ["out|water_supply_basin|freshwater_basin|rural_recycle|M1"]
    urban_transfer = ["in|water_supply_basin|freshwater_basin|urban_t_d|M1"]
    urban_transfer_eff = ["in|water_supply_basin|freshwater_basin|urban_t_d|Mf"]
    rural_transfer = ["in|water_supply_basin|freshwater_basin|rural_t_d|M1"]
    rural_transfer_eff = ["in|water_supply_basin|freshwater_basin|rural_t_d|Mf"]
    irr_c = ["in|water_supply|freshwater|irrigation_cereal|M1"]
    irr_o = ["in|water_supply|freshwater|irrigation_oilcrops|M1"]
    irr_s = ["in|water_supply|freshwater|irrigation_sugarcrops|M1"]

    # ── Dynamic variables: derived from actual raw_vars via pattern matching ──
    region_withdr = [
        v for v in raw_vars
        if v.startswith("in|water_supply_basin|freshwater_basin|basin_to_reg")
    ]
    # Hydro generation/capacity: matched by pattern rather than a fixed
    # technology list because hydro_hc/hydro_lc (and their commodity/level)
    # come from the base energy model this water module is built onto, not
    # from anything defined in this module -- see _build_water_report_df's
    # WATER_TECS, which pulls in "any technology already in the scenario
    # whose name contains 'hydro'" for exactly this reason.
    hydro_generation = [
        v for v in raw_vars
        if v.startswith("out|") and "|electr|" in v
        and re.search(r"\|hydro[^|]*\|", v)
    ]
    hydro_capacity = [v for v in raw_vars if v.startswith("CAP|capacity|hydro")]

    # Real irrigation withdrawal (irr_<method>_<crop>, message_ix_models.
    # model.water.data.crops.irrigation_tech) -- built from the module's own
    # (method, crop) enumeration rather than guessed from the variable
    # string, since a method name itself contains underscores (e.g.
    # "irr_canal_lining_flood") so splitting "irr_canal_lining_flood_wheat"
    # into method vs. crop isn't reliably recoverable by regex alone.
    # Guarded: builds without this module (or its config/CSVs) still report
    # everything else normally.
    irrigation_withdrawal_by_crop: dict[str, list[str]] = {}
    irrigation_losses_by_crop: dict[str, list[str]] = {}
    try:
        from message_ix_models.model.water.data.crops.common import (
            CROPS as _CROPS,
            IRRIGATION_METHODS as _IRRIGATION_METHODS,
            irr_tech_name as _irr_tech_name,
        )
    except Exception:
        log.info("_apply_iamc_mapping: crops/irrigation_tech module unavailable")
    else:
        for _crop in _CROPS:
            _techs = {_irr_tech_name(m, _crop) for m in _IRRIGATION_METHODS}
            irrigation_withdrawal_by_crop[_crop] = [
                v for v in raw_vars
                if v.startswith("in|water_avail_basin|surfacewater_basin|")
                and v.split("|")[3] in _techs
            ]
            irrigation_losses_by_crop[_crop] = [
                v for v in raw_vars
                if v.startswith("out|water_avail_basin|groundwater_basin|")
                and v.split("|")[3] in _techs
            ]
    irrigation_withdrawal_total = [
        v for crop_vars in irrigation_withdrawal_by_crop.values() for v in crop_vars
    ]
    irrigation_losses_total = [
        v for crop_vars in irrigation_losses_by_crop.values() for v in crop_vars
    ]

    cooling_fresh_water = [
        v for v in raw_vars
        if v.startswith("in|water_supply|surfacewater|") and "fresh" in v
    ]
    cooling_ot_fresh_water = [
        v for v in raw_vars
        if v.startswith("in|water_supply") and "__ot_fresh" in v
    ]
    cooling_cl_fresh_water = [
        v for v in raw_vars
        if v.startswith("in|water_supply") and "__cl_fresh" in v
    ]
    cooling_ot_saline_water = [
        v for v in raw_vars
        if v.startswith("in|saline_supply") and "__ot_saline" in v
    ]
    fresh_return_emissions = [
        v for v in raw_vars if v.startswith("emis|fresh_return|")
    ]
    cooling_saline_inv = [
        v for v in raw_vars if v.startswith("inv cost|") and "__saline" in v
    ]
    cooling_air_inv = [
        v for v in raw_vars if v.startswith("inv cost|") and "__air" in v
    ]
    cooling_ot_fresh_inv = [
        v for v in raw_vars if v.startswith("inv cost|") and "__ot_fresh" in v
    ]
    cooling_cl_fresh_inv = [
        v for v in raw_vars if v.startswith("inv cost|") and "__cl_fresh" in v
    ]
    cooling_inv_vars = (
        cooling_ot_fresh_inv + cooling_cl_fresh_inv
        + cooling_saline_inv + cooling_air_inv
    )

    # ── Aggregation mapping: [IAMC name, [source raw vars], unit] ─────────────
    map_agg = [
        ["Water Extraction",
         extract_gw + extract_fgw + extract_sw, "MCM/yr"],
        ["Water Extraction|Groundwater", extract_gw, "MCM/yr"],
        ["Water Extraction|Fossil Groundwater", extract_fgw, "MCM/yr"],
        ["Water Extraction|Surface Water", extract_sw, "MCM/yr"],
        ["Water Extraction|Seawater",
         extract_saline_basin + extract_saline_region, "MCM/yr"],
        ["Water Extraction|Seawater|Desalination", extract_saline_basin, "MCM/yr"],
        ["Water Extraction|Seawater|Cooling", extract_saline_region, "MCM/yr"],
        ["Water Desalination", desal_membrane + desal_distill, "MCM/yr"],
        ["Water Desalination|Membrane", desal_membrane, "MCM/yr"],
        ["Water Desalination|Distillation", desal_distill, "MCM/yr"],
        ["Water Transfer",
         urban_transfer + rural_transfer
         + urban_transfer_eff + rural_transfer_eff, "MCM/yr"],
        ["Water Transfer|Urban", urban_transfer + urban_transfer_eff, "MCM/yr"],
        ["Water Transfer|Rural", rural_transfer + rural_transfer_eff, "MCM/yr"],
        ["Water Withdrawal",
         region_withdr
         + rural_mwdem_unconnected + rural_mwdem_unconnected_eff
         + rural_mwdem_connected + rural_mwdem_connected_eff
         + urban_mwdem_connected + urban_mwdem_connected_eff
         + urban_mwdem_unconnected + urban_mwdem_unconnected_eff
         + industry_mwdem_unconnected + industry_mwdem_unconnected_eff,
         "MCM/yr"],
        ["Water Withdrawal|Energy techs & Irrigation", region_withdr, "MCM/yr"],
        ["Water Withdrawal|Irrigation|Cereal", irr_c, "MCM/yr"],
        ["Water Withdrawal|Irrigation|Oil Crops", irr_o, "MCM/yr"],
        ["Water Withdrawal|Irrigation|Sugar Crops", irr_s, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water",
         rural_infrastructure + urban_infrastructure + industry_unconnected,
         "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Extraction",
         extrt_sw_cap + extrt_gw_cap + extrt_fgw_cap, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Extraction|Surface Water",
         extrt_sw_cap, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Extraction|Groundwater",
         extrt_gw_cap + extrt_fgw_cap, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Extraction|Groundwater|Renewable",
         extrt_gw_cap, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Extraction|Groundwater|Fossil",
         extrt_fgw_cap, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Rural",
         rural_infrastructure, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Urban",
         urban_infrastructure, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Industrial",
         industry_unconnected, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Treatment & Recycling|Urban",
         urban_treatment_recycling, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Treatment & Recycling|Rural",
         rural_treatment_recycling, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Distribution|Rural",
         rural_dist, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Distribution|Urban",
         urban_dist, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Unconnected|Rural",
         rural_unconnected, "MCM/yr"],
        ["Capacity Additions|Infrastructure|Water|Unconnected|Urban",
         urban_unconnected, "MCM/yr"],
        ["Freshwater|Environmental Flow", env_flow, "MCM/yr"],
        ["Groundwater Recharge", gw_recharge, "MCM/yr"],
        ["Water Withdrawal|Municipal Water",
         rural_mwdem_unconnected + rural_mwdem_unconnected_eff
         + rural_mwdem_connected + rural_mwdem_connected_eff
         + urban_mwdem_unconnected + urban_mwdem_unconnected_eff
         + urban_mwdem_connected + urban_mwdem_connected_eff,
         "MCM/yr"],
        ["Water Withdrawal|Municipal Water|Unconnected|Rural",
         rural_mwdem_unconnected, "MCM/yr"],
        ["Water Withdrawal|Municipal Water|Unconnected|Rural Eff",
         rural_mwdem_unconnected_eff, "MCM/yr"],
        ["Water Withdrawal|Municipal Water|Connected|Rural",
         rural_mwdem_connected, "MCM/yr"],
        ["Water Withdrawal|Municipal Water|Connected|Rural Eff",
         rural_mwdem_connected_eff, "MCM/yr"],
        ["Water Withdrawal|Municipal Water|Unconnected|Urban",
         urban_mwdem_unconnected, "MCM/yr"],
        ["Water Withdrawal|Municipal Water|Unconnected|Urban Eff",
         urban_mwdem_unconnected_eff, "MCM/yr"],
        ["Water Withdrawal|Municipal Water|Connected|Urban",
         urban_mwdem_connected, "MCM/yr"],
        ["Water Withdrawal|Municipal Water|Connected|Urban Eff",
         urban_mwdem_connected_eff, "MCM/yr"],
        ["Water Withdrawal|Industrial Water|Unconnected",
         industry_mwdem_unconnected, "MCM/yr"],
        ["Water Withdrawal|Industrial Water|Unconnected Eff",
         industry_mwdem_unconnected_eff, "MCM/yr"],
        ["Final Energy|Commercial",
         electr_saline + electr_gw + electr_fgw + electr_sw
         + electr_rural_trt + electr_urban_trt
         + electr_urban_recycle + electr_rural_recycle
         + electr_urban_t_d + electr_urban_t_d_eff
         + electr_rural_t_d + electr_rural_t_d_eff + electr_irr,
         "GWa"],
        ["Final Energy|Commercial|Water",
         electr_saline + electr_gw + electr_fgw + electr_sw
         + electr_rural_trt + electr_urban_trt
         + electr_urban_recycle + electr_rural_recycle
         + electr_urban_t_d + electr_urban_t_d_eff
         + electr_rural_t_d + electr_rural_t_d_eff + electr_irr,
         "GWa"],
        ["Final Energy|Commercial|Water|Desalination", electr_saline, "GWa"],
        ["Final Energy|Commercial|Water|Groundwater Extraction",
         electr_gw + electr_fgw, "GWa"],
        ["Final Energy|Commercial|Water|Surface Water Extraction", electr_sw, "GWa"],
        ["Final Energy|Commercial|Water|Irrigation", electr_irr, "GWa"],
        ["Final Energy|Commercial|Water|Treatment",
         electr_rural_trt + electr_urban_trt, "GWa"],
        ["Final Energy|Commercial|Water|Treatment|Rural", electr_rural_trt, "GWa"],
        ["Final Energy|Commercial|Water|Treatment|Urban", electr_urban_trt, "GWa"],
        ["Final Energy|Commercial|Water|Reuse", electr_urban_recycle, "GWa"],
        ["Final Energy|Commercial|Water|Transfer",
         electr_urban_t_d + electr_urban_t_d_eff
         + electr_rural_t_d + electr_rural_t_d_eff, "GWa"],
        ["Final Energy|Commercial|Water|Transfer|Urban",
         electr_urban_t_d + electr_urban_t_d_eff, "GWa"],
        ["Final Energy|Commercial|Water|Transfer|Rural",
         electr_rural_t_d + electr_rural_t_d_eff, "GWa"],
        ["Water Waste|Collected",
         urban_collctd_wstwtr + rural_collctd_wstwtr, "MCM/yr"],
        ["Water Waste|Collected|Urban", urban_collctd_wstwtr, "MCM/yr"],
        ["Water Waste|Collected|Rural", rural_collctd_wstwtr, "MCM/yr"],
        ["Water Waste|Treated",
         urban_treated_wstwtr + rural_treated_wstwtr, "MCM/yr"],
        ["Water Waste|Treated|Urban", urban_treated_wstwtr, "MCM/yr"],
        ["Water Waste|Treated|Rural", rural_treated_wstwtr, "MCM/yr"],
        ["Water Waste|Reuse",
         urban_wstwtr_recycle + rural_wstwtr_recycle, "MCM/yr"],
        ["Water Waste|Reuse|Urban", urban_wstwtr_recycle, "MCM/yr"],
        ["Water Waste|Reuse|Rural", rural_wstwtr_recycle, "MCM/yr"],
        ["Investment|Infrastructure|Water",
         rural_infrastructure_inv + urban_infrastructure_inv
         + extrt_sw_inv + extrt_gw_inv + extrt_fgw_inv
         + saline_inv + cooling_inv_vars + industry_unconnected_inv,
         "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Extraction",
         extrt_sw_inv + extrt_gw_inv + extrt_fgw_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Extraction|Surface",
         extrt_sw_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Extraction|Groundwater",
         extrt_gw_inv + extrt_fgw_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Extraction|Groundwater|Fossil",
         extrt_fgw_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Extraction|Groundwater|Renewable",
         extrt_gw_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Desalination",
         saline_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Rural",
         rural_infrastructure_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Urban",
         urban_infrastructure_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Industrial",
         industry_unconnected_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Treatment & Recycling",
         urban_treatment_recycling_inv + rural_treatment_recycling_inv,
         "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Treatment & Recycling|Urban",
         urban_treatment_recycling_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Treatment & Recycling|Rural",
         rural_treatment_recycling_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Distribution",
         rural_dist_inv + urban_dist_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Distribution|Rural",
         rural_dist_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Distribution|Urban",
         urban_dist_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Unconnected",
         rural_unconnected_inv + urban_unconnected_inv + industry_unconnected_inv,
         "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Unconnected|Rural",
         rural_unconnected_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Unconnected|Urban",
         urban_unconnected_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Cooling",
         cooling_inv_vars, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Cooling|Once through freshwater",
         cooling_ot_fresh_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Cooling|Closed loop freshwater",
         cooling_cl_fresh_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Cooling|Once through saline",
         cooling_saline_inv, "million US$2010/yr"],
        ["Investment|Infrastructure|Water|Cooling|Air cooled",
         cooling_air_inv, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water",
         rural_infrastructure_totalom + urban_infrastructure_totalom
         + extrt_sw_om + extrt_gw_om + extrt_fgw_om
         + saline_totalom + industry_unconnected_totalom,
         "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Desalination",
         saline_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Extraction",
         extrt_sw_om + extrt_gw_om + extrt_fgw_om, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Rural",
         rural_infrastructure_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Urban",
         urban_infrastructure_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Treatment & Recycling",
         urban_treatment_recycling_totalom + rural_treatment_recycling_totalom,
         "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Treatment & Recycling|Urban",
         urban_treatment_recycling_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Treatment & Recycling|Rural",
         rural_treatment_recycling_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water| Distribution",
         rural_dist_totalom + urban_dist_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Distribution|Rural",
         rural_dist_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Distribution|Urban",
         urban_dist_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Unconnected",
         rural_unconnected_totalom + urban_unconnected_totalom
         + industry_unconnected_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Unconnected|Rural",
         rural_unconnected_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Unconnected|Urban",
         urban_unconnected_totalom, "million US$2010/yr"],
        ["Total Operation Management Cost|Infrastructure|Water|Unconnected|Industry",
         industry_unconnected_totalom, "million US$2010/yr"],
        # Cooling rows
        ["Water Withdrawal|Electricity|Cooling|Fresh Water",
         cooling_fresh_water, "MCM/yr"],
        ["Water Withdrawal|Electricity|Cooling|Once Through|Fresh Water",
         cooling_ot_fresh_water, "MCM/yr"],
        ["Water Withdrawal|Electricity|Cooling|Closed Loop|Fresh Water",
         cooling_cl_fresh_water, "MCM/yr"],
        ["Water Withdrawal|Electricity|Cooling|Once Through|Saline Water",
         cooling_ot_saline_water, "MCM/yr"],
        ["Water Return|Electricity|Cooling", fresh_return_emissions, "MCM/yr"],
        ["Generation|Electricity|Hydro", hydro_generation, "GWa/yr"],
        ["Capacity|Electricity|Hydro", hydro_capacity, "GW"],
        ["Water Withdrawal|Irrigation", irrigation_withdrawal_total, "MCM/yr"],
        ["Irrigation Losses|Groundwater Recharge", irrigation_losses_total, "MCM/yr"],
    ] + [
        [f"Water Withdrawal|Irrigation|{crop.title()}", crop_vars, "MCM/yr"]
        for crop, crop_vars in irrigation_withdrawal_by_crop.items()
    ]

    # ── Apply aggregation using pandas groupby (no pyam) ─────────────────────
    id_cols = ["model", "scenario", "region", "year"]
    result_dfs = []
    for iamc_name, src_vars, unit in map_agg:
        if not src_vars:
            continue
        sub = raw_df[raw_df["variable"].isin(src_vars)]
        if sub.empty:
            continue
        agg = sub.groupby(id_cols, as_index=False)["value"].sum()
        agg["variable"] = iamc_name
        agg["unit"] = unit
        result_dfs.append(agg)

    # ── Hydro water withdrawal (ratio applied to generation, not a plain
    # sum, so it can't live in map_agg above) ────────────────────────────────
    # Port of multiply_electricity_output_of_hydro's calculation, which only
    # ever ran in the slow/legacy report() branch *after* the fast-path
    # early return below -- i.e. it never actually executed for the basin
    # models (Pakistan IRB) this repo builds. Reproduced here in pandas so
    # hydro's water use appears in the CSV every other Water Withdrawal|*
    # variable does.
    if hydro_generation:
        perf_path = package_data_path(
            "water", "ppl_cooling_tech", "tech_water_performance_ssp_msg.csv"
        )
        perf = pd.read_csv(perf_path)
        ratio_by_tech = (
            perf.set_index("technology_name")["water_withdrawal_mid_m3_per_output"]
            * m3_GJ_TO_MCM_GWa
        )
        hydro_rows = raw_df[raw_df["variable"].isin(hydro_generation)].copy()
        hydro_rows["tech"] = hydro_rows["variable"].str.extract(r"\|(hydro[^|]*)\|")
        hydro_rows["ratio"] = hydro_rows["tech"].map(ratio_by_tech)
        unmatched = sorted(set(hydro_rows.loc[hydro_rows["ratio"].isna(), "tech"]))
        if unmatched:
            log.warning(
                "_apply_iamc_mapping: no water-withdrawal performance data for "
                f"hydro technologies {unmatched}; excluded from "
                "Water Withdrawal|Electricity|Hydro*"
            )
        hydro_rows = hydro_rows.dropna(subset=["ratio"])
        if not hydro_rows.empty:
            hydro_rows["value"] = hydro_rows["value"] * hydro_rows["ratio"]
            total = hydro_rows.groupby(id_cols, as_index=False)["value"].sum()
            total["variable"] = "Water Withdrawal|Electricity|Hydro"
            total["unit"] = "MCM/yr"
            result_dfs.append(total)
            for tech, sub in hydro_rows.groupby("tech"):
                t_agg = sub.groupby(id_cols, as_index=False)["value"].sum()
                t_agg["variable"] = f"Water Withdrawal|Electricity|Hydro|{tech}"
                t_agg["unit"] = "MCM/yr"
                result_dfs.append(t_agg)

    if not result_dfs:
        log.warning("_apply_iamc_mapping: no mappable variables found — returning raw")
        return raw_df

    out = pd.concat(result_dfs, ignore_index=True)
    return out[["model", "scenario", "region", "variable", "unit", "year", "value"]]


def report(sc: Scenario, reg: str, ssp: str, sdgs: bool = False) -> None:
    """Report nexus module results

    Parameters
    ----------
    sc : Scenario
        Scenario to report
    reg : str
        Region to report
    ssp : str
        SSP scenario (e.g., :obj:`"SSP1"`, :obj:`"SSP2"`, :obj:`"SSP3"`)
    sdgs : bool, optional
        If :obj:`True`, add population with access to water and sanitation for SDG6
    """
    log.info(f"Regions given as {reg}; no warranty if it's not in ['R11','R12']")

    # Build report_df directly from sc.var()/sc.par() — bypasses message::default.
    # message::default fails on large models in two ways:
    #   1. Computes results for all 500+ technologies → hangs for hours
    #   2. Raises pint.DimensionalityError on USD_2005/MCM water cost units
    # _build_water_report_df() produces the identical wide-format DataFrame
    # with the same variable naming convention, but only for water technologies.
    report_df = _build_water_report_df(sc)

    # ── FAST PATH ────────────────────────────────────────────────────────────
    # For large basin models (Pakistan IRB, 32 nodes, 500+ technologies) the
    # pyam aggregation loop below calls aggregate_region() ~1000 times and
    # hangs for 25+ minutes.  Skip it: melt the wide-format DataFrame to long
    # format, apply IAMC variable name mapping via pure pandas, then write CSV.
    if not report_df.empty:
        _id_cols = ["Model", "Scenario", "Region", "Variable", "Unit"]
        _yr_cols = [c for c in report_df.columns if c not in _id_cols]
        _long = report_df.melt(
            id_vars=_id_cols, value_vars=_yr_cols,
            var_name="year", value_name="value",
        )
        _long = _long.dropna(subset=["value"])
        _long = _long[_long["value"] != 0]
        _long["year"] = _long["year"].astype(int)
        _long.columns = [c.lower() for c in _long.columns]

        # Convert raw MESSAGE variable names → IAMC-style aggregated names.
        # This replicates map_agg_pd aggregation without slow pyam calls.
        _iamc_long = _apply_iamc_mapping(_long)

        _out_path = package_data_path().parents[0] / "reporting_output"
        _out_path.mkdir(exist_ok=True)
        # Use versioned filename so re-runs never overwrite an existing CSV.
        _base = f"{sc.model}_{sc.scenario}_nexus"
        _out_file = _out_path / f"{_base}.csv"
        if _out_file.exists():
            _i = 1
            while (_out_path / f"{_base}_v{_i}.csv").exists():
                _i += 1
            _out_file = _out_path / f"{_base}_v{_i}.csv"

        _iamc_long.to_csv(_out_file, index=False)
        log.info(f"CSV saved (fast path, IAMC names): {_out_file}")
        return
    # ── END FAST PATH ────────────────────────────────────────────────────────

    # Reporter is still needed for the demand query below (demand:n-c-l-y-h)
    rep = Reporter.from_scenario(sc)

    # Removing duplicate region names (handles "IRB|Pakistan" → "Pakistan" etc.)
    if not report_df.empty:
        report_df["Region"] = remove_duplicate(report_df)

    # Adding Water availability as resource in demands
    # This is not automatically reported using message:default
    rep_dm = Reporter.from_scenario(sc)
    rep_dm.set_filters(l="water_avail_basin")

    rep_dm2 = rep.get("demand:n-c-l-y-h")
    rep_dm_df = rep_dm2.to_dataframe()
    rep_dm_df.reset_index(inplace=True)
    df_dmd = rep_dm_df[rep_dm_df["l"] == "water_avail_basin"]
    # Detect scenario mode (subannual and cooling-only)
    suban, cooling_only = detect_scenario_mode(df_dmd)

    # if subannual, get and subsittute variables
    report_iam = report_iam_definition(sc, rep, df_dmd, rep_dm, report_df, suban)

    # mapping model outputs for aggregation
    urban_infrastructure = [
        "CAP_NEW|new capacity|urban_recycle",
        "CAP_NEW|new capacity|urban_sewerage",
        "CAP_NEW|new capacity|urban_t_d",
        "CAP_NEW|new capacity|urban_treatment",
        "CAP_NEW|new capacity|urban_unconnected",
        "CAP_NEW|new capacity|urban_untreated",
    ]

    rural_infrastructure = [
        "CAP_NEW|new capacity|rural_recycle",
        "CAP_NEW|new capacity|rural_sewerage",
        "CAP_NEW|new capacity|rural_t_d",
        "CAP_NEW|new capacity|rural_treatment",
        "CAP_NEW|new capacity|rural_unconnected",
        "CAP_NEW|new capacity|rural_untreated",
    ]

    urban_treatment_recycling = [
        "CAP_NEW|new capacity|urban_recycle",
        "CAP_NEW|new capacity|urban_sewerage",
        "CAP_NEW|new capacity|urban_treatment",
    ]

    rural_treatment_recycling = [
        "CAP_NEW|new capacity|rural_recycle",
        "CAP_NEW|new capacity|rural_sewerage",
        "CAP_NEW|new capacity|rural_treatment",
    ]

    rural_dist = ["CAP_NEW|new capacity|rural_t_d"]
    urban_dist = ["CAP_NEW|new capacity|urban_t_d"]

    rural_unconnected = [
        "CAP_NEW|new capacity|rural_unconnected",
        "CAP_NEW|new capacity|rural_untreated",
    ]

    urban_unconnected = [
        "CAP_NEW|new capacity|urban_unconnected",
        "CAP_NEW|new capacity|urban_untreated",
    ]

    industry_unconnected = [
        "CAP_NEW|new capacity|industry_unconnected",
        "CAP_NEW|new capacity|industry_untreated",
    ]

    extrt_sw_cap = ["CAP_NEW|new capacity|extract_surfacewater"]
    extrt_gw_cap = ["CAP_NEW|new capacity|extract_groundwater"]
    extrt_fgw_cap = ["CAP_NEW|new capacity|extract_gw_fossil"]

    extrt_sw_inv = ["inv cost|extract_surfacewater"]
    extrt_gw_inv = ["inv cost|extract_groundwater"]
    # Calculating fossil groundwater invwatments
    # 163.56 million USD/KM^3 x 2 times the renewable gw costs

    report_iam = report_iam.append(
        report_iam.multiply(
            "CAP_NEW|new capacity|extract_gw_fossil",
            163.56 * USD_KM3_TO_USD_MCM,
            "Fossil GW inv",
            ignore_units=True,
        )
    )
    extrt_fgw_inv = report_iam.filter(variable="Fossil GW inv").variable

    rural_infrastructure_inv = [
        "inv cost|rural_recycle",
        "inv cost|rural_sewerage",
        "inv cost|rural_t_d",
        "inv cost|rural_treatment",
        "inv cost|rural_unconnected",
        "inv cost|rural_untreated",
    ]

    urban_infrastructure_inv = [
        "inv cost|urban_recycle",
        "inv cost|urban_sewerage",
        "inv cost|urban_t_d",
        "inv cost|urban_treatment",
        "inv cost|urban_unconnected",
        "inv cost|urban_untreated",
    ]

    urban_treatment_recycling_inv = [
        "inv cost|urban_recycle",
        "inv cost|urban_sewerage",
        "inv cost|urban_treatment",
    ]

    rural_treatment_recycling_inv = [
        "inv cost|rural_recycle",
        "inv cost|rural_sewerage",
        "inv cost|rural_treatment",
    ]

    rural_dist_inv = ["inv cost|rural_t_d"]
    urban_dist_inv = ["inv cost|urban_t_d"]

    rural_unconnected_inv = [
        "inv cost|rural_unconnected",
        "inv cost|rural_untreated",
    ]

    urban_unconnected_inv = [
        "inv cost|urban_unconnected",
        "inv cost|urban_untreated",
    ]

    industry_unconnected_inv = [
        "inv cost|industry_unconnected",
        "inv cost|industry_untreated",
    ]

    saline_inv = [
        "inv cost|membrane",
        "inv cost|distillation",
    ]
    saline_totalom = [
        "total om cost|membrane",
        "total om cost|distillation",
    ]

    extrt_sw_om = ["total om cost|extract_surfacewater"]
    extrt_gw_om = ["total om cost|extract_groundwater"]
    extrt_fgw_om = ["total om cost|extract_gw_fossil"]

    urban_infrastructure_totalom = [
        "total om cost|urban_recycle",
        "total om cost|urban_sewerage",
        "total om cost|urban_t_d",
        "total om cost|urban_treatment",
        "total om cost|urban_unconnected",
        "total om cost|urban_untreated",
    ]

    rural_infrastructure_totalom = [
        "total om cost|rural_recycle",
        "total om cost|rural_sewerage",
        "total om cost|rural_t_d",
        "total om cost|rural_treatment",
        "total om cost|rural_unconnected",
        "total om cost|rural_untreated",
    ]

    rural_treatment_recycling_totalom = [
        "total om cost|rural_recycle",
        "total om cost|rural_sewerage",
        "total om cost|rural_treatment",
    ]

    urban_treatment_recycling_totalom = [
        "total om cost|urban_recycle",
        "total om cost|urban_sewerage",
        "total om cost|urban_treatment",
    ]

    rural_dist_totalom = ["total om cost|rural_t_d"]
    urban_dist_totalom = ["total om cost|urban_t_d"]

    rural_unconnected_totalom = [
        "total om cost|rural_unconnected",
        "total om cost|rural_untreated",
    ]

    urban_unconnected_totalom = [
        "total om cost|urban_unconnected",
        "total om cost|urban_untreated",
    ]

    industry_unconnected_totalom = [
        "total om cost|industry_unconnected",
        "total om cost|industry_untreated",
    ]

    extract_sw = ["in|water_avail_basin|surfacewater_basin|extract_surfacewater|M1"]

    extract_gw = ["in|water_avail_basin|groundwater_basin|extract_groundwater|M1"]
    extract_fgw = ["out|water_supply_basin|freshwater_basin|extract_gw_fossil|M1"]

    desal_membrane = ["out|water_supply_basin|freshwater_basin|membrane|M1"]
    desal_distill = ["out|water_supply_basin|freshwater_basin|distillation|M1"]
    env_flow = ["in|water_avail_basin|surfacewater_basin|return_flow|M1"]
    gw_recharge = ["in|water_avail_basin|groundwater_basin|gw_recharge|M1"]

    rural_mwdem_unconnected = ["out|final|rural_disconnected|rural_unconnected|M1"]
    rural_mwdem_unconnected_eff = ["out|final|rural_disconnected|rural_unconnected|Mf"]
    rural_mwdem_connected = ["out|final|rural_mw|rural_t_d|M1"]
    rural_mwdem_connected_eff = ["out|final|rural_mw|rural_t_d|Mf"]
    urban_mwdem_unconnected = ["out|final|urban_disconnected|urban_unconnected|M1"]
    urban_mwdem_unconnected_eff = ["out|final|urban_disconnected|urban_unconnected|Mf"]
    urban_mwdem_connected = ["out|final|urban_mw|urban_t_d|M1"]
    urban_mwdem_connected_eff = ["out|final|urban_mw|urban_t_d|Mf"]
    industry_mwdem_unconnected = ["out|final|industry_mw|industry_unconnected|M1"]
    industry_mwdem_unconnected_eff = ["out|final|industry_mw|industry_unconnected|Mf"]
    electr_gw = ["in|final|electr|extract_groundwater|M1"]
    electr_fgw = ["in|final|electr|extract_gw_fossil|M1"]
    electr_sw = ["in|final|electr|extract_surfacewater|M1"]
    extract_saline_region = ["out|saline_supply|saline_ppl|extract_salinewater_cool|M1"]
    extract_saline_basin = [
        "out|water_avail_basin|salinewater_basin|extract_salinewater_basin|M1"
    ]
    electr_rural_trt = ["in|final|electr|rural_sewerage|M1"]
    electr_urban_trt = ["in|final|electr|urban_sewerage|M1"]
    electr_urban_recycle = ["in|final|electr|urban_recycle|M1"]
    electr_rural_recycle = ["in|final|electr|rural_recycle|M1"]
    electr_saline = [
        "in|final|electr|distillation|M1",
        "in|final|electr|membrane|M1",
    ]

    electr_urban_t_d = ["in|final|electr|urban_t_d|M1"]
    electr_urban_t_d_eff = ["in|final|electr|urban_t_d|Mf"]
    electr_rural_t_d = ["in|final|electr|rural_t_d|M1"]
    electr_rural_t_d_eff = ["in|final|electr|rural_t_d|Mf"]

    electr_irr = [
        "in|final|electr|irrigation_cereal|M1",
        "in|final|electr|irrigation_oilcrops|M1",
        "in|final|electr|irrigation_sugarcrops|M1",
    ]

    urban_collctd_wstwtr = ["in|final|urban_collected_wst|urban_sewerage|M1"]
    rural_collctd_wstwtr = ["in|final|rural_collected_wst|rural_sewerage|M1"]

    urban_treated_wstwtr = ["in|water_treat|urban_collected_wst|urban_recycle|M1"]
    rural_treated_wstwtr = ["in|water_treat|rural_collected_wst|rural_recycle|M1"]

    urban_wstwtr_recycle = ["out|water_supply_basin|freshwater_basin|urban_recycle|M1"]
    rural_wstwtr_recycle = ["out|water_supply_basin|freshwater_basin|rural_recycle|M1"]

    urban_transfer = ["in|water_supply_basin|freshwater_basin|urban_t_d|M1"]
    urban_transfer_eff = ["in|water_supply_basin|freshwater_basin|urban_t_d|Mf"]
    rural_transfer = ["in|water_supply_basin|freshwater_basin|rural_t_d|M1"]
    rural_transfer_eff = ["in|water_supply_basin|freshwater_basin|rural_t_d|Mf"]

    # irr_water = ["out|water_irr|freshwater|irrigation|M1"]

    irr_c = ["in|water_supply|freshwater|irrigation_cereal|M1"]
    irr_o = ["in|water_supply|freshwater|irrigation_oilcrops|M1"]
    irr_s = ["in|water_supply|freshwater|irrigation_sugarcrops|M1"]

    region_withdr = report_iam.filter(
        variable="in|water_supply_basin|freshwater_basin|basin_to_reg|*"
    ).variable

    # Process cooling technologies
    report_iam, cooling_rows = compute_cooling_technologies(report_iam, sc)

    # mapping for aggregation
    map_agg_pd = pd.DataFrame(
        [
            ["Water Extraction", extract_gw + extract_fgw + extract_sw, "MCM/yr"],
            ["Water Extraction|Groundwater", extract_gw, "MCM/yr"],
            [
                "Water Extraction|Fossil Groundwater",
                extract_fgw,
                "MCM/yr",
            ],
            ["Water Extraction|Surface Water", extract_sw, "MCM/yr"],
            [
                "Water Extraction|Seawater",
                extract_saline_basin + extract_saline_region,
                "MCM/yr",
            ],
            ["Water Extraction|Seawater|Desalination", extract_saline_basin, "MCM/yr"],
            ["Water Extraction|Seawater|Cooling", extract_saline_region, "MCM/yr"],
            ["Water Desalination", desal_membrane + desal_distill, "MCM/yr"],
            ["Water Desalination|Membrane", desal_membrane, "MCM/yr"],
            ["Water Desalination|Distillation", desal_distill, "MCM/yr"],
            [
                "Water Transfer",
                urban_transfer
                + rural_transfer
                + urban_transfer_eff
                + rural_transfer_eff,
                "MCM/yr",
            ],
            ["Water Transfer|Urban", urban_transfer + urban_transfer_eff, "MCM/yr"],
            ["Water Transfer|Rural", rural_transfer + rural_transfer_eff, "MCM/yr"],
            [
                "Water Withdrawal",
                region_withdr
                + rural_mwdem_unconnected
                + rural_mwdem_unconnected_eff
                + rural_mwdem_connected
                + rural_mwdem_connected_eff
                + urban_mwdem_connected
                + urban_mwdem_connected_eff
                + urban_mwdem_unconnected
                + urban_mwdem_unconnected_eff
                + industry_mwdem_unconnected
                + industry_mwdem_unconnected_eff,
                "MCM/yr",
            ],
            ["Water Withdrawal|Energy techs & Irrigation", region_withdr, "MCM/yr"],
            # ["Water Withdrawal|Irrigation", irr_c + irr_o + irr_s, "MCM/yr"],
            ["Water Withdrawal|Irrigation|Cereal", irr_c, "MCM/yr"],
            ["Water Withdrawal|Irrigation|Oil Crops", irr_o, "MCM/yr"],
            ["Water Withdrawal|Irrigation|Sugar Crops", irr_s, "MCM/yr"],
            [
                "Capacity Additions|Infrastructure|Water",
                # Removed sub-components to avoid double-counting since:
                # - *_infrastructure
                # already includes
                # *_treatment_recycling, *_dist, *_unconnected
                rural_infrastructure + urban_infrastructure + industry_unconnected,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Extraction",
                extrt_sw_cap + extrt_gw_cap + extrt_fgw_cap,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Extraction|Surface Water",
                extrt_sw_cap,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Extraction|Groundwater",
                extrt_gw_cap + extrt_fgw_cap,
                "MCM/yr",
            ],
            [
                (
                    "Capacity"
                    " Additions|Infrastructure|Water|Extraction|Groundwater|Renewable"
                ),
                extrt_gw_cap,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Extraction|Groundwater|Fossil",
                extrt_fgw_cap,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Rural",
                rural_infrastructure,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Urban",
                urban_infrastructure,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Industrial",
                industry_unconnected,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Treatment & Recycling|Urban",
                urban_treatment_recycling,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Treatment & Recycling|Rural",
                rural_treatment_recycling,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Distribution|Rural",
                rural_dist,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Distribution|Urban",
                urban_dist,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Unconnected|Rural",
                rural_unconnected,
                "MCM/yr",
            ],
            [
                "Capacity Additions|Infrastructure|Water|Unconnected|Urban",
                urban_unconnected,
                "MCM/yr",
            ],
            ["Freshwater|Environmental Flow", env_flow, "MCM/yr"],
            ["Groundwater Recharge", gw_recharge, "MCM/yr"],
            [
                "Water Withdrawal|Municipal Water",
                rural_mwdem_unconnected
                + rural_mwdem_unconnected_eff
                + rural_mwdem_connected
                + rural_mwdem_connected_eff
                + urban_mwdem_unconnected
                + urban_mwdem_unconnected_eff
                + urban_mwdem_connected
                + urban_mwdem_connected_eff,
                "MCM/yr",
            ],
            [
                "Water Withdrawal|Municipal Water|Unconnected|Rural",
                rural_mwdem_unconnected,
                "MCM/yr",
            ],
            [
                "Water Withdrawal|Municipal Water|Unconnected|Rural Eff",
                rural_mwdem_unconnected_eff,
                "MCM/yr",
            ],
            [
                "Water Withdrawal|Municipal Water|Connected|Rural",
                rural_mwdem_connected,
                "MCM/yr",
            ],
            [
                "Water Withdrawal|Municipal Water|Connected|Rural Eff",
                rural_mwdem_connected_eff,
                "MCM/yr",
            ],
            [
                "Water Withdrawal|Municipal Water|Unconnected|Urban",
                urban_mwdem_unconnected,
                "MCM/yr",
            ],
            [
                "Water Withdrawal|Municipal Water|Unconnected|Urban Eff",
                urban_mwdem_unconnected_eff,
                "MCM/yr",
            ],
            [
                "Water Withdrawal|Municipal Water|Connected|Urban",
                urban_mwdem_connected,
                "MCM/yr",
            ],
            [
                "Water Withdrawal|Municipal Water|Connected|Urban Eff",
                urban_mwdem_connected_eff,
                "MCM/yr",
            ],
            [
                "Water Withdrawal|Industrial Water|Unconnected",
                industry_mwdem_unconnected,
                "MCM/yr",
            ],
            [
                "Water Withdrawal|Industrial Water|Unconnected Eff",
                industry_mwdem_unconnected_eff,
                "MCM/yr",
            ],
            # ["Water Withdrawal|Irrigation", irr_water, "MCM/yr"],
            [
                "Final Energy|Commercial",
                electr_saline
                + electr_gw
                + electr_fgw
                + electr_sw
                + electr_rural_trt
                + electr_urban_trt
                + electr_urban_recycle
                + electr_rural_recycle
                + electr_urban_t_d
                + electr_urban_t_d_eff
                + electr_rural_t_d
                + electr_rural_t_d_eff
                + electr_irr,
                "GWa",
            ],
            [
                "Final Energy|Commercial|Water",
                electr_saline
                + electr_gw
                + electr_fgw
                + electr_sw
                + electr_rural_trt
                + electr_urban_trt
                + electr_urban_recycle
                + electr_rural_recycle
                + electr_urban_t_d
                + electr_urban_t_d_eff
                + electr_rural_t_d
                + electr_rural_t_d_eff
                + electr_irr,
                "GWa",
            ],
            ["Final Energy|Commercial|Water|Desalination", electr_saline, "GWa"],
            [
                "Final Energy|Commercial|Water|Groundwater Extraction",
                electr_gw + electr_fgw,
                "GWa",
            ],
            [
                "Final Energy|Commercial|Water|Surface Water Extraction",
                electr_sw,
                "GWa",
            ],
            ["Final Energy|Commercial|Water|Irrigation", electr_irr, "GWa"],
            [
                "Final Energy|Commercial|Water|Treatment",
                electr_rural_trt + electr_urban_trt,
                "GWa",
            ],
            [
                "Final Energy|Commercial|Water|Treatment|Rural",
                electr_rural_trt,
                "GWa",
            ],
            ["Final Energy|Commercial|Water|Treatment|Urban", electr_urban_trt, "GWa"],
            ["Final Energy|Commercial|Water|Reuse", electr_urban_recycle, "GWa"],
            [
                "Final Energy|Commercial|Water|Transfer",
                electr_urban_t_d
                + electr_urban_t_d_eff
                + electr_rural_t_d
                + electr_rural_t_d_eff,
                "GWa",
            ],
            [
                "Final Energy|Commercial|Water|Transfer|Urban",
                electr_urban_t_d + electr_urban_t_d_eff,
                "GWa",
            ],
            [
                "Final Energy|Commercial|Water|Transfer|Rural",
                electr_rural_t_d + electr_rural_t_d_eff,
                "GWa",
            ],
            [
                "Water Waste|Collected",
                urban_collctd_wstwtr + rural_collctd_wstwtr,
                "MCM/yr",
            ],
            ["Water Waste|Collected|Urban", urban_collctd_wstwtr, "MCM/yr"],
            ["Water Waste|Collected|Rural", rural_collctd_wstwtr, "MCM/yr"],
            [
                "Water Waste|Treated",
                urban_treated_wstwtr + rural_treated_wstwtr,
                "MCM/yr",
            ],
            ["Water Waste|Treated|Urban", urban_treated_wstwtr, "MCM/yr"],
            ["Water Waste|Treated|Rural", rural_treated_wstwtr, "MCM/yr"],
            [
                "Water Waste|Reuse",
                urban_wstwtr_recycle + rural_wstwtr_recycle,
                "MCM/yr",
            ],
            ["Water Waste|Reuse|Urban", urban_wstwtr_recycle, "MCM/yr"],
            ["Water Waste|Reuse|Rural", rural_wstwtr_recycle, "MCM/yr"],
            [
                "Investment|Infrastructure|Water",
                rural_infrastructure_inv
                + urban_infrastructure_inv
                + extrt_sw_inv
                + extrt_gw_inv
                + extrt_fgw_inv
                + saline_inv
                + (getattr(report_iam, "metadata", {}).get("cooling_inv_vars", []))
                + industry_unconnected_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Extraction",
                extrt_sw_inv + extrt_gw_inv + extrt_fgw_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Extraction|Surface",
                extrt_sw_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Extraction|Groundwater",
                extrt_gw_inv + extrt_fgw_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Extraction|Groundwater|Fossil",
                extrt_fgw_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Extraction|Groundwater|Renewable",
                extrt_gw_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Desalination",
                saline_inv,
                "million US$2010/yr",
            ],
            # [
            #     "Investment|Infrastructure|Water",
            #     rural_infrastructure_inv + urban_infrastructure_inv,
            #     "million US$2010/yr",
            # ],
            [
                "Investment|Infrastructure|Water|Rural",
                rural_infrastructure_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Urban",
                urban_infrastructure_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Industrial",
                industry_unconnected_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Treatment & Recycling",
                urban_treatment_recycling_inv + rural_treatment_recycling_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Treatment & Recycling|Urban",
                urban_treatment_recycling_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Treatment & Recycling|Rural",
                rural_treatment_recycling_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Distribution",
                rural_dist_inv + urban_dist_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Distribution|Rural",
                rural_dist_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Distribution|Urban",
                urban_dist_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Unconnected",
                rural_unconnected_inv
                + urban_unconnected_inv
                + industry_unconnected_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Unconnected|Rural",
                rural_unconnected_inv,
                "million US$2010/yr",
            ],
            [
                "Investment|Infrastructure|Water|Unconnected|Urban",
                urban_unconnected_inv,
                "million US$2010/yr",
            ],
            [
                "Total Operation Management Cost|Infrastructure|Water",
                # Main aggregation including all infrastructure O&M costs
                rural_infrastructure_totalom
                + urban_infrastructure_totalom
                + extrt_sw_om
                + extrt_gw_om
                + extrt_fgw_om
                + saline_totalom
                + industry_unconnected_totalom,
                "million US$2010/yr",
            ],
            [
                "Total Operation Management Cost|Infrastructure|Water|Desalination",
                saline_totalom,
                "million US$2010/yr",
            ],
            [
                "Total Operation Management Cost|Infrastructure|Water|Extraction",
                extrt_sw_om + extrt_gw_om + extrt_fgw_om,
                "million US$2010/yr",
            ],
            [
                "Total Operation Management Cost|Infrastructure|Water|Rural",
                rural_infrastructure_totalom,
                "million US$2010/yr",
            ],
            [
                "Total Operation Management Cost|Infrastructure|Water|Urban",
                urban_infrastructure_totalom,
                "million US$2010/yr",
            ],
            [
                (
                    "Total Operation Management Cost|Infrastructure|Water|Treatment &"
                    " Recycling"
                ),
                urban_treatment_recycling_totalom + rural_treatment_recycling_totalom,
                "million US$2010/yr",
            ],
            [
                (
                    "Total Operation Management Cost|Infrastructure|Water|Treatment &"
                    " Recycling|Urban"
                ),
                urban_treatment_recycling_totalom,
                "million US$2010/yr",
            ],
            [
                (
                    "Total Operation Management Cost|Infrastructure|Water|Treatment &"
                    " Recycling|Rural"
                ),
                rural_treatment_recycling_totalom,
                "million US$2010/yr",
            ],
            [
                "Total Operation Management Cost|Infrastructure|Water| Distribution",
                rural_dist_totalom + urban_dist_totalom,
                "million US$2010/yr",
            ],
            [
                (
                    "Total Operation Management"
                    " Cost|Infrastructure|Water|Distribution|Rural"
                ),
                rural_dist_totalom,
                "million US$2010/yr",
            ],
            [
                (
                    "Total Operation Management"
                    " Cost|Infrastructure|Water|Distribution|Urban"
                ),
                urban_dist_totalom,
                "million US$2010/yr",
            ],
            [
                "Total Operation Management Cost|Infrastructure|Water|Unconnected",
                rural_unconnected_totalom
                + urban_unconnected_totalom
                + industry_unconnected_totalom,
                "million US$2010/yr",
            ],
            [
                (
                    "Total Operation Management"
                    " Cost|Infrastructure|Water|Unconnected|Rural"
                ),
                rural_unconnected_totalom,
                "million US$2010/yr",
            ],
            [
                (
                    "Total Operation Management"
                    " Cost|Infrastructure|Water|Unconnected|Urban"
                ),
                urban_unconnected_totalom,
                "million US$2010/yr",
            ],
            [
                (
                    "Total Operation Management"
                    " Cost|Infrastructure|Water|Unconnected|Industry"
                ),
                industry_unconnected_totalom,
                "million US$2010/yr",
            ],
        ]
        + cooling_rows,  # Add cooling rows here
        columns=["names", "list_cat", "unit"],
    )

    # Add water prices, ad-hoc procedure
    wp = sc.var(
        "PRICE_COMMODITY", {"commodity": ["urban_mw", "rural_mw", "freshwater"]}
    )
    wp["value"] = wp["lvl"]
    wp["unit"] = "US$2010/m3"  # MillionUSD/MCM = USD/m^3
    wp = wp.rename(columns={"node": "region"})
    # get withdrawals for weighted mean
    ww = prepare_ww(ww_input=report_iam.as_pandas(), suban=suban)
    # irrigation water, at regional level
    # need to update for global model now we have 3 irrigation
    # probably will need to do a scaled agerave with the ww, no basin level
    # for country model, still to be defined
    # TODOOOO
    # wp_irr = wp[wp.level == "water_irr"]
    # wp_irr["variable"] = "Price|Irrigation Water"
    # wp_irr = wp_irr.drop(columns={"level", "lvl", "mrg"})
    # driking water
    wr_dri = wp[wp.commodity.isin(["urban_mw", "rural_mw"])]
    wr_dri = wr_dri.drop(columns={"level", "lvl", "mrg"})
    wr_dri = wr_dri.rename(columns={"time": "subannual"}) if suban else wr_dri
    wr_dri = wr_dri.merge(ww, how="left")
    wr_dri["variable"] = np.where(
        wr_dri.commodity == "urban_mw",
        "Price|Drinking Water|Urban",
        "Price|Drinking Water|Rural",
    )

    _gcols = (
        ["region", "unit", "year"] if not suban
        else ["region", "unit", "year", "subannual"]
    )
    _tmp_wa = wr_dri[_gcols + ["value", "wdr"]].copy()
    _tmp_wa.index = range(len(_tmp_wa))
    _tmp_wa["wdr"] = _tmp_wa["wdr"].fillna(0)
    _tmp_wa["_vw"] = _tmp_wa["value"] * _tmp_wa["wdr"]

    # Use SeriesGroupBy (not DataFrameGroupBy.apply) — always returns a proper
    # Series regardless of pandas version. Avoids the pandas 2.2 regression
    # where DataFrameGroupBy.apply with a scalar-returning function returns an
    # empty DataFrame instead of a Series.
    _gb = _tmp_wa.groupby(_gcols)
    _vw_sum = _gb["_vw"].sum()   # sum of value*weight per group
    _w_sum = _gb["wdr"].sum()    # sum of weights per group
    _v_mean = _gb["value"].mean()  # fallback when all weights are zero
    _wavg = np.where(_w_sum.values == 0, _v_mean.values, _vw_sum.values / _w_sum.values)

    if isinstance(_vw_sum.index, pd.MultiIndex):
        wr_dri_m = pd.DataFrame({c: _vw_sum.index.get_level_values(c) for c in _gcols})
    elif not _vw_sum.index.empty:
        wr_dri_m = pd.DataFrame({_gcols[0]: _vw_sum.index.values})
    else:
        wr_dri_m = pd.DataFrame(columns=_gcols)
    wr_dri_m["value"] = _wavg
    wr_dri_m["variable"] = "Price|Drinking Water"

    wp = pd.concat(
        [
            wr_dri,
            # wp_irr, # TEMP
            wr_dri_m,
        ]
    )

    wp["model"] = sc.model
    wp["scenario"] = sc.scenario
    col_ex = report_iam.as_pandas().columns[report_iam.as_pandas().columns != "exclude"]
    wp = wp[col_ex]

    wp = wp.drop_duplicates()
    wp_iam = pyam.IamDataFrame(wp)
    # Merge both dataframes in pyam
    report_iam = report_iam.append(wp_iam)

    # Fetching nodes from the scenario to aggregate to MESSAGE energy region definition
    map_node = sc.set("map_node")
    map_node = map_node[map_node["node_parent"] != map_node["node"]]
    map_node_dict = map_node.groupby("node_parent")["node"].apply(list).to_dict()

    for index, row in map_agg_pd.iterrows():
        log.info(f"Processing {row['names']}")
        # Aggregates variables as per standard reporting
        report_iam.aggregate(row["names"], components=row["list_cat"], append=True)

        if row["names"] in (
            "Water Extraction|Seawater|Cooling",
            "Investment|Infrastructure|Water",
            "Water Extraction|Seawater",
        ):
            report_iam.aggregate_region(row["names"], append=True)
        else:
            for rr in map_node_dict:
                report_iam.aggregate_region(
                    row["names"], region=rr, subregions=map_node_dict[rr], append=True
                )
    # Aggregates variables separately that are not included map_agg_pd
    for rr in map_node_dict:
        report_iam.aggregate_region(
            "Water Resource|*", region=rr, subregions=map_node_dict[rr], append=True
        )
        report_iam.aggregate_region(
            "Price|*",
            method="mean",
            region=rr,
            subregions=map_node_dict[rr],
            append=True,
        )

    # Remove duplicate variables
    varsexclude = [
        "Investment|Infrastructure|Water",
        "Investment|Infrastructure|Water|Extraction",
        "Investment|Infrastructure|Water|Other",
        "Investment|Infrastructure|Water|Extraction|Groundwater",
    ]
    report_iam.filter(variable=varsexclude, unit="unknown", keep=False, inplace=True)
    # prepare data for loading timeserie
    report_pd = report_iam.as_pandas()
    # old code left, to be revised
    # report_pd = report_pd.drop(columns=["exclude"])
    # all initial variables form Reporte will be filtered out
    d = report_df.Variable.unique()
    d1 = pd.DataFrame({"variable": d})
    d1[["to_keep"]] = "No"
    # filter out initial variables
    report_pd = report_pd.merge(d1, how="left")
    report_pd = report_pd[report_pd["to_keep"] != "No"]
    report_pd = report_pd.drop(columns=["to_keep"])

    # ecluded other intermediate variables added later to report_iam
    if hasattr(report_iam, "metadata") and "water_hydro_var" in report_iam.metadata:
        report_pd = report_pd[
            -report_pd.variable.isin(report_iam.metadata["water_hydro_var"])
        ]

    # add water population
    pop_sdg6 = pop_water_access(sc, reg, ssp, sdgs)
    report_pd = pd.concat([report_pd, pop_sdg6])

    # add units wo loop to reduce complexity
    unit_mapping = dict(zip(map_agg_pd["names"], map_agg_pd["unit"]))

    # Add units for new water access variables
    water_access_units = {
        "Connection Rate|Drinking Water": "percent",
        "Connection Rate|Drinking Water|Urban": "percent",
        "Connection Rate|Drinking Water|Rural": "percent",
        "Treatment Rate|Sanitation": "percent",
        "Treatment Rate|Sanitation|Urban": "percent",
        "Treatment Rate|Sanitation|Rural": "percent",
        "Population|Drinking Water Access": "million",
        "Population|Drinking Water Access|Urban": "million",
        "Population|Drinking Water Access|Rural": "million",
        "Population|Sanitation Access": "million",
        "Population|Sanitation Access|Urban": "million",
        "Population|Sanitation Access|Rural": "million",
        "Population": "million",
        "Population|Urban": "million",
        "Population|Rural": "million",
    }

    # Merge both unit mappings
    unit_mapping.update(water_access_units)

    report_pd["unit"] = (
        report_pd["variable"].map(unit_mapping).fillna(report_pd["unit"])
    )

    # Unit conversions (GWa → EJ, million USD → billion USD) are for global
    # energy models. The Pakistan water nexus model only has MCM/yr and MUSD/yr
    # variables — pyam's IamDataFrame constructor hangs on those non-standard
    # units. Skip the pyam round-trip entirely; apply only the EJ rename.
    if not report_pd.empty and "GWa" in report_pd["unit"].values:
        try:
            df_unit = pyam.IamDataFrame(report_pd)
            df_unit.convert_unit("GWa", to="EJ", inplace=True)
            df_unit_inv = df_unit.filter(variable="Investment*")
            df_unit_inv.convert_unit(
                "million US$2010/yr", to="billion US$2010/yr",
                factor=0.001, inplace=True,
            )
            df_unit = df_unit.as_pandas()
            df_unit = df_unit[~df_unit["variable"].str.contains("Investment")]
            df_unit_inv = df_unit_inv.as_pandas()
            report_pd = pd.concat([df_unit, df_unit_inv])
        except Exception as exc:
            log.warning(f"Unit conversion skipped ({exc!r}) — saving as-is")
    report_pd["unit"].replace("EJ", "EJ/yr", inplace=True)
    # for country model
    if reg not in ["R11", "R12"] and suban:
        country_n = map_node_dict["World"][0]
        grouped = report_pd.groupby(
            ["model", "scenario", "variable", "unit", "year", "subannual"]
        )
        renamed_df = pd.DataFrame(columns=report_pd.columns)
        # Step 2: Check if there is at least one "world" row and one "country"
        # row for each group
        for name, group in grouped:
            if (
                "World" in group["region"].values
                and country_n in group["region"].values
            ):
                report_pd.drop(group.index, inplace=True)
                # Step 4: Rename "world" to "country" and remove rows
                # with region = "country"
                group = group[group["region"] == "World"]
                group.loc[group["region"] == "World", "region"] = country_n
                # Step 5: Update the original dataframe with the modified group
                renamed_df = pd.concat([renamed_df, group])

        # Step 8: Concatenate the new dataframe with the original dataframe
        report_pd = pd.concat([report_pd, renamed_df])

    out_path = package_data_path().parents[0] / "reporting_output/"
    out_path.mkdir(exist_ok=True)

    out_file = out_path / f"{sc.model}_{sc.scenario}_nexus.csv"
    report_pd.to_csv(out_file, index=False)
    log.info(f"CSV saved to {out_file}")
    # Timeseries upload to ixmp is skipped — it causes a multi-minute JDBC
    # bulk-insert hang for large models (Pakistan IRB, 32 basin nodes).
    # The CSV above is the authoritative output; Cell 3B reads it directly.


def report_full(sc: Scenario, reg: str, ssp: str, sdgs=False) -> None:
    """Combine old and new reporting workflows

    Parameters
    ----------
    sc : Scenario
        Scenario to report
    reg : str
        Region to report
    ssp : str
        SSP scenario (e.g., :obj:`"SSP1"`, :obj:`"SSP2"`, :obj:`"SSP3"`)
    sdgs : bool, optional
        If :obj:`True`, add population with access to water and sanitation for SDG6
    """
    # Remove any stale timeseries from a previous run
    a = sc.timeseries()
    a = a[a.year >= 2020]
    if not a.empty:
        sc.check_out(timeseries_only=True)
        log.info("Remove any previous timeseries")
        sc.remove_timeseries(a)
        log.info("Finished removing timeseries, now commit..")
        sc.commit("Remove existing timeseries")

    report(sc, reg, ssp, sdgs)
    log.info("overall reporting completed")

    # report() saves the CSV directly as {model}_{scenario}_nexus.csv.
    # No timeseries round-trip needed — read the file report() already wrote.
    out_path = package_data_path().parents[0] / "reporting_output/"
    out_path.mkdir(exist_ok=True)
    out_file = out_path / f"{sc.model}_{sc.scenario}_nexus.csv"
    log.info(f"Report complete → {out_file}")
