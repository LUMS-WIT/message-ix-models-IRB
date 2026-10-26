import logging
from collections.abc import Mapping
from functools import lru_cache, partial

import pandas as pd
from sdmx.model.v21 import Code

from message_ix_models import Context, ScenarioInfo
from message_ix_models.model import build
from message_ix_models.model.structure import get_codes
from message_ix_models.util import broadcast, package_data_path

from .utils import filter_basins_by_region, read_config

log = logging.getLogger(__name__)


def cat_tec_cooling(context: Context) -> tuple[pd.DataFrame, list[str]]:
    """
    Categorize cooling technologies based on predefined types and match them with
    parent technologies present in the scenario.

    This function extracts cooling technology data from a CSV file, filters them
    based on parent technologies available in the scenario, and categorizes each
    cooling technology into a predefined type. It also retrieves a list of unique
    region nodes from the scenario parameter data.

    Parameters
    ----------
    context : Context
        Provides access to the current scenario and configuration.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - cat_tec: A DataFrame with columns:
            - 'type_tec': Cooling technology category.
            - 'tec': Name of the cooling technology.
        - regions_df: A list of unique region nodes from the scenario.
    """
    # Define cooling type categories and their corresponding strings
    cooling_types = {
        "share_cooling_ot_fresh_tot": ["ot_fresh", "cl_fresh", "air", "ot_saline"],
        "share_cooling_cl_fresh_tot": ["ot_fresh", "cl_fresh", "air", "ot_saline"],
        "share_cooling_air_tot": ["ot_fresh", "cl_fresh", "air", "ot_saline"],
        "share_cooling_ot_saline_tot": ["ot_fresh", "cl_fresh", "air", "ot_saline"],
        "share_cooling_ot_fresh_share": ["ot_fresh"],
        "share_cooling_cl_fresh_share": ["cl_fresh"],
        "share_cooling_air_share": ["air"],
        "share_cooling_ot_saline_share": ["ot_saline"],
    }

    FILE = "tech_water_performance_ssp_msg.csv"
    path = package_data_path("water", "ppl_cooling_tech", FILE)
    df = pd.read_csv(path)
    cooling_df = df.loc[df["technology_group"] == "cooling"].copy(deep=True)
    # Separate a column for parent technologies of respective cooling
    # techs
    cooling_df["parent_tech"] = (
        cooling_df["technology_name"]
        .apply(lambda x: pd.Series(str(x).split("__")))
        .drop(columns=1)
    )
    # Extract unique technologies
    sc = context.get_scenario()
    #  get df = sc.par("input") for technollgies in cooling_df(parent_tach)
    df = sc.par("input", filters={"technology": cooling_df["parent_tech"].unique()})
    missing_tec = cooling_df["parent_tech"][
        ~cooling_df["parent_tech"].isin(df["technology"])
    ]
    # some techs only have output, like csp
    ref_output = sc.par("output", {"technology": missing_tec})
    ref_output.columns = df.columns
    # merge ref_input and ref_output
    df = pd.concat([df, ref_output])
    parent_tech_sc = df["technology"].unique()
    regions_df = df["node_loc"].unique().tolist()

    # Assertion check for valid data
    assert len(parent_tech_sc) > 0, (
        "No matching parent technologies found in the scenario."
    )
    assert len(regions_df) > 0, "No unique nodes (regions) found in the scenario."

    # not filter cooling_tec with only parent_tech matching parent_tech_sc
    cooling_df = cooling_df.loc[cooling_df["parent_tech"].isin(parent_tech_sc)].copy()
    unique_technologies = cooling_df["technology_name"].unique()

    # Create a list to store rows for the cat_tec DataFrame
    cat_tec_rows = []

    # Iterate through unique technologies
    for tech in unique_technologies:
        for type_tec, keywords in cooling_types.items():
            for keyword in keywords:
                if keyword in tech:
                    # Add a row to the cat_tec list with type_tec and technology
                    cat_tec_rows.append({"type_tec": type_tec, "tec": tech})

    # Create the cat_tec DataFrame
    cat_tec = pd.DataFrame(cat_tec_rows)

    return cat_tec, regions_df


def share_map_cool(
    share_keys: list[str],
    type_tec_keys: list[str],
    regions_df: list[str],
    commodity_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Helper function to create the share mapping DataFrame for both 'tot' and 'share'
    levels.

    Parameters:
    ----------
    share_keys : list
        List of share keys (e.g., 'share_calib_*')
    type_tec_keys : list
        List of type_tec keys (e.g., 'share_calib_*_tot' or '_share')
    regions_df : list
        List of region nodes
    commodity_mapping : dict, optional
        If provided, maps each share_key to a specific commodity.

    Returns:
    -------
    list
        List of share mapping rows
    """
    # Assign commodities correctly
    commodities = [
        (
            commodity_mapping[key]
            if commodity_mapping
            else ["ot_fresh", "air", "ot_saline", "cl_fresh"]
        )
        for key in share_keys
    ]

    # Expand rows in case of multiple commodities per key
    expanded_rows = []
    for key, type_tec, commodity_list in zip(share_keys, type_tec_keys, commodities):
        for commodity in (
            commodity_list if isinstance(commodity_list, list) else [commodity_list]
        ):
            expanded_rows.append(
                {
                    "shares": key,
                    "node_share": None,
                    "node": None,
                    "type_tec": type_tec,
                    "mode": "M1",
                    "commodity": commodity,
                    "level": "share",
                }
            )

    df_share = pd.DataFrame(expanded_rows).pipe(broadcast, node_share=regions_df)
    df_share["node"] = df_share["node_share"]
    return df_share[
        ["shares", "node_share", "node", "type_tec", "mode", "commodity", "level"]
    ]


def cat_tec_cooling_calib(
    context: Context,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Categorize cooling technologies based on predefined types and match them with
    parent technologies present in the scenario.

    Parameters
    ----------
    context : Context
        Provides access to the current scenario and configuration.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - cat_tec: A DataFrame with columns:
            - 'type_tec': Cooling technology category.
            - 'tec': Name of the cooling technology.
        - regions_df: A list of unique region nodes from the scenario.
    """
    FILE1 = (
        "cooltech_cost_and_shares_"
        + (f"ssp_msg_{context.regions}" if context.type_reg == "global" else "country")
        + ".csv"
    )
    path1 = package_data_path("water", "ppl_cooling_tech", FILE1)
    cool_df = pd.read_csv(path1)

    # Extract region nodes
    # read columns that start with "mix_" from cool_df
    mix_cols = [col for col in cool_df.columns if col.startswith("mix_")]
    # remove "mix_" from the column names
    regions_df = [col.replace("mix_", "") for col in mix_cols]

    # Prepare lists for share definitions
    share_keys = []
    type_tec_tot = []
    type_tec_share = []
    commodity_mapping = {}

    # Create a dictionary to store cooling techs per parent_tec
    cooling_by_parent = cool_df.groupby("utype")["cooling"].unique().to_dict()

    # Iterate over the rows to define share constraints
    for _, row in cool_df.iterrows():
        parent_tec = row["utype"]
        cool_tec = row["cooling"]

        share_key = f"share_calib_{parent_tec}_{cool_tec}"
        share_keys.append(share_key)

        type_tec_tot.append(f"{share_key}_tot")
        type_tec_share.append(f"{share_key}_share")

        # Assign the commodity based on the cooling type
        commodity_mapping[share_key] = (
            cool_tec  # Assigning the correct single commodity
        )

    # Build cat_tec DataFrame
    cat_tec_rows = []

    for parent_tec, cool_tecs in cooling_by_parent.items():
        for cool_tec in cool_tecs:
            share_key_tot = f"share_calib_{parent_tec}_{cool_tec}_tot"
            share_key_share = f"share_calib_{parent_tec}_{cool_tec}_share"

            # "tot" maps to all cooling technologies for that parent
            for other_cool in cool_tecs:
                cat_tec_rows.append(
                    {"type_tec": share_key_tot, "tec": f"{parent_tec}__{other_cool}"}
                )

            # "share" maps only to the specific cooling technology
            cat_tec_rows.append(
                {"type_tec": share_key_share, "tec": f"{parent_tec}__{cool_tec}"}
            )

    cat_tec = pd.DataFrame(cat_tec_rows)

    # Create share constraint mappings
    map_share_commodity_tot = share_map_cool(share_keys, type_tec_tot, regions_df)
    map_share_commodity_share = share_map_cool(
        share_keys, type_tec_share, regions_df, commodity_mapping
    )

    return (
        cat_tec,
        pd.DataFrame(map_share_commodity_tot),
        pd.DataFrame(map_share_commodity_share),
    )


def get_spec(context: Context) -> Mapping[str, ScenarioInfo]:
    """Return the specification for nexus implementation

    Parameters
    ----------
    context : .Context
        The key ``regions`` determines the regional aggregation used.
    """

    context = read_config()

    require = ScenarioInfo()
    remove = ScenarioInfo()
    add = ScenarioInfo()

    # cooling data included by default
    # Merge technology.yaml with set.yaml
    context["water set"]["cooling"]["technology"]["add"] = context["water technology"][
        "cooling"
    ]
    # Update the ScenarioInfo objects with required and new set elements
    for set_name, config in context["water set"]["cooling"].items():
        # Required elements
        require.set[set_name].extend(config.get("require", []))

        # Elements to remove
        remove.set[set_name].extend(config.get("remove", []))

        # Elements to add
        add.set[set_name].extend(config.get("add", []))

    if context.nexus_set == "nexus":
        # Merge technology.yaml with set.yaml
        context["water set"]["nexus"]["technology"]["add"] = context[
            "water technology"
        ]["nexus"]
        # Update the ScenarioInfo objects with required and new set elements
        for set_name, config in context["water set"]["nexus"].items():
            # Required elements
            require.set[set_name].extend(config.get("require", []))

            # Elements to remove
            remove.set[set_name].extend(config.get("remove", []))

            # Elements to add
            add.set[set_name].extend(config.get("add", []))

        # The set of required nodes varies according to context.regions
        n_codes = get_codes(f"node/{context.regions}")
        nodes = list(map(str, n_codes[n_codes.index(Code(id="World"))].child))
        require.set["node"].extend(nodes)

        # Inter-basin network technologies (water transfer + electricity
        # transmission) are generated dynamically from
        # basin_links_<regions>.csv rather than statically enumerated in
        # technology.yaml like other nexus technologies, so their names are
        # registered here instead of via context["water technology"]["nexus"].
        from .data.network import network_technology_names

        network_techs = network_technology_names(context)
        if network_techs:
            add.set["technology"].extend(network_techs)
            add.set["type_tec"].append("water_network")
            add.set["cat_tec"].extend([["water_network", t] for t in network_techs])

        # Crop/rainfed/irrigation technologies (real-data port of the
        # legacy R model's crop submodel) are likewise data-driven (one
        # technology per crop found in the real crop data, not statically
        # enumerable in technology.yaml), and introduce genuinely new
        # commodities/levels the rest of the water module doesn't already
        # have -- registered dynamically here for the same reason as the
        # network technologies above.
        from .data.crops.common import CONFIG as _crops_config
        from .data.crops.land import crop_commodity_names, crop_technology_names
        from .data.crops.rainfed import rainfed_commodity_names, rainfed_technology_names

        crop_techs = crop_technology_names(context)
        if crop_techs:
            add.set["technology"].extend(crop_techs)
            add.set["type_tec"].append("water_crops")
            add.set["cat_tec"].extend([["water_crops", t] for t in crop_techs])
            add.set["commodity"].extend(crop_commodity_names())
            add.set["level"].append("crop")
            # Phase 4: crop_<crop>'s real fertilizer emission_factor uses a
            # new combined-GWP emission category, not one of the base
            # model's individual-species codes -- same pattern the water
            # module already uses for its own "fresh_return" category
            # above. See land.py's module docstring.
            if _crops_config.get("emission_factor_enabled", False):
                add.set["emission"].append(_crops_config["emission_category"])
                add.set["type_emission"].append(_crops_config["type_emission"])
                add.set["cat_emission"].append(
                    [_crops_config["type_emission"], _crops_config["emission_category"]]
                )

        rainfed_techs = rainfed_technology_names(context)
        if rainfed_techs:
            add.set["technology"].extend(rainfed_techs)
            add.set["type_tec"].append("water_crops")
            add.set["cat_tec"].extend([["water_crops", t] for t in rainfed_techs])
            add.set["commodity"].extend(rainfed_commodity_names())
            add.set["level"].extend(["area", "crop_yield"])

        from .data.crops.irrigation_tech import (
            irrigation_commodity_names,
            irrigation_relation_names,
            irrigation_technology_names,
        )

        irrigation_techs = irrigation_technology_names(context)
        if irrigation_techs:
            add.set["technology"].extend(irrigation_techs)
            add.set["type_tec"].append("water_crops")
            add.set["cat_tec"].extend([["water_crops", t] for t in irrigation_techs])
            add.set["commodity"].extend(irrigation_commodity_names())
            add.set["level"].extend(["area", "crop_yield"])
            # The historical-withdrawal demand-pull relation (see
            # irrigation_tech.py module docstring) -- one per basin.
            add.set["relation"].extend(irrigation_relation_names(context))

        # food_import_<crop>: the backstop that keeps add_crop_demand()'s
        # hard demand from being able to break the solve outright -- see
        # food_demand.py's module docstring ("a backstop, not an optional
        # extra"). Registered whenever the crop technologies above are
        # (same basins-with-crop-data gate), not conditioned on them
        # specifically since food_demand.py has its own independent guard.
        from .data.crops.food_demand import food_import_technology_names

        food_import_techs = food_import_technology_names(context)
        if food_import_techs:
            add.set["technology"].extend(food_import_techs)
            add.set["type_tec"].append("water_crops")
            add.set["cat_tec"].extend([["water_crops", t] for t in food_import_techs])
            # food_demand.py's var_cost uses "USD/kt" -- unlike the other
            # cost units this module relies on (USD/Mha, USD/MCM, ...),
            # nothing else in the base model or the rest of the water build
            # ever registers this one on the Platform, so a real build fails
            # with "The unit 'USD/kt' does not exist in the database!" the
            # first time add_par_data() tries to commit var_cost. Registering
            # it here (apply_spec() adds every spec["add"].set["unit"] entry
            # to the Platform before calling the data callback, see
            # model/build.py's apply_spec()) is the same mechanism already
            # used for every other new set element this module introduces --
            # not a one-off mp.add_unit() workaround.
            add.set["unit"].append("USD/kt")

        # bottom_up_crop_water_<basin>: OPT-IN (context.bottom_up_water_demand_enabled,
        # default False -- a no-op for every existing build) third water-
        # demand relation, alongside (not replacing) the historical-
        # withdrawal floor above -- see
        # crops/bottom_up_water_demand.py's module docstring.
        from .data.crops.bottom_up_water_demand import bottom_up_relation_names

        bottom_up_relations = bottom_up_relation_names(context)
        if bottom_up_relations:
            add.set["relation"].extend(bottom_up_relations)

        # Irrigation sw/gw diversion layer: OPT-IN (context.irrigation_source_layer)
        # -- see crops/irrigation_source.py's module docstring.
        from .data.crops.irrigation_source import (
            FOSSIL_MODE,
            irrigation_source_commodity_names,
            irrigation_source_level_names,
            irrigation_source_technology_names,
        )

        source_techs = irrigation_source_technology_names(context)
        if source_techs:
            add.set["technology"].extend(source_techs)
            add.set["type_tec"].append("water_crops")
            add.set["cat_tec"].extend([["water_crops", t] for t in source_techs])
            add.set["commodity"].extend(irrigation_source_commodity_names())
            add.set["level"].extend(irrigation_source_level_names())
            add.set["mode"].append(FOSSIL_MODE)
            add.set["unit"].append("USD/MCM")

        # Fossil groundwater accounting/limit: OPT-IN (context.fossil_gw_accounting
        # or context.fossil_gw_limit) -- see fossil_groundwater.py.
        from .data import fossil_groundwater as _fgw

        fossil_techs = _fgw.fossil_gw_technology_names(context)
        if fossil_techs:
            add.set["emission"].append(_fgw.EMISSION)
            add.set["type_emission"].append(_fgw.TYPE_EMISSION)
            add.set["cat_emission"].append([_fgw.TYPE_EMISSION, _fgw.EMISSION])
            add.set["type_tec"].append(_fgw.TYPE_TEC)
            add.set["cat_tec"].extend([[_fgw.TYPE_TEC, t] for t in fossil_techs])
            for type_year in _fgw.fossil_gw_type_years(context):
                add.set["type_year"].append(type_year)
                add.set["cat_year"].append([type_year, int(type_year)])

        # Reservoir/dam storage technologies (real-data port of the legacy
        # R model's storage_capacity.r, using MESSAGEix's native storage
        # formulation) -- data-driven per basin like the network/crop
        # technologies above, so registered dynamically here rather than
        # statically enumerated in technology.yaml.
        from .data.storage import get_spec_storage

        storage_spec = get_spec_storage(context)
        for set_name, elements in storage_spec["add"].set.items():
            add.set[set_name].extend(elements)

        # Share commodity for groundwater
        results = {}
        df_node = context.all_nodes
        n = len(df_node.values)

        d = {
            "shares": ["share_low_lim_GWat"] * n,
            "node_share": df_node,
            "node": df_node,
            "type_tec": ["share_low_lim_GWat_share"] * n,
            "mode": ["M1"] * n,
            "commodity": ["groundwater_basin"] * n,
            "level": ["water_avail_basin"] * n,
        }

        df_share = pd.DataFrame(data=d)
        df_list = df_share.values.tolist()
        results["map_shares_commodity_share"] = df_list

        d = {
            "shares": ["share_low_lim_GWat"] * n,
            "node_share": df_node,
            "node": df_node,
            "type_tec": ["share_low_lim_GWat_total"] * n,
            "mode": ["M1"] * n,
            "commodity": ["surfacewater_basin"] * n,
            "level": ["water_avail_basin"] * n,
        }

        df_share = pd.DataFrame(data=d)

        d2 = {
            "shares": ["share_low_lim_GWat"] * n,
            "node_share": df_node,
            "node": df_node,
            "type_tec": ["share_low_lim_GWat_total"] * n,
            "mode": ["M1"] * n,
            "commodity": ["groundwater_basin"] * n,
            "level": ["water_avail_basin"] * n,
        }

        df_share2 = pd.DataFrame(data=d2)

        df_share = pd.concat([df_share, df_share2])
        df_list = df_share.values.tolist()

        results["map_shares_commodity_total"] = df_list

        for set_name, config in results.items():
            # Sets  to add
            add.set[set_name].extend(config)

        results = {}

        # Share commodity for urban water recycling
        d = {
            "shares": ["share_wat_recycle"] * n,
            "node_share": df_node,
            "node": df_node,
            "type_tec": ["share_wat_recycle_share"] * n,
            "mode": ["M1"] * n,
            "commodity": ["urban_collected_wst"] * n,
            "level": ["water_treat"] * n,
        }

        df_share = pd.DataFrame(data=d)
        df_list = df_share.values.tolist()
        results["map_shares_commodity_share"] = df_list

        d = {
            "shares": ["share_wat_recycle"] * n,
            "node_share": df_node,
            "node": df_node,
            "type_tec": ["share_wat_recycle_total"] * n,
            "mode": ["M1"] * n,
            "commodity": ["urban_collected_wst"] * n,
            "level": ["water_treat"] * n,
        }

        df_share = pd.DataFrame(data=d)

        d2 = {
            "shares": ["share_wat_recycle"] * n,
            "node_share": df_node,
            "node": df_node,
            "type_tec": ["share_wat_recycle_total"] * n,
            "mode": ["M1"] * n,
            # I think this should be something else TODO NEXUS
            "commodity": ["urban_collected_wst"] * n,
            "level": ["water_treat"] * n,
        }

        df_share2 = pd.DataFrame(data=d2)

        df_share = pd.concat([df_share, df_share2])
        df_list = df_share.values.tolist()

        results["map_shares_commodity_total"] = df_list

        for set_name, config in results.items():
            # Sets  to add
            add.set[set_name].extend(config)

    # for both cooling and nexus add share contraints for cooling technologies
    # cat_tec

    results = {}
    cat_tec, nodes_cooling = cat_tec_cooling(context)

    n = len(nodes_cooling)
    # Share commodity for urban water recycling
    shares_cool = [
        "share_cooling_ot_fresh",
        "share_cooling_cl_fresh",
        "share_cooling_air",
        "share_cooling_ot_saline",
    ]
    commodity_cool = ["ot_fresh", "cl_fresh", "air", "ot_saline"]

    type_tec_share = [
        "share_cooling_ot_fresh_share",
        "share_cooling_cl_fresh_share",
        "share_cooling_air_share",
        "share_cooling_ot_saline_share",
    ]
    df_share = pd.DataFrame(
        {
            "shares": shares_cool,
            "node_share": [None] * len(shares_cool),  # Placeholder for node_share
            "node": [None] * len(shares_cool),  # Placeholder for node
            "type_tec": type_tec_share,
            "mode": "M1",  # Repeat mode
            "commodity": commodity_cool,
            "level": "share",  # Repeat level
        }
    ).pipe(broadcast, node_share=nodes_cooling)

    df_share["node"] = df_share["node_share"]

    df_share = df_share[
        ["shares", "node_share", "node", "type_tec", "mode", "commodity", "level"]
    ]

    # for total
    type_tec_tot = [
        "share_cooling_ot_fresh_tot",
        "share_cooling_cl_fresh_tot",
        "share_cooling_air_tot",
        "share_cooling_ot_saline_tot",
    ]
    df_tot = pd.DataFrame(
        {
            "shares": shares_cool,
            "node_share": [None] * len(shares_cool),  # Placeholder for node_share
            "node": [None] * len(shares_cool),  # Placeholder for node
            "type_tec": type_tec_tot,
            "mode": "M1",  # Repeat mode
            "commodity": [None] * len(shares_cool),
            "level": "share",  # Repeat level
        }
    ).pipe(broadcast, node_share=nodes_cooling, commodity=commodity_cool)

    df_tot["node"] = df_tot["node_share"]

    df_tot = df_tot[
        ["shares", "node_share", "node", "type_tec", "mode", "commodity", "level"]
    ]
    # FIXME : Temporarily commenting out share calibration
    # Causes 4X size explosion in the model, likely problematic.
    # calibration cooling contraints on single parent techs
    # (
    #     cat_tec_calib_cool,
    #     map_com_tot_calib_cool,
    #     map_com_share_calib_cool,
    # ) = cat_tec_cooling_calib(context)

    # cat_tec_list = pd.concat([cat_tec, cat_tec_calib_cool]).values.tolist()

    # results["cat_tec"] = cat_tec_list

    # map_share_commodity_tot_list = pd.concat(
    #     [df_tot, map_com_tot_calib_cool]
    # ).values.tolist()
    # results["map_shares_commodity_total"] = map_share_commodity_tot_list

    # map_share_commodity_share_list = pd.concat(
    #     [df_share, map_com_share_calib_cool]
    # ).values.tolist()
    # results["map_shares_commodity_share"] = map_share_commodity_share_list

    for set_name, config in results.items():
        # Sets to add
        add.set[set_name].extend(config)

    # clean the remove.set from things that are actually not in the scenario
    # this saves building time significantly, as remove is slow
    scen = context.get_scenario()
    for category, elements in ((k, v) for k, v in remove.set.items() if k != "unit"):
        # Get the corresponding set from the scenario
        scen_set = scen.set(category)

        # Filter elements to keep only those present in the scenario set
        remove.set[category] = [elem for elem in elements if elem in scen_set.values]

    return dict(require=require, remove=remove, add=add)


@lru_cache()
def generate_set_elements(set_name, match=None):
    codes = read_config()["water set"][set_name].get("add", [])

    hierarchical = set_name in {"technology"}

    results = []
    for code in codes:
        if match and code.id != match:
            continue
        elif hierarchical:
            results.extend(code)

    return results


def map_basin(context: Context) -> Mapping[str, ScenarioInfo]:
    """Return specification for mapping basins to regions

    The basins are spatially consolidated from HydroSHEDS basins delineation
    database.This delineation is then intersected with MESSAGE regions to form new
    water sector regions for the nexus module.
    The nomenclature for basin names is <basin_id>|<MESSAGEregion> such as R1|AFR
    """
    #context = read_config()

    add = ScenarioInfo()

    require = ScenarioInfo()

    remove = ScenarioInfo()

    # define an empty dictionary
    results = {}
    # read csv file for basin names and region mapping
    # reading basin_delineation
    FILE = f"basins_by_region_simpl_{context.regions}.csv"
    PATH = package_data_path("water", "delineation", FILE)

    df = pd.read_csv(PATH)

    # Apply basin filter to reduce number of basins per region
    df = filter_basins_by_region(df, context)

    # Keep the full BCU_name (basin id + region) so that basins within the
    # same region get distinct mode identifiers. Splitting on "|" and taking
    # only the region suffix collapses e.g. "12|PAKISTAN" and "13|PAKISTAN"
    # onto the identical mode "MPAKISTAN", which silently merges every
    # basin in a multi-basin region into one technology/mode and breaks
    # share_mode_up (SHARE_CONSTRAINT_MODE_UP reduces to ACT <= share*ACT,
    # infeasible for any share < 1 once there is only one mode).
    df["clean_basin"] = df["BCU_name"].apply(lambda x: str(x).replace("|", "_"))

    # ✅ Use CLEAN names for nodes
    df["node"] = "B" + df["BCU_name"].astype(str)
    df["mode"] = "M" + df["clean_basin"]

    # Region mapping
    df["region"] = (
        context.map_ISO_c[context.regions]
        if context.type_reg == "country"
        else df["REGION"]
    )

    results["node"] = df["node"]
    results["mode"] = df["mode"]

    # Translate CSV REGION names to MESSAGEix node IDs where they differ.
    # IRB.yaml uses R12_PAK (not PAKISTAN) as the Pakistan region node,
    # so the CSV value "PAKISTAN" must be mapped before building map_node.
    REGION_NODE_MAP = {"PAKISTAN": "R12_PAK"}
    df["node_parent_region"] = df["REGION"].replace(REGION_NODE_MAP)

    # map nodes as per dimensions
    df1 = pd.DataFrame({"node_parent": df["node_parent_region"], "node": df["node"]})
    df2 = pd.DataFrame({"node_parent": df["node"], "node": df["node"]})

    df_node = pd.concat([df1, df2], ignore_index=True)
    nodes = df_node.values.tolist()

    results["map_node"] = nodes
    for set_name, config in results.items():
        add.set[set_name].extend(config)

    # Add region nodes — must match the node/IRB codelist children of World
    # so that get_spec()'s require check passes
    parent_nodes = df["node_parent_region"].unique().tolist()
    add.set["node"].extend(parent_nodes)

    context.all_nodes = df["node"]
    # Store the filtered basin names for use in other functions
    context.valid_basins = set(df["BCU_name"].astype(str))

    for set_name, config in results.items():
        # Sets to add
        add.set[set_name].extend(config)


    return dict(require=require, remove=remove, add=add)


def main(context: Context, scenario, **options):
    """Set up MESSAGEix-Nexus on `scenario`."""
    from .data import add_data

    log.info("Set up MESSAGEix-Nexus")

    # =========================
    # 1. Basin structure (OK)
    # =========================
    if context.nexus_set == "nexus":
        try:
            spec = map_basin(context)
            build.apply_spec(scenario, spec, **options)
        except Exception as e:
            raise RuntimeError(
                f"Water build failed during map_basin/apply_spec in "
                f"message_ix_models.model.water.build: {type(e).__name__}: {e}"
            ) from e

    # =========================
    # 2. Core water structure
    # =========================
    try:
        spec1 = get_spec(context)
    except Exception as e:
        raise RuntimeError(
            f"Water build failed during get_spec in "
            f"message_ix_models.model.water.build: {type(e).__name__}: {e}"
        ) from e
    scenario.check_out()
    # 🔥 STEP 1: Collect ALL technologies from spec1 (raw)
    all_techs = set()

    for section in ["add", "require", "remove"]:
        if "technology" in spec1[section].set:
            all_techs.update(spec1[section].set["technology"])

        for df in spec1[section].par.values():
            if hasattr(df, "columns") and "technology" in df.columns:
                all_techs.update(df["technology"].astype(str).unique())

    # 🔥 STEP 2: Register them in scenario BEFORE apply_spec
    existing = set(scenario.set("technology"))

    for tech in all_techs:
        if tech not in existing:
            scenario.add_set("technology", tech)

    # 🔥 STEP 3: CLEAN + MAP technologies
    def clean_spec_technologies(spec):

        def map_tech(t):
            return str(t).split("__")[0]

        # --- CLEAN SETS ---
        for section in ["add", "require", "remove"]:
            if "technology" in spec[section].set:
                spec[section].set["technology"] = [
                    map_tech(t) for t in spec[section].set["technology"]
                ]

        # --- CLEAN PARAMETERS ---
        for section in ["add", "require", "remove"]:
            for par_name, df in spec[section].par.items():
                if hasattr(df, "columns") and "technology" in df.columns:
                    df = df.copy()
                    df["technology"] = df["technology"].apply(map_tech)
                    spec[section].par[par_name] = df

        return spec

    # 🔥 APPLY CLEANING
    spec1 = clean_spec_technologies(spec1)

    # =========================
    # 3. Apply structure + data
    # =========================
    try:
        build.apply_spec(
            scenario,
            spec1,
            partial(add_data, context=context),
            **options
        )
    except Exception as e:
        raise RuntimeError(
            f"Water build failed during core apply_spec in "
            f"message_ix_models.model.water.build: {type(e).__name__}: {e}"
        ) from e

    # Uncomment to dump for debugging
    # scenario.to_excel('debug.xlsx')
