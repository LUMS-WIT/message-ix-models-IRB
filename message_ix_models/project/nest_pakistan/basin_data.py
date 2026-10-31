"""What input data a river basin needs, where it can come from, and which
files exist for a given basin code.

Every basin-specific file of the water/land module is named with the basin
code (``*_IRB.csv`` for the Indus). :data:`REQUIREMENTS` lists them with the
open global dataset that can produce each one and the script in
``model/water/data/pre_processing`` that does it. :func:`status` checks a
basin code against the package data; the CLI command
``mix-models nest-pak data --basin <code>`` prints it and writes it to CSV.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from message_ix_models.util import package_data_path


@dataclass(frozen=True)
class Item:
    step: str          # pipeline area
    pattern: str       # path under data/water (or data/ for "node/"), {code} = basin code
    content: str
    need: str          # "required", "recommended" or "optional"
    source: str        # open global dataset(s) that can produce it
    script: str = ""   # pre_processing script that builds it, if any
    global_ok: str = "yes"  # "yes", "partly" (global data + local check) or "no" (local data needed)


REQUIREMENTS = [
    # 1. Spatial structure
    Item("1 Spatial structure", "../node/{code}.yaml",
         "Node codelist: basin units (e.g. '12|PAKISTAN') and their parent region/energy nodes",
         "required", "Written by hand from the delineation below", "", "yes"),
    Item("1 Spatial structure", "delineation/basins_by_region_simpl_{code}.csv",
         "Basin units: BCU_name, REGION (country part), area, outlet",
         "required", "HydroBASINS (HydroSHEDS) level 4-6 intersected with GADM country borders",
         "_shapefile.py", "yes"),
    Item("1 Spatial structure", "delineation/basins_country_{code}.csv",
         "Basin unit -> country shares (for downscaling national data)",
         "required", "HydroBASINS x GADM area overlay", "", "yes"),
    Item("1 Spatial structure", "network/basin_links_{code}.csv",
         "Upstream -> downstream river links (and inter-basin canals)",
         "required", "HydroRIVERS / HydroSHEDS flow direction; canals from national irrigation maps",
         "build_basin_links_IRB.py", "partly"),

    # 2. Water availability (hydrology)
    Item("2 Water availability", "availability/qtot_5y_{rcp}_{rel}_{code}.csv",
         "Total runoff per basin unit, 5-year steps (MCM/yr), per climate scenario and reliability",
         "required", "ISIMIP2b/ISIMIP3b global hydrological models (CWatM, H08, WaterGAP2, LPJmL), 0.5 deg;"
                     " aggregate grid cells with SUM over each basin unit (not mean)",
         "hydro_agg_spatial.R, hydro_agg_raster.py, hydro_agg_basin.py", "yes"),
    Item("2 Water availability", "availability/qr_5y_{rcp}_{rel}_{code}.csv",
         "Groundwater recharge per basin unit (MCM/yr)",
         "required", "ISIMIP qr (groundwater recharge) from the same models, summed per basin unit",
         "hydro_agg_spatial.R", "yes"),
    Item("2 Water availability", "availability/dis_5y_{rcp}_{rel}_{code}.csv",
         "River discharge at basin outlets (used for environmental flows)",
         "recommended", "ISIMIP dis (routed discharge)", "hydro_agg_spatial.R", "yes"),
    Item("2 Water availability", "availability/e-flow_{rcp}_{code}.csv",
         "Environmental flow requirement (Variable Monthly Flow method)",
         "optional", "Computed from ISIMIP monthly discharge", "generate_water_constraints.R", "yes"),
    Item("2 Water availability", "availability/qtot_monthly_{rcp}_{code}.csv",
         "Monthly runoff, for seasonal time slices (makes reservoirs active)",
         "optional", "ISIMIP monthly qtot", "generate_water_constraints_monthly.R", "yes"),
    Item("2 Water availability", "availability/gw_energy_intensity_depth_{code}.csv",
         "Electricity needed to pump groundwater, from water-table depth",
         "recommended", "Global water-table depth (Fan et al. 2013) + pump efficiency", "groundwater_harmonize.R",
         "yes"),
    Item("2 Water availability", "availability/historical_new_cap_gw_sw_km3_year_{code}.csv",
         "Historical surface/groundwater extraction capacity",
         "recommended", "Huang et al. (2018) gridded sectoral withdrawals; AQUASTAT; national statistics",
         "groundwater_harmonize.R", "partly"),

    # 3. Demands
    Item("3 Demands", "demands/harmonized/{code}",
         "Urban, rural and industrial withdrawals, return flows, connection and treatment rates (SSP)",
         "required", "MESSAGEix-Nexus global harmonised demands (Huang et al. 2018 + SSP population/GDP),"
                     " downscaled to basin units", "", "yes"),

    # 4. Crops, land and irrigation
    Item("4 Crops & irrigation", "crops/crop_input_data_{code}.csv",
         "Crop yields, irrigated/rainfed area, costs per basin unit",
         "required", "SPAM 2010/2020 (area, yield), GAEZ v4, FAOSTAT", "build_crop_data_IRB.py, "
         "build_spam2020_crop_harvested_area_IRB.py", "yes"),
    Item("4 Crops & irrigation", "crops/crop_irrigation_water_{code}.csv",
         "Crop irrigation water requirement per hectare and month",
         "required", "GAEZ v4 / CROPWAT, or ISIMIP crop models (LPJmL, PEPIC) irrigation water", "build_crop_data_IRB.py",
         "yes"),
    Item("4 Crops & irrigation", "crops/historical_irrigation_withdrawals_{code}.csv",
         "Observed irrigation withdrawals (canal and tubewell): the irrigation floor",
         "required", "AQUASTAT; LGRIP30 irrigated area x water requirement; Huang et al. (2018);"
                     " national canal-head and tubewell statistics are much better where they exist",
         "build_irrigation_demand_lgrip30.py, build_lgrip30_precise_IRB.py", "partly"),
    Item("4 Crops & irrigation", "crops/irr_tech_data_{code}.csv",
         "Irrigation methods: efficiency, electricity, costs (flood, drip, sprinkler, ...)",
         "required", "FAO / literature defaults; the Indus file is a usable starting point", "", "partly"),
    Item("4 Crops & irrigation", "crops/crop_tech_data_{code}.csv",
         "Crop technology costs and lifetimes", "required", "Literature defaults; Indus file reusable", "", "partly"),
    Item("4 Crops & irrigation", "crops/land_availability_{code}.csv",
         "Cultivable land per basin unit", "recommended", "GAEZ suitability, ESA CCI land cover", "", "yes"),
    Item("4 Crops & irrigation", "crops/irrigation_demand_growth_{code}.csv",
         "Growth index of the irrigation floor (off by default)", "optional", "SSP population/GDP",
         "build_irrigation_demand_growth_IRB.py", "yes"),
    Item("4 Crops & irrigation", "crops/population_group_{code}.csv",
         "Population per region group (SSP)", "required", "IIASA SSP database, WorldPop downscaling",
         "build_pop_gdp_group_IRB.py", "yes"),
    Item("4 Crops & irrigation", "crops/gdp_per_capita_group_{code}.csv",
         "GDP per capita per region group (SSP)", "required", "IIASA SSP database", "build_pop_gdp_group_IRB.py",
         "yes"),
    Item("4 Crops & irrigation", "crops/crop_food_demand_{code}.csv",
         "Food demand per crop (only if crop demand is switched on)", "optional", "FAOSTAT food balance + SSP",
         "build_crop_food_demand_IRB.py", "yes"),
    Item("4 Crops & irrigation", "crops/irrigation_electricity_fraction_{code}.csv",
         "Share of irrigation pumping that uses grid electricity (vs diesel)", "recommended",
         "National agricultural census; no reliable global source", "", "no"),
    Item("4 Crops & irrigation", "crops/basin_irrigation_transfers_{code}.csv",
         "Canal command areas that move water between basin units", "optional",
         "National irrigation department maps", "", "no"),
    Item("4 Crops & irrigation", "crops/emission_factor_{code}.csv",
         "Crop emission factors", "optional", "IPCC Tier 1 defaults", "", "yes"),

    # 5. Infrastructure
    Item("5 Infrastructure", "infrastructure/historical_capacity_desalination_km3_year_{code}.csv",
         "Installed desalination capacity", "recommended", "Jones et al. (2019) global desalination plants",
         "desalination.R", "yes"),
    Item("5 Infrastructure", "infrastructure/projected_desalination_potential_km3_year_{code}.csv",
         "Desalination potential (coastal access)", "recommended", "Distance to coast + Jones et al. (2019)",
         "desalination.R", "yes"),
    Item("5 Infrastructure", "storage/dam_storage_capacity_GRANDv1.1_M_m3_PID.csv",
         "Reservoir storage per basin unit (one file for all basins today; split per code for a new basin)",
         "recommended", "GRanD v1.3 global reservoir and dam database", "", "yes"),

    # 6. Power plant cooling
    Item("6 Power plant cooling", "ppl_cooling_tech/cooltech_cost_and_shares_country_{code}.csv",
         "Cooling technology shares and costs per country", "required",
         "MESSAGEix-Nexus global defaults (Raptis & Pfister 2016 power plant database)",
         "calculate_ppl_cooling_technology_shares.R", "yes"),
    Item("6 Power plant cooling", "ppl_cooling_tech/power_plant_cooling_impact_MESSAGE_{code}.csv",
         "Climate impact on cooling capacity", "optional", "MESSAGEix-Nexus global data", "", "yes"),
]

#: What the national energy model needs; not a file of this package.
ENERGY_MODEL_NOTE = (
    "The national energy model is the one piece that needs national data: a MESSAGEix scenario with "
    "electricity generation, fuels and end-use demands for the country that owns the energy node "
    "(basin.region_node_map). Starting points: a MESSAGEix-GLOBIOM R12 region scenario, or a national "
    "model built from IEA/IRENA statistics. Other countries in the basin can stay without an energy model: "
    "their water sector gets the costed external electricity supply."
)


def _path(pattern: str, code: str, rcp: str, rel: str) -> Path:
    rel_path = pattern.format(code=code, rcp=rcp, rel=rel)
    return package_data_path("water", rel_path) if not rel_path.startswith("../") else (
        package_data_path("node", rel_path.split("/", 2)[2]))


def status(code: str, rcp: str = "2p6", rel: str = "low") -> pd.DataFrame:
    """One row per requirement: whether the file exists for basin ``code``."""
    rows = []
    for it in REQUIREMENTS:
        p = _path(it.pattern, code, rcp, rel)
        rows.append({"step": it.step, "file": it.pattern.format(code=code, rcp=rcp, rel=rel), "need": it.need,
                     "present": p.exists(), "content": it.content, "global data": it.global_ok,
                     "source": it.source, "script": it.script})
    return pd.DataFrame(rows)
