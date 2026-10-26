"""Prepare data for the generic, region-tier ``irrigation_<crop-category>``
technologies used on the **global** (R11/R12/R14/...) MESSAGEix-GLOBIOM
build path only.

This is the older, coarser of the two irrigation mechanisms in this
package -- **not** the one used for the Pakistan/IRB nexus build. It is
only invoked via ``DATA_FUNCTIONS`` in :mod:`.data` (the global list); the
IRB/country path (``DATA_FUNCTIONS_COUNTRY``) uses
:mod:`.crops.irrigation_tech` instead, which is the real-data,
basin-level, per-(method, crop) port of the legacy R+GAMS indus_ix model
described in that module's own docstring. See that module for the
mechanism actually exercised by the Pakistan build (real crop water
requirements, method-specific efficiencies, irrigation losses routed to
groundwater recharge, etc.) -- none of that applies here.

Data model
----------
Three broad, GLOBIOM-aligned crop categories -- ``irrigation_cereal``,
``irrigation_oilcrops``, ``irrigation_sugarcrops`` -- each a single
technology per MESSAGE **region** (``node_loc = df_node["region"]``, e.g.
``R12_PAK``), not per basin. Every basin belonging to a region collapses
onto the same technology instance; there is no basin-level differentiation
here (contrast :mod:`.crops.irrigation_tech`, which is basin-indexed).

- **Input**: 1 unit of ``freshwater`` (level ``water_supply``) per unit of
  activity, unscaled -- an arbitrary, uncalibrated ratio, not derived from
  any real crop water requirement. An electricity input (commented out,
  never activated -- see the Diaz et al. 2011 reference and the dead code
  below it) was evidently planned but never wired in, so as it stands this
  technology draws water only, no pumping energy.
- **Output**: the same ``freshwater`` commodity, just re-leveled per crop
  category (``irr_cereal`` / ``irr_sugarcrops`` / ``irr_oilcrops``) --
  effectively relabeling water into a category-specific "irrigation final
  demand" slot for GLOBIOM's crop-demand linkage to consume downstream.
  Unlike :mod:`.crops.irrigation_tech`, there is no representation of
  irrigation losses or return flow to groundwater here.

In short: this is a coarse, demand-side placeholder sized for a global,
GLOBIOM-linked model where basin-level hydrology isn't the point. Do not
use it as a reference for how irrigation water demand is actually balanced
in the Pakistan nexus build -- for that, read
:mod:`.crops.irrigation_tech`.
"""

import pandas as pd
from message_ix import make_df

from message_ix_models import Context
from message_ix_models.model.water.utils import REGION_NODE_MAP
from message_ix_models.util import broadcast, package_data_path


# water & electricity for irrigation
def add_irr_structure(context: "Context") -> dict[str, pd.DataFrame]:
    """Add irrigation withdrawal infrastructure
    The irrigation demands are added in

    Parameters
    ----------
    context : .Context

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        Keys are MESSAGE parameter names such as 'input', 'fix_cost'.
        Values are data frames ready for :meth:`~.Scenario.add_par`.
        Years in the data include the model horizon indicated by
        ``context["water build info"]``, plus the additional year 2010.
    """

    # define an empty dictionary
    results = {}

    # reading basin_delineation
    FILE2 = f"basins_by_region_simpl_{context.regions}.csv"
    PATH = package_data_path("water", "delineation", FILE2)
    df_node = pd.read_csv(PATH)

    # Filter to only include valid basins
    df_node = df_node[df_node["BCU_name"].isin(context.valid_basins)]

    # Assigning proper nomenclature
    df_node["node"] = "B" + df_node["BCU_name"].astype(str)
    # Keep full BCU_name so basins within one region get distinct modes
    # (splitting to the region suffix collapses all basins in a region
    # onto one mode, see build.py clean_basin comment).
    df_node["clean_basin"] = df_node["BCU_name"].astype(str).str.replace("|", "_", regex=False)
    df_node["mode"] = "M" + df_node["clean_basin"]
    # df_node["region"] = (
    #     context.map_ISO_c[context.regions]
    #     if context.type_reg == "country"
    #     else f"{context.regions}_" + df_node["REGION"].astype(str)
    # )
    # Map raw CSV REGION names to MESSAGEix node IDs (must match IRB.yaml / build.py)
    df_node["region"] = df_node["REGION"].replace(REGION_NODE_MAP)
    # Reference to the water configuration
    info = context["water build info"]

    # probably can be removed
    year_wat = [2010, 2015]
    year_wat.extend(info.Y)

    inp = make_df(
        "input",
        technology="irrigation_cereal",
        value=1,
        unit="-",
        level="water_supply",
        commodity="freshwater",
        mode="M1",
        time="year",
        time_origin="year",
        node_origin=df_node["region"],
        node_loc=df_node["region"],
    ).pipe(broadcast, year_vtg=info.Y)

    inp = pd.concat(
        [
            inp,
            make_df(
                "input",
                technology="irrigation_oilcrops",
                value=1,
                unit="-",
                level="water_supply",
                commodity="freshwater",
                mode="M1",
                time="year",
                time_origin="year",
                node_origin=df_node["region"],
                node_loc=df_node["region"],
            ).pipe(broadcast, year_vtg=info.Y),
        ]
    )

    inp = pd.concat(
        [
            inp,
            make_df(
                "input",
                technology="irrigation_sugarcrops",
                value=1,
                unit="-",
                level="water_supply",
                commodity="freshwater",
                mode="M1",
                time="year",
                time_origin="year",
                node_origin=df_node["region"],
                node_loc=df_node["region"],
            ).pipe(broadcast, year_vtg=info.Y),
        ]
    )
    # year_act = year_vts for tecs with 1 time-step lifetime
    inp["year_act"] = inp["year_vtg"]

    # Electricity values per unit of irrigation water supply
    # Reference: Evaluation of Water and Energy Use in
    # Pressurized Irrigation Networks in Southern Spain
    # Diaz et al. 2011 https://ascelibrary.org/
    # doi/10.1061/%28ASCE%29IR.1943-4774.0000338
    # Low Value :0.04690743
    # Average Value :0.101598174
    # High Value : 0.017123288

    # inp = pd.concat([inp,
    #     make_df(
    #         "input",
    #         technology="irrigation_sugarcrops",
    #         value=0.04690743,
    #         unit="-",
    #         level="final",
    #         commodity="electr",
    #         mode="M1",
    #         time="year",
    #         time_origin="year",
    #         node_origin=df_node["region"],
    #         node_loc=df_node["region"],
    #     ).pipe(broadcast, year_vtg=year_wat, year_act=year_wat)
    # ])
    #
    # inp = pd.concat([inp,
    #     make_df(
    #         "input",
    #         technology="irrigation_oilcrops",
    #         value=0.04690743,
    #         unit="-",
    #         level="final",
    #         commodity="electr",
    #         mode="M1",
    #         time="year",
    #         time_origin="year",
    #         node_origin=df_node["region"],
    #         node_loc=df_node["region"],
    #     ).pipe(broadcast, year_vtg=year_wat, year_act=year_wat)
    # ])
    #
    # inp = pd.concat([inp,
    #     make_df(
    #         "input",
    #         technology="irrigation_cereal",
    #         value=0.04690743,
    #         unit="-",
    #         level="final",
    #         commodity="electr",
    #         mode="M1",
    #         time="year",
    #         time_origin="year",
    #         node_origin=df_node["region"],
    #         node_loc=df_node["region"],
    #     ).pipe(broadcast, year_vtg=year_wat, year_act=year_wat)
    # ])
    # inp.loc[(inp['node_loc'] == 'R11_SAS') &
    #         (inp['commodity'] == 'electr'),
    #         "value",
    # ] *= 0.00004690743

    results["input"] = inp

    irr_out = make_df(
        "output",
        technology="irrigation_cereal",
        value=1,
        unit="MCM/year",
        level="irr_cereal",
        commodity="freshwater",
        mode="M1",
        time="year",
        time_dest="year",
        node_loc=df_node["region"],
        node_dest=df_node["region"],
    ).pipe(broadcast, year_vtg=info.Y)

    irr_out = pd.concat(
        [
            irr_out,
            make_df(
                "output",
                technology="irrigation_sugarcrops",
                value=1,
                unit="MCM/year",
                level="irr_sugarcrops",
                commodity="freshwater",
                mode="M1",
                time="year",
                time_dest="year",
                node_loc=df_node["region"],
                node_dest=df_node["region"],
            ).pipe(
                broadcast,
                year_vtg=info.Y,
            ),
        ]
    )

    irr_out = pd.concat(
        [
            irr_out,
            make_df(
                "output",
                technology="irrigation_oilcrops",
                value=1,
                unit="MCM/year",
                level="irr_oilcrops",
                commodity="freshwater",
                mode="M1",
                time="year",
                time_dest="year",
                node_loc=df_node["region"],
                node_dest=df_node["region"],
            ).pipe(
                broadcast,
                year_vtg=info.Y,
            ),
        ]
    )

    irr_out["year_act"] = irr_out["year_vtg"]

    results["output"] = irr_out

    return results
