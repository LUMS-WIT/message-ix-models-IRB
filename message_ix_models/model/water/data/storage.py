"""Reservoir/dam storage technologies for the water-energy-land nexus.

Ports the legacy R model's ``storage_capacity.r`` (see the sibling ``NEST``
repo, ``NEST/MESSAGEix/input_data_scripts/storage_capacity.r``) into
MESSAGEix's native storage formulation (``storage_tec``/``level_storage``/
``map_tec_storage``/``storage_initial``/``storage_self_discharge`` -- see
``message_ix.tests.test_feature_storage`` for the reference pattern this
follows; that test confirms the feature is present and working in the
installed message_ix 3.11.1, so no core message_ix changes are needed here).

Data model
----------
Three MESSAGEix technologies represent one basin's aggregate reservoir/dam
capacity (there is no need for one technology per named dam -- capacity is
basin-indexed data on shared technologies, the same convention
:mod:`water_supply` uses for ``extract_surfacewater`` etc.):

``reservoir_storage``
    The storage reservoir itself (``storage_tec``). Its own input/output use
    a dummy commodity/level solely so it is a valid MESSAGE technology; the
    real state variable it governs is ``STORAGE`` at
    (level=``reservoir_stor``, commodity=``reservoir_water``), which
    ``bound_activity_up`` (from GRanD dam capacity, Mm^3) caps -- see
    ``test_feature_storage.py``'s rule 5 ("max activity of charger <= max
    activity of storage").
``reservoir_charge``
    Draws real basin water (``surfacewater_basin``/``water_avail_basin``,
    the same commodity/level :mod:`water_supply` uses) and outputs it into
    the storage commodity/level.
``reservoir_release``
    The inverse: draws storage, outputs back to ``surfacewater_basin``/
    ``water_avail_basin`` for downstream use (irrigation, hydropower,
    municipal supply -- whatever the rest of the basin's water balance
    already represents).

Four CSVs (originally produced by ``storage_capacity.r`` from the GRanD dam
database v1.1 and WAPDA reservoir-level records; bundled here as
already-processed outputs, found via a synced OneDrive copy of the project's
``NEST Indus/Input/storage/`` folder -- the raw GIS/raster inputs the R
script itself needs, e.g. the GRanD shapefile database and CWatM evaporation
rasters, are not available in this repository so the R pipeline can't be
re-run, only its outputs ported):

``dam_storage_capacity_GRANDv1.1_M_m3_PID.csv``
    Cumulative dam storage capacity (Mm^3) per basin PID per year --
    ``bound_activity_up`` on ``reservoir_storage``.
``average_storage_multiplier.csv``
    Fraction of capacity dams are assumed to hold on average
    (Tarbela/Mangla-derived, applied basin-wide) -- ``storage_initial``
    (a fraction of capacity, per ``test_feature_storage.py``).
``evapo_loss_reservoir_per_unit_volume.csv``
    Per-unit-volume evaporation loss factor per basin/month/climate
    model+scenario -- annual-mean per basin becomes
    ``storage_self_discharge`` (also a fraction).
``rule_curves_values.csv``
    Monthly fill-ratio rule curves for the two named dams (Tarbela, Mangla)
    only. This model runs with ``context.time == ["year"]`` (no sub-annual
    time steps are set up anywhere in the nexus build -- see
    ``water_integration1_model.py`` in the sibling NEST-Pakistan repo, and
    ``cli.py``'s ``water_ini()`` which derives ``context.time`` from the
    scenario's own "time" set), so there is no monthly slot to bind a
    monthly rule curve to without inventing a sub-annual structure the rest
    of the module doesn't have. Rather than force it into
    ``storage_initial``/``storage_self_discharge`` under a false annual
    reading, it is kept as bundled reference data and used only as a
    plausibility cross-check in :func:`validate_storage_initial`. Wiring an
    actual monthly rule-curve constraint is future work, gated on the
    module adopting sub-annual time.

Node mapping
------------
The CSVs use the legacy R model's PID node codes (``PAK_8``, ``AFG_2``,
...). :data:`message_ix_models.model.water.data.pre_processing._pid_crosswalk.CROSSWALK`
(the same crosswalk :mod:`build_basin_links_IRB` and the crop/irrigation
build scripts use) converts these to this project's ``BCU_name`` values,
filtered to basins in ``context.valid_basins`` (set by
:func:`~message_ix_models.model.water.build.map_basin`) so only basins
actually present in the target region set are used. If none of the dam
data's basins are in scope (e.g. building for a region set without a
Pakistan/Indus node), every function here degrades to a no-op rather than
raising.
"""

import logging

import pandas as pd
from message_ix import make_df

from message_ix_models import Context, ScenarioInfo
from message_ix_models.model.water.data.pre_processing._pid_crosswalk import CROSSWALK
from message_ix_models.model.water.utils import filter_basins_by_region
from message_ix_models.util import broadcast, package_data_path, same_node

log = logging.getLogger(__name__)

STORAGE_TEC = "reservoir_storage"
CHARGE_TEC = "reservoir_charge"
RELEASE_TEC = "reservoir_release"
TECHNOLOGIES = [STORAGE_TEC, CHARGE_TEC, RELEASE_TEC]

STORAGE_LEVEL = "reservoir_stor"
STORAGE_COMMODITY = "reservoir_water"
DUMMY_LEVEL = "reservoir_dummy"
DUMMY_COMMODITY = "reservoir_dummy"

# Same basin water commodity/level used throughout the rest of the water
# module (water_supply.add_water_supply, demands.py, ...).
BASIN_COMMODITY = "surfacewater_basin"
BASIN_LEVEL = "water_avail_basin"


def _basin_nodes(context: "Context") -> pd.DataFrame:
    """Valid basin BCU_names with their MESSAGEix node id ('B' + BCU_name)."""
    FILE = f"basins_by_region_simpl_{context.regions}.csv"
    df = pd.read_csv(package_data_path("water", "delineation", FILE))
    df = filter_basins_by_region(df, context)
    valid_basins = getattr(context, "valid_basins", None)
    if valid_basins:
        df = df[df["BCU_name"].astype(str).isin(valid_basins)]
    df = df.assign(node=("B" + df["BCU_name"].astype(str)))
    return df[["BCU_name", "node"]].drop_duplicates()


def _pid_to_node(context: "Context") -> pd.DataFrame:
    """Map legacy PID node codes (e.g. 'PAK_8') to this project's node ids,
    restricted to basins actually in scope for this build."""
    nodes = _basin_nodes(context)
    in_scope = set(nodes["BCU_name"])
    rows = [
        {"PID": pid, "BCU_name": bcu, "node": "B" + bcu}
        for pid, bcu in CROSSWALK.items()
        if bcu in in_scope
    ]
    return pd.DataFrame(rows, columns=["PID", "BCU_name", "node"])


def read_dam_capacity(context: "Context") -> pd.DataFrame:
    """Dam storage capacity (Mm^3) per node per year, filtered to basins in scope."""
    path = package_data_path(
        "water", "storage", "dam_storage_capacity_GRANDv1.1_M_m3_PID.csv"
    )
    df = pd.read_csv(path)
    crosswalk = _pid_to_node(context)
    df = df.merge(crosswalk, left_on="node", right_on="PID", how="inner")
    return df[["node_y", "year", "value"]].rename(
        columns={"node_y": "node", "value": "capacity_mcm"}
    )


def read_storage_initial_fraction(context: "Context") -> pd.DataFrame:
    """Average fraction of capacity held (storage_initial input), per node."""
    path = package_data_path("water", "storage", "average_storage_multiplier.csv")
    df = pd.read_csv(path)
    crosswalk = _pid_to_node(context)
    df = df.merge(crosswalk, left_on="node", right_on="PID", how="inner")
    return df[["node_y", "avg_strg"]].rename(
        columns={"node_y": "node", "avg_strg": "fraction"}
    )


def read_self_discharge_fraction(context: "Context") -> pd.DataFrame:
    """Annual-mean evaporation loss fraction (storage_self_discharge input), per node.

    The source CSV is per (basin, year, month, climate model, climate
    scenario); this model has no sub-annual time steps (see module
    docstring), so months/models/scenarios are averaged into one annual
    figure per basin.
    """
    path = package_data_path(
        "water", "storage", "evapo_loss_reservoir_per_unit_volume.csv"
    )
    df = pd.read_csv(path)
    crosswalk = _pid_to_node(context)
    df = df.merge(crosswalk, left_on="node", right_on="PID", how="inner")
    return (
        df.groupby("node_y", as_index=False)["value"]
        .mean()
        .rename(columns={"node_y": "node", "value": "fraction"})
    )


def reservoir_node_list(context: "Context") -> list:
    """Basin nodes that have positive dam storage capacity data."""
    cap = read_dam_capacity(context)
    if cap.empty:
        return []
    return sorted(cap.loc[cap["capacity_mcm"] > 0, "node"].unique().tolist())


def validate_storage_initial(context: "Context") -> None:
    """Log a warning if the computed storage_initial fraction deviates from
    the WAPDA rule curves for the two named/verified dams (Tarbela, Mangla).

    Diagnostic only -- see the module docstring for why the monthly rule
    curves aren't wired into a model constraint directly.
    """
    try:
        path = package_data_path("water", "storage", "rule_curves_values.csv")
        rule_curves = pd.read_csv(path)
        annual_ratio = rule_curves.groupby("DAM_NAME")["ratio_avg_max"].mean()

        fractions = read_storage_initial_fraction(context).set_index("node")["fraction"]
        crosswalk = _pid_to_node(context).set_index("PID")["node"]

        named_dam_pid = {"Tarbela": "PAK_8", "Mangla": "PAK_1"}
        for dam, pid in named_dam_pid.items():
            if pid not in crosswalk.index:
                continue
            node = crosswalk[pid]
            if node not in fractions.index:
                continue
            modeled = fractions[node]
            expected = annual_ratio.get(dam)
            if expected is not None and abs(modeled - expected) > 0.15:
                log.warning(
                    f"storage_initial fraction for {dam} ({node}) = "
                    f"{modeled:.3f}, but its WAPDA rule-curve annual-mean "
                    f"fill ratio is {expected:.3f} (difference > 0.15) -- "
                    f"the basin-wide average_storage_multiplier may be a "
                    f"poor fit for this dam"
                )
    except Exception as e:  # pragma: no cover - diagnostic only, must not block build
        log.warning(f"Could not validate storage_initial against rule curves: {e}")


def get_spec_storage(context: "Context") -> dict:
    """Return the ``add``/``require``/``remove`` set elements for reservoir
    storage, in the shape :func:`~message_ix_models.model.water.build.get_spec`
    merges into its own spec.
    """
    add = ScenarioInfo()
    require = ScenarioInfo()
    remove = ScenarioInfo()

    nodes = reservoir_node_list(context)
    if not nodes:
        log.info(
            "No basins with dam storage capacity data in scope; "
            "skipping reservoir storage"
        )
        return dict(require=require, remove=remove, add=add)

    add.set["technology"].extend(TECHNOLOGIES)
    add.set["type_tec"].append("water_storage")
    add.set["cat_tec"].extend([["water_storage", t] for t in TECHNOLOGIES])
    add.set["level"].extend([STORAGE_LEVEL, DUMMY_LEVEL])
    add.set["commodity"].extend([STORAGE_COMMODITY, DUMMY_COMMODITY])
    add.set["storage_tec"].append(STORAGE_TEC)
    add.set["level_storage"].append(STORAGE_LEVEL)

    for node in nodes:
        for charger_discharger in (CHARGE_TEC, RELEASE_TEC):
            add.set["map_tec_storage"].append(
                [
                    node,
                    charger_discharger,
                    "M1",
                    STORAGE_TEC,
                    "M1",
                    STORAGE_LEVEL,
                    STORAGE_COMMODITY,
                    "year",
                ]
            )

    return dict(require=require, remove=remove, add=add)


def _tec_io(
    technology: str,
    direction: str,
    level: str,
    commodity: str,
    year_wat: list,
    nodes: list,
) -> pd.DataFrame:
    """Build a broadcast input/output frame for one technology at 1:1 ratio."""
    kwargs = dict(
        technology=technology,
        value=1,
        unit="MCM",
        year_vtg=year_wat,
        year_act=year_wat,
        level=level,
        commodity=commodity,
        mode="M1",
        time="year",
    )
    kwargs["time_origin" if direction == "input" else "time_dest"] = "year"
    return (
        make_df(direction, **kwargs).pipe(broadcast, node_loc=nodes).pipe(same_node)
    )


def add_reservoir_storage(context: "Context") -> dict:
    """Build parameter data for reservoir storage technologies:
    ``input``, ``output``, ``bound_activity_up``, ``storage_initial``,
    ``storage_self_discharge``.
    """
    info = context["water build info"]
    year_wat = sorted({*range(2010, info.Y[0] + 1, 5), *info.Y})

    nodes = reservoir_node_list(context)
    results: dict = {}
    if not nodes:
        return results

    results["input"] = pd.concat(
        [
            _tec_io(STORAGE_TEC, "input", DUMMY_LEVEL, DUMMY_COMMODITY, year_wat, nodes),
            _tec_io(CHARGE_TEC, "input", BASIN_LEVEL, BASIN_COMMODITY, year_wat, nodes),
            _tec_io(RELEASE_TEC, "input", STORAGE_LEVEL, STORAGE_COMMODITY, year_wat, nodes),
        ],
        ignore_index=True,
    )
    results["output"] = pd.concat(
        [
            _tec_io(STORAGE_TEC, "output", DUMMY_LEVEL, DUMMY_COMMODITY, year_wat, nodes),
            _tec_io(CHARGE_TEC, "output", STORAGE_LEVEL, STORAGE_COMMODITY, year_wat, nodes),
            _tec_io(RELEASE_TEC, "output", BASIN_LEVEL, BASIN_COMMODITY, year_wat, nodes),
        ],
        ignore_index=True,
    )

    # --- Capacity bound: bound_activity_up on reservoir_storage ----------
    cap = read_dam_capacity(context)
    cap = cap[cap["node"].isin(nodes)]
    cap_wide = cap.pivot_table(index="node", columns="year", values="capacity_mcm")
    cap_wide = cap_wide.reindex(columns=sorted(set(cap_wide.columns) | set(year_wat)))
    cap_wide = cap_wide.ffill(axis=1).bfill(axis=1)

    bound_rows = []
    for node in nodes:
        for year in year_wat:
            value = cap_wide.loc[node, year] if node in cap_wide.index else 0.0
            if pd.isna(value) or value <= 0:
                continue
            bound_rows.append(
                {
                    "node_loc": node,
                    "technology": STORAGE_TEC,
                    "year_act": year,
                    "mode": "M1",
                    "time": "year",
                    "value": float(value),
                    "unit": "MCM",
                }
            )
    results["bound_activity_up"] = pd.DataFrame(
        bound_rows,
        columns=["node_loc", "technology", "year_act", "mode", "time", "value", "unit"],
    )

    # --- storage_initial (fraction of capacity) ---------------------------
    frac = read_storage_initial_fraction(context).set_index("node")["fraction"]
    default_frac = float(frac.mean()) if len(frac) else 0.5
    results["storage_initial"] = pd.DataFrame(
        [
            {
                "node": node,
                "technology": STORAGE_TEC,
                "mode": "M1",
                "level": STORAGE_LEVEL,
                "commodity": STORAGE_COMMODITY,
                "year": year,
                "time": "year",
                "value": float(frac.get(node, default_frac)),
                "unit": "%",
            }
            for node in nodes
            for year in year_wat
        ],
        columns=[
            "node", "technology", "mode", "level", "commodity", "year", "time",
            "value", "unit",
        ],
    )

    # --- storage_self_discharge (fraction, annual mean of monthly evap) --
    disch = read_self_discharge_fraction(context).set_index("node")["fraction"]
    default_disch = float(disch.mean()) if len(disch) else 0.0
    results["storage_self_discharge"] = pd.DataFrame(
        [
            {
                "node": node,
                "technology": STORAGE_TEC,
                "mode": "M1",
                "level": STORAGE_LEVEL,
                "commodity": STORAGE_COMMODITY,
                "year": year,
                "time": "year",
                "value": float(disch.get(node, default_disch)),
                "unit": "%",
            }
            for node in nodes
            for year in year_wat
        ],
        columns=[
            "node", "technology", "mode", "level", "commodity", "year", "time",
            "value", "unit",
        ],
    )

    validate_storage_initial(context)

    return results
