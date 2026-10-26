"""A third, independent, OPT-IN water-demand mechanism: a per-basin
relation built bottom-up from real crop area (SPAM2020) x real crop water
requirement (this project's own ``crop_irrigation_water_IRB.csv``), as an
alternative lower bound alongside :mod:`.irrigation_tech`'s existing
historical-withdrawal floor -- NOT a replacement for it, and disabled by
default (same per-run kill-switch pattern :mod:`.food_demand` already uses
for ``context.crop_demand_enabled``, here ``context.bottom_up_water_demand_enabled``).

Why this exists
-----------------
The historical-withdrawal floor (:mod:`.irrigation_tech`) answers "how much
water was withdrawn here historically." This answers a different question,
built from first principles instead of a historical observation: "how much
water does the crop area actually growing here need." Comparing the two is
a genuine validation signal (see
``NEST-Pakistan/NEST_IRB/outputs/crosscheck_bottom_up_vs_historical_withdrawal_IRB.csv``)
-- they do NOT currently agree closely (the bottom-up number comes out to
roughly 10-25% of the historical figure in most basins), for reasons
documented there (net crop water requirement vs. gross withdrawal; SPAM has
no "fodder" category, so this project's "fodder" bucket here is really
oilseeds+tobacco+catch-all, not real fodder demand). This mechanism exists
to make that comparison live inside the actual model/solve, not just a
spreadsheet exercise -- NOT because the bottom-up number is asserted to be
more correct than the existing floor.

Design: reuse, don't duplicate, :mod:`.irrigation_tech`'s coefficients
--------------------------------------------------------------------------
A relation constraining the same ``irr_<method>_<crop>`` technologies'
withdrawal activity needs the exact same ``relation_activity`` coefficients
:func:`.irrigation_tech.add_irrigation_techs` already computes (each
technology's real withdrawal rate, MCM per Mha). Rather than re-derive that
computation independently (risking subtle drift from the proven, tested
version), this module calls :func:`.irrigation_tech.add_irrigation_techs`
itself and reuses its returned ``relation_activity`` rows directly under a
new relation name -- :func:`.irrigation_tech.add_irrigation_techs` itself
is NOT modified, so its own existing tests and behavior are unaffected
regardless of whether this module is enabled.

Softened the same way
------------------------
Uses the same ``SLACK_RELATION_BOUND_LO`` mechanism as the historical
floor -- this relation can never make a scenario infeasible outright.

A real limitation, stated plainly
------------------------------------
SPAM2020 and the CWR data are both single-year snapshots (2020-nearest);
unlike the historical floor, this mechanism applies ONE flat value across
every model year -- no population/GDP growth index is applied here. A
basin's bottom-up floor does not grow over the model horizon the way the
historical floor does.
"""

import logging

import pandas as pd
from message_ix import make_df

from message_ix_models import Context

from .common import read_bottom_up_crop_water_demand
from .irrigation_tech import _sanitize, add_irrigation_techs

log = logging.getLogger(__name__)

#: Relation name prefix -- deliberately distinct from
#: irrigation_tech.py's "hist_irr_withdrawal_" so the two can coexist on
#: the same scenario without colliding.
RELATION_PREFIX = "bottom_up_crop_water_"


def bottom_up_relation_names(context: "Context") -> list[str]:
    """One ``bottom_up_crop_water_<basin>`` relation per basin with real
    bottom-up demand data -- empty list entirely if the mechanism is
    disabled (the default), so build.py registers nothing for it."""
    if not getattr(context, "bottom_up_water_demand_enabled", False):
        return []
    demand = read_bottom_up_crop_water_demand(context)
    if demand.empty:
        return []
    positive_nodes = demand.loc[demand["value"] > 0, "node"]
    return [f"{RELATION_PREFIX}{_sanitize(n)}" for n in sorted(set(positive_nodes))]


def add_bottom_up_crop_water_demand(context: "Context") -> dict[str, pd.DataFrame]:
    """Build the ``relation_activity``/``relation_lower`` data for the
    bottom-up water-demand relation. Returns ``{}`` if disabled (the
    default), if there's no real bottom-up demand data for this context's
    valid basins, or if :func:`.irrigation_tech.add_irrigation_techs` has
    no relation_activity to reuse (e.g. no real historical-withdrawal data
    either, in which case there's nothing to attach a withdrawal rate to).
    """
    if not getattr(context, "bottom_up_water_demand_enabled", False):
        return {}
    if not getattr(context, "valid_basins", None):
        return {}

    demand = read_bottom_up_crop_water_demand(context)
    demand = demand[demand["value"] > 0]
    if demand.empty:
        return {}
    demand_lookup = demand.set_index("node")["value"]

    irr_results = add_irrigation_techs(context)
    base_relation_activity = irr_results.get("relation_activity")
    if base_relation_activity is None or base_relation_activity.empty:
        return {}

    rel_act_frames, rel_lo_frames = [], []
    for node, floor_mcm in demand_lookup.items():
        node_loc = "B" + node
        node_rows = base_relation_activity[base_relation_activity["node_loc"] == node_loc]
        if node_rows.empty:
            continue
        relation = f"{RELATION_PREFIX}{_sanitize(node)}"
        rel_act_frames.append(node_rows.assign(relation=relation))
        for year in sorted(node_rows["year_rel"].unique()):
            rel_lo_frames.append(
                make_df(
                    "relation_lower",
                    relation=relation,
                    node_rel=node_loc,
                    year_rel=int(year),
                    value=float(floor_mcm),
                    unit="MCM",
                )
            )

    if not rel_act_frames:
        return {}

    return {
        "relation_activity": pd.concat(rel_act_frames, ignore_index=True),
        "relation_lower": pd.concat(rel_lo_frames, ignore_index=True),
    }
