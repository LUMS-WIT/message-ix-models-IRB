"""Shared R-model PID -> this project's BCU_name crosswalk.

Factored out of :mod:`build_basin_links_IRB.py` so every ``build_*_IRB.py``
conversion script (network data, crop/irrigation data, ...) uses the exact
same mapping instead of maintaining separate copies that could silently
drift apart.

See :mod:`build_basin_links_IRB` for the full evidence behind this mapping
and its confidence level -- it is domain-agnostic (basin identity only, not
tied to network/crop/any other data), so it belongs here rather than in any
one domain's build script.
"""

#: R-model PID -> this project's BCU_name.
#:
#: Spatially verified (2026-10-20). The earlier ordinal mapping was wrong for
#: 16 of 24 basins (IND_1..3, all PAK_*). Each R basin
#: (network/raw/Indus_bcu.shp) was intersected with the delineation behind
#: the hydrology data (thesis "ISIMP Water Model/Indus Shape Files/
#: Indus_bcu.shp", whose row i is row i of basins_by_region_simpl_IRB.csv and
#: of every qtot/qr CSV): every pair overlaps 99.8-100 %, next best <= 0.2 %,
#: one-to-one.
CROSSWALK = {
    "AFG_1": "1|AFGHAN_SOUTH", "AFG_2": "2|AFGHAN_NORTH",
    "CHN_1": "3|CHINA", "CHN_2": "4|CHINA", "CHN_3": "5|CHINA",
    "IND_1": "7|INDIA_WEST", "IND_2": "8|INDIA_WEST", "IND_3": "6|INDIA_WEST",
    "IND_4": "9|INDIA_EAST", "IND_5": "10|INDIA_WEST", "IND_6": "11|INDIA_WEST",
    "PAK_1": "13|PAKISTAN", "PAK_2": "14|PAKISTAN", "PAK_3": "15|PAKISTAN",
    "PAK_4": "17|PAKISTAN", "PAK_5": "18|PAKISTAN", "PAK_6": "19|PAKISTAN",
    "PAK_7": "21|PAKISTAN", "PAK_8": "22|PAKISTAN", "PAK_9": "12|PAKISTAN",
    "PAK_10": "16|PAKISTAN", "PAK_11": "24|PAKISTAN", "PAK_12": "20|PAKISTAN",
    "PAK_13": "23|PAKISTAN",
}
assert len(CROSSWALK) == 24
assert len(set(CROSSWALK.values())) == 24
