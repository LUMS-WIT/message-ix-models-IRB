# Setting up another river basin

The water and land module reads every basin-specific input from files named
with a basin code (`*_IRB.csv` for the Indus). A new basin needs the same files
under a new code, plus a national energy model for at least one country.
`mix-models nest-pak --set basin.code=XYZ data` lists each file, whether it
exists, and where it can come from.

## What can come from global data

Most of the water and land inputs can be built from open global datasets.
Three things need national data: the energy model, irrigation withdrawals
(global data exists but is coarse), and canals.

| Input | Global source | Global data enough? |
|---|---|---|
| Basin units and their countries | HydroBASINS (HydroSHEDS) level 4-6 × GADM borders | Yes |
| River links between basin units | HydroRIVERS / HydroSHEDS flow direction | Yes (natural rivers) |
| Inter-basin canals | National irrigation maps | **No** |
| Runoff `qtot`, recharge `qr`, discharge `dis` | **ISIMIP2b / ISIMIP3b** global hydrological models | Yes |
| Environmental flows | Computed from ISIMIP discharge (Variable Monthly Flow) | Yes |
| Groundwater pumping energy | Water-table depth (Fan et al. 2013) | Yes |
| City, village and industry water demand (SSP) | MESSAGEix-Nexus harmonised demands (Huang et al. 2018 + SSP) | Yes |
| Crop area and yields | SPAM 2010/2020, GAEZ v4, FAOSTAT | Yes |
| Crop irrigation water requirement | GAEZ v4 / CROPWAT, or ISIMIP crop models (LPJmL, PEPIC) | Yes |
| Irrigated area | LGRIP30 (30 m), GMIA | Yes |
| Historical irrigation withdrawals (the irrigation floor) | AQUASTAT, Huang et al. 2018, LGRIP30 × water requirement | **Partly**: national canal/tubewell statistics are much better |
| Irrigation method costs and efficiencies | FAO and literature; the Indus files are a usable default | Partly |
| Share of pumping on grid electricity | National agricultural census | **No** (assume, then test) |
| Population, GDP (SSP) | IIASA SSP database, WorldPop | Yes |
| Reservoirs | GRanD v1.3 | Yes |
| Desalination plants and potential | Jones et al. (2019) | Yes |
| Power plant cooling shares and costs | MESSAGEix-Nexus global defaults (Raptis & Pfister 2016) | Yes |
| National energy model | National data; or start from a MESSAGEix-GLOBIOM R12 region | **No** (one per modelled country) |

## Hydrology from ISIMIP

ISIMIP provides daily and monthly runoff (`qtot`), groundwater recharge (`qr`)
and routed discharge (`dis`) on a 0.5° grid (about 55 km) from several global
hydrological models driven by several climate models:

- **ISIMIP2b**: RCP2.6 / RCP6.0 / RCP8.5; GCMs GFDL-ESM2M, HadGEM2-ES,
  IPSL-CM5A-LR, MIROC5; hydrological models H08, LPJmL, PCR-GLOBWB, WaterGAP2.
- **ISIMIP3b**: SSP1-2.6 / SSP3-7.0 / SSP5-8.5; CMIP6 GCMs; models such as
  CWatM, H08, WaterGAP2. The Indus files use the `2p6` / `7p0` / `no_climate`
  naming.

Steps (`model/water/data/pre_processing/`):

1. Download `qtot`, `qr`, `dis` for the scenarios and models you want
   (https://data.isimip.org).
2. Aggregate grid cells to basin units with `hydro_agg_spatial.R`,
   `hydro_agg_raster.py` or `hydro_agg_basin.py`. **Use the sum over a basin
   unit's cells, not the mean.** The mean gives the volume of one average
   cell; this bug once left the Indus with a small fraction of its real water.
3. Average to 5-year steps. Take the ensemble quantiles across GCM × model
   runs as the `low` / `med` / `high` reliability levels.
4. Optionally, monthly files, which let reservoirs shift water between seasons.
   The ISIMIP monthly values are a **rate in km³/year** for each month, so a
   month's volume is rate × days / 365.
   `pre_processing/build_monthly_data_IRB.py` (or `mix-models nest-pak
   monthly-data`, which also draws them) writes every monthly input as a
   volume per month to `data/water/monthly/`. These inputs are hydrology with
   reliability levels per calendar month, irrigation, sectoral profiles and
   reservoirs.

Note: the annual `low` / `med` / `high` files are percentiles of the
*monthly* rates taken across all months (not annual means). This puts annual
runoff about 40 % below the mean from 2025 on.

Limits to keep in mind:

- 0.5° cells are coarse for small basin units. A unit should cover tens of
  cells.
- Global models don't represent glacier melt or snow storage well in some
  basins (for example, the upper Indus). Compare runoff with gauged flows at
  the basin outlets before trusting it.
- Recharge and runoff overlap: part of `qr` reappears as baseflow inside
  `qtot`. The model can pump natural recharge that is also counted in the
  rivers. For the Indus the upper bound is about 14 km³/yr against 131 km³/yr
  of runoff; see the river balance check in each report.

## Steps for a new basin

1. **Choose the basin units.** Cut HydroBASINS at a level that gives 10-40
   units, and split them at country borders. Name them `n|COUNTRY`
   (`12|PAKISTAN`). Write `delineation/basins_by_region_simpl_<CODE>.csv` and
   `delineation/basins_country_<CODE>.csv`.
2. **Node list.** Write `message_ix_models/data/node/<CODE>.yaml` with every
   basin unit and its parent region node. Use the energy model's node as the
   parent for the country that has one (`R12_PAK`).
3. **River network.** Write `network/basin_links_<CODE>.csv`, with upstream →
   downstream links and any canals; see `build_basin_links_IRB.py`.
4. **Hydrology** from ISIMIP (above): `availability/qtot_5y_*`, `qr_5y_*`,
   `dis_5y_*`, plus `e-flow_*` if environmental flows are used.
5. **Demands**: `demands/harmonized/<CODE>/`, downscaled from the global
   MESSAGEix-Nexus demands.
6. **Crops and irrigation**: the `crops/*_<CODE>.csv` files (SPAM, GAEZ,
   LGRIP30; see `build_crop_data_IRB.py` and `build_irrigation_demand_lgrip30.py`).
   Spend the most care on historical irrigation withdrawals: they set the
   irrigation floor that drives most results.
7. **Infrastructure and cooling**: desalination (Jones et al.), reservoirs
   (GRanD), cooling shares (global defaults).
8. **Energy model**: a MESSAGEix scenario for the country that owns the energy
   node. Other countries in the basin can stay without one; their water
   sector then buys electricity from a costed external supply. Set a price
   per region in `nexus.context.external_electricity_cost` (USD/kWh, with its
   currency year), using the economic cost of grid supply rather than the
   subsidised farm tariff. Make sure `model_currency_year` matches the
   currency of the energy model. The amount supplied is reported as
   "Additional electricity needed", and its cost as "Cost of additional electricity".
9. **Config**: copy `config.yaml` and set `basin.code`, `basin.name`,
   `basin.region_node_map`, `report.groups`, the energy workbook and the model
   name.
10. **Run**: `data` (all required files present?) → `build-energy` → `build` →
    `check` (fix every FAIL) → `solve` → `report`.

## Code that is still Indus-specific

These parts need work before a second basin runs end to end:

- The `pre_processing/*_IRB.py` scripts and `_pid_crosswalk.py` contain Indus
  basin IDs. Copy and adapt them, or generalise them.
- `storage/dam_storage_capacity_GRANDv1.1_M_m3_PID.csv` is one file without a
  basin code; split it per code.
- The calibration fixes in `config.yaml` (`calibration.bound_activity_lo`) are
  specific to the Pakistan energy data. Re-derive them with `check` on the new
  energy model.
- Transboundary water agreements (for example the Indus Waters Treaty) are not
  modelled. Upstream countries pass on whatever they don't use.
