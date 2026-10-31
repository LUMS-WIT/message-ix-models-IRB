# NEST-Pakistan workflow

A national energy model (MESSAGEix-Pakistan) coupled to the NEST water and
land module of a river basin (the Indus by default), run from the command
line. It replaces the step-by-step notebooks (`Water_Integration2.ipynb`,
`pakistan_baseline_org.ipynb`, `water_integration1_model.py`), which stay in
the NEST-Pakistan repository as legacy and are no longer needed to run the model.

## The pipeline

```
 national energy model        NEST water/land module         checks & solve            reports
 ─────────────────────        ───────────────────────        ──────────────            ───────
 Excel workbook               basin data (data/water/*_IRB)  consistency check         PDF + CSV
   │ build-energy               │                              │ calibration             basin / national /
   ▼                            ▼                              ▼                         upper basins / whole basin
 LTS ──relax──► clone ──► water_build ──► LTS_nexus ──► calibrate ──► solve ──► report / compare
```

| Step | Command | Code |
|---|---|---|
| Which input files the basin has | `mix-models nest-pak data` | `basin_data.py` |
| Build the national energy model from Excel | `mix-models nest-pak build-energy` | `workflow.build_energy` |
| Relax, add water/land, calibrate | `mix-models nest-pak build -s NAME` | `workflow.relax_energy`, `build_nexus`, `calibration.py` |
| Check inputs and structure before solving | `mix-models nest-pak check -s NAME` | `checks.py` |
| Solve | `mix-models nest-pak solve -s NAME` | `workflow.solve` |
| All of the above + report | `mix-models nest-pak run -s NAME` | `workflow.run` |
| Report one scenario | `mix-models nest-pak report -s NAME` | `report/` |
| Compare scenarios | `mix-models nest-pak compare -s A -s B` | `report.compare` |
| Scenarios in the database | `mix-models nest-pak list` | |
| Settings in effect | `mix-models nest-pak config` | `config.py` |

`-s` takes `NAME` or `NAME:VERSION`.

## Configuration

Every assumption is in [`config.yaml`](config.yaml): database, model name,
energy workbook and the constraints relaxed on it, the basin, climate (`rcp`),
SSP, data reliability, the water-module switches (irrigation sources, fossil
groundwater, external electricity cost), calibration fixes, solver options,
reporting years and region groups.

Change a setting for one run with `--set`, before the command name:

```
mix-models nest-pak --set nexus.rcp=7p0 --set nexus.context.external_electricity_cost=350 run -s LTS_nexus_rcp70_elec350
```

Electricity prices in the regions without an energy model (India, China,
Afghanistan) are set per region in `nexus.context.external_electricity_cost`,
as a number or a path `{year: price}`, in `external_electricity_cost_unit`
(USD/kWh) and dollars of `external_electricity_currency_year`. They are
converted to the model's unit (× 8760 → million USD/GWa) and currency year
(`model_currency_year`, US GDP deflator). The `check` report prints each price
before and after conversion; `report` shows the price and the electricity bill
per region. For example, cheaper power in China after 2030:

```
mix-models nest-pak --set "nexus.context.external_electricity_cost.CHINA={2030: 0.065, 2050: 0.04}" run -s LTS_nexus_cn_cheap
```

or keep a scenario's settings in its own file, which only needs the keys it changes:

```
mix-models nest-pak --config scenarios/fossil_cap.yaml run -s LTS_nexus_fossilcap
```

## Reports

`report` writes to `report.output_dir/<scenario>/`:

- `report_<scenario>.pdf`: title page, headline table, river water balance
  check, calibration log, then one section per theme with charts for the
  national region, the region comparison for 2030/2050/2070, the upper
  basins, and a basin-by-year map.
- `results_nodes.csv`: every indicator per basin node and year.
- `results_regions.csv`: the same summed to Pakistan, India, China,
  Afghanistan, "Upper basins" and the whole basin (prices averaged).
- `river_balance_check.csv`: the double-counting check.

Themes (`--theme`, repeatable): `water`, `energy-for-water`,
`water-for-energy`, `energy`, `land`, `investment`, `costs`, `emissions`.
Levels (`--level`): `regions`, `basins`. `--node 16|PAKISTAN` adds a one-basin report.

Every number comes from one table built by `report/extract.py`. An indicator
named `"Family: item"` is drawn automatically as a stacked chart of its
family, so adding an indicator there is enough to get it into every report.

## Running notes

- The HSQLDB database can be open in **one process at a time**. Close
  notebooks that use it (`mp.close_db()`) before running a command; the
  command says so if the database is locked.
- The database is about 3 GB and is held in the Java heap: `platform.jvmargs`
  defaults to `-Xmx10G`.
- GAMS is taken from `platform.gams_dir`.
- A full `run` takes about 25-50 minutes (build ~15, solve ~10-30).

## A different river basin

See [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md): the files a basin needs,
which can come from global datasets (ISIMIP and others), and the steps to set
it up. `mix-models nest-pak --set basin.code=XYZ data` lists what is missing.
