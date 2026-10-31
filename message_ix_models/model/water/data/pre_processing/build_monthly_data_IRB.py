"""Monthly input data for the Indus water module, in one consistent form.

The raw monthly inputs come in three different conventions:

- ISIMIP hydrology (``availability/qtot|qr|e-flow*_5y_m_*``, ``*_monthly_*``)
  is a flow *rate* in km³/year for each month (``hydro_agg_raster.py``
  converts the kg/m²/s flux to km³/year), so a month's volume is
  ``rate x days / 365``;
- irrigation (``crops/historical_irrigation_withdrawals_IRB.csv``,
  ``crops/crop_irrigation_water_IRB.csv``) is MCM/day per month;
- reservoir data (``storage/*``) uses the legacy PID codes (``PAK_8``).

This script writes every monthly input as a **volume in that month** (or a
share / fraction), keyed by ``BCU_name``, to ``data/water/monthly/``:

``hydrology_monthly_<CODE>.csv``
    variable (qtot = river runoff, qr = groundwater recharge, e-flow =
    environmental flow from runoff, e-flow_qr = from recharge), rcp, rel,
    node, year, month, value, unit (km3). ``rel``:

    - ``mean``: 5-year mean of each calendar month (the ``_5y_m_`` files);
    - ``low`` / ``med`` / ``high``: 50th / 30th / 10th percentile of each
      calendar month over a 20-year window centred on the year (clipped to
      2015-2100), i.e. the flow reached in 50 / 70 / 90 % of years, the
      same reliability levels the annual files are named after. Computed per
      calendar month, so a dry month is compared with other years' same
      month, not with the monsoon;
    - e-flows have ``rel = all``.

``irrigation_monthly_<CODE>.csv``
    source (historical canal / tubewell withdrawal; crop water requirement
    by year = MCM/day per Mha x the basin's 2015 irrigated area of each
    crop, a net requirement at the field), node, year, month, value (MCM in
    the month), share (of the node's annual total).
``sectoral_monthly_profile_<CODE>.csv``
    Monthly share of city, village and industry water demand. No monthly
    data exist for the Indus: the share is the month's days / 365 (flat).
    Replace with measured profiles when available.
``storage_monthly_<CODE>.csv``
    Reservoir evaporation (fraction of the stored volume lost in the month,
    ensemble mean, per basin, rcp and year) and the WAPDA rule curves of
    Tarbela and Mangla (fill ratio at month end).

Every output can be grouped into any set of time slices (seasons, quarters)
by summing volumes over months.

Run: ``python -m message_ix_models.model.water.data.pre_processing.build_monthly_data_IRB``
"""

import logging

import numpy as np
import pandas as pd

from message_ix_models.util import package_data_path

from ._pid_crosswalk import CROSSWALK

log = logging.getLogger(__name__)

CODE = "IRB"
RCPS = ["2p6", "7p0", "no_climate"]
YEARS = list(range(2015, 2101, 5))
#: Reliability level -> percentile of a calendar month's flow across years.
RELIABILITY = {"low": 0.5, "med": 0.3, "high": 0.1}
WINDOW = 20  # years
#: ISIMIP2b names in the evaporation file -> this model's climate names.
EVAP_RCP = {"historical": "no_climate", "rcp26": "2p6", "rcp60": "7p0"}
#: Dam -> basin, as in ``storage.validate_storage_initial``.
DAM_PID = {"Tarbela": "PAK_8", "Mangla": "PAK_1"}

#: Month groupings used for summaries (and later for time slices).
SCHEMES = {
    "seasonal": {"kharif (Apr-Sep)": [4, 5, 6, 7, 8, 9], "rabi (Oct-Mar)": [10, 11, 12, 1, 2, 3]},
    "quarterly": {"Apr-Jun": [4, 5, 6], "Jul-Sep": [7, 8, 9], "Oct-Dec": [10, 11, 12], "Jan-Mar": [1, 2, 3]},
}

DAYS = pd.Series({1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31})


def _path(*parts):
    return package_data_path("water", *parts)


def basins(code: str = CODE) -> list[str]:
    """BCU_name in the row order of the hydrology files."""
    return pd.read_csv(_path("delineation", f"basins_by_region_simpl_{code}.csv"))["BCU_name"].tolist()


def _read_wide(name: str, code: str = CODE) -> pd.DataFrame:
    """A wide basin x date file as long rows: node, year, month, rate."""
    df = pd.read_csv(_path("availability", name)).drop(columns="Unnamed: 0", errors="ignore")
    df.index = basins(code)
    df.columns = pd.to_datetime(df.columns)
    long = df.stack().rename("rate").reset_index()
    long.columns = ["node", "date", "rate"]
    return long.assign(year=long["date"].dt.year, month=long["date"].dt.month).drop(columns="date")


def _volume(df: pd.DataFrame, col: str = "rate") -> pd.Series:
    """km³/year rate in a month -> km³ in that month."""
    return df[col] * df["month"].map(DAYS) / 365


def hydrology(code: str = CODE) -> pd.DataFrame:
    parts = []
    for rcp in RCPS:
        for var in ("qtot", "qr"):
            m = _read_wide(f"{var}_5y_m_{rcp}_low_{code}.csv", code)
            parts.append(m.assign(variable=var, rcp=rcp, rel="mean", value=_volume(m)))
        for var, f in (("e-flow", f"e-flow_5y_m_{rcp}_{code}.csv"), ("e-flow_qr", f"e-flow_qr_5y_m_{rcp}_{code}.csv")):
            m = _read_wide(f, code)
            parts.append(m.assign(variable=var, rcp=rcp, rel="all", value=_volume(m)))
        for var in ("qtot", "qr"):
            parts.append(_reliability(var, rcp, code))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["year"].isin(YEARS)].assign(unit="km3")
    return df[["variable", "rcp", "rel", "node", "year", "month", "value", "unit"]].sort_values(
        ["variable", "rcp", "rel", "node", "year", "month"]).reset_index(drop=True)


def _reliability(var: str, rcp: str, code: str) -> pd.DataFrame:
    """Percentiles of each calendar month's volume over a 20-year window."""
    if rcp == "no_climate":
        # Present-day climate held constant: both scenarios' 2015-2034 months.
        s = pd.concat([_read_wide(f"{var}_monthly_{r}_{code}.csv", code) for r in ("2p6", "7p0")])
        s = s[s["year"].between(2015, 2034)].assign(value=lambda d: _volume(d))
        q = s.groupby(["node", "month"])["value"]
        return pd.concat([pd.concat([q.quantile(p).reset_index().assign(year=y) for y in YEARS]).assign(
            variable=var, rcp=rcp, rel=rel) for rel, p in RELIABILITY.items()])
    s = _read_wide(f"{var}_monthly_{rcp}_{code}.csv", code)
    s = s.assign(value=_volume(s))
    first, last = s["year"].min(), s["year"].max()
    rows = []
    for y in YEARS:
        lo = max(first, min(y - WINDOW // 2, last - WINDOW + 1))
        w = s[s["year"].between(lo, lo + WINDOW - 1)].groupby(["node", "month"])["value"]
        for rel, p in RELIABILITY.items():
            rows.append(w.quantile(p).reset_index().assign(year=y, variable=var, rcp=rcp, rel=rel))
    return pd.concat(rows)


def irrigation(code: str = CODE) -> pd.DataFrame:
    h = pd.read_csv(_path("crops", f"historical_irrigation_withdrawals_{code}.csv"))
    h = h.assign(source=h["tec"].map({"irrigation_sw_diversion": "historical canal withdrawal",
                                      "irrigation_gw_diversion": "historical tubewell withdrawal"}),
                 year=np.nan, month=h["time"].astype(int))
    # Crop water requirement is MCM/day per Mha of crop: weight by each
    # basin's irrigated area of that crop (2015) to get a volume.
    c = pd.read_csv(_path("crops", f"crop_irrigation_water_{code}.csv"))
    a = pd.read_csv(_path("crops", f"crop_input_data_{code}.csv"))
    a = a[a["par"] == "crop_irr_land_2015"][["crop", "node", "value"]].rename(columns={"value": "Mha"})
    c = c.merge(a, on=["crop", "node"], how="inner").assign(value=lambda d: d["value"] * d["Mha"])
    c = c.groupby(["node", "year", "time"], as_index=False)["value"].sum()
    c = c.assign(source="crop water requirement (2015 irrigated area)", month=c["time"].astype(int))
    df = pd.concat([h, c], ignore_index=True)[["source", "node", "year", "month", "value"]]
    df["value"] = df["value"] * df["month"].map(DAYS)  # MCM/day -> MCM in the month
    total = df.groupby(["source", "node", "year"], dropna=False)["value"].transform("sum")
    df["share"] = (df["value"] / total).fillna(0)
    return df.assign(unit="MCM").sort_values(["source", "node", "year", "month"]).reset_index(drop=True)


def sectoral_profile() -> pd.DataFrame:
    rows = [(s, k, m, DAYS[m] / 365) for s in ("urban", "rural", "manufacturing")
            for k in ("withdrawal", "return") for m in DAYS.index]
    return pd.DataFrame(rows, columns=["sector", "type", "month", "share"]).assign(
        note="assumed flat (days/365); no monthly data for this basin")


def storage(code: str = CODE) -> pd.DataFrame:
    e = pd.read_csv(_path("storage", "evapo_loss_reservoir_per_unit_volume.csv"))
    e = e[(e["model"] == "ensemble") | (e["scenario"] == "historical")]
    e = e.assign(node=e["node"].map(CROSSWALK), rcp=e["scenario"].map(EVAP_RCP), month=e["time"].astype(int))
    e = e.groupby(["rcp", "node", "year", "month"], as_index=False)["value"].mean().assign(
        variable="evaporation loss", unit="fraction of stored volume")
    r = pd.read_csv(_path("storage", "rule_curves_values.csv"))
    r = r.assign(node=r["DAM_NAME"].map(DAM_PID).map(CROSSWALK), month=r["time"].astype(int),
                 variable="rule curve: " + r["DAM_NAME"], value=r["ratio_avg_max"], unit="fill ratio",
                 capacity_MCM=r["V_max"], rcp=np.nan, year=np.nan)
    cols = ["variable", "rcp", "node", "year", "month", "value", "unit"]
    return pd.concat([e[cols], r[cols + ["capacity_MCM"]]], ignore_index=True)


def group_months(df: pd.DataFrame, scheme: dict, value: str = "value") -> pd.DataFrame:
    """Sum a monthly ``value`` into the slices of ``scheme`` (name -> months)."""
    m2s = {m: s for s, months in scheme.items() for m in months}
    keys = [c for c in df.columns if c not in (value, "month", "share")]
    return df.assign(slice=df["month"].map(m2s)).groupby(keys + ["slice"], as_index=False, dropna=False)[value].sum()


def build(code: str = CODE, write: bool = True) -> dict[str, pd.DataFrame]:
    out = {f"hydrology_monthly_{code}": hydrology(code), f"irrigation_monthly_{code}": irrigation(code),
           f"sectoral_monthly_profile_{code}": sectoral_profile(), f"storage_monthly_{code}": storage(code)}
    if write:
        d = _path("monthly")
        d.mkdir(exist_ok=True)
        for name, df in out.items():
            df.to_csv(d / f"{name}.csv", index=False)
            log.info(f"wrote {d / name}.csv ({len(df)} rows)")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for name, df in build().items():
        print(name, df.shape)
