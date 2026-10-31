"""Monthly input data: build it and draw it, before any monthly model run.

``mix-models nest-pak monthly-data`` runs
:mod:`.pre_processing.build_monthly_data_IRB` (unit-consistent monthly files
in ``data/water/monthly/``) and writes ``monthly_data_<code>.pdf`` with the
seasonal pattern of water supply and demand per region, the reliability
levels, climate change by month, and the reservoir data.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from message_ix_models.model.water.data.pre_processing import build_monthly_data_IRB as bm

from .config import output_dir
from .report.pdf import A4, INK, MUTED, Pdf

log = logging.getLogger(__name__)

#: Indus water year: kharif first.
ORDER = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]
LABELS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
COLORS = {"river runoff": "#2a6fb0", "groundwater recharge": "#7fb3d5", "environmental flow": "#9e9e9e",
          "historical canal withdrawal": "#d9822b", "historical tubewell withdrawal": "#8c564b",
          "crop water requirement": "#3a9d5d"}


def region_of(node: str, groups: dict) -> str:
    for g, subs in groups.items():
        if any(s in node for s in subs):
            return g
    return "other"


def _monthly(df: pd.DataFrame) -> pd.Series:
    return df.groupby("month")["value"].sum().reindex(ORDER, fill_value=0)


def _axis(ax, title, unit):
    ax.set_xticks(range(12), LABELS, fontsize=8)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(unit, fontsize=8, color=MUTED)
    ax.axvspan(-0.5, 5.5, color="#fff4e0", zorder=0)  # kharif
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=8)


def shares(data: dict, groups: dict, rcp: str, year: int) -> pd.DataFrame:
    """Share of each series in the two seasons and four quarters, per region."""
    h, i = data["hydro"], data["irr"]
    series = {
        "river runoff": h[(h.variable == "qtot") & (h.rel == "mean") & (h.rcp == rcp) & (h.year == year)],
        "groundwater recharge": h[(h.variable == "qr") & (h.rel == "mean") & (h.rcp == rcp) & (h.year == year)],
        "historical canal + tubewell withdrawal": i[i.source.str.startswith("historical")],
        "crop water requirement": i[i.source.str.startswith("crop") & (i.year == 2020)],
    }
    rows = []
    for name, df in series.items():
        for reg in ["Whole basin"] + list(groups):
            d = df if reg == "Whole basin" else df[df.node.map(lambda n: region_of(n, groups)) == reg]
            tot = d["value"].sum()
            if tot <= 0:
                continue
            row = {"series": name, "region": reg}
            for scheme in bm.SCHEMES.values():
                for s, months in scheme.items():
                    row[s] = 100 * d[d.month.isin(months)]["value"].sum() / tot
            rows.append(row)
    return pd.DataFrame(rows)


def storage_need(h: pd.DataFrame, i: pd.DataFrame, rcp: str, year: int) -> pd.DataFrame:
    """Runoff - e-flow - canal withdrawal per month for the whole basin, and
    the total shortfall (water storage would have to supply) per reliability."""
    s = h[(h.rcp == rcp) & (h.year == year)]
    ef = _monthly(s[s.variable == "e-flow"])
    canal = _monthly(i[i.source == "historical canal withdrawal"]) / 1000
    rows = []
    for rel in ("mean", "low", "med"):
        left = _monthly(s[(s.variable == "qtot") & (s.rel == rel)]) - ef - canal
        rows.append({"runoff": rel, **dict(zip(LABELS, left)), "shortfall": -left.clip(upper=0).sum(),
                     "surplus": left.clip(lower=0).sum()})
    return pd.DataFrame(rows)


def report(cfg: dict, rcp: str | None = None, year: int = 2030, build: bool = True):
    code = cfg["basin"]["code"]
    rcp = rcp or cfg["nexus"]["rcp"]
    groups = cfg["report"]["groups"]
    files = bm.build(code, write=True) if build else {
        n: pd.read_csv(bm._path("monthly", f"{n}_{code}.csv")) for n in
        ("hydrology_monthly", "irrigation_monthly", "sectoral_monthly_profile", "storage_monthly")}
    pick = lambda prefix: next(v for k, v in files.items() if k.startswith(prefix))  # noqa: E731
    data = {"hydro": pick("hydrology"), "irr": pick("irrigation"), "storage": pick("storage")}
    h, i, st = data["hydro"], data["irr"], data["storage"]
    h = h.assign(region=h.node.map(lambda n: region_of(n, groups)))
    i = i.assign(region=i.node.map(lambda n: region_of(n, groups)))

    out = output_dir(cfg, f"data_{code}")
    path = out / f"monthly_data_{code}.pdf"
    tab = shares(data, groups, rcp, year)
    tab.round(1).to_csv(out / f"monthly_shares_{code}.csv", index=False)

    pdf = Pdf(path, f"Monthly input data - {cfg['basin']['name']}")
    pdf.title_page("Monthly input data", f"{cfg['basin']['name']} ({code}) - climate {rcp}, year {year}", [
        "Every series is a volume in the month (km³ or MCM), built from the raw inputs by",
        "model/water/data/pre_processing/build_monthly_data_IRB.py into data/water/monthly/.",
        "Months run in the Indus water year: kharif (Apr-Sep, shaded) then rabi (Oct-Mar).",
        "", "Contents: units and findings; supply and demand by month per region; seasonal and",
        "quarterly shares; reliability levels; climate change by month; reservoirs."])

    pdf.text_page("Units and findings", [
        "River runoff (qtot), groundwater recharge (qr) and environmental flows come from ISIMIP3b (CWatM, "
        "GFDL-ESM4) as a flow rate in km³/year for each month. A month's volume is rate x days / 365. Read as "
        "volumes, the monthly files add up to the annual mean flow: 217 km³/yr of runoff in 2020.",
        "The annual files used by the current (annual) model are not annual means from 2025 on: 'low' is the "
        "20-year median of the monthly rates, 'med' the 30th and 'high' the 10th percentile, taken across all "
        "months together. Because a few monsoon months carry most of the flow, the median month is far below "
        "the mean, so annual runoff drops from 216.5 km³ (2020, a true mean) to about 120-130 km³ (2025 on). "
        "That drop is a statistical artefact, not climate change. The monthly files here do not have it: "
        "reliability is taken per calendar month across years (a dry March compared with other Marches).",
        "Irrigation: historical canal and tubewell withdrawals are MCM/day per month; crop water requirement is "
        "MCM/day per Mha, weighted here by each basin's 2015 irrigated area of every crop (a net field "
        "requirement, smaller than gross withdrawal). The monthly shape of historical withdrawals is the crop "
        "requirement's shape (the source spread annual withdrawals with it): modelled timing, not observed. "
        "Crop requirement does not change with year or climate, and its total (about 31 km³) is small against "
        "202 km³ of withdrawals, which implies a field efficiency of about 15 %: check the area or the requirement.",
        "City, village and industry demand has no monthly data for this basin: assumed flat (days / 365).",
        "Reservoir evaporation (ISIMIP2b ensemble; rcp26 -> 2p6, rcp60 -> 7p0) and the WAPDA rule curves of "
        "Tarbela and Mangla are mapped from the legacy PID codes to basin names.",
        "Check against observations: gauged rim-station inflows carry roughly 80-85 % of the year's flow in "
        "kharif. The modelled runoff carries about 71 %, so global hydrology likely spreads the summer melt peak "
        "too thinly; it also dips in May and rises in February-March, earlier than the observed melt. Compare "
        "with WAPDA rim-station flows before trusting the timing.",
        "The rule curves keep Tarbela above 92 % full all year, while the real reservoir is drawn close to dead "
        "level each spring: check what 'V_avg' in rule_curves_values.csv measures before using it as a bound."])

    # Supply and demand by month, whole basin and per region.
    for regions in (["Whole basin"], list(groups)):
        n = 1 if len(regions) == 1 else 2
        fig, axes = plt.subplots(n, n, figsize=A4, squeeze=False)
        for ax, reg in zip(axes.flat, regions):
            sel = (lambda d: d) if reg == "Whole basin" else (lambda d, r=reg: d[d.region == r])
            hy = sel(h[(h.rcp == rcp) & (h.year == year)])
            ir = sel(i)
            x = range(12)
            ax.bar(x, _monthly(hy[(hy.variable == "qtot") & (hy.rel == "mean")]), color=COLORS["river runoff"],
                   alpha=0.85, label="river runoff (local)")
            ax.bar(x, _monthly(hy[(hy.variable == "qr") & (hy.rel == "mean")]), color=COLORS["groundwater recharge"],
                   alpha=0.9, width=0.4, label="groundwater recharge")
            ax.plot(x, _monthly(hy[hy.variable == "e-flow"]), color=COLORS["environmental flow"], lw=1.5,
                    ls="--", label="environmental flow")
            hist = ir[ir.source.str.startswith("historical")]
            ax.plot(x, _monthly(hist) / 1000, color=COLORS["historical canal withdrawal"], lw=2.2, marker="o",
                    ms=3, label="historical withdrawal (canal + tubewell)")
            crop = ir[ir.source.str.startswith("crop") & (ir.year == 2020)]
            ax.plot(x, _monthly(crop) / 1000, color=COLORS["crop water requirement"], lw=2, marker="s", ms=3,
                    label="crop water requirement (net)")
            _axis(ax, reg, "km³ in the month")
        axes.flat[0].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Water supply and irrigation demand by month ({rcp}, {year})", fontsize=13, x=0.05, ha="left")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.save(fig)

    pdf.table("Share of the year in each season and quarter (%)", tab, note=(
        f"Runoff and recharge: {rcp}, {year}, 5-year mean. Withdrawal: historical. Crop requirement: 2020. "
        "Upstream inflow is not included in a region's local runoff."), rows_per_page=22)

    pdf.table("Whole-basin river balance by month: how much storage is needed", storage_need(h, i, rcp, year), note=(
        "River runoff - environmental flow - historical canal withdrawal, km³. Negative months need water moved "
        "from surplus months (storage). Groundwater and upstream timing (travel time) are ignored."),
        rows_per_page=10)

    # Reliability and climate.
    fig, axes = plt.subplots(1, 2, figsize=A4)
    hb = h[(h.rcp == rcp) & (h.year == year) & (h.variable == "qtot")]
    for rel, c in (("mean", "#2a6fb0"), ("low", "#4c9be8"), ("med", "#9bc4ee"), ("high", "#d0e3f7")):
        axes[0].plot(range(12), _monthly(hb[hb.rel == rel]), marker="o", ms=3, color=c,
                     label=f"{rel} ({'5-year mean' if rel == 'mean' else f'reached in {100 - int(bm.RELIABILITY[rel] * 100)}% of years'})")
    _axis(axes[0], f"River runoff by reliability level ({rcp}, {year})", "km³ in the month")
    axes[0].legend(fontsize=7)
    for r, ls in (("2p6", "-"), ("7p0", "--")):
        for y, c in ((2020, "#555555"), (2050, "#2a9d8f"), (2090, "#e76f51")):
            d = h[(h.rcp == r) & (h.year == y) & (h.variable == "qtot") & (h.rel == "mean")]
            axes[1].plot(range(12), _monthly(d), ls=ls, color=c, label=f"{r} {y}")
    _axis(axes[1], "River runoff by month: climate change", "km³ in the month")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf.save(fig)

    # Reservoirs.
    fig, axes = plt.subplots(1, 2, figsize=A4)
    for dam, c in (("Tarbela", "#2a6fb0"), ("Mangla", "#d9822b")):
        d = st[st.variable == f"rule curve: {dam}"].set_index("month")["value"].reindex(ORDER)
        cap = st[st.variable == f"rule curve: {dam}"]["capacity_MCM"].iloc[0]
        axes[0].plot(range(12), d * 100, marker="o", ms=3, color=c, label=f"{dam} ({cap / 1000:.1f} km³)")
    _axis(axes[0], "Rule curves: average fill (% of capacity)", "%")
    axes[0].legend(fontsize=8)
    ev = st[(st.variable == "evaporation loss") & (st.rcp == ("no_climate" if rcp == "no_climate" else rcp))]
    ev = ev[ev.year == ev.year[ev.year <= year].max()] if (ev.year <= year).any() else ev
    for reg in groups:
        d = ev[ev.node.map(lambda n: region_of(n, groups)) == reg]
        if not d.empty:
            axes[1].plot(range(12), d.groupby("month")["value"].mean().reindex(ORDER) * 100, marker="o", ms=3,
                         label=reg)
    _axis(axes[1], "Reservoir evaporation (% of stored volume, mean of basins)", "%")
    axes[1].legend(fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf.save(fig)
    pdf.close()
    return path, tab
