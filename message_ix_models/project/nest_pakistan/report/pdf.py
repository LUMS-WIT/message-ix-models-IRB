"""PDF reports built from the tidy result tables of :mod:`.extract`.

Indicators named ``"Family: item"`` (e.g. ``"Withdrawal: River"``) are drawn
as one stacked chart per family; indicators without a colon are drawn as
lines. So a new indicator added to :mod:`.extract` appears in the reports
with no plotting code.
"""

import logging
import textwrap
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

log = logging.getLogger(__name__)

A4 = (11.69, 8.27)  # landscape
INK, MUTED, ACCENT = "#1b2732", "#74828f", "#1d6aaa"
PALETTE = ["#1d6aaa", "#4d7a26", "#b87a08", "#c0392b", "#6b4fa0", "#2a9d8f", "#8d6e63", "#e76f51",
           "#5c7c99", "#a3b18a", "#d4a373", "#7f7f7f"]

THEMES = {
    "water": ("Water", "Natural water, withdrawals by source and use, irrigation demand met, city demands, "
                       "unused water leaving the basins, and water shadow prices."),
    "energy-for-water": ("Energy for water", "Electricity used by water supply, irrigation, treatment and "
                                             "distribution, split by where it comes from: the national energy "
                                             "model, or the costed external supply for regions without one."),
    "water-for-energy": ("Water for energy", "Basin water sent to power plants, cooling water by cooling type, "
                                             "process water of fuel production, and water returned to rivers."),
    "energy": ("Energy system", "Electricity generation and capacity by fuel, useful energy demand and the "
                                "electricity price in the national energy model."),
    "land": ("Land and crops", "Crop area by irrigation method and by crop."),
    "investment": ("Investment", "Investment per year by sector (new capacity x investment cost) and new power "
                                 "capacity by fuel."),
    "costs": ("Operating costs", "Fixed O&M (capacity x fixed cost) and variable costs (activity x variable "
                                 "cost) by sector."),
    "emissions": ("Emissions", "Emissions reported by the model at each node."),
}

#: Indicator family drawn on the basin map of each theme.
BASIN_MAP = {"water": "Withdrawal", "energy-for-water": "Electricity for", "water-for-energy": "Basin water sent",
             "land": "Crop area", "investment": "Investment", "costs": "Fixed O&M"}


def family(indicator: str) -> tuple[str, str]:
    if ": " in indicator:
        a, b = indicator.split(": ", 1)
        return a, b
    return indicator, ""


class Pdf:
    """A landscape A4 PDF with title, text, table and figure pages."""

    def __init__(self, path, title: str):
        self.path, self.title, self.n = path, title, 0
        self.pdf = PdfPages(path)

    def close(self):
        d = self.pdf.infodict()
        d["Title"], d["Creator"] = self.title, "message_ix_models.project.nest_pakistan"
        self.pdf.close()
        log.info(f"wrote {self.path} ({self.n} pages)")

    def _footer(self, fig):
        self.n += 1
        fig.text(0.02, 0.015, self.title, fontsize=7, color=MUTED)
        fig.text(0.98, 0.015, str(self.n), fontsize=7, color=MUTED, ha="right")

    def save(self, fig):
        self._footer(fig)
        self.pdf.savefig(fig)
        plt.close(fig)

    def title_page(self, title, subtitle, lines):
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.78, title, fontsize=26, weight="bold", color=INK)
        fig.text(0.06, 0.72, subtitle, fontsize=13, color=MUTED)
        y = 0.62
        for line in lines:
            fig.text(0.06, y, line, fontsize=10.5, color=INK)
            y -= 0.035
        fig.text(0.06, 0.08, f"Generated {datetime.now():%Y-%m-%d %H:%M}", fontsize=8, color=MUTED)
        self.save(fig)

    def text_page(self, heading, paragraphs):
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.9, heading, fontsize=18, weight="bold", color=INK)
        y = 0.82
        for p in paragraphs:
            for line in textwrap.wrap(p, 120) or [""]:
                fig.text(0.06, y, line, fontsize=10, color=INK)
                y -= 0.028
            y -= 0.015
        self.save(fig)

    def table(self, heading, df: pd.DataFrame, note: str = "", rows_per_page: int = 24, col_width=None):
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:,.3g}" if abs(v) < 100 else f"{v:,.0f}")
        df = df.astype(str).apply(lambda s: s.str.slice(0, 70))
        pages = max(1, int(np.ceil(len(df) / rows_per_page)))
        for i in range(pages):
            chunk = df.iloc[i * rows_per_page:(i + 1) * rows_per_page]
            fig = plt.figure(figsize=A4)
            fig.text(0.04, 0.93, heading + (f" ({i + 1}/{pages})" if pages > 1 else ""), fontsize=15, weight="bold",
                     color=INK)
            if note:
                fig.text(0.04, 0.895, note, fontsize=9, color=MUTED)
            ax = fig.add_axes([0.04, 0.06, 0.92, 0.81])
            ax.axis("off")
            if len(chunk):
                t = ax.table(cellText=chunk.values, colLabels=list(chunk.columns), loc="upper left",
                             cellLoc="left", colLoc="left", colWidths=col_width)
                t.auto_set_font_size(False)
                t.set_fontsize(7.5)
                t.scale(1, 1.25)
                for (r, _), cell in t.get_celld().items():
                    cell.set_edgecolor("#d5dce3")
                    if r == 0:
                        cell.set_facecolor("#eef2f5")
                        cell.set_text_props(weight="bold")
                    txt = cell.get_text().get_text()
                    if txt in ("FAIL", "WARN", "PASS"):
                        cell.set_facecolor({"FAIL": "#fbe5e2", "WARN": "#fbf0d9", "PASS": "#e8f1df"}[txt])
            else:
                ax.text(0, 1, "(no rows)", fontsize=10, color=MUTED, va="top")
            self.save(fig)


# ------------------------------------------------------------------ charts
def _stacked(ax, wide: pd.DataFrame, title, unit):
    wide = wide.loc[:, wide.abs().sum() > 1e-9]
    if wide.empty:
        ax.set_visible(False)
        return
    order = wide.sum().sort_values(ascending=False).index
    wide = wide[order]
    bottom_pos = np.zeros(len(wide))
    bottom_neg = np.zeros(len(wide))
    x = np.arange(len(wide))
    for i, c in enumerate(wide.columns):
        v = wide[c].values
        b = np.where(v >= 0, bottom_pos, bottom_neg)
        ax.bar(x, v, bottom=b, color=PALETTE[i % len(PALETTE)], width=0.72, label=c)
        bottom_pos += np.where(v >= 0, v, 0)
        bottom_neg += np.where(v < 0, v, 0)
    ax.set_xticks(x, wide.index, fontsize=8)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_ylabel(unit, fontsize=8, color=MUTED)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#e5e9ee", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))


def _lines(ax, wide: pd.DataFrame, title, unit):
    for i, c in enumerate(wide.columns):
        ax.plot(wide.index, wide[c], marker="o", ms=3, color=PALETTE[i % len(PALETTE)], label=c)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_ylabel(unit, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=8)
    ax.grid(color="#e5e9ee", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))


def charts_for(df: pd.DataFrame, theme: str, where: str):
    """(title, unit, wide table, kind) for every indicator family of ``theme``
    in the rows ``df`` (one region or node)."""
    d = df[df["theme"] == theme]
    out = []
    d = d.assign(fam=d["indicator"].map(lambda s: family(s)[0]), item=d["indicator"].map(lambda s: family(s)[1]))
    for (fam, unit), g in d.groupby(["fam", "unit"], sort=False):
        if (g["item"] != "").any():
            wide = g.pivot_table(index="year", columns="item", values="value", aggfunc="sum").fillna(0.0)
            kind = "lines" if "price" in fam.lower() else "stacked"
            out.append((f"{fam} - {where}", unit, wide, kind))
    singles = d[d["item"] == ""]
    for unit, g in singles.groupby("unit"):
        wide = g.pivot_table(index="year", columns="indicator", values="value", aggfunc="sum").fillna(0.0)
        out.append((f"{', '.join(wide.columns[:3])}{' ...' if wide.shape[1] > 3 else ''} - {where}", unit, wide,
                    "lines"))
    return out


def chart_pages(pdf: Pdf, df: pd.DataFrame, theme: str, where: str, per_page: int = 2):
    charts = charts_for(df, theme, where)
    for i in range(0, len(charts), per_page):
        fig, axes = plt.subplots(per_page, 1, figsize=A4, squeeze=False)
        fig.subplots_adjust(left=0.07, right=0.72, top=0.92, bottom=0.07, hspace=0.45)
        for ax, (title, unit, wide, kind) in zip(axes[:, 0], charts[i:i + per_page]):
            (_stacked if kind == "stacked" else _lines)(ax, wide, title, unit)
        for ax in axes[len(charts[i:i + per_page]):, 0]:
            ax.set_visible(False)
        pdf.save(fig)


def region_page(pdf: Pdf, reg: pd.DataFrame, theme: str, year: int, regions: list[str]):
    """Each indicator family: one bar per region group in ``year``."""
    d = reg[(reg["theme"] == theme) & (reg["year"] == year) & reg["region"].isin(regions)]
    d = d[~d["indicator"].str.contains("price", case=False)]
    if d.empty:
        return
    d = d.assign(fam=d["indicator"].map(lambda s: family(s)[0]))
    fams = list(dict.fromkeys(d["fam"]))[:4]
    fig, axes = plt.subplots(2, 2, figsize=A4, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, hspace=0.55, wspace=0.3)
    fig.text(0.07, 0.94, f"{THEMES[theme][0]} by region, {year}", fontsize=15, weight="bold", color=INK)
    for ax, fam in zip(axes.flat, fams):
        g = d[d["fam"] == fam]
        wide = g.pivot_table(index="region", columns="indicator", values="value", aggfunc="sum").reindex(
            [r for r in regions if r in set(g["region"])]).fillna(0.0)
        wide.columns = [family(c)[1] or c for c in wide.columns]
        _stacked(ax, wide, fam, g["unit"].iloc[0])
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(fontsize=6, frameon=False, loc="upper right")
    for ax in list(axes.flat)[len(fams):]:
        ax.set_visible(False)
    pdf.save(fig)


def basin_map_page(pdf: Pdf, nodes: pd.DataFrame, theme: str):
    """Basin x year heat map of the theme's headline family (sum of items)."""
    fam = BASIN_MAP.get(theme)
    if not fam:
        return
    d = nodes[(nodes["theme"] == theme) & nodes["indicator"].str.startswith(fam + ":")
              & nodes["node"].astype(str).str.startswith("B")]
    if d.empty:
        return
    unit = d["unit"].iloc[0]
    wide = d.pivot_table(index="node", columns="year", values="value", aggfunc="sum").fillna(0.0)
    wide = wide.reindex(sorted(wide.index, key=lambda n: int("".join(c for c in n.split("|")[0] if c.isdigit()) or 0)))
    fig = plt.figure(figsize=A4)
    fig.text(0.05, 0.93, f"{THEMES[theme][0]}: {fam} by basin ({unit}, sum of all items)", fontsize=14,
             weight="bold", color=INK)
    ax = fig.add_axes([0.16, 0.08, 0.72, 0.8])
    im = ax.imshow(wide.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(wide)), wide.index, fontsize=7)
    ax.set_xticks(range(wide.shape[1]), wide.columns, fontsize=8)
    vmax = np.nanmax(wide.values) if wide.size else 0
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            v = wide.values[i, j]
            ax.text(j, i, f"{v:,.0f}" if abs(v) >= 10 else f"{v:.2g}", ha="center", va="center", fontsize=6,
                    color="white" if vmax and v > 0.6 * vmax else INK)
    fig.colorbar(im, ax=ax, fraction=0.03)
    pdf.save(fig)


# ---------------------------------------------------------------- reports
def headline_table(reg: pd.DataFrame, regions: list[str], years: list[int]) -> pd.DataFrame:
    """A few numbers that summarise a scenario, per region and year."""
    def total(prefix, theme=None, exact=False):
        d = reg if theme is None else reg[reg["theme"] == theme]
        d = d[d["indicator"] == prefix] if exact else d[d["indicator"].str.startswith(prefix)]
        return d.groupby(["region", "year"])["value"].sum()

    t = pd.DataFrame({
        "Withdrawn km3": total("Withdrawal: ") / 1000,
        "Fossil gw km3": total("Withdrawal: Fossil groundwater", exact=True) / 1000,
        "Irrig. unmet km3": total("Irrigation demand unmet", exact=True) / 1000,
        "Elec. for water GWa": total("Electricity for water from ", "energy-for-water"),
        "Water to power km3": total("Basin water sent", "water-for-energy") / 1000,
        "Invest. bn USD/yr": total("Investment: ") / 1000,
    }).fillna(0.0).reset_index()
    t = t[t["region"].isin(regions) & t["year"].isin(years)]
    t["region"] = pd.Categorical(t["region"], regions, ordered=True)
    return t.sort_values(["region", "year"]).round(2)


def scenario_report(path, nodes: pd.DataFrame, reg: pd.DataFrame, cfg: dict, meta: dict,
                    themes=None, levels=("regions", "basins"), balance: pd.DataFrame | None = None,
                    calibration: pd.DataFrame | None = None):
    """The full report of one solved scenario."""
    from .extract import region_order

    themes = [t for t in (themes or THEMES) if t in THEMES]
    regions = [r for r in region_order(cfg) if r in set(reg["region"])]
    home = regions[0] if regions else None
    years = cfg["report"]["years"]
    key_years = [y for y in (2030, 2050, 2070) if y in years] or years[-1:]
    pdf = Pdf(path, f"NEST {cfg['basin']['name']} - {meta['scenario']}")
    pdf.title_page(f"{cfg['basin']['name']}: {meta['scenario']}", f"{cfg['model']} v{meta.get('version')}",
                   [f"Objective function: {meta.get('objective', float('nan')):,.6g}",
                    f"Climate {cfg['nexus']['rcp']}, {cfg['nexus']['ssp']}, data reliability {cfg['nexus']['rel']}",
                    f"Years: {years[0]}-{years[-1]}", f"Themes: {', '.join(THEMES[t][0] for t in themes)}",
                    f"Regions: {', '.join(regions)}", "",
                    "Units: MCM = million m3 (1 km3 = 1,000 MCM); GWa = GW-year (1 GWa = 8.76 TWh);",
                    "costs in the model's currency unit (million USD)."])
    pdf.table("Headline results", headline_table(reg, regions, key_years),
              note="Sums over each region group's basins (and its energy node). Upper basins = all but the first group.")
    if balance is not None and not balance.empty:
        b = balance.assign(MCM=balance["MCM"].round(2))
        gap = balance.loc[balance["side"] == "in", "MCM"].sum() - balance.loc[balance["side"] == "out", "MCM"].sum()
        pdf.table(f"River water balance check, {key_years[0]} (gap {gap:,.3f} MCM)", b,
                  note="All river pools together. River links appear on both sides; a gap would mean water is "
                       "created or lost (double counting).")
    if calibration is not None and not calibration.empty:
        pdf.table("Calibration changes applied before solving", calibration)

    for theme in themes:
        name, text = THEMES[theme]
        if not (reg["theme"] == theme).any():
            continue
        pdf.text_page(name, [text])
        if "regions" in levels and home:
            chart_pages(pdf, reg[reg["region"] == home], theme, home)
            for y in key_years:
                region_page(pdf, reg, theme, y, regions)
            if len(regions) > 1:
                chart_pages(pdf, reg[reg["region"] == "Upper basins"], theme, "Upper basins")
        if "basins" in levels:
            basin_map_page(pdf, nodes, theme)
    pdf.close()
    return path


def node_report(path, nodes: pd.DataFrame, cfg: dict, meta: dict, node: str):
    """Every theme for one basin node."""
    d = nodes[nodes["node"] == node]
    pdf = Pdf(path, f"NEST {meta['scenario']} - {node}")
    pdf.title_page(node, f"{meta['scenario']} v{meta.get('version')}", [f"{len(d)} values"])
    for theme in THEMES:
        if (d["theme"] == theme).any():
            pdf.text_page(f"{THEMES[theme][0]} - {node}", [THEMES[theme][1]])
            chart_pages(pdf, d, theme, node)
    pdf.close()
    return path


def checks_report(path, checks, cfg: dict, scenario: str | None):
    from .. import checks as ch

    pdf = Pdf(path, f"NEST input checks - {scenario or cfg['basin']['code']}")
    s = ch.summary(checks)
    pdf.title_page("Input and assumption checks", f"{cfg['basin']['name']} - {scenario or 'data files only'}",
                   [f"{(s['status'] == st).sum()} {st}" for st in ("FAIL", "WARN", "PASS", "INFO")])
    pdf.table("Summary", s, col_width=[0.22, 0.06, 0.72])
    for c in checks:
        if not c.table.empty:
            pdf.table(f"{c.status}: {c.name}", c.table, note=c.finding)
    pdf.close()
    return path


def comparison_report(path, reg: pd.DataFrame, cfg: dict, scenarios: list[str], objectives: dict,
                      region: str | None = None, themes=None):
    """Several scenarios side by side: headline numbers, then every indicator
    family as lines (one per scenario) for one region."""
    from .extract import region_order

    regions = region_order(cfg)
    region = region or regions[0]
    years = cfg["report"]["years"]
    key_years = [y for y in (2030, 2050, 2070) if y in years]
    pdf = Pdf(path, f"NEST scenario comparison - {region}")
    pdf.title_page("Scenario comparison", f"{cfg['basin']['name']} - {region}",
                   [f"{s}: objective {objectives.get(s, float('nan')):,.6g}" for s in scenarios])
    rows = []
    for s in scenarios:
        h = headline_table(reg[reg["scenario"] == s], [r for r in regions if r in set(reg["region"])], key_years)
        rows.append(h[h["region"] == region].assign(scenario=s))
    if rows:
        t = pd.concat(rows).drop(columns="region")
        pdf.table(f"Headline results, {region}", t[["scenario"] + [c for c in t.columns if c != "scenario"]])
    d = reg[reg["region"] == region]
    for theme in themes or THEMES:
        sub = d[d["theme"] == theme]
        if sub.empty:
            continue
        pdf.text_page(f"{THEMES[theme][0]}: scenarios compared", [THEMES[theme][1],
                      "One line per scenario; each chart is the total of an indicator family."])
        fams = sub.assign(fam=sub["indicator"].map(lambda s: family(s)[0]))
        groups = list(fams.groupby(["fam", "unit"], sort=False))
        for i in range(0, len(groups), 4):
            fig, axes = plt.subplots(2, 2, figsize=A4, squeeze=False)
            fig.subplots_adjust(left=0.07, right=0.8, top=0.9, bottom=0.08, hspace=0.45, wspace=0.35)
            for ax, ((fam, unit), g) in zip(axes.flat, groups[i:i + 4]):
                agg = "mean" if "price" in fam.lower() else "sum"
                wide = g.pivot_table(index="year", columns="scenario", values="value", aggfunc=agg).reindex(
                    columns=scenarios)
                _lines(ax, wide, fam, unit)
            for ax in list(axes.flat)[len(groups[i:i + 4]):]:
                ax.set_visible(False)
            pdf.save(fig)
    pdf.close()
    return path
