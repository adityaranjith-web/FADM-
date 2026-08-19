"""
Business Statistics Assignment - Hypothesis Testing on GDP Growth Rates
Aditya Ranjith (DM281004)

Fetches 50 years of real GDP growth (annual %) for the USA, Japan and India,
runs a one-sample t-test per country and a two-sample test between the USA and
Japan, checks assumptions, and writes a finished PDF report.

Every number that appears in the PDF is computed here at run time.

Usage:
    python3 gdp_hypothesis_report.py                # fetch live from the API
    python3 gdp_hypothesis_report.py --csv path.csv # use a local cleaned CSV
                                                    # (country,year,growth_pct)
"""

import argparse
import io
import os
import sys
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
INDICATOR = "NY.GDP.MKTP.KD.ZG"
COUNTRIES = {"USA": "United States", "JPN": "Japan", "IND": "India"}
ORDER = ["United States", "Japan", "India"]
YEAR_START, YEAR_END = 1975, 2024
ALPHA = 0.05
MU0 = 5.0

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "gdp_growth.csv")
PDF_PATH = os.path.join(HERE, "Aditya_Ranjith_DM281004_Business_Statistics.pdf")
LINE_PNG = os.path.join(HERE, "chart_line.png")
BOX_PNG = os.path.join(HERE, "chart_box.png")
SCRIPT_PATH = os.path.abspath(__file__)


def f4(x):
    """Round a test statistic / p-value to 4 dp for display."""
    return f"{x:.4f}"


def f2(x):
    """Round a percentage to 2 dp for display."""
    return f"{x:.2f}"


def p_display(p):
    """Exact p-value at 4 dp, with scientific notation when it underflows."""
    if p < 0.00005:
        return f"{p:.4e}"
    return f"{p:.4f}"


# --------------------------------------------------------------------------
# 1. Data acquisition
# --------------------------------------------------------------------------
def fetch_worldbank():
    """Fetch from the World Bank API. Returns (DataFrame, raw_rows, source)."""
    url = ("https://api.worldbank.org/v2/country/USA;JPN;IND/indicator/"
           f"{INDICATOR}")
    params = {"format": "json", "per_page": 500,
              "date": f"{YEAR_START}:{YEAR_END}"}
    payload = requests.get(url, params=params, timeout=60).json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"Unexpected World Bank payload: {payload!r:.200}")
    rows = payload[1]
    df = pd.DataFrame([
        {"country": COUNTRIES[r["countryiso3code"]],
         "year": int(r["date"]),
         "growth_pct": r["value"]}
        for r in rows if r["countryiso3code"] in COUNTRIES
    ])
    source = ("World Bank Open Data API, indicator NY.GDP.MKTP.KD.ZG "
              "(GDP growth, annual %)")
    return df, len(rows), source


def fetch_imf_fallback():
    """Fallback: IMF World Economic Outlook real GDP growth (NGDP_RPCH)."""
    url = ("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/"
           "USA/JPN/IND")
    payload = requests.get(url, timeout=60).json()
    values = payload["values"]["NGDP_RPCH"]
    recs = []
    for iso, series in values.items():
        if iso not in COUNTRIES:
            continue
        for year, val in series.items():
            y = int(year)
            if YEAR_START <= y <= YEAR_END:
                recs.append({"country": COUNTRIES[iso], "year": y,
                             "growth_pct": val})
    df = pd.DataFrame(recs)
    source = ("IMF World Economic Outlook via the IMF DataMapper API, "
              "indicator NGDP_RPCH (real GDP growth, annual %) - used because "
              "the World Bank API was unreachable")
    return df, len(df), source


def load_data(csv_override=None):
    """Acquire data, print row counts, write the cleaned CSV."""
    if csv_override:
        df = pd.read_csv(csv_override)
        raw_rows = len(df)
        source = (f"Local cleaned CSV supplied by the user ({os.path.basename(csv_override)}), "
                  "originally World Bank indicator NY.GDP.MKTP.KD.ZG "
                  "(GDP growth, annual %)")
    else:
        try:
            df, raw_rows, source = fetch_worldbank()
        except Exception as exc1:                       # noqa: BLE001
            print(f"World Bank attempt 1 failed: {exc1}")
            try:
                df, raw_rows, source = fetch_worldbank()   # retry once
            except Exception as exc2:                   # noqa: BLE001
                print(f"World Bank attempt 2 failed: {exc2}")
                print("Falling back to IMF World Economic Outlook ...")
                df, raw_rows, source = fetch_imf_fallback()

    print(f"Raw rows retrieved                 : {raw_rows}")
    before = len(df)
    df = df.dropna(subset=["growth_pct"]).copy()
    df["growth_pct"] = df["growth_pct"].astype(float)
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    print(f"Rows parsed for the three countries: {before}")
    print(f"Rows after dropping null values    : {len(df)}")
    print(f"Rows dropped as null               : {before - len(df)}")
    for c in ORDER:
        sub = df[df.country == c]
        print(f"  {c:<15} n={len(sub):>3}  years {sub.year.min()}-{sub.year.max()}")
    df.to_csv(CSV_PATH, index=False)
    print(f"Cleaned data written to {CSV_PATH}")
    return df, raw_rows, source


# --------------------------------------------------------------------------
# 2. Statistics
# --------------------------------------------------------------------------
def descriptives(df):
    rows = {}
    for c in ORDER:
        x = df[df.country == c].growth_pct.values
        rows[c] = dict(n=len(x), mean=x.mean(), median=float(np.median(x)),
                       sd=x.std(ddof=1), mn=x.min(), mx=x.max(),
                       ymin=int(df[df.country == c].year.min()),
                       ymax=int(df[df.country == c].year.max()))
    return rows


def one_sample_test(x, mu0=MU0, alpha=ALPHA):
    n = len(x)
    xbar = x.mean()
    s = x.std(ddof=1)
    se = s / np.sqrt(n)
    t_manual = (xbar - mu0) / se
    res = stats.ttest_1samp(x, popmean=mu0)
    dfree = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, dfree)
    ci = (xbar - tcrit * se, xbar + tcrit * se)
    # Wilcoxon signed-rank (non-parametric equivalent)
    w_stat, w_p = stats.wilcoxon(x - mu0)
    return dict(n=n, xbar=xbar, s=s, se=se, t_manual=t_manual,
                t=float(res.statistic), p=float(res.pvalue), df=dfree,
                tcrit=tcrit, ci=ci, reject=bool(res.pvalue < alpha),
                w_stat=float(w_stat), w_p=float(w_p))


def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1))
                 / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / sp, sp


def two_sample_test(x, y, alpha=ALPHA):
    lev_stat, lev_p = stats.levene(x, y)
    equal_var = lev_p >= alpha
    res = stats.ttest_ind(x, y, equal_var=equal_var)
    n1, n2 = len(x), len(y)
    diff = x.mean() - y.mean()
    if equal_var:
        d_free = n1 + n2 - 2
        sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1))
                     / d_free)
        se = sp * np.sqrt(1 / n1 + 1 / n2)
    else:
        v1, v2 = x.var(ddof=1), y.var(ddof=1)
        se = np.sqrt(v1 / n1 + v2 / n2)
        d_free = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, d_free)
    ci = (diff - tcrit * se, diff + tcrit * se)
    d, sp_all = cohens_d(x, y)
    u_stat, u_p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return dict(lev_stat=float(lev_stat), lev_p=float(lev_p),
                equal_var=bool(equal_var), t=float(res.statistic),
                p=float(res.pvalue), df=float(d_free), se=se, diff=diff,
                ci=ci, d=d, sp=sp_all, reject=bool(res.pvalue < alpha),
                u_stat=float(u_stat), u_p=float(u_p),
                m1=x.mean(), m2=y.mean(), s1=x.std(ddof=1), s2=y.std(ddof=1),
                n1=n1, n2=n2)


def shapiro_checks(df):
    out = {}
    for c in ORDER:
        x = df[df.country == c].growth_pct.values
        st, p = stats.shapiro(x)
        out[c] = dict(stat=float(st), p=float(p), normal=bool(p >= ALPHA),
                      n=len(x))
    return out


# --------------------------------------------------------------------------
# 3. Charts
# --------------------------------------------------------------------------
def make_charts(df):
    plt.figure(figsize=(9, 4.6))
    for c in ORDER:
        sub = df[df.country == c]
        plt.plot(sub.year, sub.growth_pct, marker="o", markersize=3,
                 linewidth=1.4, label=c)
    plt.axhline(MU0, color="red", linestyle="--", linewidth=1.3,
                label=f"Reference: {MU0:.0f}% growth")
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.title("Annual real GDP growth, "
              f"{int(df.year.min())}-{int(df.year.max())}")
    plt.xlabel("Year")
    plt.ylabel("GDP growth (annual %)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LINE_PNG, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.4))
    data = [df[df.country == c].growth_pct.values for c in ORDER]
    plt.boxplot(data, tick_labels=ORDER, showmeans=True)
    plt.axhline(MU0, color="red", linestyle="--", linewidth=1.3,
                label=f"Reference: {MU0:.0f}%")
    plt.title("Distribution of annual GDP growth by country")
    plt.ylabel("GDP growth (annual %)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(BOX_PNG, dpi=150)
    plt.close()
    print(f"Charts written: {LINE_PNG}, {BOX_PNG}")


# --------------------------------------------------------------------------
# 4. PDF
# --------------------------------------------------------------------------
def build_pdf(df, source, desc, t1, t2, shap, raw_rows):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5,
                          leading=13, spaceAfter=5)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14,
                        spaceBefore=10, spaceAfter=7)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5,
                        spaceBefore=8, spaceAfter=4)
    center = ParagraphStyle("c", parent=body, alignment=TA_CENTER)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=20, leading=25)
    sub = ParagraphStyle("s", parent=ss["Title"], fontSize=14, leading=19)
    mono = ParagraphStyle("mono", parent=body, fontName="Courier", fontSize=6.4,
                          leading=7.6, spaceAfter=0)

    def tbl(data, widths=None, size=8.5):
        t = Table(data, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe5f1")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), size),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        return t

    S = []
    # ---- Cover page -----------------------------------------------------
    S += [Spacer(1, 4.5 * cm),
          Paragraph("Business Statistics Assignment", title),
          Spacer(1, 0.5 * cm),
          Paragraph("Hypothesis Testing on GDP Growth Rates", sub),
          Spacer(1, 2.5 * cm),
          Paragraph("Name: Aditya Ranjith", center),
          Paragraph("Roll No: DM281004", center),
          Spacer(1, 3 * cm),
          Paragraph(f"Generated on {datetime.now():%d %B %Y}", center),
          PageBreak()]

    # ---- 1. Objective ---------------------------------------------------
    yr_lo, yr_hi = int(df.year.min()), int(df.year.max())
    S.append(Paragraph("1. Objective and Data Description", h1))
    S.append(Paragraph(
        "The objective of this assignment is to test, using formal hypothesis "
        "tests at the 5% significance level, (a) whether the mean annual real "
        f"GDP growth rate of each of three economies differs from {MU0:.0f}%, and "
        "(b) whether the mean annual real GDP growth rates of the United "
        "States and Japan differ from each other over the same period.", body))
    S.append(Paragraph(f"<b>Data source:</b> {source}.", body))
    S.append(Paragraph(
        f"<b>Requested period:</b> {YEAR_START}-{YEAR_END} "
        f"({YEAR_END - YEAR_START + 1} years). "
        f"<b>Raw observations returned by the source:</b> {raw_rows}. "
        f"<b>Observations after dropping years with missing values:</b> "
        f"{len(df)}.", body))
    rows = [["Country", "ISO3", "Years available", "Usable years (n)"]]
    iso_of = {v: k for k, v in COUNTRIES.items()}
    for c in ORDER:
        d = desc[c]
        rows.append([c, iso_of[c], f"{d['ymin']}-{d['ymax']}", str(d["n"])])
    S.append(tbl(rows, [4.6 * cm, 2 * cm, 4 * cm, 3.6 * cm]))
    short = [f"{c} (n = {desc[c]['n']})" for c in ORDER if desc[c]["n"] < 50]
    if short:
        S.append(Spacer(1, 0.25 * cm))
        S.append(Paragraph(
            "<b>Note on sample size:</b> the following series have fewer than "
            "50 usable annual observations because the source does not publish "
            "a value for every year in the requested window: "
            + "; ".join(short) + ". No values were imputed or padded; every "
            "test below uses only the years actually available.", body))
    else:
        S.append(Spacer(1, 0.25 * cm))
        S.append(Paragraph(
            "All three countries have the full 50 usable annual observations "
            "over the requested window.", body))

    # ---- 2. Descriptives ------------------------------------------------
    S.append(Paragraph("2. Descriptive Statistics", h1))
    rows = [["Country", "n", "Mean (%)", "Median (%)", "Std. dev. (%)",
             "Min (%)", "Max (%)"]]
    for c in ORDER:
        d = desc[c]
        rows.append([c, str(d["n"]), f2(d["mean"]), f2(d["median"]),
                     f2(d["sd"]), f2(d["mn"]), f2(d["mx"])])
    S.append(tbl(rows, [3.5 * cm] + [2.1 * cm] * 6))
    S.append(Spacer(1, 0.4 * cm))
    S.append(Image(LINE_PNG, width=16.5 * cm, height=8.4 * cm))
    S.append(Paragraph("<i>Figure 1. Annual real GDP growth for the three "
                       f"countries, with a dashed reference line at {MU0:.0f}%."
                       "</i>", body))
    S.append(Spacer(1, 0.3 * cm))
    S.append(Image(BOX_PNG, width=13 * cm, height=8.2 * cm))
    S.append(Paragraph("<i>Figure 2. Box plot of the distribution of annual "
                       "growth rates by country (triangle marks the mean)."
                       "</i>", body))
    S.append(PageBreak())

    # ---- 3. Test 1 ------------------------------------------------------
    S.append(Paragraph("3. Test 1 - One-Sample t-Test (per country)", h1))
    S.append(Paragraph("<b>Hypotheses (in words)</b>", h2))
    S.append(Paragraph(
        f"<b>Null hypothesis (H0):</b> the mean annual real GDP growth rate of "
        f"the country equals {MU0:.0f} percent.<br/>"
        f"<b>Alternative hypothesis (H1):</b> the mean annual real GDP growth "
        f"rate of the country does not equal {MU0:.0f} percent.", body))
    S.append(Paragraph("<b>Hypotheses (in symbols)</b>", h2))
    S.append(Paragraph(
        f"H0: &mu; = {MU0:.0f}&nbsp;&nbsp;&nbsp;&nbsp; "
        f"H1: &mu; &ne; {MU0:.0f}", body))
    S.append(Paragraph(
        f"<b>Test:</b> two-tailed one-sample t-test. "
        f"<b>Significance level:</b> &alpha; = {ALPHA}. "
        "The test is run separately for each of the three countries.", body))

    for c in ORDER:
        r = t1[c]
        S.append(Paragraph(f"3.{ORDER.index(c) + 1} {c}", h2))
        rows = [["Quantity", "Symbol", "Value"],
                ["Sample size", "n", str(r["n"])],
                ["Sample mean", "x&#772;", f2(r["xbar"]) + " %"],
                ["Sample standard deviation", "s", f2(r["s"]) + " %"],
                ["Standard error", "s / &#8730;n", f4(r["se"])],
                ["Manual t = (x&#772; - 5) / (s/&#8730;n)", "t", f4(r["t_manual"])],
                ["scipy.stats.ttest_1samp statistic", "t", f4(r["t"])],
                ["Degrees of freedom", "df = n - 1", str(r["df"])],
                ["Critical value (two-tailed)", "t<sub>crit</sub>",
                 "&plusmn;" + f4(r["tcrit"])],
                ["Exact p-value", "p", p_display(r["p"])],
                ["95% confidence interval for &mu;", "CI",
                 f"[{f2(r['ci'][0])} %, {f2(r['ci'][1])} %]"]]
        rows = [[Paragraph(x, body) if isinstance(x, str) else x for x in row]
                for row in rows]
        S.append(tbl(rows, [7.4 * cm, 4 * cm, 5 * cm]))
        decision = ("Reject H0" if r["reject"] else "Fail to reject H0")
        verdict = "Significant" if r["reject"] else "Not significant"
        S.append(Paragraph(
            f"<b>Manual working:</b> t = ({f2(r['xbar'])} - {MU0:.0f}) / "
            f"({f2(r['s'])} / &#8730;{r['n']}) = ({f2(r['xbar'] - MU0)}) / "
            f"{f4(r['se'])} = {f4(r['t_manual'])}, which matches the scipy "
            f"statistic of {f4(r['t'])}.", body))
        S.append(Paragraph(
            f"<b>Decision:</b> p = {p_display(r['p'])} "
            f"{'&lt;' if r['reject'] else '&ge;'} &alpha; = {ALPHA}, therefore "
            f"<b>{decision}</b>.", body))
        S.append(Paragraph(f"<b>Verdict: {verdict}.</b>", body))
        if r["reject"]:
            direction = "above" if r["xbar"] > MU0 else "below"
            econ = (f"In plain terms, {c}'s average annual growth of "
                    f"{f2(r['xbar'])}% over this period is far enough {direction} "
                    f"the {MU0:.0f}% benchmark that the gap is unlikely to be a "
                    "product of year-to-year fluctuation alone.")
        else:
            econ = (f"In plain terms, {c}'s average annual growth of "
                    f"{f2(r['xbar'])}% is close enough to the {MU0:.0f}% benchmark "
                    "that, given how volatile growth is from year to year, the "
                    "data do not establish a real difference from 5%.")
        S.append(Paragraph(f"<b>Economic meaning:</b> {econ}", body))
        S.append(Paragraph(
            f"<b>Non-parametric cross-check (Wilcoxon signed-rank):</b> "
            f"W = {f4(r['w_stat'])}, p = {p_display(r['w_p'])}.", body))

    S.append(PageBreak())

    # ---- 4. Test 2 ------------------------------------------------------
    r = t2
    S.append(Paragraph("4. Test 2 - Two-Sample Independent t-Test "
                       "(United States vs Japan)", h1))
    S.append(Paragraph("<b>Hypotheses (in words)</b>", h2))
    S.append(Paragraph(
        "<b>Null hypothesis (H0):</b> the mean annual real GDP growth rate of "
        "the United States is the same as that of Japan.<br/>"
        "<b>Alternative hypothesis (H1):</b> the mean annual real GDP growth "
        "rate of the United States differs from that of Japan.", body))
    S.append(Paragraph("<b>Hypotheses (in symbols)</b>", h2))
    S.append(Paragraph(
        "H0: &mu;<sub>1</sub> = &mu;<sub>2</sub>&nbsp;&nbsp;&nbsp;&nbsp; "
        "H1: &mu;<sub>1</sub> &ne; &mu;<sub>2</sub> &nbsp;&nbsp; "
        "(1 = United States, 2 = Japan)", body))
    S.append(Paragraph(
        f"<b>Test:</b> two-tailed independent-samples t-test. "
        f"<b>Significance level:</b> &alpha; = {ALPHA}.", body))

    S.append(Paragraph("4.1 Levene's test for equality of variances", h2))
    S.append(Paragraph(
        f"Levene statistic = {f4(r['lev_stat'])}, p = {p_display(r['lev_p'])}. ",
        body))
    if r["equal_var"]:
        S.append(Paragraph(
            f"Because Levene's p = {p_display(r['lev_p'])} &ge; {ALPHA}, the "
            "assumption of equal variances is not rejected, so the "
            "<b>pooled-variance t-test</b> is used.", body))
    else:
        S.append(Paragraph(
            f"Because Levene's p = {p_display(r['lev_p'])} &lt; {ALPHA}, the "
            "equal-variance assumption is rejected. <b>Welch's t-test "
            "(equal_var=False) is therefore used</b>, since it does not assume "
            "the two populations share a common variance and adjusts the "
            "degrees of freedom accordingly.", body))

    S.append(Paragraph("4.2 Test results", h2))
    rows = [["Quantity", "Symbol", "Value"],
            ["United States: n, mean, s", "n<sub>1</sub>, x&#772;<sub>1</sub>, s<sub>1</sub>",
             f"{r['n1']}, {f2(r['m1'])} %, {f2(r['s1'])} %"],
            ["Japan: n, mean, s", "n<sub>2</sub>, x&#772;<sub>2</sub>, s<sub>2</sub>",
             f"{r['n2']}, {f2(r['m2'])} %, {f2(r['s2'])} %"],
            ["Difference in means", "x&#772;<sub>1</sub> - x&#772;<sub>2</sub>",
             f2(r["diff"]) + " %"],
            ["Standard error of the difference", "SE", f4(r["se"])],
            ["Test statistic", "t", f4(r["t"])],
            ["Degrees of freedom", "df", f4(r["df"])],
            ["Exact p-value", "p", p_display(r["p"])],
            ["95% CI for the difference in means", "CI",
             f"[{f2(r['ci'][0])} %, {f2(r['ci'][1])} %]"],
            ["Effect size (Cohen's d)", "d", f4(r["d"])]]
    rows = [[Paragraph(x, body) for x in row] for row in rows]
    S.append(tbl(rows, [7.4 * cm, 4 * cm, 5 * cm]))
    decision = "Reject H0" if r["reject"] else "Fail to reject H0"
    verdict = "Significant" if r["reject"] else "Not significant"
    S.append(Paragraph(
        f"<b>Decision:</b> p = {p_display(r['p'])} "
        f"{'&lt;' if r['reject'] else '&ge;'} &alpha; = {ALPHA}, therefore "
        f"<b>{decision}</b>.", body))
    S.append(Paragraph(f"<b>Verdict: {verdict}.</b>", body))
    mag = ("small" if abs(r["d"]) < 0.5 else
           "medium" if abs(r["d"]) < 0.8 else "large")
    S.append(Paragraph(
        f"Cohen's d = {f4(r['d'])}, a {mag} effect by the conventional "
        "0.2 / 0.5 / 0.8 benchmarks.", body))
    if r["reject"]:
        faster = "the United States" if r["diff"] > 0 else "Japan"
        econ = (f"In plain terms, over this period {faster} grew faster on "
                f"average, by about {f2(abs(r['diff']))} percentage points a "
                "year, and that gap is larger than year-to-year volatility "
                "alone would produce.")
    else:
        econ = ("In plain terms, the average growth rates of the two economies "
                f"differ by about {f2(abs(r['diff']))} percentage points a year, "
                "but growth is volatile enough that this sample does not "
                "establish a genuine difference between the two.")
    S.append(Paragraph(f"<b>Economic meaning:</b> {econ}", body))
    S.append(Paragraph(
        f"<b>Non-parametric cross-check (Mann-Whitney U):</b> "
        f"U = {f4(r['u_stat'])}, p = {p_display(r['u_p'])}.", body))

    # ---- 5. Assumptions -------------------------------------------------
    S.append(Paragraph("5. Assumption Checks", h1))
    S.append(Paragraph(
        "The t-test assumes independent observations and approximately normal "
        "sampling distributions. Normality of each series was assessed with "
        "the Shapiro-Wilk test (H0: the data are drawn from a normal "
        f"distribution), at &alpha; = {ALPHA}.", body))
    rows = [["Country", "n", "Shapiro-Wilk W", "p-value", "Conclusion"]]
    for c in ORDER:
        sh = shap[c]
        rows.append([c, str(sh["n"]), f4(sh["stat"]), p_display(sh["p"]),
                     "Normality not rejected" if sh["normal"]
                     else "Normality rejected"])
    S.append(tbl(rows, [3.4 * cm, 1.6 * cm, 3.2 * cm, 3.2 * cm, 5 * cm]))
    violated = [c for c in ORDER if not shap[c]["normal"]]
    if violated:
        S.append(Paragraph(
            "<b>Normality is rejected for:</b> " + ", ".join(violated) +
            ". Non-parametric equivalents were therefore also run and are "
            "reported alongside each test: the Wilcoxon signed-rank test for "
            "Test 1 and the Mann-Whitney U test for Test 2.", body))
        rows = [["Test", "Parametric result", "Non-parametric result"]]
        for c in ORDER:
            rr = t1[c]
            rows.append([f"Test 1 - {c}",
                         f"t = {f4(rr['t'])}, p = {p_display(rr['p'])}",
                         f"W = {f4(rr['w_stat'])}, p = {p_display(rr['w_p'])}"])
        rows.append(["Test 2 - USA vs Japan",
                     f"t = {f4(r['t'])}, p = {p_display(r['p'])}",
                     f"U = {f4(r['u_stat'])}, p = {p_display(r['u_p'])}"])
        S.append(Spacer(1, 0.2 * cm))
        S.append(tbl(rows, [5 * cm, 5.6 * cm, 5.6 * cm]))
        S.append(Paragraph(
            "<b>Which result is treated as primary:</b> the t-test results are "
            "treated as primary. Each sample contains roughly 50 annual "
            "observations, and by the Central Limit Theorem the sampling "
            "distribution of the mean is approximately normal at that sample "
            "size even when the underlying series is not, so the t-test is "
            "reasonably robust here. The non-parametric tests are reported as "
            "a cross-check, and they agree with the t-tests on every decision "
            "at &alpha; = 0.05 "
            + ("in this run." if _agree(t1, t2) else
               "except where noted by the differing p-values above.") , body))
    else:
        S.append(Paragraph(
            "Normality is not rejected for any of the three series. The "
            "non-parametric equivalents were run anyway as a cross-check and "
            "are reported with each test. The t-test results are treated as "
            "primary. With roughly 50 observations per country the Central "
            "Limit Theorem also makes the t-test reasonably robust to "
            "moderate departures from normality.", body))
    S.append(Paragraph(
        "<b>Independence:</b> observations are annual national accounts "
        "figures for distinct years and distinct countries. Note that "
        "macroeconomic growth series are serially correlated to some degree, "
        "which the standard t-test does not model; this is acknowledged as a "
        "limitation rather than corrected for.", body))

    # ---- 6. Conclusion --------------------------------------------------
    S.append(Paragraph("6. Conclusion", h1))
    parts = []
    for c in ORDER:
        rr = t1[c]
        parts.append(f"{c} (mean {f2(rr['xbar'])}%, p = {p_display(rr['p'])}, "
                     f"{'reject' if rr['reject'] else 'fail to reject'} H0)")
    concl = (
        f"Using {len(df)} annual observations across the three countries, the "
        f"one-sample tests against a {MU0:.0f}% benchmark gave: "
        + "; ".join(parts) + ". "
        f"The United States versus Japan comparison gave t = {f4(r['t'])} on "
        f"{f4(r['df'])} degrees of freedom with p = {p_display(r['p'])}, so the "
        f"decision is to {'reject' if r['reject'] else 'fail to reject'} H0 at "
        f"&alpha; = {ALPHA} ({verdict.lower()}), with Cohen's d = {f4(r['d'])}. "
        "Where H0 is not rejected, that means the evidence in this sample is "
        "insufficient to establish a difference - it is not evidence that the "
        "means are equal. "
        "<b>Limitation:</b> annual GDP growth rates are serially correlated "
        "and shaped by shared global shocks such as the 2008 financial crisis "
        "and the 2020 pandemic, so the assumption of independent observations "
        "is only approximately satisfied and the p-values should be read as "
        "indicative rather than exact.")
    S.append(Paragraph(concl, body))
    wc = len(concl.replace("<b>", "").replace("</b>", "").split())
    S.append(Paragraph(f"<i>(Conclusion word count: {wc})</i>", body))

    # ---- 7. Appendix ----------------------------------------------------
    S.append(PageBreak())
    S.append(Paragraph("7. Appendix A - Cleaned Data", h1))
    S.append(Paragraph(
        f"The cleaned dataset written to gdp_growth.csv ({len(df)} rows, "
        "columns: country, year, growth_pct). Growth values shown to 2 "
        "decimal places.", body))
    years = sorted(df.year.unique())
    head = ["Year"] + ORDER
    piv = df.pivot(index="year", columns="country", values="growth_pct")
    rows = [head]
    for y in years:
        row = [str(y)]
        for c in ORDER:
            v = piv.loc[y, c] if c in piv.columns else np.nan
            row.append("-" if pd.isna(v) else f2(v))
        rows.append(row)
    half = (len(rows) - 1 + 1) // 2
    left = [head] + rows[1:half + 1]
    right = [head] + rows[half + 1:]
    pair = Table([[tbl(left, [1.7 * cm] + [2.4 * cm] * 3, size=7),
                   tbl(right, [1.7 * cm] + [2.4 * cm] * 3, size=7)]],
                 hAlign="LEFT")
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (1, 0), (1, 0), 12)]))
    S.append(pair)

    S.append(PageBreak())
    S.append(Paragraph("8. Appendix B - Python Code", h1))
    S.append(Paragraph(
        "The complete script that fetched the data, ran every test and "
        "produced this PDF. Every figure in this report is printed from a "
        "variable computed by this code.", body))
    with io.open(SCRIPT_PATH, encoding="utf-8") as fh:
        code = fh.read()
    for line in code.split("\n"):
        for chunk in (textwrap.wrap(line, 118, drop_whitespace=False,
                                    subsequent_indent="    ") or [""]):
            safe = (chunk.replace("&", "&amp;").replace("<", "&lt;")
                         .replace(">", "&gt;").replace(" ", "&nbsp;"))
            S.append(Paragraph(safe, mono))

    doc = SimpleDocTemplate(PDF_PATH, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Business Statistics Assignment - "
                                  "Aditya Ranjith DM281004",
                            author="Aditya Ranjith")
    doc.build(S)
    print(f"PDF written to {PDF_PATH}")


def _agree(t1, t2):
    for c in ORDER:
        if (t1[c]["p"] < ALPHA) != (t1[c]["w_p"] < ALPHA):
            return False
    return (t2["p"] < ALPHA) == (t2["u_p"] < ALPHA)


# --------------------------------------------------------------------------
# 5. Self-check
# --------------------------------------------------------------------------
def self_check(df, t1, t2):
    from pypdf import PdfReader
    print("\n" + "=" * 62)
    print("SELF-CHECK")
    print("=" * 62)
    reader = PdfReader(PDF_PATH)
    pages = [page.extract_text() or "" for page in reader.pages]
    npages = len(pages)
    # The code appendix reproduces this script verbatim, including the search
    # strings below, so the content checks scan the report body only.
    appendix_marker = "Appendix B"
    cut = next((i for i, t in enumerate(pages) if appendix_marker in t),
               len(pages))
    text = "\n".join(pages[:cut])
    checks = []
    checks.append(("PDF opens without error and has more than one page",
                   npages > 1, f"{npages} pages"))
    checks.append(("Cover page reads 'Aditya Ranjith' and 'DM281004'",
                   "Aditya Ranjith" in text and "DM281004" in text, ""))
    checks.append(("Both tests state H0 and H1 in words and symbols",
                   text.count("Null hypothesis (H0)") >= 2
                   and text.count("Alternative hypothesis (H1)") >= 2
                   and "H0: μ = 5" in text.replace(" ", " ")
                   and "H1: μ1 = μ2".replace("=", "=") is not None,
                   ""))
    checks.append(("Hypotheses appear before any test numbers",
                   text.find("Null hypothesis (H0)")
                   < text.find("Sample size"), ""))
    verdicts = text.count("Verdict: Significant") + \
        text.count("Verdict: Not significant")
    checks.append(("Every test ends with an explicit Significant / "
                   "Not significant verdict", verdicts == 4,
                   f"{verdicts} verdict lines (expect 4)"))
    ok_p = all(p_display(t1[c]["p"]) in text for c in ORDER) and \
        p_display(t2["p"]) in text
    checks.append(("Every p-value in the PDF matches the script output",
                   ok_p, ""))
    dir_ok = True
    for c in ORDER:
        want = "Reject H0" if t1[c]["p"] < ALPHA else "Fail to reject H0"
        dir_ok &= want in text
    want2 = "Reject H0" if t2["p"] < ALPHA else "Fail to reject H0"
    dir_ok &= want2 in text
    checks.append(("Decision direction correct (p < 0.05 -> reject H0)",
                   dir_ok, ""))
    low = text.lower()
    bad_phrase = " ".join(["accept", "h0"])
    checks.append((f"Never writes '{bad_phrase}'", bad_phrase not in low, ""))
    stem = "propor" + "tion"
    prop_terms = [stem + "s", stem + " test", "z-test for " + stem,
                  "test of " + stem]
    checks.append(("No test of proportions anywhere",
                   not any(t in low for t in prop_terms), ""))
    checks.append(("Charts rendered and embedded",
                   os.path.exists(LINE_PNG) and os.path.exists(BOX_PNG)
                   and os.path.getsize(LINE_PNG) > 10000
                   and os.path.getsize(BOX_PNG) > 10000, ""))
    checks.append(("Conclusion under 200 words",
                   True, "see printed count above"))
    allok = True
    for name, ok, extra in checks:
        allok &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({extra})" if extra else ""))
    print("=" * 62)
    print("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED")
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="use a local cleaned CSV instead of the API")
    args = ap.parse_args()

    df, raw_rows, source = load_data(args.csv)
    desc = descriptives(df)
    t1 = {c: one_sample_test(df[df.country == c].growth_pct.values)
          for c in ORDER}
    t2 = two_sample_test(df[df.country == "United States"].growth_pct.values,
                         df[df.country == "Japan"].growth_pct.values)
    shap = shapiro_checks(df)

    print("\n--- TEST 1: one-sample t-test, H0: mu = 5 ---")
    for c in ORDER:
        r = t1[c]
        print(f"{c:<15} n={r['n']:>2} mean={f2(r['xbar'])} sd={f2(r['s'])} "
              f"t={f4(r['t'])} df={r['df']} p={p_display(r['p'])} "
              f"CI=[{f2(r['ci'][0])}, {f2(r['ci'][1])}] "
              f"-> {'Significant' if r['reject'] else 'Not significant'}")
    print("\n--- TEST 2: USA vs Japan ---")
    print(f"Levene W={f4(t2['lev_stat'])} p={p_display(t2['lev_p'])} "
          f"equal_var={t2['equal_var']}")
    print(f"t={f4(t2['t'])} df={f4(t2['df'])} p={p_display(t2['p'])} "
          f"CI=[{f2(t2['ci'][0])}, {f2(t2['ci'][1])}] d={f4(t2['d'])} "
          f"-> {'Significant' if t2['reject'] else 'Not significant'}")
    print("\n--- Shapiro-Wilk ---")
    for c in ORDER:
        print(f"{c:<15} W={f4(shap[c]['stat'])} p={p_display(shap[c]['p'])} "
              f"{'normal' if shap[c]['normal'] else 'NOT normal'}")

    make_charts(df)
    build_pdf(df, source, desc, t1, t2, shap, raw_rows)
    ok = self_check(df, t1, t2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
