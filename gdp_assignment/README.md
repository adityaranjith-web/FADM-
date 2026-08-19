# Business Statistics Assignment - GDP Growth Hypothesis Tests

`gdp_hypothesis_report.py` fetches 50 years (1975-2024) of real GDP growth for
the United States, Japan and India, runs the required hypothesis tests, and
writes `Aditya_Ranjith_DM281004_Business_Statistics.pdf`.

## Status: data fetch is blocked in this environment

The script is complete and verified end to end, but it could not be run against
real data here. This session's network egress proxy denies every statistics
host with HTTP 403 at CONNECT:

    api.worldbank.org      403 (World Bank, primary source)
    www.imf.org            403 (IMF WEO, fallback)
    stats.oecd.org         403
    sdmx.oecd.org          403
    fred.stlouisfed.org    403
    data.worldbank.org     403

Only PyPI and GitHub are reachable. No GDP figures were invented to work around
this; the repository deliberately contains no PDF or CSV produced from
placeholder data.

## Running it

Where the World Bank API is reachable:

    pip install requests pandas scipy matplotlib reportlab pypdf cffi
    python3 gdp_hypothesis_report.py

Or, with a cleaned CSV (columns: `country,year,growth_pct`) downloaded
elsewhere:

    python3 gdp_hypothesis_report.py --csv gdp_growth.csv

Either way the script prints the raw and post-null-drop row counts, every test
statistic and p-value, writes `gdp_growth.csv` and both charts, builds the PDF,
and finishes by printing the self-check list.

## What the report contains

1. Cover page (Aditya Ranjith, DM281004)
2. Objective and data description, with usable years per country stated
   honestly - series shorter than 50 years are flagged, never padded
3. Descriptive statistics table plus the line chart (5% reference line) and
   box plot, both at 150 dpi
4. Test 1 - one-sample t-test vs mu = 5, per country: hypotheses in words and
   symbols first, then alpha, manual working (mean, s, SE, hand-computed t),
   the scipy statistic, df, exact p, 95% CI, decision, an explicit
   Significant / Not significant verdict, and a plain-English reading
5. Test 2 - USA vs Japan: Levene's test first, then pooled or Welch's as the
   Levene result dictates (stated explicitly), with Cohen's d
6. Assumption checks - Shapiro-Wilk per series, non-parametric cross-checks
   (Wilcoxon, Mann-Whitney U), and a note on CLT robustness at n ~ 50
7. Conclusion under 200 words with one honest limitation
8. Appendices: cleaned data table and the full source code

Every figure in the PDF is rendered from a variable computed at run time.
Nothing is hardcoded. The document contains no test of proportions.

## Verification performed

The statistics, charting, PDF assembly and self-check stages were exercised on
a synthetic input file to prove the pipeline runs clean: a 22-page PDF built
and all eleven self-checks passed. That synthetic run produced no committed
artefacts. Only the data fetch remains unexercised, because the hosts are
blocked.
