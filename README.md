# QuantLab Factor Library

Factor research toolkit that reads cleaned Parquet outputs from the data pipeline (`../data/data-processed/*.parquet`) and writes factor signals/analytics to `../data/factors/`. Includes templates, loaders, transforms, analytics, and an example notebook.

## Quickstart
- Configure paths in `config/config.json` if needed (`data_root`, `final_dir`, `factors_dir`); defaults point to `../data`.
- Factor cleaning defaults can also be tweaked in `config/config.json` under `factor_defaults` (winsor_limits, min_coverage, fill_method, neutralize_method); per-factor calls can still override.
- Create env: `conda env create -f quantlab_env/environment.yml` (includes numpy, pandas, pyarrow, scipy, etc.).
- Run default factors:  
  `python -m quantlab_factor_library.run_factors`
- Example : [`notebooks/factor_parallel_demo.ipynb`](notebooks/factor_parallel_demo.ipynb)
- Cleaning defaults and per-factor overrides can be tweaked in `config/config.json` (`factor_defaults` and `factor_overrides`, including forward_fill flags for fundamentals); per-factor calls can still override in code.

## Modular run steps
You no longer need to run end-to-end every time:
- `compute_factors(parallel=False, max_workers=4)`: compute/clean factors, save factor files and LS PnL; returns factors, ls_returns, ff, fwd_returns.
- `run_analytics_only(factors, fwd_returns, ff=None)`: IC/IR, LS stats, FF regression; writes diagnostics/registry.
- `compute_correlations_only(factors, ls_returns=None, ff=None)`: factor cross-corr and LS vs FF correlation.
- `run_time_effects(factors, fwd_returns, window=252, step=21)`: rolling IC/IC IR over time.
Use the notebooks to see the sequence; re-run analytics/correlations/rolling without recomputing factors.

## What’s inside
| Path | Purpose |
| --- | --- |
| `quantlab_factor_library/paths.py` | Resolve repo/data roots (configurable via `config/config.json`). |
| `quantlab_factor_library/data_loader.py` | Load long-format parquet, pivot to wide price/sector, compute forward returns; load FF factors. |
| `quantlab_factor_library/base.py` | `FactorBase` enforcing `compute_raw_factor` + `post_process`; shared cleaning via `compute`. |
| `quantlab_factor_library/factors/` | Parameterized starters: Momentum, Volatility, MeanReversion, DollarVolume. |
| `quantlab_factor_library/factor_definitions.py` | Single place to declare the default factor set; `run_factors` and demos import from here. |
| `quantlab_factor_library/transforms.py` | Coverage filter, winsorize, fill (median/sector-median), neutralize (sector/global), z-score, drop-all-NaN; `clean_factor` helper. |
| `quantlab_factor_library/analytics.py` | IC (Spearman), autocorr, decile monotonicity, LS diagnostic (Sharpe/max DD/mean/std), FF regression (alpha/betas + t-stats/p-values), factor correlation, diagnostics/registry writers. |
| `quantlab_factor_library/run_factors.py` | Runs default factors, saves outputs, updates analytics registry, writes correlations/FF time series; optional `parallel=True` (ThreadPool via `concurrent.futures`) to fan out per-factor computations. |
| `notebooks/factor_demo.ipynb` | End-to-end demo (load → compute → transparent pipeline → analytics → correlation → save factors/diagnostics). |
| `notebooks/factor_parallel_demo.ipynb` | Same as above with optional parallel run snippet. |
| `config/config.json` | Optional path overrides. |
| FF loader | `DataLoader.load_ff_factors()` reads `data/data-processed/FAMA_FRENCH_FACTORS.parquet` (mktrf, smb, hml, rmw, cma, rf, umd). |
| Note | FF regression uses lightweight OLS (numpy + scipy for p-values) to keep the pipeline lean. |

## Default factors (methodology)
- Momentum (12m, skip 1m): past 12m return excluding the most recent month, shifted one day.
- Volatility (60d): rolling std of daily returns, shifted one day.
- Downside Volatility (60d): rolling std of negative returns, shifted one day.
- Mean Reversion (5d): negative 5-day return (short-term reversal), shifted one day.
- 52w High Proximity: price / 252d rolling max − 1, shifted one day.
- Dollar Volume (20d): rolling mean of price × volume (liquidity), shifted one day.
- Amihud Illiquidity (20d): rolling mean of |ret| / dollar volume, shifted one day.
- Residual Volatility (252d): rolling std of residuals after regressing returns on market (mktrf), shifted one day.
- Size: log market cap = price × quarterly shares outstanding (no annual fallback), shifted one day.
- Earnings Yield: trailing net income (TTM) / market cap using quarterly fundamentals and price, forward-filled between reports, shifted one day.
- Profitability: Return on Equity from annual fundamentals (net income / shareholder equity), forward-filled between reports, shifted one day.
- Dividend Yield: trailing 12-month dividends from dividend history divided by price, shifted one day.
- Beta (252d): rolling beta vs FF mktrf, shifted one day.
- Downside Beta (252d): rolling beta vs FF mktrf using only down-market days, shifted one day.
- Book-to-Price: book value per share (annual equity / shares) divided by price, shifted one day.
- Cashflow Yield: annual operating cashflow divided by market cap (price × annual shares), forward-fill between reports, shifted one day.
- Free Cashflow Yield: (annual operating cashflow − capex) divided by market cap (price × annual shares), forward-fill between reports, shifted one day.
- Accruals: (Net Income − Operating Cash Flow) / Total Assets (annual), forward-fill between reports, shifted one day.
- ROA: netIncome / totalAssets (prefers annual fundamentals), shifted one day.
- Leverage: totalLiabilities / totalAssets (prefers annual fundamentals), shifted one day.
- Sales Growth: y/y growth in totalRevenue (prefers annual fundamentals), shifted one day.
- Asset Growth: y/y growth in totalAssets (prefers annual fundamentals), shifted one day.
- R&D Intensity: R&D / totalRevenue (annual), forward-fill between reports, shifted one day.
- Net Issuance: percent change in shares outstanding from annual balance sheet, forward-fill between reports, shifted one day.
- Return Skewness: rolling skew of returns, shifted one day.
- Return Kurtosis: rolling kurtosis of returns, shifted one day.
- Dividend Growth: y/y dividend growth (annual sums), forward-fill between reports, shifted one day.
- Analyst Revision (30d): EPS estimate revisions up minus down (trailing 30d), pivoted by date, shifted one day.
- Earnings Surprise: surprise percentage from earnings releases, prefers quarterly dataset (filters `period_type=quarterly` if present), pivoted by report date, shifted one day; uses relaxed coverage/no fill for sparse event data.
- Turnover: volume / shares outstanding (static shares), shifted one day.

## Outputs
- Factors: `../data/factors/factor_<name>.parquet` (long format: Date, Ticker, Value).
- Registry: `../data/factors/factor_analytics_summary.parquet` (mean IC, IC t-stat, IC IR, mean autocorr, decile spread, LS stats, FF alpha/betas).
- Diagnostics: `../data/factors/factor_step_diagnostics.parquet` (IC/IR, decile spreads, LS stats, FF betas/t-stats/p-values) and `../data/factors/factor_correlation.parquet`. Reference copies (parquet + CSV) are git-tracked at [`diagnostics/factor_step_diagnostics.parquet`](diagnostics/factor_step_diagnostics.parquet) and [`diagnostics/factor_step_diagnostics.csv`](diagnostics/factor_step_diagnostics.csv) for quick inspection in a browser.
- FF time series: `../data/factors/factor_ff_timeseries.parquet` for benchmarking/orthogonalization.

## Default cleaning/neutralization (used by `compute` and the notebook)
- Coverage filter (drop dates with <30% non-NaN coverage).
- Winsorize (1st/99th pct).
- Fill (cross-sectional median).
- Sector neutralization (fallback to global if sector map missing).
- Z-score cross-sectionally; drop all-NaN dates.

## Extending
- Build new factors by subclassing `FactorBase` or parameterizing existing classes (e.g., Momentum with different lookback/skip).
- Use FF factors for benchmarking/orthogonalization via `load_ff_factors()` and `regress_on_ff`.
