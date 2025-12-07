# QuantLab Factor Library

Factor research toolkit that reads cleaned Parquet outputs from the data pipeline (`../data/data-processed/*.parquet`) and writes factor signals/analytics to `../data/factors/`. Includes templates, loaders, transforms, analytics, and an example notebook.

## Quickstart
- Configure paths in `config/config.json` if needed (`data_root`, `final_dir`, `factors_dir`); defaults point to `../data`.
- Create env: `conda env create -f quantlab_env/environment.yml` (includes numpy, pandas, pyarrow, scipy, etc.).
- Run default factors:  
  `python -m quantlab_factor_library.run_factors`
- Example usage: [`notebooks/factor_demo.ipynb`](notebooks/factor_demo.ipynb)

## What’s inside
- `quantlab_factor_library/paths.py` – resolves repo and data roots (configurable).
- `quantlab_factor_library/data_loader.py` – loads long-format parquet, pivots to wide price/sector, computes forward returns; loads FF factors.
- `quantlab_factor_library/base.py` – `FactorBase` enforcing `compute_raw_factor` + `post_process`; shared cleaning via `compute`.
- `quantlab_factor_library/factors/` – parameterized starters: Momentum, Volatility, MeanReversion, DollarVolume.
- `quantlab_factor_library/transforms.py` – coverage filter, winsorize, fill (median/sector-median), neutralize (sector/global), z-score, drop-all-NaN; `clean_factor` helper.
- `quantlab_factor_library/analytics.py` – IC (Spearman), autocorr (lag-1 rank persistence), decile monotonicity, LS diagnostic (top/bottom pct) with Sharpe, max drawdown, mean/std, FF regression (alpha/betas + t-stats/p-values), factor correlation, diagnostics/registry writers.
- `quantlab_factor_library/run_factors.py` – runs default factors, saves outputs, updates analytics registry, writes factor correlations, FF time series.
- `notebooks/factor_demo.ipynb` – end-to-end demo (load → compute → transparent pipeline → analytics → correlation → save factors/diagnostics).
- `config/config.json` – optional path overrides.
- FF loader: `DataLoader.load_ff_factors()` reads `data/data-processed/FAMA_FRENCH_FACTORS.parquet` (mktrf, smb, hml, rmw, cma, rf, umd).
- FF regression uses a lightweight OLS (numpy + scipy for p-values) to keep the pipeline lean and fast.

## Outputs
- Factors: `../data/factors/factor_<name>.parquet` (long format: Date, Ticker, Value).
- Registry: `../data/factors/factor_analytics_summary.parquet` (mean IC, IC t-stat, IC IR, mean autocorr, decile spread, LS stats, FF alpha/betas).
- Diagnostics: `../data/factors/factor_step_diagnostics.parquet` (IC/IR, decile spreads, LS stats, FF betas/t-stats/p-values) and `../data/factors/factor_correlation.parquet`.
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
