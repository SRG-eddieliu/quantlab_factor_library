# QuantLab Factor Library
QuantLab Factor Library is a plug‑and‑play toolkit for building and evaluating equity factor signals on top of the cleaned outputs from the data pipeline. It loads wide price/sector data and Fama‑French benchmarks, computes forward returns, and provides 50+ prebuilt factors. Each factor subclasses a common FactorBase via inheritance, runs its raw computation, and then passes through a standardized cleaning pipeline (coverage filter, winsorization, fill, sector/global neutralization, z‑score) with overrides via config/config.json.

The runner supports modular steps: compute factors (optionally in parallel), run analytics (IC/IR, decile monotonicity, LS Sharpe/DD with FF regressions), compute cross-factor and FF correlations, and generate rolling IC/IR time effects. Outputs are written to data/factors as long-format Parquet (one file per factor and LS PnL), with registry and correlation files plus CSV mirrors in diagnostics for quick inspection. Two notebooks (factor_demo, factor_parallel_demo) show the full workflow and how to rerun analytics without recomputing factors. Use the default catalog or extend it by adding new factor classes and tuning cleaning/neutralization per factor.

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

#### Factor table
| # | Factor | Methodology | Intuition |
| --- | --- | --- | --- |
| 1 | momentum_12m | 12-1m return skip last 21d | Trend persistence |
| 2 | residual_momentum_12m | estimated rolling beta; computed residual_ret=stock_ret-beta*mkt_ret; Sum(residual_ret) over rolling window skip last 21d | Pure idio trend |
| 3 | efficiency_ratio_252d | 252d total return / sum(abs(daily returns)) | Higher efficiency ratios are often associated with cleaner persistent trends  |
| 4 | industry_momentum | Sector 6–1m return mapped to members | Sector momentum |
| 5 | max_daily_return_1m | 21d max single-day return | Lottery spike / reversal pointing lower forward return |
| 6 | high52w_proximity | Price / 252d high − 1 | Strength if current price near year highs |
| 7 | mean_reversion_5d | Negative 5d return | Short-term reversal |
| 8 | vwap_dev_21d | Price / 21d VWAP -1 | high menas overbuying likely revert |
| 9 | hurst_252d | Hurst exponent on 252d returns measuring persistence | Trend (>0.5) vs mean-revert (<0.5) can be used as filter |
| 10 | volatility_60d | Rolling std of returns | Low-vol premium |
| 11 | downside_vol_60d | Std of negative returns | Downside risk premium |
| 12 | residual_vol_252d | Std of market residuals; calculated by std(stock_ret - beta*mkt) | Idio risk penalty |
| 13 | ivol_60d | Short-window residual_vol | Faster idio risk |
| 14 | atr_14d | Average true range measured by avg of max(high−low,high−prev_close,low−prev_close) | Absolute move size |
| 15 | skewness_60d | Rolling skew | Lottery vs crash asymmetry |
| 16 | kurtosis_60d | Rolling kurtosis | Tail heaviness |
| 17 | beta_252d | Market beta | Systematic risk |
| 18 | downside_beta_252d | Beta on down-market days | Bad-beta exposure |
| 19 | coskewness_252d | Beta to squared market returns measuring how stock co-move with mkt vol | high coskewness lower fwd ret |
| 20 | dollar_volume_20d | 20d avg price×volume | Liquidity capacity; can be used as liquidity filter |
| 21 | amihud_illiq_20d | Mean(abs(ret)/dollar vol) 20d | Price impact per trading volume (illiquidity), high idicating illiquidity lower fwd ret |
| 22 | amihud_illiq_log_20d | Log-stabilized illiquidity 20d | Same as above, smoother |
| 23 | amihud_illiq_252d | Annual illiquidity | Structural illiquidity premium |
| 24 | turnover | Volume/shares | Attention/churn |
| 25 | obv | on balance volume; cumulative sum(volume_updays-volume_downdays) | Flow pressure |
| 26 | earnings_yield | TTM net income / mkt cap | Cheapness (value) |
| 27 | book_to_price | Book per share / price | Cheap vs book |
| 28 | ev_to_ebitda_inv | Inverse EV/EBITDA (EV = price×shares+debt−cash) | Cheap vs operating cash |
| 29 | cashflow_yield | Operating CF / mkt cap | Cheap vs cash |
| 30 | free_cashflow_yield | (Operating CF − capex) / mkt cap | Cheap vs free cash |
| 31 | accruals | (NI − CFO) / assets | Earnings quality (lower better) |
| 32 | size_log_mktcap | Log(price×shares) | Size exposure |
| 33 | profitability_roe | NI / equity | Quality profitability |
| 34 | roa | NI / assets | Efficiency of assets |
| 35 | gross_profitability | Gross profit / assets | Quality (Novy-Marx) |
| 36 | leverage | Liabilities / assets | Balance-sheet risk |
| 37 | sales_growth | YoY revenue growth | Growth strength |
| 38 | sales_growth_accel | QoQ change in YoY revenue growth | Growth acceleration |
| 39 | asset_growth | YoY assets | Investment intensity (high often bad) |
| 40 | investment_to_assets | Δ(PPE+inventory over 4q) / assets | Investment aggressiveness |
| 41 | rd_intensity | R&D / revenue | Innovation spend |
| 42 | net_issuance | YoY share change (annual) | Dilution signal |
| 43 | net_buyback_yield | Negative 4q share growth | Buyback support |
| 44 | dividend_yield_ttm | Trailing 12m dividends / price | Income/defensive |
| 45 | dividend_growth | YoY dividend growth | Payout momentum |
| 46 | piotroski_fscore | 0–9 composite quality | Balance-sheet/earnings health |
| 47 | analyst_revision_eps_30d | Net EPS estimate revisions (#analyst increased EPS est - decrease) 30d, carried forward max 30 trading days | Info drift from revisions |
| 48 | earnings_surprise | qtr; (reportedEPS - estEPS)/estEPS | Post-earnings drift |
| 49 | sue | Surprise / rolling 8q std | Standardized surprise |
| 50 | benford_chi2_d1 | First-digit Benford chi-square | Accounting conformity (lower better) |
| 51 | benford_chi2_d2 | Second-digit Benford chi-square | Accounting conformity (lower better) |
| 52 | composite_momentum | Exp-weighted mix of 21/63/126/252d stock returns (skip last 21d) | Recency-weighted trend |
| 53 | industry_co_momentum | Sector average returns, exp-weighted 21/63/126/252d (skip 21d), same score to members | Sector trend/leadership |
| 54 | volume_inclusive_icm | Sector average of return×volume, exp-weighted buckets (21/63/126/252d), skip 21d | Flow-weighted sector momentum |
| 55 | industry_co_reversal | Sector reversal: exp-weighted 21/63d sector returns, skip 5d, sign flipped | Sector mean-revert tilt |
| 56 | size_log_total_assets | Log total assets (quarterly, ffill) | Size proxy (smaller = higher expected return) |
| 57 | size_log_enterprise_value | Log enterprise value (price×shares + debt − cash, quarterly, ffill) | Size proxy (smaller EV = higher expected return) |
| 58 | size_log_revenue | Log total revenue (quarterly, ffill) | Size proxy (smaller revenue = higher expected return) |

### Thematic composites (config-driven)
- Defined in `config/config.json` under `composites`: name, factors, optional `sign` map (+1/-1), and `weight_method` (`equal`, `inv_vol`, `ic_ir`).
- Weighting options:  
  - `equal`: equal weights.  
  - `inv_vol`: 1 / volatility (prefers LS return std if available; else factor std), normalized.  
  - `ic_ir`: weights proportional to IC IR (from factor analytics).
- Build + diagnose via `run_composite_pipeline` (build, full analytics, save diagnostics) or `build_composites_from_config` + `analyze_composites`.
- Included themes:  
  - `theta_momentum_riskadj`: momentum_12m, residual_momentum_12m, volatility_60d (-), downside_vol_60d (-), industry_momentum.  
  - `theta_pure_value`: earnings_yield, book_to_price, ev_to_ebitda_inv, cashflow_yield, free_cashflow_yield.  
  - `theta_pure_quality`: profitability_roe, roa, gross_profitability, piotroski_fscore, accruals (-).  
  - `theta_shortterm_reversal`: mean_reversion_5d, max_daily_return_1m (-), high52w_proximity (-).  
  - `theta_growth_acceleration`: sales_growth, sales_growth_accel, rd_intensity, dividend_growth.  
  - `theta_info_forensic_drift`: sue, earnings_surprise, analyst_revision_eps_30d, benford_chi2_d1 (-), benford_chi2_d2 (-).  
  - `theta_structural_liquidity`: dollar_volume_20d (-), amihud_illiq_252d, amihud_illiq_log_20d, turnover (-).  
  - `theta_systematic_risk`: beta_252d (-), residual_vol_252d (-), downside_beta_252d (-), leverage (-), asset_growth (-), investment_to_assets (-), net_issuance (-).  
  - `theta_size_smallcap`: size_log_mktcap (-), size_log_total_assets (-), size_log_enterprise_value (-), size_log_revenue (-).

Notebook flow (see `notebooks/factor_parallel_demo.ipynb`):
- Build/diagnose composites under all weightings (equal/inv_vol/ic_ir), using `run_composite_pipeline` with `ic_map` (IC IR) and `ls_vol_map` (LS vol) from base factor analytics.
- Compare IC IR across weightings; select the best weighting per composite via `select_best_weights`.
- Re-run best composites, saving full diagnostics to `diagnostics/`:
  - `composite_analytics_summary.parquet/csv` (IC/IR, LS stats, FF regression, deciles).  
  - `composite_correlation.parquet/csv` and `composite_ff_correlation.parquet/csv`.
- Weighting for composites uses a 50/50 blend of full-sample IC IR and 12m IC when `blend_12m=True` (see notebook), plus LS vol for `inv_vol` weighting.
- Each composite and its LS PnL are also saved to `data/factors/` (wide→long parquet) alongside the raw factors.

## Outputs
- Factors: `../data/factors/factor_<name>.parquet` (long format: Date, Ticker, Value).
- Registry: `../data/factors/factor_analytics_summary.parquet` (mean IC, IC t-stat, IC IR, mean autocorr, decile spread, LS stats, FF alpha/betas).
- Correlation: `../data/factors/factor_correlation.parquet` (and FF corr) with CSV mirrors in `diagnostics/` for quick inspection.
- FF time series: `../data/factors/factor_ff_timeseries.parquet` for benchmarking/orthogonalization.

## Default cleaning/neutralization (used by `compute` and the notebook)
- Coverage filter (drop dates with <30% non-NaN coverage).
- Winsorize (1st/99th pct).
- Fill (cross-sectional median).
- Sector neutralization. (sector de-mean)
- Z-score cross-sectionally. (stock selection alpha instead of global alpha overlay)
- drop all-NaN dates.

## Data handling notes
- Fundamental tables are treated as quarterly-only in factor code; date alignment comes from the pipeline’s `fiscalDateEnding + 2 business days` when available.
- Slow-moving fundamentals (e.g., earnings yield, size proxies) are forward-filled to daily to avoid empty windows.
- Analyst revisions (`analyst_revision_eps_30d`) fill missing up/down counts with 0 before netting to avoid losing dates, then only carry forward for 30 trading days to avoid stale signals.
- Event-style factors (earnings_surprise, SUE) remain point-in-time; avoid forward fill to reduce look-ahead.

## Extending
- Abstract Base Class (ABC) architecture being used with base class `FactorBase` and mandatory methods `compute_raw_factor` and `post_process` which every specific factor implementation must inherit and override. The main execution script dynamically iterate through a list of factor classes, instantiate them, and call their respective methods without needing to know the complex internal calculation logic of each specific factor. This achieves the goal of modularity: allowing new factors to be added simply by creating a new class file without ever modifying the main processing loop.
- Use FF factors for benchmarking/orthogonalization via `load_ff_factors()` and `regress_on_ff`.
- FF regression uses lightweight OLS (numpy + scipy for p-values) to keep the pipeline lean and enhance running efficiency.
- parallel run option implemented for efficient computation leveraging python built-in concurrent method `ThreadPoolExecutor`.

## References
- Analytics registry (CSV): [`diagnostics/factor_analytics_summary.csv`](diagnostics/factor_analytics_summary.csv)
- Rolling analytics (CSV): [`diagnostics/factor_rolling_analytics.csv`](diagnostics/factor_rolling_analytics.csv)
- Correlation matrices (CSV): [`diagnostics/factor_correlation.csv`](diagnostics/factor_correlation.csv) and [`diagnostics/factor_ff_correlation.csv`](diagnostics/factor_ff_correlation.csv)
