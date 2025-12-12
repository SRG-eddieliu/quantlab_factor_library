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

### Full factor list
Momentum & Trend  
- momentum_12m — 12-1m return (skip last 21d); trend persistence.  
- residual_momentum_12m — momentum orthogonal to market; pure idio trend.  
- efficiency_ratio_252d — smoothness of 252d path (higher = cleaner trend).  
- industry_momentum — sector 6–1m return mapped to members.  
- max_daily_return_1m — 21d max single-day jump; lottery/reversal flag.  
- high52w_proximity — proximity to 52w high; trend strength.

Reversal & Path  
- mean_reversion_5d — negative 5d return; short-term reversal.  
- vwap_dev_21d — price vs 21d VWAP; stretched vs flow anchor.  
- hurst_252d — Hurst exponent on 252d returns; >0.5 trending, <0.5 mean-revert.

Volatility & Tail Risk  
- volatility_60d — rolling std of returns; low-vol anomaly proxy.  
- downside_vol_60d — std of negative returns; downside risk focus.  
- residual_vol_252d — std of market residuals; idiosyncratic risk.  
- ivol_60d — short-window idiosyncratic vol.  
- atr_14d — average true range; absolute move size.  
- skewness_60d — rolling skew; lottery vs crash asymmetry.  
- kurtosis_60d — rolling kurtosis; tail heaviness.  
- beta_252d — market beta.  
- downside_beta_252d — beta on down-market days.  
- coskewness_252d — beta to squared market returns; tail co-movement.

Liquidity & Flow  
- dollar_volume_20d — 20d avg price×volume; trading capacity.  
- amihud_illiq_20d — |ret|/dollar vol (20d); price impact.  
- amihud_illiq_log_20d — log-stabilized illiquidity (20d).  
- amihud_illiq_252d — annual illiquidity.  
- turnover — volume ÷ shares; attention/churn.  
- obv — cumulative signed volume; flow pressure.

Value  
- earnings_yield — TTM net income / mkt cap (quarterly ffill); cheapness.  
- book_to_price — book per share / price; value.  
- ev_to_ebitda_inv — inverse EV/EBITDA (EV = price×shares + debt − cash).  
- cashflow_yield — operating CF / mkt cap.  
- free_cashflow_yield — (op CF − capex) / mkt cap.  
- accruals — (NI − CFO) / assets; earnings quality.

Quality & Profitability  
- size_log_mktcap — log(price×shares).  
- profitability_roe — NI / equity.  
- roa — NI / assets.  
- gross_profitability — gross profit / assets.  
- leverage — liabilities / assets.

Growth & Investment  
- sales_growth — YoY revenue growth.  
- sales_growth_accel — quarterly YoY growth minus YoY four quarters ago.  
- asset_growth — YoY assets.  
- investment_to_assets — Δ(PPE+inventory over 4q) / assets.  
- rd_intensity — R&D / revenue.

Capital Actions  
- net_issuance — YoY share change (annual).  
- net_buyback_yield — negative 4q share growth (buybacks positive).  
- dividend_yield_ttm — trailing 12m dividends / price.  
- dividend_growth — YoY dividend growth.

Composite Quality  
- piotroski_fscore — 0–9 composite quality score.

Estimates & Events  
- analyst_revision_eps_30d — net EPS estimate revisions over 30d.  
- earnings_surprise — surprise % vs estimates (quarterly).  
- sue — standardized unexpected earnings (surprise / 8q std).

Forensic & Integrity  
- benford_chi2_d1 — first-digit Benford chi-square (quarterly fundamentals).  
- benford_chi2_d2 — second-digit Benford chi-square (quarterly fundamentals).

The asymmetry of the return distribution. High positive skew means frequent small losses but occasional large gains.

Short: Retail traders tend to overpay for positively skewed stocks (the "lottery ticket" effect), leading to poor long-term returns.

kurtosis_60d

The "fat-tailedness" of the return distribution. High kurtosis means more extreme events (crashes or spikes).

Short: High tail risk (fat tails) is generally penalized by investors.

beta_252d

Market Beta: measures the stock's sensitivity to the overall market return.

Short: Used to short high-beta stocks and long low-beta stocks to capture the low-beta/low-vol anomaly.

downside_beta_252d

Beta calculated only on days when the market itself was down.

Short: Stocks that fall disproportionately more than the market during stress are highly risky and should be shorted.

coskewness_252d

Measures a stock's sensitivity to market volatility (squared market returns).

Short: Stocks that suffer most when market volatility rises are poor hedges and should be avoided or shorted.

IV. Liquidity & Flow

These factors capture the ease of trading and the supply/demand dynamics of a stock.

Factor Name

Measures

Trading Intuition (High Value)

dollar_volume_20d

The average daily traded dollar value over 20 days.

Short: High dollar volume means high liquidity. Low liquidity stocks often carry an illiquidity premium that can be exploited by going long.

amihud_illiq_20d

Amihud's illiquidity measure: price impact per unit of volume (high value = high price impact).

Long: Stocks with high illiquidity (high Amihud) are poorly rewarded and often shorted to capture the illiquidity premium.

amihud_illiq_log_20d

Log-transformed version of Amihud's measure for stabilizing the distribution.

Long: Same intuition as amihud_illiq_20d.

amihud_illiq_252d

The yearly version of the illiquidity measure.

Long: Captures long-term structural illiquidity premium.

turnover

Trading volume divided by shares outstanding (how frequently shares change hands).

Short: High turnover means high trading interest; low turnover stocks are often neglected and may carry a premium.

obv

On-Balance Volume: a cumulative sum of volume that adds on up days and subtracts on down days.

Long: Rising OBV indicates buying pressure (accumulation) that often precedes price appreciation.

V. Value

These are classic factors that identify stocks trading cheaply relative to their fundamentals, based on the principle of mean reversion.

Factor Name

Measures

Trading Intuition (High Value)

earnings_yield

TTM Net Income / Market Cap (the inverse of the P/E ratio).

Long: High earnings yield means the stock is cheap relative to its earnings power (classic Value signal).

book_to_price

Book Equity per Share / Price (the inverse of the P/B ratio).

Long: High B/P means the stock is cheap relative to its historical accounting value (classic Value signal).

ev_to_ebitda_inv

Inverse of Enterprise Value to EBITDA. (EV accounts for debt and cash).

Long: A preferred Value metric that is less susceptible to accounting tricks than P/E. High value means cheap relative to operating cash flow.

cashflow_yield

Operating Cash Flow / Market Cap.

Long: Cash flow is harder to manipulate than net income. High value means cheap relative to actual cash generation.

free_cashflow_yield

(Operating CF - Capex) / Market Cap.

Long: The purest form of cash flow: the money the company has left after funding necessary maintenance. High value means excellent cash generation relative to price.

accruals

(Net Income - Operating Cash Flow) / Assets.

Short: Measures the use of non-cash accounting entries (aggressiveness). High accruals predict lower future earnings (forensic accounting signal).

VI. Quality & Profitability

These factors reward companies with high profitability, strong balance sheets, and consistent performance.

Factor Name

Measures

Trading Intuition (High Value)

size_log_mktcap

Logarithm of Market Capitalization.

Long/Short: Typically used as a risk factor (Small stocks tend to outperform, hence we short large caps or use this to orthogonalize other factors).

profitability_roe

Return on Equity (Net Income / Equity).

Long: High ROE indicates high return for shareholders' investment (Quality signal).

roa

Return on Assets (Net Income / Total Assets).

Long: Measures management's efficiency in using the company's total assets (Quality signal).

gross_profitability

Gross Profit / Total Assets.

Long: A highly robust quality metric (Novy-Marx) that correlates with high returns.

leverage

Total Liabilities / Total Assets.

Short: High leverage increases bankruptcy risk and reduces financial flexibility (a "Quality" detractor).

VII. Growth & Investment

These factors focus on the speed of business expansion and investment decisions.

Factor Name

Measures

Trading Intuition (High Value)

sales_growth

Year-over-Year (YoY) Revenue Growth.

Long: Rewards companies with expanding sales, a key driver of long-term value.

sales_growth_accel

Quarterly YoY growth minus the YoY growth four quarters ago (the acceleration or deceleration of growth).

Long: Rewards companies whose growth rate is currently increasing (often a stronger signal than the absolute growth rate).

asset_growth

Year-over-Year (YoY) Total Asset Growth.

Short: Companies that grow assets aggressively (high investment) often destroy shareholder value (Investment Anomaly).

investment_to_assets

Change in capital assets (PPE + inventory) relative to total assets.

Short: Same as asset growth; captures aggressive investment which often leads to underperformance (misallocation of capital).

rd_intensity

Research & Development expense / Revenue.

Long: Rewards companies investing heavily in future growth and innovation, often a long-term Quality/Growth signal.

VIII. Capital Actions

These factors capture management's confidence and shareholder return policy.

Factor Name

Measures

Trading Intuition (High Value)

net_issuance

YoY change in shares outstanding.

Short: High net issuance suggests equity is being issued to finance projects, often diluting existing shareholders or signaling that management thinks the stock is overvalued.

net_buyback_yield

The negative of share growth (positive value means net share reduction/buybacks).

Long: Buybacks signal management confidence and directly reduce the share count, boosting earnings per share.

dividend_yield_ttm

Trailing 12-month dividends / Price.

Long: Rewards income-generating stocks, often indicating maturity and stability.

dividend_growth

YoY dividend growth rate.

Long: Suggests management confidence in future earnings stability.

IX. Composite Quality & Estimates/Events

These groups contain pre-packaged composites and information shock indicators.

Factor Name

Measures

Trading Intuition (High Value)

piotroski_fscore

A 0-9 score assessing the strength of a company's financial position (profitability, leverage, liquidity, operating efficiency).

Long: Rewards high-quality stocks based on fundamental composite criteria.

analyst_revision_eps_30d

Net change in consensus EPS estimates over the last 30 days.

Long: Positive revisions predict future stock price appreciation (Post-Earnings Announcement Drift, PEAD).

earnings_surprise

The magnitude of the quarterly earnings surprise relative to consensus.

Long: Captures the immediate impact of an earnings beat or miss.

sue

Standardized Unexpected Earnings (the surprise scaled by historical standard deviation).

Long: A robust measure of the information shock, predicting future positive returns (PEAD).

X. Forensic & Integrity

These factors look for accounting irregularities or signs of manipulation.

Factor Name

Measures

Trading Intuition (High Value)

benford_chi2_d1

A Chi-square score measuring how much the first digits of quarterly fundamentals deviate from Benford's Law (expected distribution).

Short: High chi-square suggests that the numbers may have been manipulated or smoothed, indicating high accounting risk.

benford_chi2_d2

Same test, but on the second digit distribution.

Short: Same intuition; a high score indicates data integrity risk.


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

## Extending
- Build new factors by subclassing `FactorBase` or parameterizing existing classes (e.g., Momentum with different lookback/skip).
- Use FF factors for benchmarking/orthogonalization via `load_ff_factors()` and `regress_on_ff`.
- FF regression uses lightweight OLS (numpy + scipy for p-values) to keep the pipeline lean.

## References
- Analytics registry (CSV): [`diagnostics/factor_analytics_summary.csv`](diagnostics/factor_analytics_summary.csv)
- Rolling analytics (CSV): [`diagnostics/factor_rolling_analytics.csv`](diagnostics/factor_rolling_analytics.csv)
- Correlation matrices (CSV): [`diagnostics/factor_correlation.csv`](diagnostics/factor_correlation.csv) and [`diagnostics/factor_ff_correlation.csv`](diagnostics/factor_ff_correlation.csv)
