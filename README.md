# Portfolio Optimisation Toolkit

This repository hosts a set of Python scripts for constructing equity portfolios using a range of quantitative valuation models and optimisation techniques. The workflow pulls market and fundamental data, generates forecasts with multiple statistical approaches and produces optimised portfolio allocations. The results are written to Excel workbooks for inspection or further analysis.

The repository follows a classical quantitative finance workflow:

1) Acquire market, fundamental, and alternative data.
2) Use various models to forecast equity returns and buy and sell signals.
3) Optimising asset weights based on the forecasts and risk models.
4) Analysing characteristics of the resulting portfolio.

The techniques used are a sophisticated blend of traditional financial modeling, econometric time series analysis, modern machine learning and my own proprietary models

## Installation

The project requires Python 3.10+ and a number of scientific packages. Install the dependencies using:

```bash
pip install -r requirements.txt
```

Several scripts expect local Excel files as inputs/outputs (e.g. `Portfolio_Optimisation_Data_YYYY-MM-DD.xlsx`). Update the paths inside the modules if your files live elsewhere.

Financial reports that are imported are obtained from [Stock Analysis ](https://stockanalysis.com/).
## Repository Layout

```
fetch_data/       - Download analyst fundamentals, historical prices and macro data
data_processing/  - Load and clean financial statements, ratios and macro series
forecasts/        - Forecasting and valuation models (DCF, DCFE, Prophet, SARIMAX, etc.)
functions/        - Reusable utilities (regression, Black–Litterman, CAPM, covariance)
Optimiser/        - Portfolio optimisation routines and reporting
indicators/       - Technical indicators and Reddit sentiment scrapers
maps/             - Mapping tables for currencies, sectors and indexes
rel_val/          - Relative valuation models (PE, PS, PBV, EV/Sales, Graham, etc.)
```

## Typical Workflow
The major components are described below.

## Data Collection (`fetch_data`)

* **`financial_data.py`** – Retrieves analyst estimates and financial statements
  via `yfinance`.  Processes historical draws and exports metrics such as EPS,
  revenue growth and analyst price targets.

  It assigns a standard error metric for the average price prediction via the equation $\displaystyle \sigma = \frac{\text{Max Price Prediction } - \text{ Min Price Prediction}}{2 \cdot Z \cdot \text{Price}}$ where $\alpha \;=\frac{1}{N_{\text{analysts}}} \quad\Longrightarrow\quad Z = Z_{1 - \alpha/2}$
  
* **`fetch_macro_data.py`** – Downloads macroeconomic time series (interest
  rates, CPI, GDP, unemployment) from FRED and major index prices.
* **`factor_data.py`** – Loads Fama–French factor returns from the Fama–French
  database for use in factor models.
* **`collecting_data.py`** – Downloads historical open, close, low, high and volume data, computes basic
  technical indicators with the `ta` package and scrapes economic forecasts from
  Trading Economics. 

* These scripts populate the Excel workbooks used in later stages.

## Data Processing (`data_processing`)

* **`financial_forecast_data.py`** – Handles per‑ticker financial statements and
  analyst forecasts.

  Provides convenience methods for currency adjustment,
  outlier removal and generating forward‑looking KPIs used by the models.
  
* **`macro_data.py`** – Wrapper around macroeconomic series giving easy access to
  inflation, GDP growth and other regressors.
  
* **`ratio_data.py`** – Loads historical financial ratios and derives growth
  metrics used for regressions.
  
* **`ind_data_processing.py`** – Cleans the stock screener output and aggregates
  industry/region level data.

## Forecast Models (`forecasts`)

### Machine-Learning:

* **`prophet_model.py`** – Utilises Facebook Prophet with piecewise linear and logistic trends.

  Financial and macro regressors extend the additive model, and cross-validation tunes changepoints and seasonality. Weekly seasonal trends are enabled. Daily and yearly seasonality is disabled due to the   substantial noise created.

  Scenario draws create probabilistic price paths.

* **`Sarimax.py`** – Fits SARIMAX (Seasonal Autoregressive Integrated Moving Average + Exogenous Variables) time-series models with exogenous macro factors.

  Candidate ARIMA (Autoregressive Integrated Moving Average) orders are weighted by AIC (Akaike Information Criterion) to form an ensemble. 

  Future macro scenarios are drawn from a VAR (Vector Auto Regressive) process via Cholesky simulation and propagated through the model.

  Monte-Carlo simulation is then used to generate correlated draws from the VAR models output
  
* **`lstm.py`** – Builds a recurrent LSTM (Long Short-Term Memory) network on rolling windows of returns and engineered factors.

  Robust scaling, dropout layers and early stopping help regularise the model.

  Bootstrapped datasets yield an ensemble of forecasts for each ticker.

  Macro Forecasts obtained from Trading Economics are used, as well as revenue and eps forecasts that are obtained from Yahoo Finance and Stock Analysis.
  
* **`returns_reg_pred.py`** – Trains a gradient‑boosting regression on engineered features to predict twelve‑month returns.

  Hyperparameters are tuned with grid search, and bootstrapped samples produce an ensemble of models.

  Macro Forecasts obtained from Trading Economics are used, as well as revenue and eps forecasts that are obtained from Yahoo Finance and Stock Analysis.

### Intrinsic Valuation:

* **`dcf.py`** – Performs discounted cash‑flow valuation to determine the enterprise value. Cash flows are forecast using elastic‑net regression (see `fast_regression.py`) and then discounted.

  Monte‑Carlo scenarios for growth generate a distribution of intrinsic values. The scenarios are used to help gauge the uncertainty of the valuation.
  
* **`dcfe.py`** – Similar to `dcf.py` but values equity directly via discounted cash‑flow to equity.  Constrained regression ensures realistic relationships between drivers.

  Monte-Carlo simulation is used for the aformentioned reason.
  
* **`ri.py`** – Implements a residual income model where future book value is grown and excess returns are discounted using the cost of equity.

  Monte-Carlo simulation is once again used for the aformentioned reason.


### Relative Valuation and Factor Models:

* **`relative_valuation_and_capm.py`** – A script to compute the relative value and then the stock price via valuation ratios and analyst earnings and revenue estimates.

  The script also computes factor model forecasts (CAPM, Fama-French 3 factor model and Fama-French 5 factor model) to derive expected returns. Betas are estimated by OLS, factor paths are simulated with VAR, and Black–Litterman views adjust expected market returns.

* **`Combination_Forecast.py`** – Aggregates all of the above model outputs into
  a Bayesian ensemble, applying weights and producing an overall score table.

  Weights are are assigned for each models prediction based on the inverse of the standard error or volatility, i.e.

$$
w_i = \frac{\frac{1}{\mathrm{SE}_i^2}}{\sum{\frac{1}{{SE}_i^2}}}
$$

  These weights are capped at 10% per model, unless there are not enough valid models, in which case the cap is $ \frac{1}{\text{number of valid models}}$.

  The score is inspired by the Pitroski F-score.

  It includes all 9 of the Pitroski F-Score variables:

  1: Positive Return on Assets
  2: Positive Operating Cash Flow
  3: ROA year on year growth
  4: Operating Cash Flow higher than Net Income
  5: Negative Long Term Debt year on year Growth
  6: Current Ratio year on year growth
  7: No new shares Issued
  8: Higher year on year Gross Margin
  9: Higher Asset Turnover Ratio year on year Growth.

  I adapt this in the following way

  - Negative Return on Assets $\Rightarrow$- 1
  - Return on Assets > Industry Average $\Rightarrow$ + 1
  - Return on Assets < Industry Average $\Rightarrow$ - 1
  - Previous Return on Assets < Current Return on Assets $\Rightarrow$ - 1
  - Previous Current Ratio > Current Ratio $\Rightarrow$ - 1

  I then add the following scores relating to financials as well. These were back tested to see the significance.

  - 5% increase in percentage of shares shorted month on month $\Rightarrow$ - 1
  - 5% decrease in percentage of shares shorted month on month $\Rightarrow$ + 1
  - Insider Purchases $\Rightarrow$ + 2
  - Insider Selling $\Rightarrow$ - 1
  - Positive Earnings Growth $\Rightarrow$ + 1
  - Negative Earnings Growth $\Rightarrow$ - 1
  - Top 25% Earnings Growth $\Rightarrow$ + 1
  - Earnings Growth > Industry Average Earnings Growth $\Rightarrow$ + 1
  - Analyst Average Predicted EPS > Current EPS $\Rightarrow$ + 1
  - Analyst Average Predicted EPS < Current EPS $\Rightarrow$ - 1
  - Revenue Growth > Industry Average Revenue Growth $\Rightarrow$ + 1
  - Revenue Growth < Industry Average Revenue Growth $\Rightarrow$ - 1 
  - Analyst Average Predicted Revenue > Current Revenue $\Rightarrow$ + 1
  - Analyst Average Predicted Revenue < Current Revenue $\Rightarrow$ - 1
  - Positive Return on Equity $\Rightarrow$ + 1
  - ROE > Industry Average Return on Equity $\Rightarrow$ + 1
  - Return on Equity < Industry Average Return on Equity $\Rightarrow$ - 1
  - 0 < Price to Book <= 1 $\Rightarrow$ - 1
  - Negative Price to Book $\Rightarrow$ - 1
  - Price to Book < Industry Price to Book $\Rightarrow$ + 1
  - Price to Book > Industry Price to Book $\Rightarrow$ - 1
  - Trailing 12 month Price to Earnings > Industry Price to Earnings $\Rightarrow$ - 1 
  - Forward Price to Earnings < Trailing 12 month Price to Earnings $\Rightarrow$ + 1
  - Forward Price to Earnings > Trailing 12 month Price to Earnings $\Rightarrow$ - 1

  I then consider Analyst recommendations, to gather a sense of professional sentiment:

  - Strong Buy Recommendation -> +3
  - Hold Recommendation -> -1
  - Sell or Strong Sell Recommendation -> -5
 
  I then consider the stock prices movement within the market:

  - Positive Skewness based on last 5 years weekly returns -> +1
  - Sharpe Ratio based on last year weekly returns > 1.0 -> +1
  - Sortin Ratio based on last year weekly returns > 1.0 -> +1
  - Upside Ratio > 1.5 and Downside Ratio < 0.5 based on last 5 years weekly returns with respect to the S&P500 -> +1
  - Upside Ratio > 1.0 and Downside Ratio < 0.5 based on last 5 years weekly returns with respect to the S&P500 -> +1
  - Upside Ratio > 1.0 and Downside Ratio < 1.0 based on last 5 years weekly returns with respect to the S&P500 -> +1
  - Upside Ratio > 1.5 and Downside Ratio < 1.0 based on last 5 years weekly returns with respect to the S&P500 -> +1
  - Downside Ratio > 1.0 and Upside Ratio < Downside Ratio based on last 5 years weekly returns with respect to the S&P500 -> -1
  - Downside Ratio > 1.5 and Upside Ratio < Downside Ratio based on last 5 years weekly returns with respect to the S&P500 -> -1
  - Positive Jensen's Alpha over last 5 years with respect to the S&P500 -> +1, Negative Jensens Alpha over last 5 years with respect to the S&P500 -> -1
  - Negative Predicted Jensen's Alpha -> -5
 
  I then consider daily sentiment scores from webscraping r/wallstreetbets. This is to capture the sentiment amongst retail investors, which have an increasing importance in influencing the market:

  - Positive Average Sentiment -> +1, Negative Average Sentiment -> -1
  - Positive Average Sentiment and over 4 mentions -> +1, Negative Average Sentiment and over 4 mentions -> -1
  - Average Sentiment > 0.2 and over 4 mentions -> +1, Average Sentiment < 0.2 and over 4 mentions -> -1
  - Average Sentiment > 0.2 and over 10 mentions -> +1, Average Sentiment < 0.2 and over 10 mentions -> -1

  I then add the scores from the technical buy and sell indicators to these scores.

## Utility Functions (`functions`)

* **`fast_regression.py`** – An elastic‑net solver built with CVXPY used to forecast cash flows in `dcf.py` and `dcfe.py`.

  It applies Huber loss and L1 (Lasso) / L2 (Ridge) penalties and performs grid‑search cross‑validation, optionally enforcing accounting sign constraints.

* **`cov_functions.py`** – Implements covariance estimators including constant‑correlation and Ledoit–Wolf shrinkage.

  Predicted covariances are derived from multi‑horizon scaling with an extended Stein shrinkage variant.

* **`black_litterman_model.py`** – Implements the Black–Litterman Bayesian update combining equilibrium market returns with subjective views to obtain posterior means and covariances.

* **`capm.py`** – Helper implementing the CAPM formula:

$$
\quad
\mathbb{E}[R_i]
= R_f + \beta_i \bigl(\mathbb{E}[R_m] - R_f\bigr),
$$


* **`coe.py`** – Calculates the cost of equity per ticker by combining country risk premiums and currency risk with the standard CAPM estimate.

* **`fama_french_3_pred.py` / `fama_french_5_pred.py`** – Estimate expected returns using the Fama–French 3 factor and Fama-French 5 factor models using OLS Betas and simulated future factor values.

  Fama-French 3 factor model is given by:
  
$$
\quad
\mathbb{E}[R_i]
= R_f + \beta_{i,m} \bigl(\mathbb{E}[R_m] - R_f\bigr) + \beta_{i,\mathrm{SMB}} \mathbb{E}[\mathrm{SMB}] + \beta_{i,\mathrm{HML}} \mathbb{E}[\mathrm{HML}],
$$

  Fama-French 5 factor model is given by:
  
$$
\quad
\mathbb{E}[R_i]
= R_f + \beta_{i,m} \bigl(\mathbb{E}[R_m] - R_f\bigr) + \beta_{i,\mathrm{SMB}} \mathbb{E}[\mathrm{SMB}] + \beta_{i,\mathrm{HML}} \mathbb{E}[\mathrm{HML}] + \beta_{i,\mathrm{RMW}} \mathbb{E}[\mathrm{RMW}] + \beta_{i,\mathrm{CMA}} \mathbb{E}[\mathrm{CMA}],
$$

  
* **`factor_simulations.py`** – Generates future factor realisations by fitting a VAR model and applying Cholesky shocks. These simulated paths feed into the Fama–French forecasts.
  
* **`export_forecast.py`** – Writes DataFrames to Excel with conditional formatting and table styling.
  
* **`read_pred_file.py`** – Reads previously generated forecast sheets and updates them with latest market prices.

## Technical Indicators and Sentiment (`indicators`)

* **`technical_indicators.py`** – Calculates technical Buy and Sell stock indicators, scoring each ticker and saving results to
  Excel.

  These indicators include:
  
  - MACD (Moving Average Convergence/Divergence)
  - RSI (Relative Strength Index) with Buy and Sell thresholds of 30 and 70 respectively over a 14 day period window.
  - EMA (Exponential Moving Average) Crossover Signals with Fast and Slow moving averages of 12 and 26 respectively.
  - Bollinger Signals with a Bollinger Band window of 20 with Bollinger Band standard deviation of 2.
  - Stochastic Signals with Slow and Fast windows of 14 and 3 respectively, and with buy and sell values of 20 and 80 respectively.
  - ATR (Average True Range) Breakout with ATR window of 14, ATR Breakout window of 20 and ATR Multiplier of 1.5.
  - OBV (On-Balance Volume) Divergence with OBV lookback of 20.
  - True Wilder ADX (Average Directional Index) with ADX window and ADX threshold of 14 and 25 respectively.
  - MFI (Money Flow Index) with MDI window of 14 and Buy and Sell thresholds of 20 and 80 respectively.
  - VWAP (Volume Weighted Average Price) with VWAP window of 14

 Buy and Sell signals are given a score of ±1
 
* **`wallstreetbets_scrapping.py`** – Scrapes posts and comments from r/wallstreetbets.

  Ticker mentions are analysed with NLTK’s (Natural Language Toolkit) VADER (Valence Aware Dictionary and Sentioment Reasoner) sentiment classifier and aggregated scores are saved.

  I have tuned this dictionary to account for relevant slang and market related terms frequently used. For example, "bullish", "buy the dip" and "yolo".

## Relative Valuation (`rel_val`)

Provides multiple models blending peer multiples and fundamental data:

* **`pe.py`, `price_to_sales.py`, `pbv.py`, `ev.py`** – Compute valuations using peer multiples such as P/E, P/S, P/BV and EV/Sales based on industry and regional medians as well as the own tickers respective metric.
  
* **`graham_model.py`** – Implements a Graham‑style intrinsic value combining earnings and book value metrics. This does not use 22.5, and instead uses the industry averages.
  
* **`relative_valuation.py`** – Consolidates all relative valuation signals into a single fair value estimate.

## Portfolio Optimisation (`Optimiser`)

* **`portfolio_functions.py`** – Utility functions for:

  - Portfolio Return
  - Portfolio Volatility,
  - Portfolio Downside Deviation
  - Tracking Error
  - Portfolio Beta
  - Treynor Ratio
  - Portfolio Score
  - Sharpe Ratio
  - Annualised Volatility
  - Drawdown
  - Skewness
  - Kurtosis
  - VaR Gaussian
  - VaR
  - CVaR
  - Information Ratio
  - Annualised Returns
  - Ucler Index
  - CDaR
  - Jensen's Alpha
  - Capture Ratios
  - Sortino Ratios
  - Calmar Ratio
  - Omega Ratio
  - Modigliani Ratio
  - MAR Ratio
  - Pain Index
  - Pain Ratio
  - Tail Ratio
  - RAROC
  - Percentage of Portfolio Positive Streaks
  - Geometric Brownian Motion
  - Portfolio Simulation
  - Simulation and Portfolio Metrics Report

* **`portfolio_optimisers.py`** – Implements portfolio optimisers subject to constraints using `scipy.optimize`. These optimisers include:

  - Max Sharpe Portfolio
  - Max Sortino Portfolio
  - Max Information Ratio Portfolio
  - Equal Risk Portfolio
  - Max Sharpe Portfolio using Black Litterman returns and covariance
  - Max Risk Adjusted Score Portfolio
  - Custom Portfolio

  The custom portfolio maximise's the scaled Sharpe Ratio, Sortino Ratio and the Sharpe Ratio using Black Litterman returns and covariance, and then adds a penalty term for deviations from the Max Information Ratio Portfolio. This optimiser uses empirical CDF transform for scaling.
  
  I have also included constraint on sectors, with the a maximum of 15% of the portfolio being in a single sector, with the exception of Healthcare, which has a upper limit of 10% and Technology which has a limit of 30%.

* **`Portfolio_Optimisation.py`** – Orchestrates data loading, covariance estimation and optimisation runs, then exports weights, attribution and performance statistics.

  There is also my proprietary function for portfolio constraints to minimise portfolio risk and return forecast errors. Each ticker that has a positive expected return and a positive score is assigned an initial weight value of the s of the square root of the tickers market cap / forecasting standard error.
  
  The sum of all of these values is the calculated and an initial weight of 

$$
\tilde{w}_i
= \frac{\frac{\sqrt{\mathrm{Market Cap}_i}}{\mathrm{SE}_i}}{\sum{\frac{\mathrm{Market Cap}_i}{{SE}_i}}}
$$

  The lower and upper portfolio weight constraints are then given by:


$$
\mathrm{Upper}_i
= \sqrt{\tilde{w}_i} \cdot \frac{\mathrm{score}_i}{\max_i \{\mathrm{score}_i\}},
\qquad
\mathrm{Lower}_i
= \tilde{w}_i \cdot \frac{\mathrm{score}_i}{\max_i \{\mathrm{score}_i\}}.
$$

  These bounds are subject to constraints. I have a minimum value of $$\frac{2}{\text{Money in Portfolio}}$$ constraint on the lower bound and the upper constraint is 10%, with the excepetion of tickers that are in the Healthcare sector which have an upper bound of 2.5%.

## Running the Toolkit

A typical end-to-end workflow is:

1. **Data download** – Run the scripts under `fetch_data/` to gather the latest
   market, fundamental and macro data.
2. **Indicator generation** – Execute the scripts in `indicators/` to compute
   technical and sentiment indicators.
3. **Forecasting** – Produce valuation forecasts using the models in
   `forecasts/`.  Individual models may take time depending on the amount of
   data and the number of Monte‑Carlo simulations.
4. **Portfolio optimisation** – Run
   `python Optimiser/Portfolio_Optimisation.py` to build the covariance matrix
   and produce optimised portfolios.

Intermediate results are written to dated Excel files such as
`Portfolio_Optimisation_Forecast_<date>.xlsx` for transparency.

## Notes

Some modules reference absolute paths tailored to my own machine
and expect input Excel workbooks that are not part of this repository.  Adjust
`config.py` to point at your own data locations before running the scripts.

## License

This project is released under the MIT License.
