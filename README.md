# Portfolio Optimisation Toolkit

## Table of Contents
- [Installation](#Installation)
- [Repository Layout](#Repository-Layout)
- [Data Collection](#Data-Collection)
- [Data Processing](#Data-Processing)
- [Forecast Models](#Forecast-Models)
- [Machine-Learning](#Machine-Learning)
- [Intrinsic Valuation](#Intrinsic-Valuation)
- [Relative Valuation and Factor Models](#Relative-Valuation-and-Factor-Models)
- [Relative Valuation](#Relative-Valuation)
- [Forecast Ensemble and Score](#Forecast-Ensemble-and-Score)
- [Utility Functions](#Utility-Functions)
- [Technical Indicators and Sentiment](#Technical-Indicators-and-Sentiment)
- [Portfolio Optimisation](#Portfolio-Optimisation)
- [Running the Toolkit](#Running-the-Toolkit)
- [Notes](#Notes)
- [License](#License)

This repository hosts a set of Python scripts for constructing equity portfolios using a range of quantitative valuation models and optimisation techniques. The workflow pulls market and fundamental data, generates forecasts with multiple statistical approaches and produces optimised portfolio allocations. The results are exported into Excel workbooks for inspection or further analysis.

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

  Provides convenience methods for currency adjustment, outlier removal and generating forward‑looking KPIs used by the models.
  
* **`macro_data.py`** – Wrapper around macroeconomic series giving easy access to inflation, GDP growth and other regressors.
  
* **`ratio_data.py`** – Loads historical financial ratios and derives growth metrics used for regressions.
  
* **`ind_data_processing.py`** – Cleans the stock screener output and aggregates industry/region level data.

## Forecast Models (`forecasts`)

### Machine-Learning (`machine_learning`):

* **`prophet_model.py`**

Utilises Facebook Prophet with piecewise linear and logistic trends.

Financial and macro regressors extend the additive model, and cross-validation tunes changepoints and seasonality. Weekly seasonal trends are enabled. Daily and yearly seasonality is disabled due to the   substantial noise created.

  The forecast $\(\hat y(t)\)$ is given by
```math
  \hat y(t)
  = g(t)
  + s_w(t)
  + \sum_{j=1}^K \beta_j\,X_j(t)
  + \varepsilon_t,
```
where $g(t)$ is a piecewise-linear trend, $s_w(t)$ is weekly seasonality, $X_j(t)$ are external regressors, and $\varepsilon_t\sim N(0,\sigma^2)$. Using changepoints $\{\tau_\ell\}$ and indicators $a_\ell(t)=\mathbf{1}\{t\ge\tau_\ell\}$ gives:
```math

  g(t)
  = \Bigl(k + \sum_{\ell=1}^L \delta_\ell\,a_\ell(t)\Bigr)\,t
    \;+\;
    \Bigl(m - \sum_{\ell=1}^L \delta_\ell\,\tau_\ell\Bigr),
  \quad
  \delta_\ell\sim N\bigl(0,\sigma_{\delta}^2\bigr).
```
For weekly seasonality, a fourier series of order $N$ with Gaussian priors on the Fourier coeficients is used:
```math
  s_w(t)
  = \sum_{n=1}^N
    \Bigl[
      a_n\cos\!\bigl(2\pi n\,t/7\bigr)
      + b_n\sin\!\bigl(2\pi n\,t/7\bigr)
    \Bigr],
  \quad
  a_n,b_n\sim N(0,\sigma_{\rm seas}^2).
```
Each regressor $X_j(t)$ enters linearly:
```math
  \hat y(t) \supset \beta_j\,X_j(t),
  \quad
  \beta_j\sim N(0,\sigma_{\beta}^2).
```
All regressors are standardized prior to fitting. Using initial window $T_0$, period $\Delta$, horizon $H$ MSE is averaged and used to report RMSE:
```math
  \mathrm{MSE}^{(k)}
  = \frac1{|I_{\text{test}}^{(k)}|}
    \sum_{i\in I_{\text{test}}^{(k)}}(y_i-\hat y_i)^2,
  \quad
  \mathrm{RMSE}=\sqrt{\mathrm{MSE}}.
```
The series from $x_1$ to $x_H$ over $H$ points is given by:
```math
  x_t
  = x_1 + \frac{t-1}{H-1}\,(x_H - x_1),
  \quad t=1,\dots,H.
```
Forecasts over all $(revenue, eps)$ scenarios are computed to give probabilistic price paths. Scenario SE is given by $\sigma_{\text{scen}}/\sqrt{N_{\rm analysts}}$ and is combined with CV uncertainty to give the Total SE as:
```math
\mathrm{SE}_{\rm total} = \sqrt{\sigma_{\text{scen}}^2 + (\mathrm{RMSE})^2}
```

* **`Sarimax.py`**
Fits SARIMAX (Seasonal Autoregressive Integrated Moving Average + Exogenous Variables) time-series models with exogenous macro factors.

Candidate ARIMA (Autoregressive Integrated Moving Average) orders are weighted by AIC (Akaike Information Criterion) to form an ensemble. 

Future macro scenarios are drawn from a VAR (Vector Auto Regressive) process via Cholesky simulation and propagated through the model.

Monte-Carlo simulation is then used to generate correlated draws from the VAR models output

Fit $VAR(p)$ residuals $\varepsilon_t$ and implied covariance $\Sigma_u$. Compute
```math
\alpha
= \frac{\mathrm{tr}\bigl(\widehat\Sigma_\varepsilon\bigr)}
       {\mathrm{tr}\bigl(\Sigma_u\bigr)},
\quad
\widehat\Sigma_\varepsilon
= \frac{1}{T}\sum_t \varepsilon_t\varepsilon_t^\top.
```
For each model $m$ with $\mathrm{AIC}_m$, let $\Delta_m = \mathrm{AIC}_m - \min_j \mathrm{AIC}_j$.  Then
```math
w_m
= \frac{\exp(-\tfrac12\,\Delta_m)}
       {\sum_j \exp(-\tfrac12\,\Delta_j)}.
```
Forecast log-returns $\hat r_{t+1:\,t+H}$ and set
```math
P_{\mathrm{pred}}
= P_t \exp\Bigl(\sum_{h=1}^H \hat r_{t+h}\Bigr).
```
Over $F$ folds,  
```math
\mathrm{RMSE}
= \sqrt{\frac1F\sum_{f=1}^F\bigl(P_{\mathrm{true}}^{(f)}-P_{\mathrm{pred}}^{(f)}\bigr)^2}.
```
$VAR(p)$ dynamics are:
```math
x_{t+h}
= c + \sum_{\ell=1}^p A_\ell\,x_{t+h-\ell} + \varepsilon_{t+h}.
```
Two shock covariances:
```math
\Sigma_w = \alpha\,\Sigma_u,\quad
\Sigma_q = S\,\Sigma_u,
```
with $S$ =shock interval. Draw $\varepsilon\sim N(0,\Sigma_q)$ every $S$ steps, else $N(0,\Sigma_w)$. Use antithetic sampling: $\tilde x^{(j+N/2)} = 2\,x_{\rm last}-x^{(j)}$.


For each macro path:
1. Sample model $m$ with $\Pr(m)=w_m$.  
2. Draw parameters $\beta^*\sim N(\hat\beta_m,\mathrm{Cov}(\hat\beta_m))$.  
3. Forecast $\{\mu_{t+h},\sigma^2_{t+h}\}$, simulate $r_{t+h} = \mu_{t+h} + \sigma_{t+h}\,z_{t+h},\quad z_{t+h}\sim N(0,1)$
4. Propagate $P_{t+h} = P_{t+h-1}\,\exp(r_{t+h})$, then clip $P$ into $[\ell,u]$.
  
* **`lstm.py`** – Builds a recurrent LSTM (Long Short-Term Memory) network on rolling windows of returns and engineered factors.

  Robust scaling, dropout layers and early stopping help regularise the model.

  Bootstrapped datasets yield an ensemble of forecasts for each ticker.

  Macro Forecasts obtained from Trading Economics are used, as well as revenue and eps forecasts that are obtained from Yahoo Finance and Stock Analysis.
  
* **`returns_reg_pred.py`**

Trains a gradient‑boosting regression on engineered features to predict twelve‑month returns.

Hyperparameters are tuned with grid search, and bootstrapped samples produce an ensemble of models.

Macro Forecasts obtained from Trading Economics are used, as well as revenue and eps forecasts that are obtained from Yahoo Finance and Stock Analysis.

The model fits $M$ successive trees:
```math
  F_m(x) = F_{m-1}(x) + \nu\,h_m(x),
  \quad
  F_0(x) = \bar y,
```

where each tree $h_m$ is fit to the residuals $r_i^{(m)} = y_i - F_{m-1}(x_i)$, and $\nu$ is the learning rate. For hyper-parameter selection via time-series cross validation, it searchs over $M\in\{100,200\},\quad \nu\in\{0.05,0.10\}$ using forward-chaining splits. For each setting, the following is computed and the combination with minimising the cross-validated MSE is selected $\mathrm{MSE} = \frac1N\sum_{i=1}^N\bigl(y_i - \hat y_i\bigr)^2$ . $B$ bootstrap samples are drawn, and $F^{(b)}$ is fitted to each, then at any $x$:
```math
  \mathrm{SE}_{\mathrm{boot}}(x)
  = \sqrt{\frac1{B-1}\sum_{b=1}^B\bigl(F^{(b)}(x)-\bar F(x)\bigr)^2},
  \quad
  \bar F(x)=\frac1B\sum_bF^{(b)}(x).
```
Form $P$ scenarios from the Cartesian product of revenue and EPS labels, yielding feature vectors $\{x_j\}_{j=1}^P$.

- Base predictions: $\hat F_j = F(x_j)$.  
- Bootstrap SE: $\mathrm{SE}_{\mathrm{boot},\,j}$.  
- Scenario variance: $\displaystyle \sigma_{\mathrm{scen}}^2 = \frac1P\sum_{j=1}^P(\hat F_j - \bar F)^2$
- Final SE for each scenario:
```math
    \mathrm{SE}_j
    = \sqrt{\mathrm{SE}_{\mathrm{boot},\,j}^2 + \sigma_{\mathrm{scen}}^2}.
```


### Intrinsic Valuation (`intrinsic_value`):

* **`dcf.py`**

Performs discounted cash‑flow valuation to determine the enterprise value given by:
```math
Enterprise Value = \sum_{i=1}^{n-1} \frac{FCFE_i}{\bigl(1 + WACC\bigr)^{\frac{t_i - t_0}{365}}} + \frac{TV}{\bigl(1 + WACC\bigr)^{\frac{t_n - t_0}{365}}}
```
where $t_0$ is todays date, $t_i$ is the date of the i'th cash flow forecast and $t_n$ is the date of the terminal forecast date.
Cash flows are forecast using elastic‑net regression (see `fast_regression.py`) and then discounted.
Monte‑Carlo scenarios for growth generate a distribution of intrinsic values. The scenarios are used to help gauge the uncertainty of the valuation.
  
* **`dcfe.py`**

Similar to `dcf.py` but values equity directly via discounted cash‑flow to equity.  Constrained regression ensures realistic relationships between drivers. Equity value is given by:
```math
Equity Value = \sum_{i=1}^{n-1} \frac{FCFF_i}{\bigl(1 + COE\bigr)^{\frac{t_i - t_0}{365}}} + \frac{TV}{\bigl(1 + COE\bigr)^{\frac{t_n - t_0}{365}}}
```
Monte-Carlo simulation is used for the aformentioned reason.
  
* **`ri.py`**

Implements a residual income model where future book value is grown and excess returns are discounted using the cost of equity. Equity value is given by:
```math
Equity Value = BVPS_0 + \sum_{i=1}^{n-1} \frac{EPS_i - \bigl(COE \cdot BVPS_{i-1} \bigr)}{\bigl(1 + COE\bigr)^{\frac{t_i - t_0}{365}}} + \frac{TV}{\bigl(1 + COE\bigr)^{\frac{t_n - t_0}{365}}}
```
Monte-Carlo simulation is once again used for the aformentioned reason.


### Relative Valuation and Factor Models (`rel_val`):

Provides multiple models blending peer multiples and fundamental data:

* **`pe.py`, `price_to_sales.py`, `pbv.py`, `ev.py`** – Compute valuations using peer multiples such as P/E, P/S, P/BV and EV/Sales based on industry and regional medians as well as the own tickers respective metric.
  
* **`graham_model.py`** – Implements a Graham‑style intrinsic value combining earnings and book value metrics. This does not use 22.5, and instead uses the industry averages.
  
* **`relative_valuation.py`** – Consolidates all relative valuation signals into a single fair value estimate.

* **`relative_valuation_and_capm.py`** – A script to compute the relative value and then the stock price via valuation ratios and analyst earnings and revenue estimates.

  The script also computes factor model forecasts (CAPM, Fama-French 3 factor model and Fama-French 5 factor model) to derive expected returns. Betas are estimated by OLS, factor paths are simulated with VAR, and Black–Litterman views adjust expected market returns.


## Forecast Ensemble and Score

* **`Combination_Forecast.py`**

Fuses all of the model return forecasts and standard errors into a Bayesian ensemble, applying weights and producing an overall score.

Weights are assigned for each models prediction based on the inverse of the standard error or volatility, i.e.

$$
w_i = \frac{\frac{1}{\mathrm{SE}_i^2}}{\sum{\frac{1}{{SE}_i^2}}}
$$

These weights are capped at 10% per model, unless there are not enough valid models, in which case the cap is $\displaystyle \frac{1}{\text{number of valid models}}$.

The score is inspired by the Pitroski F-score.

It includes all 9 of the Pitroski F-Score variables:

  1. Positive Return on Assets $\Rightarrow$ + 1
  2. Positive Operating Cash Flow $\Rightarrow$ + 1
  3. ROA year on year growth $\Rightarrow$ + 1
  4. Operating Cash Flow higher than Net Income $\Rightarrow$ + 1
  5. Negative Long Term Debt year on year Growth $\Rightarrow$ + 1
  6. Current Ratio year on year growth $\Rightarrow$ + 1
  7. No new shares Issued $\Rightarrow$ + 1
  8. Higher year on year Gross Margin $\Rightarrow$ + 1
  9. Higher Asset Turnover Ratio year on year Growth $\Rightarrow$ + 1

I have adapted this in the following way:

  - Negative Return on Assets $\Rightarrow$- 1
  - Return on Assets > Industry Average $\Rightarrow$ + 1
  - Return on Assets < Industry Average $\Rightarrow$ - 1
  - Previous Return on Assets < Current Return on Assets $\Rightarrow$ - 1
  - Previous Current Ratio > Current Ratio $\Rightarrow$ - 1

I then added the following scores relating to financials as well. These were back tested to see the significance.

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

Then I considered Analyst recommendations, to gather a sense of professional sentiment:

  - Strong Buy Recommendation $\Rightarrow$ + 3
  - Hold Recommendation $\Rightarrow$ - 1
  - Sell or Strong Sell Recommendation $\Rightarrow$ - 5
 
I further consider the stock prices movement within the market:

  - Positive Skewness based on last 5 years weekly returns $\Rightarrow$ + 1
  - Sharpe Ratio based on last year weekly returns > 1.0 $\Rightarrow$ + 1
  - Sortin Ratio based on last year weekly returns > 1.0 $\Rightarrow$ + 1
  - Upside Capture Ratio > 1.5 and Downside Capture Ratio < 0.5 based on last 5 years weekly returns with respect to the S&P500 $\Rightarrow$ + 1
  - Upside Capture Ratio > 1.0 and Downside Capture Ratio < 0.5 based on last 5 years weekly returns with respect to the S&P500 $\Rightarrow$ + 1
  - Upside Capture Ratio > 1.0 and Downside Capture Ratio < 1.0 based on last 5 years weekly returns with respect to the S&P500 $\Rightarrow$ + 1
  - Upside Capture Ratio > 1.5 and Downside Capture Ratio < 1.0 based on last 5 years weekly returns with respect to the S&P500 $\Rightarrow$ + 1
  - Downside Capture Ratio > 1.0 and Upside Capture Ratio < Downside Capture Ratio based on last 5 years weekly returns with respect to the S&P500 $\Rightarrow$ - 1
  - Downside Capture Ratio > 1.5 and Upside Capture Ratio < Downside Capture Ratio based on last 5 years weekly returns with respect to the S&P500 $\Rightarrow$ - 1
  - Positive Jensen's Alpha over last 5 years with respect to the S&P500 $\Rightarrow$ + 1
  - Negative Jensens Alpha over last 5 years with respect to the S&P500 $\Rightarrow$ - 1
  - Negative Predicted Jensen's Alpha $\Rightarrow$ - 5
 
I also consider daily sentiment scores from webscraping r/wallstreetbets. This is to capture the sentiment amongst retail investors, which have an increasing importance in influencing the market:

  - Positive Average Sentiment $\Rightarrow$ + 1
  - Negative Average Sentiment $\Rightarrow$ - 1
  - Positive Average Sentiment and over 4 mentions $\Rightarrow$ + 1
  - Negative Average Sentiment and over 4 mentions $\Rightarrow$ - 1
  - Average Sentiment > 0.2 and over 4 mentions $\Rightarrow$ + 1
  - Average Sentiment < 0.2 and over 4 mentions $\Rightarrow$ - 1
  - Average Sentiment > 0.2 and over 10 mentions $\Rightarrow$ + 1
  - Average Sentiment < 0.2 and over 10 mentions $\Rightarrow$ - 1

I then add the scores from the technical buy and sell indicators to these scores.

## Utility Functions (`functions`)

* **`fast_regression.py`**

An elastic‑net solver built with CVXPY used to forecast cash flows in `dcf.py` and `dcfe.py`.

It applies Huber loss and L1 (Lasso) / L2 (Ridge) penalties and performs grid‑search cross‑validation, optionally enforcing accounting sign constraints.

Given data $\(\{(x_i, \quad y_i)\}_{i=1}^n\)$ with $\(x_i \in \mathbb{R}^p\)$, we augment with an intercept by defining  

```math
X = 
\begin{pmatrix}
1 & x_1^\top\\
\vdots & \vdots\\
1 & x_n^\top
\end{pmatrix}
\;\in\;\mathbb{R}^{n\times(p+1)},
\quad
\beta = (\beta_0,\beta_1,\dots,\beta_p)^\top.
```

The residuals are $r_i = \bigl(X \beta\bigr)_i - y_i$ . For a threshold $\(M>0\)$, the Huber loss on a scalar residual $\(r\)$ is
```math
h_M(r) = 
\begin{cases}
\dfrac{1}{2}\,r^2, & |r|\le M,\\[1em]
M\bigl(|r| - \tfrac{1}{2}M\bigr), & |r|>M.
\end{cases}
```
Thus the total data‐fit term is
```math
\mathcal{L}_{\mathrm{huber}}(\beta)
\;=\;
\sum_{i=1}^n h_M\bigl(r_i\bigr).
```
Let $\(\lambda>0\)$ and $\(\alpha\in[0,1]\)$. The elastic‐net penalty with a tiny “ridge‐epsilon” $\(\varepsilon=10^{-8}\)$ added for numerical stability is:
```math
\mathcal{P}(\beta)
=\;
\lambda\!\bigl(\alpha\lVert\beta\rVert_1 + (1-\alpha)\lVert\beta\rVert_2^2\bigr)
\;+\;\varepsilon\,\lVert\beta\rVert_2^2.
```
For the constrained regression, non-negativity is imposed on the coefficients $\beta_j \ge0, j=1,\dots,p$ . This gives the overall convex program as:
```math
\min_{\beta\in\mathbb R^{p+1}}
\quad
\sum_{i=1}^n h_M\bigl((X\beta)_i - y_i\bigr)
\;+\;
\lambda\!\Bigl(\alpha\lVert\beta\rVert_1 + (1-\alpha)\lVert\beta\rVert_2^2\Bigr)
\;+\;\varepsilon\,\lVert\beta\rVert_2^2
\quad
\text{s.t. } \beta_{1:p}\ge0\;\text{(if constrained).}
```
To improve conditioning, features and target are scaled before solving:
1.  Compute
```math
\mu_x = \tfrac1n\sum_i x_i,\quad \sigma_x = \sqrt{\tfrac1n\sum_i (x_i-\mu_x)^2},\quad \mu_y = \tfrac1n\sum_i y_i,\quad \sigma_y = \sqrt{\tfrac1n\sum_i (y_i-\mu_y)^2} 
```
  replacing any zero $\(\sigma_x\)$ or $\(\sigma_y\)$ by 1.
2.  Define
```math
 x_{i}^s = \frac{x_i - \mu_x}{\sigma_x},\qquad y_i^s = \frac{y_i - \mu_y}{\sigma_y}
```
3.  Solve for $\(\beta^s\)$ on $\(\{(x_i^s,y_i^s)\}\)$. Then recover original-scale coefficients:
```math
    \beta_j = \frac{\sigma_y}{\sigma_{x_j}}\,\beta^s_j,\quad
    \beta_0 = \mu_y + \sigma_y\,\beta^s_0 \;-\;\sum_{j=1}^p \beta_j\,\mu_{x_j}
```
Results are searched over triples $\((\alpha,\lambda,M)\)$ by $\(K\)$-fold CV:

1.  Split indices into $\(\{I_{\text{train}}^{(k)},I_{\text{test}}^{(k)}\}_{k=1}^K\)$.
2.  For each $\((\alpha,\lambda,M)\)$ and each fold $k$, fit $\beta^{(k)}$ on the training set, predict $\hat y_i = (X\beta^{(k)})_i\)$ in the test set, and compute the mean-squared error
```math
    \mathrm{MSE}^{(k)}(\alpha,\lambda,M)
    = \frac1{\lvert I_{\text{test}}^{(k)}\rvert}
      \sum_{i\in I_{\text{test}}^{(k)}}(y_i - \hat y_i)^2.
```
3.  Average over folds:
```math
\overline{\mathrm{MSE}}(\alpha,\lambda,M) = \frac1K\sum_{k=1}^K \mathrm{MSE}^{(k)} $
```
4.  Select $\((\alpha^*,\lambda^*,M^*)\)$ minimizing $\(\overline{\mathrm{MSE}}\)$, then re-fit on all data.

* **`cov_functions.py`** – Implements covariance estimators including constant‑correlation and Ledoit–Wolf shrinkage.

  Predicted covariances are derived from multi‑horizon scaling with an extended Stein shrinkage variant.

* **`black_litterman_model.py`**

Implements the Black–Litterman Bayesian update combining equilibrium market returns with subjective views to obtain posterior means and covariances.

Let:
- $n$ = number of assets  
- $\Sigma\in\mathbb{R}^{n\times n}$ = prior covariance  
- $w\in\mathbb{R}^n$ = benchmark (market‐cap) weights  
- $\delta>0$ = risk aversion coefficient  
- $\tau>0$ = scaling factor for the prior covariance  
- $P\in\mathbb{R}^{k\times n}$ = “pick” matrix encoding \(k\) views  
- $q\in\mathbb{R}^k$ = view returns  
- $\kappa>0$ = confidence scalar (higher $\kappa$ ⟶ more confidence)  

```math
\pi \;=\;\delta\,\Sigma\,w
```
```math
\tilde\Sigma \;=\;\tau\,\Sigma
\quad,\quad
\Omega \;=\;\frac{\mathrm{diag}\bigl(P\,\tilde\Sigma\,P^\top\bigr)}{\kappa}
```
where $\Omega\in\mathbb{R}^{k\times k}$ is diagonal. Define $A = P\, \tilde\Sigma\, P^\top + \Omega$ . Then the adjusted expected returns are
```math
\mu_{BL}
\;=\;
\pi
\;+\;
\tilde\Sigma\,P^\top\,A^{-1}\,\bigl(q \;-\; P\,\pi\bigr)
```
with $\mu_{BL}\in\mathbb{R}^n\$ .
```math
\Sigma_{BL}
\;=\;
\Sigma
\;+\;
\tilde\Sigma
\;-\;
\tilde\Sigma\,P^\top\,A^{-1}\,P\,\tilde\Sigma
```


* **`capm.py`**

Helper implementing the CAPM formula:

$$
\quad
\mathbb{E}[R_i]
= R_f + \beta_i \bigl(\mathbb{E}[R_m] - R_f\bigr),
$$


* **`coe.py`** – Calculates the cost of equity per ticker by combining country risk premiums and currency risk with the standard CAPM estimate.

* **`fama_french_3_pred.py` / `fama_french_5_pred.py`**

Estimate expected returns using the Fama–French 3 factor and Fama-French 5 factor models using OLS Betas and simulated future factor values.

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

  
* **`factor_simulations.py`**

Generates future factor realisations by fitting a VAR model and applying Cholesky shocks. These simulated paths feed into the Fama–French forecasts.

Let $x_t\in\mathbb{R}^k$ be the factor vector. Fit
```math
  x_t = c + \sum_{\ell=1}^{p} A_\ell\,x_{t-\ell} + u_t,
  \quad
  u_t \sim \mathcal{N}(0,\Sigma_u),
```
where $p$ is chosen by minimizing AIC. Cholesky factor $L$ of $\Sigma_u$ is then computed by:
```math
  \Sigma_u = L\,L^\top,
  \quad
  u_t = L\,z_t,
  \;z_t\sim\mathcal{N}(0,I_k).
```
Using the last $p$ observations as initialization, $N$ paths are simulated over $H$ steps:
```math
  x_{t+h}^{(j)}
  = c + \sum_{\ell=1}^{p}A_\ell\,x_{t+h-\ell}^{(j)}
    + L\,z_{t+h}^{(j)},
  \quad
  z_{t+h}^{(j)}\sim\mathcal{N}(0,I_k).
```
Let $X_h = [\,x_{t+h}^{(1)},\dots,x_{t+h}^{(N)}]$, then the Mean and Covariance are:
```math
\bar x_h = \frac{1}{N}\sum_{j=1}^N x_{t+h}^{(j)}, \qquad \mathrm{Cov}_h = \frac{1}{N-1}\sum_{j=1}^N (x_{t+h}^{(j)}-\bar x_h)(x_{t+h}^{(j)}-\bar x_h)^\top
```
  
* **`export_forecast.py`** – Writes DataFrames to Excel with conditional formatting and table styling.
  
* **`read_pred_file.py`** – Reads previously generated forecast sheets and updates them with latest market prices.

## Technical Indicators and Sentiment (`indicators`)

* **`technical_indicators.py`**

Calculates technical Buy and Sell stock indicators, scoring each ticker and saving the results to Excel.

These indicators include:
  
  - MACD (Moving Average Convergence/Divergence)
  - RSI (Relative Strength Index) with Buy and Sell thresholds of 30 and 70 respectively over a 14 day period window.
  - EMA (Exponential Moving Average) Crossover Signals with Fast and Slow moving averages of 12 and 26 respectively.
  - Bollinger Signals with a Bollinger Band window of 20 with Bollinger Band standard deviation of 2.
  - Stochastic Signals with Slow and Fast windows of 14 and 3 respectively, and with buy and sell values of 20 and 80 respectively.
  - ATR (Average True Range) Breakout with ATR window of 14, ATR Breakout window of 20 and ATR Multiplier of 1.5.
  - OBV (On-Balance Volume) Divergence with OBV lookback of 20.
  - True Wilder ADX (Average Directional Index) with ADX window and ADX threshold of 14 and 25 respectively.
  - MFI (Money Flow Index) with MFI window of 14 and Buy and Sell thresholds of 20 and 80 respectively.
  - VWAP (Volume Weighted Average Price) with VWAP window of 14

Buy and Sell signals are given a score of ± 1
 
* **`wallstreetbets_scrapping.py`**

Scrapes posts and comments from r/wallstreetbets.

Ticker mentions are analysed with NLTK’s (Natural Language Toolkit) VADER (Valence Aware Dictionary and Sentioment Reasoner) sentiment classifier and aggregated scores are saved.

I have tuned this dictionary to account for relevant slang and market related terms frequently used. For example, "bullish", "buy the dip" and "yolo".


## Portfolio Optimisation (`Optimiser`)

* **`portfolio_functions.py`**

Utility functions for:

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
  - VaR (Value at Risk)
  - VaR Gaussian
  - CVaR (Conditional Value at Risk)
  - Information Ratio
  - Annualised Returns
  - Ucler Index
  - CDaR (Conditional Drawdown at Risk)
  - Jensen's Alpha
  - Capture Ratios
  - Sortino Ratios
  - Calmar Ratio
  - Omega Ratio
  - Modigliani Ratio
  - MAR (Managed Account Reports) Ratio
  - Pain Index
  - Pain Ratio
  - Tail Ratio
  - RAROC (Risk Adjusted Return on Capital)
  - Percentage of Portfolio Positive Streaks
  - Geometric Brownian Motion 
  - Portfolio Simulation 
  - Simulation and Portfolio Metrics Report

* **`portfolio_optimisers.py`**

Implements portfolio optimisers subject to constraints using `scipy.optimize`. These optimisers include:

  - Max Sharpe Portfolio
  - Max Sortino Portfolio
  - Max Information Ratio Portfolio
  - Equal Risk Portfolio
  - Max Sharpe Portfolio using Black Litterman returns and covariance
  - Max Risk Adjusted Score Portfolio
  - Custom Portfolio

The custom portfolio maximise's the scaled Sharpe Ratio, Sortino Ratio and the Sharpe Ratio using Black Litterman returns and covariance, and then adds a penalty term for deviations from the Max Information Ratio Portfolio. This optimiser uses empirical CDF transform for scaling. The objective function can be written as:

```math
\max \Biggl[\gamma_{\mathrm{Sharpe}} \Bigl(\frac{\mathbb{E}[R_i] - R_f}{\sigma_i}\Bigr) \quad + \quad \gamma_{\mathrm{Sortino}} \Bigl(\frac{\mathbb{E}[R_i] - R_f}{\mathrm{DD}_i}\Bigr) \quad + \quad \gamma_{\mathrm{Sharpe,BL}} \Bigl(\frac{\mathbb{E}[R_i]^{\mathrm{BL}} - R_f}{\sigma_i^{\mathrm{BL}}}\Bigr) \quad - \quad \gamma_{\mathrm{Information}} \sum_{i}\Bigl(w_i - w_{i,\mathrm{MIR}}\Bigr)^{2}\Biggr]
```
 
I have also included constraint on sectors, with the a maximum of 15% of the portfolio being in a single sector, with the exception of Healthcare, which has a upper limit of 10% and Technology which has a limit of 30%.

* **`Portfolio_Optimisation.py`**

Orchestrates data loading, covariance estimation and optimisation runs, then exports weights, attribution and performance statistics.

There is also my proprietary function for portfolio constraints to minimise portfolio risk and return forecast errors. Each ticker that has a positive expected return and a positive score is assigned an initial weight value of the s of the square root of the tickers market cap / forecasting standard error.
  
The lower and upper portfolio weight constraints are then given by:


$$
\mathrm{Lower}_i
= \frac{\frac{\sqrt{\mathrm{Market Cap}_i}}{\mathrm{SE}_i}}{\sum{\frac{\sqrt{\mathrm{Market Cap}_i}}{{SE}_i}}} \cdot \frac{\mathrm{score}_i}{\max \mathrm{score}},
\qquad
\mathrm{Upper}_i
= \sqrt{\frac{\frac{\sqrt{\mathrm{Market Cap}_i}}{\mathrm{SE}_i}}{\sum{\frac{\sqrt{\mathrm{Market Cap}_i}}{{SE}_i}}}} \cdot \frac{\mathrm{score}_i}{\max \mathrm{score}}.
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
