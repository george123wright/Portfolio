import pandas as pd
import numpy as np
import statsmodels.api as sm

from data_processing.ratio_data import RatioData
import config


r = RatioData()


def exp_fac_reg(
    tickers = None
) -> pd.DataFrame:
    """
    Estimate an expected-return regression per ticker using index, sector, industry,
    and factor signals. This uses exponentially weighted historical data and is more 
    a guide for current market trends as apposed to a long-term stable prediction.

    Data sources
    ------------
    - Historical panel (from RatioData.exp_factor_data): per-ticker DataFrame with
      at minimum:
        • y = "Ticker Excess Return"
        • Optional regressors, if available in the columns:
          "Index Excess Return", "Sector Return", "Industry Return",
          and factor proxies "MTUM", "QUAL", "SIZE", "USMV", "VLUE".
    - Exponentially Weighted Returns (from RatioData.exp_factors): per-ticker forecasts for:
     
        • "Index", "Sector", "Industry" (used to fill the corresponding regressors)
     
        • Factor forecasts mapped as:
            MTUM <- "Momentum"
            QUAL <- "Quality"
            SIZE <- "Size"
            USMV <- "Volatility"
            VLUE <- "Value"
    
    - Risk-free rate (config.RF): scalar used to convert predicted *excess* return
      into predicted *total* return.

    Model specification
    -------------------
    Let t index time and consider a given ticker. The dependent variable is the
    ticker's excess return:
   
        y_t = r_t^ticker - r_t^f,
   
    where r_t^f is the risk-free rate applicable to the sampling interval used to
    build the historical panel.

    The regressor vector x_t contains a constant and any subset (depending on data
    availability) of:
    
        I_t = "Index Excess Return",
    
        S_t = "Sector Return",
    
        D_t = "Industry Return",
    
        F_t^MTUM, F_t^QUAL, F_t^SIZE, F_t^USMV, F_t^VLUE (factor proxies).

    The linear model is:
       
        y_t = alpha + beta_I * I_t + beta_S * S_t + beta_D * D_t + sum_k beta_k * F_t^k + epsilon_t,
    
    where epsilon_t is an error term with mean zero and variance sigma^2.

    Estimation
    ----------
    Coefficients theta = [alpha, beta_I, beta_S, beta_D, beta_k...]' are estimated
    by OLS:
        theta_hat = argmin_theta sum_t (y_t - x_t' theta)^2.
    The fitted model yields:
        • Coefficient vector: theta_hat.
        • Residual mean squared error: MSE_resid = sigma_hat^2.
        • Coefficient covariance matrix: Cov(theta_hat).

    Out-of-sample prediction
    ------------------------
    A forward regressor vector x_pred is constructed using the exogenous forecasts:
        x_pred = [1,
                  I_pred, S_pred, D_pred,
                  MTUM_pred, QUAL_pred, SIZE_pred, USMV_pred, VLUE_pred]',
    with entries omitted if unavailable for a given ticker. The predicted *excess*
    return is:
        r_excess_pred = x_pred' theta_hat.
    The predicted *total* return adds back the risk-free rate:
        r_total_pred = r_excess_pred + RF,
    where RF = config.RF.

    Prediction uncertainty
    ----------------------
    The variance of the predicted conditional mean is:
        Var_mean = x_pred' Cov(theta_hat) x_pred.
    The variance of a new observation (i.e., including irreducible noise) is:
        Var_obs = Var_mean + sigma_hat^2.
    The standard error reported by this function is the observation-level standard
    error, annualised under the convention of 252 trading days per year:
        SE_annual = sqrt(Var_obs) * sqrt(252).

    Diagnostics and outputs
    -----------------------
    For each ticker, the function returns a row containing:
        • "Returns": r_total_pred (predicted total return for the next period),
        • "SE": SE_annual (annualised one-step-ahead observation standard error),
        • "r2": R^2 from the in-sample OLS fit, defined as
                R^2 = 1 - SSE / SST,
                where SSE is the residual sum of squares and SST is the total sum of squares,
        • beta[c]: fitted coefficient for each included regressor column c.

    Data handling and safeguards
    ----------------------------
    - Tickercode set: if `tickers` is None, uses `config.tickers`.
    - Columns are intersected with what is present in each ticker's DataFrame; any
      missing regressors are dropped for that ticker.
    - A constant term is always included in the regression design matrix.
    - Rows with any NaNs across y and included regressors are dropped.
    - A minimum of 10 observations is required; otherwise a default row with zeros
      is returned for that ticker.
    - If the dependent variable column "Ticker Excess Return" is missing, or the
      regression fails to fit, the function returns zeros for that ticker.
    - Coefficient and covariance reindexing aligns to x_pred; missing entries are
      treated as zero exposure.

    Assumptions and interpretation
    ------------------------------
    - The linear specification assumes a stable contemporaneous relationship between
      excess returns and the chosen regressors over the sample used for estimation.
    - OLS point estimates are unbiased under exogeneity, and the reported
      uncertainties rely on correct specification of the residual variance used by
      Statsmodels (non-robust by default).
    - The standard error is annualised purely by square-root-of-time scaling
      (factor sqrt(252)). This assumes independent, identically distributed
      one-period errors when annualising to a one-year horizon proxy.
    - The predicted return is a one-step-ahead expectation conditional on the input
      forecasts; it is not automatically converted to a specific time scale beyond
      the annualisation of the uncertainty. The scale of "Returns" will match the
      scale of the underlying input forecasts plus RF (e.g., daily vs weekly), so
      interpretation should be consistent with how the historical panel and forward
      signals are constructed.
    """
      
    if tickers is None:
   
        tickers = list(config.tickers)

    factor_data_dict = r.exp_factor_data(
        tickers = tickers
    )
    
    factor_preds_df = r.exp_factors(
        tickers = tickers
    )

    fac_col_to_pred_name = {
        'MTUM': 'Momentum',
        'QUAL': 'Quality',
        'SIZE': 'Size',
        'USMV': 'Volatility',
        'VLUE': 'Value'
    }

    rows = []
    
    tickers = sorted(set(factor_data_dict.keys()) & set(factor_preds_df.index))

    for t in tickers:
       
        df = factor_data_dict[t].copy()
       
        y_col = 'Ticker Excess Return'

        if y_col not in df.columns:
          
            rows.append({
                'Ticker': t, 
                'Returns': 0, 
                'r2': 0, 
                'SE': 0
            })
        
            continue

        index_regressor = ['Index Excess Return']       
     
        ind_sec_reggressors = ['Industry Return', 'Sector Return']  

        regressor_list = []

        if ind_sec_reggressors is not None:

            regressor_list.extend(ind_sec_reggressors)

        if index_regressor is not None:

            regressor_list.extend(index_regressor)

        regressor_list = [c for c in regressor_list if c in df.columns]

        factor_cols = [c for c in df.columns if c in fac_col_to_pred_name]

        X_cols = regressor_list + factor_cols

        if not X_cols:

            rows.append({
                'Ticker': t,
                'Returns': 0,
                'r2': 0, 
                'SE': 0
            })

            continue

        sub = df[[y_col] + X_cols].dropna()

        if len(sub) < 10:

            rows.append({
                'Ticker': t, 
                'Returns': 0, 
                'r2': 0, 
                'SE': 0
            })

            continue

        y = sub[y_col]

        X = sm.add_constant(sub[X_cols], has_constant = 'add')

        try:

            model = sm.OLS(y, X).fit()

        except Exception:

            rows.append({
                'Ticker': t,
                'Returns': 0, 
                'r2': 0, 
                'SE': 0
            })

            continue

        x_pred = pd.Series(0.0, index = X.columns)

        x_pred['const'] = 1.0

        if ind_sec_reggressors is not None:

            if 'Industry' in factor_preds_df.columns and 'Industry Return' in x_pred.index:

                x_pred['Industry Return'] = factor_preds_df.at[t, 'Industry']

            if 'Sector' in factor_preds_df.columns and 'Sector Return' in x_pred.index:

                x_pred['Sector Return'] = factor_preds_df.at[t, 'Sector']

        if index_regressor is not None:

            if 'Index' in factor_preds_df.columns and 'Index Excess Return' in x_pred.index:

                x_pred['Index Excess Return'] = factor_preds_df.at[t, 'Index']

        for fac_col in factor_cols:

            pred_name = fac_col_to_pred_name.get(fac_col)

            if pred_name and pred_name in factor_preds_df.columns and fac_col in x_pred.index:

                x_pred[fac_col] = factor_preds_df.at[t, pred_name]

        betas = model.params.reindex(x_pred.index).fillna(0.0)

        pred_excess = float(betas @ x_pred)

        pred = pred_excess + config.RF

        covb = model.cov_params().reindex(index = x_pred.index, columns = x_pred.index).fillna(0.0)

        x = x_pred.values

        var_mean = float(x @ covb.values @ x)

        se_obs_pred = float(np.sqrt(var_mean + model.mse_resid)) * np.sqrt(252)

        row = {
            'Ticker': t,
            'Returns': pred,
            'SE': se_obs_pred,
            'r2': model.rsquared,
        }

        for c in X_cols:

            row[f'beta[{c}]'] = betas.get(c, np.nan)

        rows.append(row)

    return pd.DataFrame(rows).set_index('Ticker').sort_index()

