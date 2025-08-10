import pandas as pd
import numpy as np
import statsmodels.api as sm
from data.processing.ratio_data import RatioData
import config


r = RatioData()


def exp_fac_reg(
    tickers = None
) -> pd.DataFrame:
   
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
          
            rows.append({'Ticker': t, 'Returns': 0, 'r2': 0, 'SE': 0})
        
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

            rows.append({'Ticker': t, 'Returns': 0, 'r2': 0, 'SE': 0})

            continue

        sub = df[[y_col] + X_cols].dropna()

        if len(sub) < 10:

            rows.append({'Ticker': t, 'Returns': 0, 'r2': 0, 'SE': 0})

            continue

        y = sub[y_col]

        X = sm.add_constant(sub[X_cols], has_constant='add')

        try:

            model = sm.OLS(y, X).fit()

        except Exception:

            rows.append({'Ticker': t, 'Returns': 0, 'r2': 0, 'SE': 0})

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

pred = exp_fac_reg(['ML.PA'])
print(pred)
