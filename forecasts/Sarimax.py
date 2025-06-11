import logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import VAR
from joblib import Parallel, delayed

from export_forecast import export_results
from ratio_data import RatioData
from macro_data3 import MacroData

REGRESSORS = ['Interest', 'Cpi', 'Gdp', 'Unemp']
SMALL_FLOOR = 1e-7
N_SIMS = 1000
half = N_SIMS // 2


def configure_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)
    return logger

logger = configure_logger()

def prepare_sarimax_model(
    df_model: pd.DataFrame,
    regressors: List[str]
) -> SARIMAX:
    y = df_model['y']

    candidates = [(1,1,0), (0,1,1), (1,1,1)]
    best_aic = np.inf
    best_order = candidates[0]
    warm_init = None

    for order in candidates:
        try:
            m = SARIMAX(y,
                        order=order,
                        seasonal_order=(0,0,0,0),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
            fit_q = m.fit(disp=False, method='lbfgs', maxfun=1000)
            if fit_q.aic < best_aic:
                best_aic, best_order = fit_q.aic, order
                warm_init = fit_q.params
        except Exception:
            continue

    p_star, d_star, q_star = best_order

    model = SARIMAX(y,
                    order=(p_star, d_star, q_star),
                    seasonal_order=(0,0,0,0),
                    exog=df_model[regressors],
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        sp = warm_init
        if sp is None or len(sp) != len(model.start_params):
            sp = None
        fit_final = model.fit(
            disp=False,
            method='lbfgs',
            maxfun=10000,
            factr=1e7,
            start_params=sp
        )

    lb = acorr_ljungbox(fit_final.resid, lags=range(1,13), return_df=True)
    if (lb['lb_pvalue'] < 0.05).any():
        p2, d2, q2 = p_star, d_star, q_star
        if p2 < 3:
            p2 += 1
        elif q2 < 3:
            q2 += 1
        else:
            return fit_final

        model2 = SARIMAX(y,
                         order=(p2, d2, q2),
                         seasonal_order=(0,0,0,0),
                         exog=df_model[regressors],
                         enforce_stationarity=False,
                         enforce_invertibility=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            try:
                fit2 = model2.fit(
                    disp=False,
                    method='lbfgs',
                    maxfun=10000,
                    factr=1e7,
                    error_dist='t',
                    dist_kwargs={'df':5},
                    start_params=fit_final.params
                )
                if fit2.aic < fit_final.aic:
                    return fit2
            except Exception:
                pass

    return fit_final


def evaluate_sarimax_cv(
    df_scaled: pd.DataFrame,
    regressors: List[str],
    fit,
    n_splits: int = 3,
    horizon: int = 52
) -> float:
    """
    Cross‐validate a multi‐step (horizon) SARIMAX forecast.
    Returns the RMSE of price predictions  horizon steps ahead.
    """
    N = len(df_scaled)
    if N <= horizon:
        return np.nan

    max_splits = max(1, N - horizon - 1)
    n_splits = min(n_splits, max_splits)
    fold_size = (N - horizon) // (n_splits + 1)
    if fold_size < 1:
        return np.nan

    sq_errors = []
    for k in range(n_splits):
        train_end = (k+1) * fold_size
        P0 = df_scaled['price'].iloc[train_end]

        exog_h = df_scaled[regressors].iloc[
            train_end + 1 : train_end + 1 + horizon
        ]
        if len(exog_h) < horizon:
            break

        pred = fit.get_forecast(steps=horizon, exog=exog_h)
        r_pred = pred.predicted_mean.values    

        r_true = df_scaled['y'].iloc[
            train_end + 1 : train_end + 1 + horizon
        ].values

        P_pred = P0 * np.exp(np.sum(r_pred))
        P_true = P0 * np.exp(np.sum(r_true))

        sq_errors.append((P_true - P_pred) ** 2)

    return np.sqrt(np.mean(sq_errors)) if sq_errors else np.nan


def simulate_price_path(
    macro_path: np.ndarray,
    fit,                
    scaler,          
    future_dates: pd.DatetimeIndex,
    cp: float,         
    param_mean: np.ndarray,
    param_cov: np.ndarray
) -> np.ndarray:
    """
    Simulate one price path:
      1) draw regression coefficients from N(param_mean, param_cov)
      2) swap into `fit.params`
      3) call get_forecast(...) to get log-returns,
      4) compound to prices.
    """
    beta_sim = np.random.multivariate_normal(param_mean, param_cov)

    orig = fit.params.copy()
    fit.params = beta_sim
    try:
        fb = pd.DataFrame({'ds': future_dates})
        fb[REGRESSORS] = macro_path
        exog = scaler.transform(fb[REGRESSORS].values)
        pred = fit.get_forecast(steps=len(future_dates), exog=exog)
        r_pred = pred.predicted_mean.values

        return cp * np.exp(np.cumsum(r_pred))

    finally:
        fit.params = orig

def simulate_macro_scenarios(
    vr: VAR,
    last_vals: np.ndarray,
    steps: int,
    shock_interval: int = 13 
) -> np.ndarray:
    """
    Simulate n_sims VAR paths using antithetic variates.
    Returns array (n_sims, steps, k).
    """
    Su = vr.sigma_u
    k = Su.shape[0]

    cov = (Su + Su.T) / 2 + 1e-6 * np.eye(k)
    L = np.linalg.cholesky(cov)
    
    sims = np.zeros((N_SIMS, steps, k), float)

    for i in range(half):
        hist = list(last_vals.copy())
        for t in range(steps):
            y_hat = sum(vr.coefs[j] @ hist[-j-1] for j in range(vr.k_ar))
            if t % shock_interval == 0:
                z   = np.random.randn(k)
                eps = L @ z
            else:
                eps = np.zeros(k)
            nxt = y_hat + eps
            sims[i, t] = nxt
            hist.append(nxt)            

    for i in range(half):
        hist = list(last_vals.copy())
        for t in range(steps):
            y_hat = sum(vr.coefs[j] @ hist[-j-1] for j in range(vr.k_ar))
            eps = -(sims[i, t] - y_hat)
            nxt = y_hat + eps
            sims[half + i, t] = nxt
            hist.append(nxt)

    return sims


def main() -> None:
    macro = MacroData()
    r = macro.r
    tickers = r.tickers
    forecast_period = 52
    cv_splits = 3
    close = r.weekly_close
    latest_prices = r.last_price
    analyst = r.analyst

    logger.info("Importing macro history …")
    raw_macro = macro.assign_macro_history_non_pct().reset_index()

    raw_macro = raw_macro.rename(
        columns={'year':'ds'} if 'year' in raw_macro.columns
        else {raw_macro.columns[1]:'ds'}
    )
    raw_macro['ds'] = (
        raw_macro['ds'].dt.to_timestamp()
        if isinstance(raw_macro['ds'].dtype, pd.PeriodDtype)
        else pd.to_datetime(raw_macro['ds'])
    )
    country_map = {t: str(c) for t,c in zip(analyst.index, analyst['country'])}
    raw_macro['country'] = raw_macro['ticker'].map(country_map)
    macro_clean = raw_macro[['ds','country'] + REGRESSORS].dropna()

    logger.info("Simulating macro scenarios for each country …")
    country_macro_paths: Dict[str, Optional[np.ndarray]] = {}
    for ctry, dfc in macro_clean.groupby('country'):
        dfm_raw = (
            dfc.set_index('ds')[REGRESSORS]
            .sort_index()
            .resample('W').mean()
            .ffill()
            .dropna()
        )

        dfm = np.log(dfm_raw).diff().dropna()
        vr = VAR(dfm).fit(maxlags=1)
        if vr.k_ar < 1:
            country_macro_paths[ctry] = None
            continue

        last_vals = dfm.values[-vr.k_ar:]

        sims = simulate_macro_scenarios(
            vr=vr,
            last_vals=last_vals,
            steps=forecast_period
        ) 

        country_macro_paths[ctry] = sims
    
    results = {tk: {} for tk in tickers}

    logger.info("Running SARIMAX Monte Carlo forecasts …")
    for tk in tickers:
        logger.info("Ticker: %s", tk)
        cp = latest_prices.get(tk, np.nan)
        if pd.isna(cp):
            logger.warning("No price for %s, skipping", tk)
            continue

        dfp = pd.DataFrame({'ds': close.index, 'price': close[tk].values})
        dfp['ds'] = pd.to_datetime(dfp['ds'])
        dfp['y'] = np.log(dfp['price']).diff()
        dfp = dfp.dropna(subset=['y']).reset_index(drop=True)

        tm = raw_macro[raw_macro['ticker'] == tk][['ds'] + REGRESSORS]
        dfm = pd.merge_asof(
            dfp.sort_values('ds'),
            tm.sort_values('ds'),
            on='ds',
            direction='backward'
        )
        dfm = (
            dfm.set_index('ds')
               .sort_index()
               .asfreq('W-SUN')     
               .ffill()
               .bfill()
        )
        dfm = dfm.dropna(subset=['price','y'] + REGRESSORS)

        df_price = dfm[['price']].copy()
        df_price['y'] = dfm['y']                   
        df_macro     = dfm[REGRESSORS].copy()       

        df_macro_ld  = np.log(df_macro).diff().dropna()

        df_comb = pd.concat([df_price, df_macro_ld], axis=1).dropna()
        if df_comb.empty:
            logger.warning("Insufficient data for %s, skipping", tk)
            continue

        scaler = StandardScaler().fit(df_comb[REGRESSORS].values)
        df_scaled = df_comb.copy()
        df_scaled[REGRESSORS] = scaler.transform(df_comb[REGRESSORS].values)

        try:
            fit = prepare_sarimax_model(df_scaled, REGRESSORS)
        except Exception as e:
            logger.error("SARIMAX failed for %s: %s", tk, e)
            continue
        
        param_mean = fit.params.copy()
        param_cov  = fit.cov_params()   
        
        rm_price = evaluate_sarimax_cv(
            df_scaled,
            REGRESSORS,
            fit,
            n_splits=cv_splits,
            horizon=forecast_period
        )
        
        last_date = df_scaled.index.max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=forecast_period,
            freq='W-SUN'
        )

        country = country_map.get(tk)
        macro_sims = country_macro_paths.get(country)
        if macro_sims is None:
            last_obs = dfm[REGRESSORS].iloc[-1].values
            det_path = np.tile(last_obs, (forecast_period, 1))
            macro_sims = np.stack([det_path] * N_SIMS)


        price_sims = Parallel(n_jobs=-1)(
            delayed(simulate_price_path)(
                macro_sims[i],
                fit,
                scaler,
                future_dates,
                cp,
                param_mean,
                param_cov
            )
            for i in range(N_SIMS)
        )
        price_sims = np.vstack(price_sims)
        
        final_sims = price_sims[:, -1]
        
        min_price = final_sims.min()
        max_price = final_sims.max()
        avg_price = final_sims.mean()
        rets = final_sims / cp - 1
        avg_ret = rets.mean()
        rets_std = rets.std(ddof=1)

        se = np.sqrt(((rm_price / cp)**2) + (rets_std**2))


        results[tk].update({'low': min_price, 'avg': avg_price, 'high': max_price, 'returns': avg_ret, 'se': se})
        
        logger.info(
            "Ticker: %s, Low: %.2f, Avg: %.2f, High: %.2f, Returns: %.4f, SE: %.4f",
            tk, min_price, avg_price, max_price, avg_ret, se
        )
        

    df_out = pd.DataFrame({
        'Ticker': tickers,
        'Current Price': [latest_prices.get(t, np.nan) for t in tickers],
        'Avg Price': [results[t].get('avg', np.nan) for t in tickers],
        'Low Price': [results[t].get('low', np.nan) for t in tickers],
        'High Price': [results[t].get('high', np.nan) for t in tickers],
        'Returns': [results[t].get('avg', np.nan) / latest_prices.get(t, np.nan) - 1
                    if latest_prices.get(t, np.nan)
                    else np.nan
                    for t in tickers],
        'SE': [results[t].get('rmse', np.nan) for t in tickers]
    }).set_index('Ticker')

    export_results({'SARIMAX Monte Carlo': df_out})
    logger.info("Monte Carlo SARIMAX run completed.")


if __name__ == '__main__':
    main()
