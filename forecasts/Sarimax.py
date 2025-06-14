import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import VAR
from joblib import Parallel, delayed

from export_forecast import export_results
from macro_data3 import MacroData

BASE_REGRESSORS = ['Interest', 'Cpi', 'Gdp', 'Unemp']
SHOCK_INTERVAL = 13   
N_SIMS = 1000
half = N_SIMS // 2
CANDIDATE_ORDERS = [(1,1,0), (0,1,1), (1,1,1)]


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

def compute_alpha_from_residuals(vr: VAR) -> float:
    """
    Compute α = trace(Σ̂_ε) / trace(Σ_u)
     - Σ̂_ε: empirical covariance of one‐step VAR residuals
     - Σ_u : vr.sigma_u from the fitted model
    """
    resid = vr.resid
    cov_eps = np.cov(resid, rowvar=False, ddof=0)
    cov_u   = vr.sigma_u
    alpha    = np.trace(cov_eps) / np.trace(cov_u)
    return alpha

def prepare_sarimax_ensemble(
    df_model: pd.DataFrame,
    regressors: List[str]
) -> Tuple[List[SARIMAX], np.ndarray]:
    """
    Fit SARIMAX models for each candidate order, return list of fit results and sampling probabilities.
    """
    fits = []
    aics = []
    y = df_model['y']
    exog = df_model[regressors]

    for order in CANDIDATE_ORDERS:
        try:
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=(0,0,0,0),
                exog=exog,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                fit = model.fit(disp=False, method='lbfgs', maxfun=5000)
            fits.append(fit)
            aics.append(fit.aic)
        except Exception:
            continue

    if not fits:
        raise RuntimeError("No SARIMAX fits succeeded")

    aics = np.array(aics)
    delta = aics - aics.min()
    weights = np.exp(-0.5 * delta)
    probs = weights / weights.sum()

    return fits, probs


def evaluate_sarimax_cv(
    df_scaled: pd.DataFrame,
    regressors: List[str],
    fits: List,
    probs: np.ndarray,
    n_splits: int = 3,
    horizon: int = 52
) -> float:
    """
    Cross‐validate multi‐step SARIMAX ensemble forecasts (uses mean forecasts only).
    Returns RMSE of price forecasts horizon steps ahead.
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

        fit = fits[np.argmin([f.aic for f in fits])]
        pred = fit.get_forecast(steps=horizon, exog=exog_h)
        r_pred = pred.predicted_mean.values

        r_true = df_scaled['y'].iloc[
            train_end + 1 : train_end + 1 + horizon
        ].values

        P_pred = P0 * np.exp(np.sum(r_pred))
        P_true = P0 * np.exp(np.sum(r_true))
        sq_errors.append((P_true - P_pred) ** 2)

    return np.sqrt(np.mean(sq_errors)) if sq_errors else np.nan


def simulate_macro_scenarios(
    vr: VAR,
    last_vals: np.ndarray,
    steps: int,
    shock_interval: int = SHOCK_INTERVAL,
    alpha: Optional[float] = None,
) -> np.ndarray:
    
    if alpha is None:
        alpha = compute_alpha_from_residuals(vr)
        
    Su = vr.sigma_u
    k = Su.shape[0]
    cov = (Su + Su.T) / 2
    cov_w = alpha * cov
    cov_q = shock_interval * cov
    jitter = 1e-6 * np.eye(k)
    L_w = np.linalg.cholesky(cov_w + jitter)
    L_q = np.linalg.cholesky(cov_q + jitter)

    sims = np.zeros((N_SIMS, steps, k), float)
    for i in range(half):
        hist = list(last_vals.copy())
        for t in range(steps):
            y_hat = sum(vr.coefs[j] @ hist[-j-1] for j in range(vr.k_ar))
            if (t % shock_interval) == 0:
                eps = L_q @ np.random.randn(k)
            else:
                eps = L_w @ np.random.randn(k)
            nxt = y_hat + eps
            sims[i, t] = nxt
            hist.append(nxt)
    for i in range(half, N_SIMS):
        sims[i] = 2*last_vals - sims[i - half]  
    return sims


def simulate_price_path(
    macro_path: np.ndarray,
    fits: List,
    fit_probs: np.ndarray,
    scaler,
    future_dates: pd.DatetimeIndex,
    cp: float,
    regressors: List[str],
    lb: float,
    ub: float,
) -> np.ndarray:
    idx = np.random.choice(len(fits), p=fit_probs)
    fit = fits[idx]
    orig_params = fit.params.copy()

    beta_sim = np.random.multivariate_normal(orig_params, fit.cov_params())
    fit.params = beta_sim

    try:
        fb = pd.DataFrame({'ds': future_dates})
        for j, name in enumerate(BASE_REGRESSORS):
            fb[name] = macro_path[:, j]
        fb['shock_dummy'] = ((np.arange(len(future_dates)) % SHOCK_INTERVAL) == 0).astype(float)

        exog_macro = scaler.transform(fb[BASE_REGRESSORS].values)
        exog = np.hstack([exog_macro, fb[['shock_dummy']].values])

        fc = fit.get_forecast(steps=len(future_dates), exog=exog)
        mu  = fc.predicted_mean
        var = fc.var_pred_mean
        r_sim = mu + np.random.randn(len(mu)) * np.sqrt(var)

        path = cp * np.exp(np.cumsum(r_sim))
        return np.clip(path, lb, ub)

    finally:
        fit.params = orig_params


def main() -> None:
    macro = MacroData()
    r = macro.r
    tickers = r.tickers
    forecast_period = 52
    cv_splits = 3

    close = r.weekly_close
    latest_prices = r.last_price
    analyst = r.analyst
    
    lb = 0.2 * latest_prices
    ub = 5.0 * latest_prices

    logger.info("Importing macro history …")
    raw_macro = macro.assign_macro_history_non_pct().reset_index()
    raw_macro = raw_macro.rename(
        columns={'year':'ds'} if 'year' in raw_macro.columns
        else {raw_macro.columns[1]:'ds'}
    )
    raw_macro['ds'] = raw_macro['ds'].dt.to_timestamp()
    country_map = {t: str(c) for t,c in zip(analyst.index, analyst['country'])}
    raw_macro['country'] = raw_macro['ticker'].map(country_map)
    macro_clean = raw_macro[['ds','country'] + BASE_REGRESSORS].dropna()

    logger.info("Simulating macro scenarios…")
    country_paths: Dict[str, Optional[np.ndarray]] = {}
    for ctry, dfc in macro_clean.groupby('country'):
        dfm_raw = (
            dfc.set_index('ds')[BASE_REGRESSORS]
               .sort_index().resample('W').mean()
               .ffill().dropna()
        )
        dfm = np.log(dfm_raw).diff().dropna()
        vr = VAR(dfm).fit(maxlags=1)
        if vr.k_ar < 1:
            country_paths[ctry] = None
            continue
        last_vals = dfm.values[-vr.k_ar:]
        sims = simulate_macro_scenarios(vr, last_vals, forecast_period)
        country_paths[ctry] = sims

    results = {tk:{} for tk in tickers}
    logger.info("Fitting SARIMAX ensemble…")

    for tk in tickers:
        cp = latest_prices.get(tk, np.nan)
        if pd.isna(cp):
            logger.warning("No price for %s, skipping", tk); continue

        dfp = pd.DataFrame({'ds': close.index, 'price': close[tk].values})
        dfp['y'] = np.log(dfp['price']).diff(); dfp.dropna(inplace=True)
        tm = raw_macro[raw_macro['ticker']==tk][['ds']+BASE_REGRESSORS]
        dfm = pd.merge_asof(dfp.sort_values('ds'), tm.sort_values('ds'), on='ds')
        dfm = dfm.set_index('ds').asfreq('W-SUN').ffill().bfill().dropna()

        df_price = dfm[['price']].copy(); df_price['y']=np.log(dfm['price']).diff(); df_price.dropna(inplace=True)
        df_macro = np.log(dfm[BASE_REGRESSORS]).diff().dropna()
        df_comb = pd.concat([df_price, df_macro], axis=1).dropna()
        if df_comb.empty:
            logger.warning("Insufficient data for %s, skipping", tk); continue

        scaler = StandardScaler().fit(df_comb[BASE_REGRESSORS].values)
        df_scaled = df_comb.copy()
        df_scaled[BASE_REGRESSORS] = scaler.transform(df_comb[BASE_REGRESSORS].values)
        df_scaled['shock_dummy'] = ((np.arange(len(df_scaled)) % SHOCK_INTERVAL) == 0).astype(float)
        regressors = BASE_REGRESSORS + ['shock_dummy']

        fits, probs = prepare_sarimax_ensemble(df_scaled, regressors)
        rmse = evaluate_sarimax_cv(df_scaled, regressors, fits, probs, n_splits=cv_splits, horizon=forecast_period)

        last_date = df_scaled.index.max()
        future_dates = pd.date_range(start=last_date+pd.Timedelta(weeks=1), periods=forecast_period, freq='W-SUN')
        country = country_map.get(tk)
        macro_sims = country_paths.get(country)
        if macro_sims is None:
            last_obs = dfm[BASE_REGRESSORS].iloc[-1].values
            macro_sims = np.stack([np.tile(last_obs, (forecast_period,1))]*N_SIMS)

        price_sims = Parallel(n_jobs=-1)(
            delayed(simulate_price_path)(
                macro_sims[i], fits, probs, scaler,
                future_dates, cp, regressors,
                lb_tk := lb[tk], ub_tk := ub[tk]
            )
            for i in range(N_SIMS)
        )
        price_sims = np.vstack(price_sims)

        final = price_sims[:,-1]
        rets = final/cp - 1
        results[tk] = {
            'low': final.min(),
            'avg': final.mean(),
            'high': final.max(),
            'returns': rets.mean(),
            'se': np.sqrt((rmse/cp)**2 + rets.std(ddof=1)**2)
        }
        logger.info("%s -> low %.2f, avg %.2f, high %.2f, se %.2f", tk, results[tk]['low'], results[tk]['avg'], results[tk]['high'], results[tk]['se'])

    df_out = pd.DataFrame({
        'Ticker': tickers,
        'Current Price': [latest_prices.get(t) for t in tickers],
        'Avg Price': [results[t]['avg'] for t in tickers],
        'Low Price': [results[t]['low'] for t in tickers],
        'High Price': [results[t]['high'] for t in tickers],
        'Returns': [(results[t]['avg']/latest_prices.get(t)-1) for t in tickers],
        'SE': [results[t]['se'] for t in tickers]
    }).set_index('Ticker')

    export_results({'SARIMAX Monte Carlo': df_out})
    logger.info("Run completed.")

if __name__ == '__main__':
    main()
