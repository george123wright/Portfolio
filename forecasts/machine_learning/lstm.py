from __future__ import annotations

import logging
import random
import sys
import gc
import psutil
import os
import time
import faulthandler
from typing import List, Dict, Optional, Tuple

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.api import VAR
from numpy.lib.stride_tricks import as_strided

from data_processing.financial_forecast_data import FinancialForecastData
import config

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

REV_KEYS = ["low_rev_y", "avg_rev_y", "high_rev_y", "low_rev", "avg_rev", "high_rev"]
EPS_KEYS = ["low_eps_y", "avg_eps_y", "high_eps_y", "low_eps", "avg_eps", "high_eps"]
SCENARIOS = [(r, e) for r in REV_KEYS for e in EPS_KEYS]

TECHNICAL_REGRESSORS = ['MA52_ret']
MACRO_REGRESSORS     = ["Interest", "Cpi", "Gdp", "Unemp"]
FIN_REGRESSORS       = ["Revenue", "EPS (Basic)"]
NON_FIN_REGRESSORS   = MACRO_REGRESSORS + TECHNICAL_REGRESSORS
ALL_REGRESSORS       = NON_FIN_REGRESSORS + FIN_REGRESSORS

HIST_WINDOW = 52
HORIZON = 52
SEQUENCE_LENGTH = HIST_WINDOW + HORIZON

w = HORIZON // 4
rem = HORIZON - 4*w
repeats_quarter = np.array([w, w, w, w+rem], dtype=int)

SMALL_FLOOR = 1e-6
L2_LAMBDA = 1e-4
PATIENCE = 5
EPOCHS = 30
BATCH = 64
N_SIMS = 100


def configure_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s"))
        logger.addHandler(h)
    return logger

logger = configure_logger()

def make_scaled_arrays(
    df: pd.DataFrame,
    regs: List[str]
) -> Tuple[np.ndarray, np.ndarray, RobustScaler, RobustScaler, np.ndarray, np.ndarray]:
    y = df["y"].to_numpy(np.float32)
    y_safe = np.maximum(y, SMALL_FLOOR)
    log_y   = np.log(y_safe)
    log_ret = np.concatenate([[0.0], np.diff(log_y)]).astype(np.float32)

    R       = df[regs].to_numpy(np.float32)
    R_safe  = np.maximum(R, SMALL_FLOOR)
    delta_R = np.diff(np.log(R_safe), axis=0).astype(np.float32)

    scaler_ret = RobustScaler().fit(log_ret[1:].reshape(-1,1))
    scaler_ret.scale_[scaler_ret.scale_ < SMALL_FLOOR] = SMALL_FLOOR
    scaled_ret = np.concatenate([[0.0],
        ((log_ret[1:] - scaler_ret.center_) / scaler_ret.scale_).ravel()
    ])

    scaler_reg = RobustScaler().fit(delta_R)
    scaler_reg.scale_[scaler_reg.scale_ < SMALL_FLOOR] = SMALL_FLOOR
    scaled_reg = ((delta_R - scaler_reg.center_) / scaler_reg.scale_).astype(np.float32)

    return (
        scaled_ret,
        scaled_reg,
        scaler_reg,
        scaler_ret,
        delta_R.min(axis=0),
        delta_R.max(axis=0),
    )


def make_windows(
    ret: np.ndarray,
    reg: np.ndarray,
    hist: int,
    hor: int
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(ret) - 1
    N = T - (hist + hor) + 1
    if N <= 0:
        return np.zeros((0, hist+hor, reg.shape[1]+1)), np.zeros((0, hor))

    def strided(a, L):
        s = a.strides[0]
        return as_strided(a, (N, L), (s, s), writeable=False)

    pr = strided(ret[:-1], hist)
    fr = strided(ret[hist:-1], hor)
    pR = as_strided(reg,   (N, hist, reg.shape[1]),
                   (reg.strides[0], reg.strides[0], reg.strides[1]), writeable=False)
    fR = as_strided(reg[hist:], (N, hor, reg.shape[1]),
                   (reg.strides[0], reg.strides[0], reg.strides[1]), writeable=False)

    X = np.zeros((N, hist+hor, 1 + reg.shape[1]), np.float32)
    X[:, :hist, 0] = pr
    X[:, :hist, 1:] = pR
    X[:, hist:, 1:] = fR

    return X, fr.copy().astype(np.float32)


def init_worker(
    _price_rec,
    _macro_rec,
    _country_slices,
    _country_var_results,
    _fd_rec_dict,
    _latest_price,
    _analyst,
    _ticker_country,
    _next_fc,
    _next_macro
):
    global price_rec, macro_rec, country_slices, country_var_results
    global fd_rec_dict, latest_price, analyst, ticker_country, next_fc, next_macro
    price_rec = _price_rec
    macro_rec = _macro_rec
    country_slices = _country_slices
    country_var_results = _country_var_results
    fd_rec_dict = _fd_rec_dict
    latest_price = _latest_price
    analyst = _analyst
    ticker_country = _ticker_country
    next_fc = _next_fc
    next_macro = _next_macro


def forecast_one(ticker: str) -> Tuple[str, float, float, float, float, float]:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    tf.config.optimizer.set_jit(True)
    tf.random.set_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    def make_tf_dataset(X, y, batch=BATCH, shuffle=False, buffer=None, repeat=False):
        ds = tf.data.Dataset.from_tensor_slices(
            (X.astype(np.float32), y.astype(np.float32))
        )
        if shuffle:
            ds = ds.shuffle(buffer or len(X), seed=SEED)
        if repeat:
            ds = ds.repeat()
        ds = ds.batch(batch)
        ds = ds.map(cast_xy, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    def build_model(n_reg: int) -> Model:
        inp = Input((SEQUENCE_LENGTH, 1 + n_reg), dtype="float32")
        x   = LSTM(64, return_sequences=True,
                   kernel_regularizer=l2(L2_LAMBDA),
                   recurrent_regularizer=l2(L2_LAMBDA),
                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                   recurrent_initializer=tf.keras.initializers.Orthogonal(seed=SEED))(inp)
        x   = Dropout(0.1, seed=SEED)(x)
        x   = LSTM(32,
                   kernel_regularizer=l2(L2_LAMBDA),
                   recurrent_regularizer=l2(L2_LAMBDA),
                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                   recurrent_initializer=tf.keras.initializers.Orthogonal(seed=SEED))(x)
        x   = Dropout(0.1, seed=SEED)(x)
        out = Dense(HORIZON, kernel_regularizer=l2(L2_LAMBDA))(x)
        model = Model(inp, out)
        model.compile(
            optimizer=Adam(5e-4),
            loss=Huber(delta=1.0),
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )
        return model

    CALLBACKS = [
        EarlyStopping("loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau("loss", patience=PATIENCE, factor=0.5, min_lr=SMALL_FLOOR),
    ]
    
    try:
        seed = SEED ^ (hash(ticker) & 0xFFFFFFFF)
        rng = np.random.default_rng(seed)
        logger.info(
            "%s: RAM usage %.1f GB / %.1f GB",
            ticker,
            psutil.Process().memory_info().rss / (1024**3),
            psutil.virtual_memory().total   / (1024**3),
        )

        logger.info("Ticker %s", ticker)
        cur_p = latest_price.get(ticker, np.nan)
        if np.isnan(cur_p):
            return ticker, *([np.nan]*5)
                
        mask_t = (price_rec['ticker']==ticker)
        pr_t   = price_rec[mask_t]
        if len(pr_t)==0:
            return ticker, *(np.nan,)*5

        ctry = analyst['country'].get(ticker)
        if ctry not in country_slices:
            return ticker, *(np.nan,)*5
        s,e = country_slices[ctry]; macro_ct = macro_rec[s:e]
        if len(macro_ct)<8:
            return ticker, *(np.nan,)*5

        idx_m = np.searchsorted(macro_ct['ds'], pr_t['ds'], 'right')-1
        valid = idx_m>=0
        pr_t, idx_m = pr_t[valid], idx_m[valid]
        if len(pr_t) < HIST_WINDOW+HORIZON:
            return ticker, *(np.nan,)*5

        yv = pr_t['y'].astype(np.float32)
        ys = np.where(yv>SMALL_FLOOR, yv, SMALL_FLOOR)
        ly = np.log(ys)
        lr = np.concatenate([[0.0], np.diff(ly)]).astype(np.float32)
        ma = np.convolve(lr, np.ones(HIST_WINDOW)/HIST_WINDOW, mode='valid')
        MA52 = np.concatenate([np.full(HIST_WINDOW-1,np.nan), ma])
        valid_ma = ~np.isnan(MA52)

        macro_vals = np.vstack([macro_ct[r][idx_m] for r in MACRO_REGRESSORS]).T
        if ticker in fd_rec_dict:
            fd_rec = fd_rec_dict[ticker]
            idx_f  = np.searchsorted(fd_rec['ds'], pr_t['ds'], 'right')-1
            valid_fd = idx_f>=0
        else:
            valid_fd = np.zeros(len(pr_t), bool)

        keep = valid_ma & valid_fd
        idx_keep = np.nonzero(keep)[0]
        if len(idx_keep) < HIST_WINDOW+HORIZON:
            return ticker, *(np.nan,)*5

        dfm = pd.DataFrame({
            'ds':       pr_t['ds'][idx_keep],
            'y':        yv[idx_keep],
            'log_ret':  lr[idx_keep],
            'Interest': macro_vals[idx_keep,0],
            'Cpi':      macro_vals[idx_keep,1],
            'Gdp':      macro_vals[idx_keep,2],
            'Unemp':    macro_vals[idx_keep,3],
            'MA52_ret': MA52[idx_keep]
        })
        if ticker in fd_rec_dict:
            dfm['Revenue']     = fd_rec['Revenue'][idx_f[idx_keep]]
            dfm['EPS (Basic)'] = fd_rec['EPS (Basic)'][idx_f[idx_keep]]
            regs, n_reg, n_ch = ALL_REGRESSORS, n_reg_all, n_channels_all
        else:
            regs, n_reg, n_ch = NON_FIN_REGRESSORS, n_reg_non_fin, n_channels_non_fin
        
        scaled_ret, scaled_reg, sc_reg, sc_ret, mn_d, mx_d = \
            make_scaled_arrays(dfm[['y'] + regs], regs)
        X_full, y_full = make_windows(scaled_ret, scaled_reg, HIST_WINDOW, HORIZON)
        N = len(X_full)
        if N == 0:
            return ticker, *(np.nan,)*5

        model = build_model(n_reg)
        init_w = model.get_weights()

        val_len   = max(1, N // 5)
        train_len = N - val_len
        ds = make_tf_dataset(X_full, y_full, batch=BATCH, shuffle=False, repeat=False)
        train_ds = make_tf_dataset(X_full[:train_len], y_full[:train_len], batch=BATCH, shuffle=True, repeat=True)
        val_ds   = make_tf_dataset(X_full[train_len:], y_full[train_len:], batch=BATCH)

        model.set_weights(init_w)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=train_len  // BATCH,
            validation_steps=val_len   // BATCH,
            callbacks=CALLBACKS,
            verbose=0,
        )

        preds_val   = model.predict(val_ds, batch_size=BATCH, verbose=0)
        resids      = (y_full[train_len:] - preds_val).ravel()
        sigma_model = np.std(resids, ddof=1) if len(resids) else np.nan

        model.set_weights(init_w)
        model.fit(
            X_full,
            y_full,
            batch_size=BATCH,
            epochs=EPOCHS,
            callbacks=CALLBACKS,
            verbose=0
        )

        K.clear_session()

        hist_df = dfm.iloc[-HIST_WINDOW:].copy()
        last_hist_macro = hist_df[MACRO_REGRESSORS].values[-1].astype(np.float32)


        rev_points_arr = np.array(
            [float(next_fc.at[ticker, rev_key]) for rev_key, _ in SCENARIOS],
            dtype=np.float32,
        )
        eps_points_arr = np.array(
            [float(next_fc.at[ticker, eps_key]) for _, eps_key in SCENARIOS],
            dtype=np.float32,
        )
        n_scn = len(SCENARIOS)
        last_hist_macro = np.where(last_hist_macro > SMALL_FLOOR, last_hist_macro, SMALL_FLOOR)

        country_t = ticker_country[ticker]
        var_tuple = country_var_results.get(country_t)

        if var_tuple is not None:
            var_initial_state, A, Σ, neqs = var_tuple
            Hq = HORIZON // 4

            P = np.empty((Hq + 1, neqs, neqs), dtype=np.float32)
            P[0] = np.eye(neqs, dtype=np.float32)
            for i in range(1, Hq + 1):
                P[i] = P[i - 1] @ A

            eps_shocks = rng.multivariate_normal(
                mean=np.zeros(neqs, dtype=np.float32),
                cov=Σ,
                size=(N_SIMS, Hq),
                method="cholesky"        
            ).astype(np.float32)
            i_idx = np.arange(Hq)[:, None]  
            j_idx = np.arange(Hq)[None, :]   
            idx   = i_idx - j_idx           

            mask = idx >= 0                 
            T_full = np.zeros((Hq, Hq, neqs, neqs), dtype=np.float32)
            T_full[mask] = P[idx[mask]]

            noise_term = np.einsum('ski,tkij->stj', eps_shocks, T_full)

            init_term = var_initial_state.astype(np.float32) @ P[1:].astype(np.float32)
            init_term = np.broadcast_to(init_term[np.newaxis, ...], (N_SIMS, Hq, neqs))

            sims_uncentered = init_term + noise_term
            sims_uncentered = np.clip(sims_uncentered, SMALL_FLOOR, None)
            sims_tiled = np.repeat(sims_uncentered[None, ...], n_scn, axis=0)
        else:
            int_array = np.array(
                next_macro.get(ticker, {}).get("InterestRate", [np.nan]), 
                dtype=np.float32
            )
            inf_array = np.array(
                next_macro.get(ticker, {}).get("Consumer_Price_Index_Cpi", [np.nan]),
                dtype=np.float32,
            )
            gdp_array = np.array(
                next_macro.get(ticker, {}).get("GDP", [np.nan, np.nan, np.nan]),
                dtype=np.float32,
            )
            unemp_array = np.array(
                next_macro.get(ticker, {}).get("Unemployment", [np.nan]), 
                dtype=np.float32
            )

            quarterly_fc = np.zeros((4, 4), dtype=np.float32)
            for i, arr in enumerate([int_array, inf_array, gdp_array, unemp_array]):
                if arr.size >= 4:
                    quarterly_fc[:, i] = arr[:4]
                else:
                    fill_count = arr.size
                    if fill_count > 0:
                        quarterly_fc[:fill_count, i] = arr
                        quarterly_fc[fill_count:, i] = arr[-1]
                    else:
                        quarterly_fc[:, i] = SMALL_FLOOR

            sims_tiled = np.broadcast_to(
                quarterly_fc[np.newaxis, np.newaxis, :, :], (n_scn, N_SIMS, 4, 4)
            )

        last_vals_all = last_hist_macro.reshape(1, 1, 1, 4).astype(np.float32)
        last_vals_all = np.broadcast_to(last_vals_all, (n_scn, N_SIMS, 1, 4))

        with np.errstate(divide="ignore", invalid="ignore"):
            log_prev_all = np.log(np.where(last_vals_all > 0, last_vals_all, SMALL_FLOOR))
            log_q_all = np.log(np.where(sims_tiled > 0, sims_tiled, SMALL_FLOOR))

        cat = np.concatenate([log_prev_all, log_q_all], axis=2)
        diffs_all = np.diff(cat, axis=2)

        quarter_shocks = diffs_all[..., :4, :]

        deltas_macro_weekly = np.repeat(quarter_shocks, repeats_quarter, axis=2)

        deltas_future_all = np.zeros((n_scn, N_SIMS, HORIZON, n_reg), dtype=np.float32)

        for m_idx, macro_name in enumerate(MACRO_REGRESSORS):
            reg_idx = regs.index(macro_name)
            slice_rhs = deltas_macro_weekly[..., m_idx]

            deltas_future_all[..., reg_idx] = slice_rhs

        if "Revenue" in regs:
            
            last_rev = float(hist_df.iloc[-1]["Revenue"])
            if last_rev <= SMALL_FLOOR:
                last_rev = SMALL_FLOOR

            last_rev_arr = np.full((n_scn,), last_rev, dtype=np.float32)

            valid_rev = rev_points_arr > SMALL_FLOOR

            mu_rev = np.zeros(n_scn, dtype=np.float32)
            to_log_points = rev_points_arr[valid_rev]
            to_log_last   = last_rev_arr[valid_rev]
            mu_rev[valid_rev] = (
                np.log(to_log_points) - np.log(to_log_last)
            ) / 4.0

            q_rev = last_rev_arr[:, None] * np.exp(
                np.cumsum(mu_rev.reshape(n_scn, 1).repeat(4, axis=1), axis=1)
            ).astype(np.float32)

            with np.errstate(divide="ignore", invalid="ignore"):
                log_q_rev = np.log(np.where(q_rev > 0, q_rev, SMALL_FLOOR))
                log_prev_rev = np.log(last_rev_arr)

            shifts_rev = np.concatenate([
                (log_q_rev[:, :1] - log_prev_rev[:, None]),    
                (log_q_rev[:, 1:] - log_q_rev[:, :-1])    
            ], axis=1)

            shifts_rev_exp = shifts_rev.reshape((n_scn, 1, 4))
            deltas_rev_weekly = np.repeat(shifts_rev_exp, repeats_quarter, axis=2)

            deltas_rev_weekly = np.repeat(deltas_rev_weekly, N_SIMS, axis=1)
            deltas_rev_weekly = deltas_rev_weekly[..., None]

            rev_idx = regs.index("Revenue")
            deltas_future_all[..., rev_idx] = deltas_rev_weekly[..., 0]

        if "EPS (Basic)" in regs:
            last_eps = float(hist_df.iloc[-1]["EPS (Basic)"])
            if last_eps <= SMALL_FLOOR:
                last_eps = SMALL_FLOOR

            last_eps_arr = np.full((n_scn,), last_eps, dtype=np.float32)

            valid_eps = eps_points_arr > SMALL_FLOOR
            mu_eps = np.zeros(n_scn, dtype=np.float32)
            mu_eps[valid_eps] = (
                np.log(eps_points_arr[valid_eps]) - np.log(last_eps_arr[valid_eps])
            ) / 4.0

            q_eps = last_eps_arr[:, None] * np.exp(
                np.cumsum(mu_eps.reshape(n_scn, 1).repeat(4, axis=1), axis=1)
            ).astype(np.float32)

            with np.errstate(divide="ignore", invalid="ignore"):
                log_q_eps = np.log(np.where(q_eps > 0, q_eps, SMALL_FLOOR))
                log_prev_eps = np.log(last_eps_arr)

            shifts_eps = np.concatenate(
                [
                    (log_q_eps[:, :1] - log_prev_eps[:, None]),
                    (log_q_eps[:, 1:] - log_q_eps[:, :-1]),
                ],
                axis=1,
            )

            shifts_eps_exp = shifts_eps.reshape((n_scn, 1, 4))
            deltas_eps_weekly = np.repeat(shifts_eps_exp, repeats_quarter, axis=2)

            deltas_eps_weekly = np.repeat(deltas_eps_weekly, N_SIMS, axis=1)
            deltas_eps_weekly = deltas_eps_weekly[..., None]

            eps_idx = regs.index("EPS (Basic)")
            deltas_future_all[..., eps_idx] = deltas_eps_weekly[..., 0]

        center = sc_reg.center_.reshape((1, 1, 1, n_reg))
        scale = sc_reg.scale_.reshape((1, 1, 1, n_reg))
        mins = mn_d.reshape((1, 1, 1, n_reg))
        maxs = mx_d.reshape((1, 1, 1, n_reg))

        deltas_future_all = (
            np.maximum(np.minimum(deltas_future_all, maxs), mins) - center
        ) / scale

        X_hist = np.zeros((1, HIST_WINDOW, n_ch), dtype=np.float32)
        last_hist_returns = (
            hist_df["log_ret"].values[-HORIZON:].astype(np.float32).reshape(-1, 1)
        )
        lr2 = (last_hist_returns - sc_ret.center_) / sc_ret.scale_
        scaled_hist_ret = lr2.ravel()
        X_hist[0, :, 0] = scaled_hist_ret

        full_regs_array = dfm[regs].astype(np.float32).values
        with np.errstate(divide="ignore", invalid="ignore"):
            log_regs_full = np.log(
                np.where(full_regs_array > SMALL_FLOOR, full_regs_array, SMALL_FLOOR)
            )
        delta_regs_full = np.diff(log_regs_full, axis=0).astype(np.float32)
        delta_regs_full = (delta_regs_full - sc_reg.center_) / sc_reg.scale_
        X_hist[0, :, 1:] = delta_regs_full[-HORIZON:, :]

        zeros_future_ret = np.zeros((n_scn, N_SIMS, HORIZON, 1), dtype=np.float32)
        X_future = np.concatenate([zeros_future_ret, deltas_future_all], axis=3) 

        hist_block = np.broadcast_to(X_hist, (n_scn, N_SIMS, HORIZON, n_ch)) 

        X_all = np.concatenate([hist_block, X_future], axis=2) 

        X_all_flat = X_all.reshape(-1, SEQUENCE_LENGTH, n_ch) 

        pred_scaled_all = model.predict(X_all_flat, batch_size=256, verbose=0)

        noise_all = rng.normal(loc=0.0, scale=sigma_model, size=pred_scaled_all.shape).astype(np.float32)

        pred_scaled_noisy = pred_scaled_all + noise_all

        median = sc_ret.center_[0]
        iqr    = sc_ret.scale_[0]
        pred_returns = pred_scaled_noisy * iqr + median

        sum_returns      = np.sum(pred_returns, axis=1)
        final_prices_flat = cur_p * np.exp(sum_returns)

        p_lower, p_median, p_upper = np.nanpercentile(final_prices_flat, [2.5, 50.0, 97.5])
        rets        = final_prices_flat / cur_p - 1.0
        avg_ret_val = np.nanmean(rets)
        std_ret_val = np.nanstd(rets, ddof=0)
        gc.collect()

        return ticker, p_lower, p_median, p_upper, avg_ret_val, std_ret_val
    except Exception as e:
        logger.error("Error processing ticker %s: %s", ticker, e)
        return ticker, np.nan, np.nan, np.nan, np.nan, np.nan

import tensorflow as _tf

def cast_xy(x, y):
    return _tf.cast(x, _tf.float32), _tf.cast(y, _tf.float32)

def main() -> None:
    
    logger.info("Importing data …")
    fdata = FinancialForecastData()
    macro = fdata.macro
    r = macro.r
    close_df = r.weekly_close
    tickers = r.tickers
    fin_raw = fdata.prophet_data
    next_fc = fdata.next_period_forecast()
    next_macro = macro.macro_forecast_dict()
    latest_price = r.last_price
    analyst = r.analyst
    ticker_country = analyst['country']
    results = []
    
    
    dates_all = close_df.index.values
    tick_list = close_df.columns.values
    price_arr = close_df.values.astype(np.float32)
    T_dates, M = price_arr.shape
    total_rows = T_dates * M

    ds_col = np.repeat(dates_all, M)
    tick_col = np.tile(tick_list, T_dates)
    y_col = price_arr.reshape(-1)
    max_tlen = max(len(t) for t in tick_list)
    price_rec = np.zeros(total_rows,
                            dtype=[('ds','datetime64[ns]'),
                                ('ticker', f'U{max_tlen}'),
                                ('y','float32')])
    price_rec['ds'] = ds_col
    price_rec['ticker'] = tick_col
    price_rec['y'] = y_col
    price_rec = price_rec[~np.isnan(price_rec['y'])]
    country_map = {t:str(c) for t,c in zip(analyst.index, analyst['country'])}
    country_arr = np.array([country_map[t] for t in price_rec['ticker']])
    price_rec = rfn.append_fields(price_rec, 'country', country_arr, usemask=False)
    price_rec = price_rec[np.lexsort((price_rec['ds'], price_rec['country']))]

    raw_macro = macro.assign_macro_history_non_pct().reset_index()
    raw_macro = raw_macro.rename(columns={'year':'ds'} if 'year' in raw_macro else
                                    {raw_macro.columns[1]:'ds'})
    raw_macro['ds'] = (raw_macro['ds'].dt.to_timestamp()
                        if isinstance(raw_macro['ds'].dtype, pd.PeriodDtype)
                        else pd.to_datetime(raw_macro['ds']))
    raw_macro['country'] = raw_macro['ticker'].map(country_map)
    macro_clean = raw_macro[['ds','country'] + MACRO_REGRESSORS].dropna()
    max_clen = max(len(c) for c in macro_clean['country'])
    macro_rec = np.zeros(len(macro_clean),
                            dtype=[('ds','datetime64[ns]'),
                                ('country', f'U{max_clen}')] +
                                [(reg,'float32') for reg in MACRO_REGRESSORS])
    macro_rec['ds'] = macro_clean['ds'].values
    macro_rec['country'] = macro_clean['country'].values
    for reg in MACRO_REGRESSORS:
        macro_rec[reg] = macro_clean[reg].values
    macro_rec = macro_rec[np.lexsort((macro_rec['ds'], macro_rec['country']))]

    unique_countries, first_idx = np.unique(macro_rec['country'], return_index=True)
    country_slices: Dict[str, Tuple[int,int]] = {}
    for i, ctry in enumerate(unique_countries):
        start = first_idx[i]
        end   = first_idx[i+1] if i+1 < len(first_idx) else len(macro_rec)
        country_slices[ctry] = (start, end)

    country_var_results: Dict[str, Optional[tuple]] = {}
    for ctry, (s, e) in country_slices.items():
        rec = macro_rec[s:e]
        dfm = pd.DataFrame({reg: rec[reg] for reg in MACRO_REGRESSORS},
                            index=pd.DatetimeIndex(rec['ds']))

        dfm = (
            dfm[~dfm.index.duplicated(keep='first')]
                .sort_index()
                .resample('W')
                .mean()
                .ffill()
                .dropna()
        )

        nonconst = dfm.std() > SMALL_FLOOR

        if dfm.shape[1] == 0 or len(dfm) <= dfm.shape[1]:
            country_var_results[ctry] = None
            continue

        try:
            vr = VAR(dfm).fit(maxlags=1)
        except Exception as ex:
            country_var_results[ctry] = None
            continue

        k_ar = vr.k_ar
        if k_ar < 1:
            country_var_results[ctry] = None
            continue

        A = vr.coefs[0].astype(np.float32)
        Σ_df = vr.resid.cov()                   
        Σ = Σ_df.to_numpy(dtype=np.float32) 
        
        Σ = 0.5 * (Σ + Σ.T)                     
        eigvals = np.linalg.eigvalsh(Σ)
        min_eig = eigvals.min()
        if min_eig < 0:
            Σ += np.eye(Σ.shape[0], dtype=Σ.dtype) * (-min_eig + 1e-8)
        neqs = vr.neqs
        last_state = rec[-k_ar:]
        init = np.column_stack([last_state[r] for r in MACRO_REGRESSORS]).astype(np.float32).ravel()
        country_var_results[ctry] = (init, A, Σ, neqs)

    fd_rec_dict: Dict[str,np.ndarray] = {}
    for t in tickers:
        df_fd = fin_raw.get(t, pd.DataFrame()).reset_index().rename(
            columns={'index':'ds','rev':'Revenue','eps':'EPS (Basic)'})
        if df_fd.empty: continue
        df_fd['ds'] = pd.to_datetime(df_fd['ds'])
        df_fd = df_fd[['ds','Revenue','EPS (Basic)']].dropna()
        if df_fd.empty: continue
        rec = np.zeros(len(df_fd),
                        dtype=[('ds','datetime64[ns]'),
                                ('Revenue','float32'),
                                ('EPS (Basic)','float32')])
        rec['ds'] = df_fd['ds'].values
        rec['Revenue'] = df_fd['Revenue'].values.astype(np.float32)
        rec['EPS (Basic)'] = df_fd['EPS (Basic)'].values.astype(np.float32)
        fd_rec_dict[t] = np.sort(rec, order='ds')
        
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=init_worker,
        initargs=(
            price_rec, macro_rec, country_slices,
            country_var_results, fd_rec_dict,
            latest_price, analyst, ticker_country,
            next_fc, next_macro
        ),
        mp_context=ctx
    ) as exe:
        results = []
        futures = [exe.submit(forecast_one, t) for t in tickers]
        for fut in as_completed(futures):
            try:
                ticker, lo, mid, hi, ret, se = fut.result()
                results.append({
                    "Ticker": ticker,
                    "Min": lo,
                    "Avg": mid,
                    "Max": hi,
                    "Return": ret,
                    "SE": se
                })
                logger.info(
                    "Ticker %s: Min: %.2f, Avg: %.2f, Max: %.2f, Return: %.4f, SE: %.4f",
                    ticker, lo, mid, hi, ret, se
                )
            except Exception as ex:
                logger.error("Worker failed for %s: %s", ticker, ex)

    summary_df = pd.DataFrame(results).set_index("Ticker")
    excel_file = os.path.expanduser("~/config.MODEL_FILE")
    with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        summary_df.to_excel(writer, sheet_name='LSTM')

    logger.info("Forecasting complete.")

    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    faulthandler.enable()
    main()
