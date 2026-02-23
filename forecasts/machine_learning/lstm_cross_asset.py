from __future__ import annotations

"""
LSTM-based Cross-Asset Forecasting with Skewed-t Modelling, Copula Simulation, and
Macro–Fundamental Scenarios
==========================================================================
 
Overview
--------
This module implements an end-to-end pipeline for multi-asset return forecasting and
scenario simulation. It trains a *global* direct-H model that predicts H future
log-return steps using a recurrent backbone (LSTM), a gated self-attention mixer,
and two probabilistic heads:
 
1) A *quantile* head that outputs per-step monotone quantiles (10%, 50%, 90%) via
   a non-decreasing parameterisation.

2) A *distribution* head that outputs per-step parameters of a Hansen (1994)
   skewed-t observation model, suitable for asymmetric heavy-tailed returns.

The system pools data across tickers with shared scalers and optional ticker
embeddings. It augments inputs with macro-economic, style-factor, graph-based,
volatility, and fundamental features, harmonised on a weekly grid. Forecasts are
post-processed with ensemble-level calibration and simulated under a dependence
structure constructed from de-noised residuals, targeted-shrinkage correlations,
and an optional t-copula. Scenario outputs include predictive price bands,
expected returns, and dispersion measures.

Notation and Data
-----------------
• Time index t increases in weeks. A forecast produces H future steps.

• Prices: P_t denotes the level; log-return r_t = log(P_t) − log(P_{t−1}).

• Features: at each week there are K regressors X_t ∈ R^K, assembled from:

  – macro levels by country, resampled weekly;

  – style factors (weekly returns);

  – graph features from rolling correlations;

  – volatility proxies (rolling σ, quarticity, bipower variation);

  – fundamentals (Revenue, EPS);

  – higher-order moments (rolling skewness/kurtosis of returns).

Regressor Innovations (“Deltas”)
--------------------------------
To promote stationarity, mixed transformations are used:

• For Interest, Unemp, Balance Of Trade, Balance On Current Account:
  first difference ΔX_t = X_t − X_{t−1}.

• For Cpi, Gdp, Corporate Profits (strictly positive levels):
  log-difference Δx_t = log(max(X_t, ε)) − log(max(X_{t−1}, ε)).

• For factors and volatility proxies: pass-through levels aligned to t
  (no differencing).

• For Revenue: log-difference as above.

• For EPS (can be negative): signed-log transform s(x) = sign(x) · log(1 + |x|);
  EPS delta is Δe_t = s(EPS_t) − s(EPS_{t−1}).

Zero deltas are treated as missing, then forward-filled within each series with
leading zeros. Deltas are robust-scaled with a global `RobustScaler` and clipped
to per-feature percentile bounds (e.g., 1st–99th).

AR(1) Baseline for Log-Returns
------------------------------
A per-ticker deterministic baseline is subtracted from training targets and added
back during simulation to simplify the distributional modelling.

• AR(1) model: 

    r_{t+1} = β₀ + β₁ r_t + ε_{t+1}.

• Long-run mean: 

    m̂ = β̂₀ / (1 − β̂₁) (if |1 − β̂₁| > small threshold; otherwise 0).

• H-step deterministic path from last observed r_t:

  μ_base(h) = m̂ + (r_t − m̂) · (β̂₁)^h,  for h = 1…H.

Model Architecture (Direct-H)
-----------------------------
Input sequence length is SEQ_LEN = HIST_WINDOW + H (past horizon plus future
conditioning frame). Channel 0 holds scaled past returns; channels 1…K hold
scaled regressor deltas. Future H rows contain zero in channel 0 and contain the
*predicted/assumed* future deltas in channels 1…K at inference time.

Backbone
~~~~~~~~
1) LSTM (returning sequences) → Layer Normalisation.

2) Optional ticker embedding e ∈ R^{d_emb}, broadcast across time and concatenated.

3) Dense projection to d_model and Multi-Head Self-Attention with residual
   connection and Layer Normalisation.

4) Gated mixer: let h be the projection and att the attention output; form

   mix = σ(g) ⊙ att + (1 − σ(g)) ⊙ h,

   where g is a learned gate and σ(·) is the logistic function. The last H time
   positions of `mix` feed the heads.

Quantile Head (Monotone by Construction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For each step h, the head outputs raw values (a_h, b_h, c_h) and constructs
ordered quantiles:

• q10_h = a_h,

• q50_h = a_h + softplus(b_h) + ε,

• q90_h = q50_h + softplus(c_h) + ε,

where ε is a small positive constant. The loss for quantile level τ is the pinball
loss L_τ(e) = max(τ·e, (τ − 1)·e), with residual e = y − q(τ). The objective
averages over τ ∈ {0.1, 0.5, 0.9}, all horizon steps, and the batch.

Skewed-t Distribution Head
~~~~~~~~~~~~~~~~~~~~~~~~~~
For each step h, the head outputs raw parameters, transformed into the Hansen
skewed-t parameters (μ_h, σ_h, ν_h, λ_h):

• σ_h = softplus(raw_σ_h) + σ_floor,

• μ_h = μ_max · tanh(raw_μ_h · σ_h),

• ν_h = ν_floor + softplus(raw_ν_h),

• λ_h = tanh(raw_λ_h).

Given residual target y_h, standardise 

    z_h = (y_h − μ_h) / σ_h 
    
and set a skew scale 

    s_h = 1 + λ_h if z_h ≥ 0, 
    
else 

    s_h = 1 − λ_h.
    
Define 

    z̃_h = z_h / s_h.

Let

  log c(ν) = lgamma((ν + 1) / 2) − lgamma(ν / 2) − 0.5 · [log(π) + log(ν − 2)].

The log-density is

  log f(y_h) = log c(ν_h) − log σ_h − log s_h − 0.5 · (ν_h + 1) · log(1 + z̃_h^2 / (ν_h − 2)).

The negative log-likelihood is the mean of −log f(y_h) over batch and horizon.

Regularisation includes:

• σ² penalty with weight λ_σ,

• tail penalty with weight λ_{1/ν} on 1 / (ν − 2),

• mild L2 penalties on μ and λ,

• horizon drift penalty: if M = ∑_{h=1}^H μ_h, add μ_sum_penalty · M².

Auxiliary Realised-Volatility Head (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
An auxiliary head predicts a horizon-level realised-volatility proxy (e.g.,
standard deviation across the H true returns), optimised with Huber loss. This
stabilises scale learning.

Training Regime
---------------
• Global pooling: all eligible tickers contribute windows built with *shared*
  global scalers. Optional ticker IDs act as embeddings.

• Time-decay sample weights: later windows can be up-weighted linearly.

• Optimiser: Adam with cosine-restart learning-rate schedule; gradient norm
  clipping at 1.0; early stopping; snapshot averaging of the last epochs.

Ensemble Calibration
--------------------
After training multiple seeds, per-step ensemble statistics are computed:

• Step means μ̄_h = average over models of μ_{k,h}.

• Step variances σ̄_h² = average over models of σ_{k,h}².

• Between-model mean variability v_μ,h = variance over models of μ_{k,h}.

A conservative dispersion is set as

  σ_step(h) = sqrt( σ̄_h² + v_μ,h ).

Aggregate over the horizon:

  μ_tot = ∑_{h=1}^H μ̄_h,   σ_tot = sqrt( ∑_{h=1}^H σ_step(h)² ).

Using calibration residuals r_cal = (∑ y_true − ∑ μ_base) − μ_tot:

• Robust dispersion scale via MAD: mad = median(|r_cal − median(r_cal)|) / 0.6745.

• sigma_scale = clip( mad / median(σ_tot), lower=0.8, upper=1.6 ).

• μ_bias = shrunken median(r_cal), clipped to a hard cap; applied evenly per step
  as bias_step = μ_bias / H.

• A central empirical band (eps_lo, eps_hi) is recorded for reference.

Dependence Modelling and Shocks
-------------------------------
Residuals for dependence estimation are constructed from aligned returns with one
of two de-noising modes:

• Factor mode: regress each asset on contemporaneous factor returns; residuals
  are y − F β̂.

• AR(1) mode: use raw returns, then fit per-asset AR(1) over the window and take
  one-step residuals 
  
    ε_t = r_t − [m̂ + (r_{t−1} − m̂) φ̂].

Cross-sectional correlation is estimated with missing-data awareness, minimum
pair counts, shrinkage R ← (1 − α)R + αI, and projection to the SPD cone. A
*targeted* shrinkage further reduces off-graph entries (defined by thresholded
and top-K correlations) more than on-graph entries.

Temporal dependence is modelled via an AR(1) coefficient ρ (median across assets
of AR(1) on the residuals). Gaussian shocks Z_{s,t} ∈ R^M are simulated as

• Z_{s,0} = L ε_{s,0},  where R_assets = L Lᵀ and ε_{s,0} ~ N(0, I).

• Z_{s,t} = ρ Z_{s,t−1} + sqrt(1 − ρ²) L ε_{s,t}.

Optionally, a *t-copula* is used: draw w_{s,t} ~ χ²_ν / ν; use mixing scale
t_mix = sqrt(ν / w) element-wise. Convert correlated normals to uniforms and then
to Student-t quantiles per step before applying the Hansen skew mapping.

Graph-Propagated Features
-------------------------
A rolling window produces correlation matrices C_t whose diagonals are zeroed.
Edges are retained if |C_t(i, j)| ≥ τ and/or among the top-K per row. Degree
normalisation constructs Ŵ_t = D^{−1/2} W_t D^{−1/2}, with D_{ii} = ∑_j |W_t(i, j)|.
Per-ticker features are neighbour-weighted sums:

• g_lr[t, j]   = ∑_i Ŵ_t(j, i) · lr[t, i],

• g_rv26[t, j] = ∑_i Ŵ_t(j, i) · rv26[t, i],

• g_rv52[t, j] = ∑_i Ŵ_t(j, i) · rv52[t, i].

Future graph features can be zeroed to avoid information leakage.

Macro and Fundamental Futures
-----------------------------
Macro deltas are simulated only on *release* weeks:

• Monthly series (every 4 weeks) and quarterly series (every 13 weeks) receive
  non-zero innovations at their cadence; other weeks are zero. Innovations are
  historical release-to-release deltas drawn via the stationary bootstrap with
  geometric block lengths (restart probability p).

Style-factor futures are sampled via stationary bootstrap over historical factor
returns, then averaged across simulations for the conditioning deltas.

Fundamentals (Revenue and EPS) are nudged by analyst targets: form a total target
change over the next reporting horizon (log-change for Revenue; signed-log
difference for EPS), draw normally with target uncertainty, and distribute across
H with a smooth cosine weight. These produce future deltas for the financial
regressors.

Macro Uncertainty Injection into Per-step σ
-------------------------------------------
Let σ_macro(h) denote an exogenous horizon-wise volatility augmentation derived,
for example, from the cross-sectional factor-future variance. Then the per-step
scale is adjusted as

  σ_h² ← σ_h² + σ_macro(h)² + (σ_macro_extra)²,

followed by a square-root to recover σ_h.

Simulation to Price Bands
-------------------------
For each seed/model and simulation path:

1) Parameters per step:
   
   μ_h ← μ_model,h + μ_base,h + bias_step,
   
   σ_h ← σ_model,h · sigma_scale, then inject macro uncertainty as above.

2) Latent draw:
   
   – If dependence is enabled, obtain uniforms from the copula (Gaussian or t),
     convert to per-step Student-t quantiles with ν_h, standardise to unit, and
     apply Hansen skew mapping: let (a, b) be constants computed from (ν_h, λ_h),
     set threshold thr = −a / b, then:
   
       z_unit = t_quantile(ν_h, u),
   
       y = b · z_unit + a,
   
       x = y / (1 − λ_h) if z_unit < thr, else y / (1 + λ_h).

   – Else, draw u ~ Uniform(0, 1) independently and proceed as above.

3) Per-step return and aggregation:
   
   step_draw = μ_h + σ_h · x;  logR = ∑_h step_draw.

4) Clamp logR to [log(lbp), log(ubp)] to avoid pathological compounding; convert
   to simple returns via R = exp(logR) − 1. Price scenarios are P_T · (1 + R).

From the simulation set, report the 5th/50th/95th percentiles (Min/Avg/Max Price),
the mean return, and the standard deviation of returns.

Stationary Bootstrap Indices (for Factors and Macro)
----------------------------------------------------
For series length L, horizon H, and restart probability p:

• At each step h, with probability p start a new block; otherwise continue the
  current block. Let G_h be the restart counter (cumulative number of restarts).

• Draw a matrix of random starts S0 ∈ {0,…,L−1}.

• Offsets since the last restart are O_h = h − last_restart_index(h).

• Indices are pos_h = (S0_{G_h} + O_h) mod L.

Implementation Notes
--------------------
• Global scalers (RobustScaler) are fitted on pooled tails of deltas and returns
  to stabilise scale across tickers and time.

• Correlation matrices are projected to SPD via eigenvalue clipping and optional
  ridge shrinkage towards τI with τ = trace(S)/p.

• All large simulation tensors (factor futures, correlated shocks, t-mix scales)
  may be stored in `multiprocessing.shared_memory` to reduce copying; clean-up
  is guaranteed in a `finally` block.

• Training uses cosine-restart learning rates, early stopping, and snapshot
  averaging of the last epochs to approximate a short-horizon SWA effect.

Assumptions and Limitations
---------------------------
• AR(1) baselines capture only low-order autocorrelation and a stable long-run
  mean; structural breaks are not explicitly modelled.

• Macro release cadences are approximated (4-week month, 13-week quarter), and
  analyst-driven fundamental deltas are smoothed heuristically across H.

• The skewed-t head models conditional one-step residuals; dependence across
  time and assets is supplied by the copula construction rather than the head.

• Quantile calibration is performed per step; horizon-level coverage is improved
  via ensemble dispersion scaling and bias correction but is not guaranteed to be
  exact in finite samples.

Configuration Summary
---------------------
Key hyper-parameters are grouped as dataclasses:

• ModelHP: history and horizon lengths, LSTM widths, dropout, L2.

• TrainHP: batch size, epochs, early-stopping patience, learning rate.

• DistHP: σ floors and bounds, ν floor, regularisation weights.

• ScenarioHP: number of simulation paths, bootstrap probability, macro σ knobs.

The `config` module provides bounds for compounded returns and the output file
path for Excel export. The top-level `run()` orchestrates state building, global
training, batched forecasting, and optional Excel persistence in a robust manner.

References
----------
Hansen, B. E. (1994). Autoregressive conditional density estimation.
Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
"""

import os as _os

_os.environ.setdefault("PYTHONHASHSEED", "42")

_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

for _v in (
    "OMP_NUM_THREADS", 
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", 
    "NUMEXPR_NUM_THREADS"
):

    _os.environ.setdefault(_v, "1")

import cProfile, faulthandler, logging, pstats, random
import signal
import threading
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from scipy.special import stdtr as t_cdf, stdtrit as t_ppf, ndtr as n_cdf

import numpy.lib.recfunctions as rfn
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pandas.api.types import is_period_dtype, is_datetime64_any_dtype
import hashlib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

import multiprocessing as mp
from multiprocessing import shared_memory
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from dataclasses import dataclass

import config
from data_processing.financial_forecast_data import FinancialForecastData
from TVP_GARCH_MC import _analyst_sigmas_and_targets_combined

from scipy.special import gammaln


import tensorflow as tf

try:

    tf.config.set_visible_devices([], "GPU")

except Exception:

    pass

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

tf.keras.utils.set_random_seed(SEED)

from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Input, LSTM, Dense, LayerNormalization, Lambda, MultiHeadAttention, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


STATE = None

SAVE_TO_EXCEL = True


@contextmanager
def _time_limit(seconds: float, label: str = "operation"):
    """
    Hard timeout for a block. On timeout:
 
    - dumps a traceback (faulthandler)
 
    - raises TimeoutError so caller can skip/continue

    Notes:
 
    - Uses SIGALRM, so it only works reliably in the MAIN THREAD on Unix/macOS.
 
    """
 
    if seconds is None or seconds <= 0:
 
        yield
 
        return

    if threading.current_thread() is not threading.main_thread():

        yield

        return

    def _handler(
        signum, 
        frame
    ):
    
        raise TimeoutError(f"Timed out after {seconds:.1f}s in {label}")


    old_handler = signal.signal(signal.SIGALRM, _handler)

    try:

        faulthandler.dump_traceback_later(seconds, repeat=False)

        signal.setitimer(signal.ITIMER_REAL, seconds)

        yield

    finally:

        signal.setitimer(signal.ITIMER_REAL, 0.0)

        faulthandler.cancel_dump_traceback_later()

        signal.signal(signal.SIGALRM, old_handler)
        
        
@dataclass(frozen = True)
class ModelHP:
    
    hist_window: int = 52
    
    horizon: int = 52
    
    lstm1: int = 96
    
    lstm2: int = 64
    
    l2_lambda: float = 1e-4
    
    dropout: float = 0.15
    

@dataclass(frozen = True)
class TrainHP:
   
    batch: int = 256
   
    epochs: int = 30
   
    patience: int = 5
   
    lr: float = 5e-4
   
    small_floor: float = 1e-6


@dataclass(frozen = True)
class DistHP:
    
    sigma_floor: float = 1e-3
    
    sigma_max: float = 0.15
    
    nu_floor: float = 6.0
    
    lambda_sigma: float = 1e-5
    
    lambda_invnu: float = 5e-4


@dataclass(frozen = True)
class ScenarioHP:
   
    n_sims: int = 100
   
    bootstrap_p: float = 1 / 6
   
    sigma_macro_extra: float = 0.0
   
    sigma_macro_alpha: float = 0.0


@dataclass(frozen = True)
class HP:
    
    model: ModelHP = ModelHP()
    
    train: TrainHP = TrainHP()
    
    dist: DistHP = DistHP()
    
    scen: ScenarioHP = ScenarioHP()


@tf.function(jit_compile = True, reduce_retracing = True)
def _pinball_loss_seq_tf(
    y_true,
    y_pred
):
    """
    Quantile (pinball) loss aggregated across a horizon for monotone per-step quantiles.

    Let 
    
        e_{t, h} = y_{t, h} − q_{t, h}(τ) 
    
    denote the residual at time index t and horizon
    step h for quantile level τ ∈ {0.1, 0.5, 0.9}. The per-observation pinball loss is

        L_τ(e) = max(τ · e, (τ − 1) · e).

    This function applies the loss to all three quantiles at each horizon step and
    returns the mean loss over batch and horizon, averaged over the three τ values.

    Parameters
    ----------
    y_true : tf.Tensor, shape (B, H, 1)
        Ground-truth future log-returns for H steps.
    y_pred : tf.Tensor, shape (B, H, 3)
        Predicted quantiles (q10, q50, q90) per step.

    Returns
    -------
    tf.Tensor (scalar)
        Mean pinball loss across batch and horizon, averaged over the three quantiles.

    Notes
    -----
    • The quantile aggregation is an equal average over τ ∈ {0.1, 0.5, 0.9}.
    
    • Using a sequence form avoids summing the whole-horizon loss into one scalar
    target, which preserves per-step calibration during training.
    """

    q = tf.constant([0.1, 0.5, 0.9], dtype = tf.float32)

    e = y_true - y_pred                

    qv = tf.reshape(q, [1, 1, 3])         

    loss = tf.maximum(qv * e, (qv - 1.0) * e)

    return tf.reduce_mean(tf.reduce_sum(loss, axis = -1), axis = 1) / tf.cast(tf.size(q), tf.float32)


@tf.function(jit_compile = True, reduce_retracing = True)
def _skewt_nll_seq_tf(
    y_true_res,
    params,
    sigma_floor, 
    mu_max, 
    lambda_sigma,
    lambda_invnu,
    mu_sum_penalty,
    nu_floor
):
    """
    Negative log-likelihood for a Hansen (1994) skewed-t observation model with mild
    regularisation and a drift-sum penalty to discourage horizon-wise bias drift.

    Model
    -----
    For each step h, conditionally on parameters (μ_h, σ_h, ν_h, λ_h), the residual
    y_h follows the Hansen skewed-t density:

    1) Standardisation:

        z_h = (y_h − μ_h) / σ_h,
       
        s_h = 1 + λ_h if z_h ≥ 0, else 1 − λ_h,
       
        z̃_h = z_h / s_h.

    2) Base Student-t(ν_h) with scale √((ν_h − 2)/ν_h), folded by the skew s_h and
    Jacobian term:

        log c(ν) = lgamma((ν + 1)/2) − lgamma(ν/2) − 0.5 · [log(π) + log(ν − 2)],

        log f(y_h) = log c(ν_h) − log σ_h − log s_h − 0.5 · (ν_h + 1) · log[1 + (z̃_h^2)/(ν_h − 2)].

    The loss is −E[log f(y_h)] averaged over batch and horizon.

    Parameterisation and constraints
    --------------------------------
    σ_h = softplus(raw_σ_h) + sigma_floor,
    
    μ_h = μ_max · tanh(raw_μ_h · σ_h),
    
    ν_h = nu_floor + softplus(raw_ν_h),
    
    λ_h = tanh(raw_λ_h).

    The μ transform bounds the per-step mean effect and makes it scale-aware via σ_h.

    Regularisation
    --------------
    • σ L2 penalty: λ_σ · E[σ_h^2].
    
    • Heavy-tail penalty: λ_{1/ν} · E[ 1 / (ν_h − 2) ].
    
    • Mild penalties on μ_h and λ_h (L2) for additional shrinkage.
    
    • Drift penalty: let M = ∑_{h=1}^H μ_h (sum over horizon). Add μ_sum_penalty · E[M^2]
    to discourage systematic horizon-wise bias accumulation.

    Parameters
    ----------
    y_true_res : tf.Tensor, shape (B, H, 1)
        Residual targets (y − μ_base) per step.
    params : tf.Tensor, shape (B, H, 4)
        Raw network outputs for (μ, σ, ν, λ) before transforms.
    sigma_floor : tf.Tensor (scalar)
        Additive floor for σ to prevent collapse.
    mu_max : tf.Tensor (scalar)
        Maximum absolute per-step mean via tanh bounding.
    lambda_sigma : tf.Tensor (scalar)
        Weight for σ^2 regularisation.
    lambda_invnu : tf.Tensor (scalar)
        Weight for 1/(ν − 2) regularisation.
    mu_sum_penalty : tf.Tensor (scalar)
        Weight for μ horizon-sum penalty.
    nu_floor : tf.Tensor (scalar)
        Minimum degrees of freedom before softplus.

    Returns
    -------
    tf.Tensor (scalar)
        Regularised negative log-likelihood averaged over batch and horizon.

    References
    ----------
    Hansen, B. E. (1994). Autoregressive conditional density estimation.
    """
    
    y = tf.cast(y_true_res, tf.float32)

    sig = tf.nn.softplus(params[..., 1:2]) + sigma_floor

    mu  = mu_max * tf.tanh(params[..., 0:1] * sig)

    nu  = nu_floor + tf.nn.softplus(params[..., 2:3])

    lam = tf.tanh(params[..., 3:4])

    z = (y - mu) / sig
  
    s = tf.where(z >= 0.0, 1.0 + lam, 1.0 - lam)
  
    zt = z / s
  
    log_base = (tf.math.lgamma((nu + 1.0) / 2.0) - tf.math.lgamma(nu / 2.0) - 0.5 * (tf.math.log(nu - 2.0) + tf.math.log(np.pi)) - tf.math.log(sig) - tf.math.log(s))
   
    log_pdf = log_base - 0.5 * (nu + 1.0) * tf.math.log1p((zt * zt) / (nu - 2.0))
    
    nll = -tf.reduce_mean(log_pdf, axis = 1)

    reg = (lambda_sigma * tf.reduce_mean(tf.square(sig)) + lambda_invnu * tf.reduce_mean(1.0 / (nu - 2.0)))
    
    reg_mu  = 5e-3 * tf.reduce_mean(tf.square(mu))
    
    reg_lam = 1e-4 * tf.reduce_mean(tf.square(lam))
    
    mu_sum   = tf.reduce_sum(mu, axis = 1, keepdims = True)
    
    drift_pen = mu_sum_penalty * tf.reduce_mean(tf.square(mu_sum))
    
    return nll + reg + reg_mu + reg_lam + drift_pen


class LSTM_Cross_Asset_Forecaster:
    
    SEED = 42
    
    random.seed(SEED)
    
    np.random.seed(SEED)

    SAVE_ARTIFACTS = False
    
    ARTIFACT_DIR = config.BASE_DIR / "lstm_artifacts"
    
    EXCEL_PATH = config.MODEL_FILE

    MU_MAX = 0.1
    
    LAMBDA_SIGMA = 1e-5
    
    LAMBDA_INVNU = 5e-4
    
    NU_FLOOR = 6.0
    
    MU_SUM_PENALTY = 5e-2 

    MACRO_REGRESSORS = [
        "Interest",
        "Cpi",
        "Gdp",
        "Unemp",
        "Balance Of Trade",
        "Corporate Profits",
        "Balance On Current Account",
    ]
    
    FACTORS = ["MTUM", "QUAL", "SIZE", "USMV", "VLUE"]
    
    FIN_REGRESSORS = ["Revenue", "EPS (Basic)"]
    
    MOMENT_COLS = ["skew_104w_lag52", "kurt_104w_lag52"]

    VOL_REGS = ["rv26", "rv52", "rq26", "rq52", "bpv26", "bpv52"]

    USE_GRAPH = True            
   
    GRAPH_ROLL = 52             
   
    GRAPH_TOPK = 5            
   
    GRAPH_TAU  = 0.20           
   
    GFEATS = ["g_lr", "g_rv26", "g_rv52"]  

    USE_CORR_SIM = True         
   
    ROLL_R = 156                
   
    NU_COPULA = 6.0         

    NON_FIN_REGRESSORS = MACRO_REGRESSORS + FACTORS + VOL_REGS + (GFEATS if USE_GRAPH else [])
   
    ALL_REGRESSORS = NON_FIN_REGRESSORS + FIN_REGRESSORS + MOMENT_COLS

    GFEATS_BLOCK_FUTURE = True
   
    USE_TIME_DECAY_WEIGHTS = True        
   
    USE_AUX_RV_TARGET = True           
   
    USE_EMBEDDING = True               
   
    EMB_DIM = 8                          
   
    USE_ENSEMBLE = True                 
   
    ENSEMBLE_SEEDS = [42, 1337, 9001]     
    
    GLOBAL_MODEL_BASENAME = "global_directH_seed"
   
    GLOBAL_MODEL_DIR = ARTIFACT_DIR / "global"
   
    GRAPH_STEP = 9  

    GRAPH_MONTHLY_REUSE = False      
   
    GRAPH_MONTHLY_REUSE_WEEKS = 5
    
    ENABLE_TICKER_TIMEOUTS = True

    TICKER_TIMEOUT_BUILDSTATE_SEC = 120.0  

    TICKER_TIMEOUT_CACHEBUILD_SEC = 120.0   

    TICKER_TIMEOUT_WINDOWS_SEC   = 120.0   


    def __init__(
        self, 
        tickers: Optional[List[str]] = None, 
        hp: Optional["HP"] = None
    ):

        self.logger = self._configure_logger()
    
        self.tickers_arg = tickers
    
        self.hp = hp or HP()

        self.HIST_WINDOW = self.hp.model.hist_window
    
        self.HORIZON = self.hp.model.horizon
    
        self.SEQUENCE_LENGTH = self.HIST_WINDOW + self.HORIZON
    
        self.L2_LAMBDA = self.hp.model.l2_lambda
    
        self._LSTM1 = self.hp.model.lstm1
    
        self._LSTM2 = self.hp.model.lstm2
    
        self._DROPOUT = self.hp.model.dropout

        self.BATCH = self.hp.train.batch
    
        self.EPOCHS = self.hp.train.epochs
    
        self.PATIENCE = self.hp.train.patience
    
        self._LR = self.hp.train.lr
    
        self.SMALL_FLOOR = self.hp.train.small_floor

        self.SIGMA_FLOOR = self.hp.dist.sigma_floor
    
        self.SIGMA_MAX = self.hp.dist.sigma_max
    
        self.NU_FLOOR = self.hp.dist.nu_floor
    
        self.LAMBDA_SIGMA = self.hp.dist.lambda_sigma
    
        self.LAMBDA_INVNU = self.hp.dist.lambda_invnu
        
        self.SIGMA_MACRO_EXTRA = float(self.hp.scen.sigma_macro_extra)
    
        self.SIGMA_MACRO_ALPHA = float(self.hp.scen.sigma_macro_alpha)

        self.N_SIMS = self.hp.scen.n_sims
    
        self._BOOT_P = self.hp.scen.bootstrap_p
                
        self._created_shms: List[shared_memory.SharedMemory] = []
        

    def _configure_logger(
        self
    ) -> logging.Logger:
        """
        Create and configure a module-scoped logger.

        Returns
        -------
        logging.Logger
            Logger emitting human-readable time-stamped messages.

        Notes
        -----
        This function attaches a single `StreamHandler` if none exist to avoid duplicate
        messages when multiple forecasters are instantiated.
        """

        logger = logging.getLogger("LSTM_Cross_Asset_Forecaster")
    
        logger.setLevel(logging.INFO)
    
        if not logger.handlers:
    
            h = logging.StreamHandler()
    
            h.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s"))
    
            logger.addHandler(h)
    
        return logger


    def choose_splits(
        self, 
        N: int, 
        min_fold: Optional[int] = None
    ) -> int:
        """
        Heuristic selection of the number of time-series CV splits based on sample size.

        Logic
        -----
        Let N denote the number of rolling sequence windows available. If N < 2 · batch,
        return 0 (not enough data). If N ≥ 4 · batch, return 2. Otherwise, return 1.

        Parameters
        ----------
        N : int
            Number of available sequence windows.
        min_fold : Optional[int]
            Minimum fold size; defaults to 2 · batch size.

        Returns
        -------
        int
            Number of splits for `TimeSeriesSplit` (0, 1, or 2).

        Rationale
        ---------
        A small number of folds reduces variance when data are scarce and avoids
        pathological validation sets that are too short for horizon learning.
        """

        if min_fold is None:
    
            min_fold = 2 * self.BATCH
    
        if N < min_fold:
    
            return 0
    
        if N >= 2 * min_fold:
    
            return 2
    
        return 1


    @staticmethod
    def _seed_from_str(
        s: str, 
        base: int = 42
    ) -> int:
        """
        Deterministic seed derivation from a string using BLAKE2b.

        Parameters
        ----------
        s : str
            Input string.
        base : int
            Base integer to xor with the BLAKE2b digest.

        Returns
        -------
        int
            Deterministic seed suitable for NumPy/TF PRNG initialisation.

        Notes
        -----
        The hash digest is 8 bytes to fit into Python int and provide stable seeds across
        processes and platforms.
        """


        h = hashlib.blake2b(s.encode("utf-8"), digest_size = 8).digest()
    
        return base ^ int.from_bytes(h, "big", signed = False)


    @staticmethod
    def _ffill_2d(
        a: np.ndarray
    ) -> np.ndarray:
        """
        Forward-fill per column in a 2-D array with NaN handling and zero pre-fill.

        Parameters
        ----------
        a : np.ndarray, shape (T, K)
            Input array with NaNs.

        Returns
        -------
        np.ndarray, shape (T, K)
            Forward-filled array; leading NaNs replaced with zero.

        Notes
        -----
        Columns that are entirely NaN are replaced with zeros, ensuring stable downstream
        scaling and differencing.
        """

        a = np.asarray(a, np.float32)
       
        if a.size == 0:
       
            return a
       
        n, k = a.shape
       
        out = a.copy()
       
        for j in range(k):
       
            col = out[:, j]
       
            mask = np.isnan(col)
       
            if mask.all():
       
                out[:, j] = 0.0
       
                continue
       
            idx = np.where(~mask, np.arange(n), 0)
       
            np.maximum.accumulate(idx, out = idx)
       
            out[:, j] = col[idx]
       
            first_valid = np.argmax(~mask)
       
            out[:first_valid, j] = 0.0
       
        return out


    @staticmethod
    def _ffill_1d(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Forward-fill a 1-D array with NaN handling and zero pre-fill.

        Parameters
        ----------
        x : np.ndarray, shape (T,)
            Input vector.

        Returns
        -------
        np.ndarray, shape (T,)
            Forward-filled vector; leading NaNs replaced with zero.
        """

        x = np.asarray(x, np.float32)
    
        if x.size == 0:
    
            return x
    
        out = x.copy()
    
        mask = np.isnan(out)
    
        if mask.all():
    
            return np.zeros_like(out)
    
        idx = np.where(~mask, np.arange(out.size), 0)
    
        np.maximum.accumulate(idx, out = idx)
    
        out = out[idx]
    
        first_valid = int(np.argmax(~mask))
    
        out[:first_valid] = 0.0
    
        return out


    def build_delta_matrix(
        self, 
        reg_mat: np.ndarray, 
        regs: list[str]
    ) -> np.ndarray:
        """
        Construct a matrix of weekly innovations ("deltas") for heterogeneous regressors.

        Transform rules
        ---------------
        • “Interest”, “Unemp”, “Balance Of Trade”, “Balance On Current Account”:
        first difference ΔX_t = X_t − X_{t−1}.

        • “Cpi”, “Gdp”, “Corporate Profits”:
        log-difference Δx_t = log(max(X_t, ε)) − log(max(X_{t−1}, ε)).

        • Factors and volatility proxies (e.g., MTUM, rv26): pass-through levels (no
        differencing), aligning to t.

        • “Revenue”: log-difference as above.

        • “EPS (Basic)”: difference of signed-log transforms
       
            s(x) = sign(x) · log(1 + |x|), i.e.
        
            Δe_t = s(EPS_t) − s(EPS_{t−1}).

        All zero deltas are treated as missing and forward-filled (with leading zeros).

        Parameters
        ----------
        reg_mat : np.ndarray, shape (T, K)
            Aligned regressor levels.
        regs : list[str]
            Regressor names ordered as in columns of `reg_mat`.

        Returns
        -------
        np.ndarray, shape (T − 1, K)
            Delta matrix aligned to (t = 1…T − 1).

        Rationale
        ---------
        Mixed transformations approximate stationarity of innovations whilst respecting
        economic scales (log-diff for strictly positive quantities) and allowing signed
        earnings effects.
        """

        R = np.asarray(reg_mat, np.float32)
    
        T, K = R.shape
    
        out = np.zeros((T - 1, K), np.float32)
    
        regs_arr = np.array(regs, dtype = object)

        idx_diff = np.isin(regs_arr, ("Interest", "Unemp", "Balance Of Trade", "Balance On Current Account"))
    
        idx_logd = np.isin(regs_arr, ("Cpi", "Gdp", "Corporate Profits"))
    
        idx_pas  = np.isin(regs_arr, self.FACTORS + self.VOL_REGS + (self.GFEATS if self.USE_GRAPH else []))

        if idx_diff.any():
    
            D = np.diff(R[:, idx_diff], axis = 0).astype(np.float32)
    
            D[D == 0.0] = np.nan
    
            out[:, idx_diff] = self._ffill_2d(
                a = D
            )
    
        if idx_logd.any():

            X = np.log(np.maximum(R[:, idx_logd], self.SMALL_FLOOR)).astype(np.float32)
    
            D = np.diff(X, axis = 0)
    
            D[D == 0.0] = np.nan
    
            out[:, idx_logd] = self._ffill_2d(
                a = D
            )

        if idx_pas.any():
    
            out[:, idx_pas] = R[1:, idx_pas]

        if "Revenue" in regs:
    
            j = regs.index("Revenue")
    
            d = np.diff(np.log(np.maximum(R[:, j], self.SMALL_FLOOR)).astype(np.float32))
           
            d[d == 0.0] = np.nan
           
            out[:, j] = self._ffill_1d(
                x = d
            )
        
        if "EPS (Basic)" in regs:
           
            j = regs.index("EPS (Basic)")
           
            prev = self.slog1p_signed(
                x = R[:-1, j]
            ).astype(np.float32)
           
            nxt  = self.slog1p_signed(
                x = R[1:,  j]
            ).astype(np.float32)
           
            d = (nxt - prev).astype(np.float32)
           
            d[d == 0.0] = np.nan
           
            out[:, j] = self._ffill_1d(
                x = d
            )
        
        return out


    @staticmethod
    def fit_ar1_baseline(
        log_ret: np.ndarray
    ) -> tuple[float, float]:
        """
        Fit a baseline AR(1) for log-returns and derive the long-run mean.

        Model
        -----
        r_{t+1} = β₀ + β₁ r_t + ε_{t+1}.

        The OLS solution on (r_t, r_{t+1}) yields β̂₀, β̂₁. The implied unconditional mean is

            m̂ = β̂₀ / (1 − β̂₁)  (if |1 − β̂₁| > 1e−8, else 0),

        and the persistence φ̂ = β̂₁.

        Parameters
        ----------
        log_ret : np.ndarray, shape (T,)
            Log-return series.

        Returns
        -------
        (m_hat, phi_hat) : tuple[float, float]
            Long-run mean and AR(1) coefficient.

        Use
        ---
        The baseline path m̂ + (r_t − m̂) φ̂^h is subtracted from targets to simplify the
        distributional head and is added back during simulation.
        """

        r = np.asarray(log_ret, float).ravel()
    
        if len(r) < 3:
    
            return 0.0, 0.0
    
        r0 = r[:-1]
        
        r1 = r[1:]
        
        X = np.column_stack([np.ones_like(r0), r0])
        
        beta, *_ = np.linalg.lstsq(X, r1, rcond = None)
        
        m = float(beta[0] / (1.0 - beta[1])) if abs(1.0 - beta[1]) > 1e-8 else 0.0
      
        phi = float(beta[1])
      
        return m, phi


    def _aligned_lr_matrix(
        self, 
        grouped_tickers,
        weekly_price_by_ticker,
        master_index
    ):
        """
        Return an aligned DataFrame of weekly log-returns across tickers.

        Parameters
        ----------
        grouped_tickers : Iterable[str]
            Ticker order.
        weekly_price_by_ticker : Dict[str, Dict[str, np.ndarray]]
            Per-ticker dicts containing 'lr' and 'index'.
        master_index : array-like of datetime64
            Timeline to reindex on.

        Returns
        -------
        pd.DataFrame
            Columns are tickers, index is `master_index`, values are log-returns with
            NaNs filled as zeros for downstream residual estimation.
        """
    
        idx = pd.DatetimeIndex(master_index)
    
        cols = {}
    
        for t in grouped_tickers:
    
            s = pd.Series(weekly_price_by_ticker[t]["lr"], index = pd.DatetimeIndex(weekly_price_by_ticker[t]["index"]))
            
            cols[t] = s.reindex(idx).astype(np.float32)
        
        lr_df = pd.DataFrame(cols, index=idx)
        
        return lr_df


    @staticmethod
    def ar1_seq(
        r_t, 
        H, 
        m_hat, 
        phi_hat
    ):
        """
        Closed-form multi-step AR(1) forecast sequence.

        For horizon h = 1…H, the mean path is

            E[r_{t+h} | r_t] = m̂ + (r_t − m̂) φ̂^h.

        Parameters
        ----------
        r_t : np.ndarray or float
            Last observed return r_t.
        H : int
            Forecast horizon.
        m_hat : float
            Long-run mean.
        phi_hat : float
            AR(1) coefficient.

        Returns
        -------
        np.ndarray, shape (H,)
            Deterministic baseline path.
        """

        r_t = np.asarray(r_t, dtype = np.float32)
    
        h = np.arange(1, H + 1, dtype = np.float32)[None, ...]  
    
        base = m_hat + (r_t[..., None] - m_hat) * (phi_hat ** h)
    
        return base.astype(np.float32)


    def _build_unscaled_cache_for_ticker(
        self,
        t: str,
        regs: List[str],
        reg_pos: Dict[str, int],
        *,
        align_cache: Dict[str, Any],
        macro_weekly_by_country: Dict[str, pd.DataFrame],
        factor_weekly_values: np.ndarray,
        moments_by_ticker: Dict[str, np.ndarray],
        fd_weekly_by_ticker: Dict[str, Dict[str, np.ndarray]],
        fd_rec_dict: Dict[str, np.ndarray],
        weekly_price_by_ticker: Dict[str, Dict[str, np.ndarray]],
        graph_feats_by_ticker: Dict[str, np.ndarray],
        analyst: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Assemble per-ticker unscaled feature caches aligned on a common weekly grid.

        Builds:
        • reg_mat : levels of all regressors.
        
        • DEL_full: delta matrix via `build_delta_matrix`, with macro deltas lag-padded
        to avoid immediate forward-fill leakage.
        
        • lr_core : aligned log-returns (excluding the first week to match deltas).
        
        • idx_keep: row indices kept after alignment.

        Parameters
        ----------
        t : str
            Ticker.
        regs : List[str]
            Regressor names (global order).
        reg_pos : Dict[str, int]
            Mapping from name to column position.
        align_cache, macro_weekly_by_country, factor_weekly_values, moments_by_ticker,
        fd_weekly_by_ticker, fd_rec_dict, weekly_price_by_ticker, graph_feats_by_ticker,
        analyst : various
            Pre-computed alignment and feature stores.

        Returns
        -------
        dict or None
            Dict with keys {"reg_mat", "DEL_full", "lr_core", "idx_keep"} if sufficient
            data exist; otherwise None.

        Data-leakage precautions
        ------------------------
        • Macro deltas at t are constructed from levels available up to t.
       
        • Future-dependent graph features are guarded elsewhere; this function merely
        aligns already-computed features.
        """

        flags = {
            "has_factors": (factor_weekly_values.shape[0] > 0),
            "has_fin": (t in fd_weekly_by_ticker),
            "has_moms": (t in moments_by_ticker),
        }

        align = align_cache.get(t)
        
        if align is None: 
            
            return None

        ctry = analyst["country"].get(t, None)
       
        dfm_ct = macro_weekly_by_country.get(ctry)
       
        if dfm_ct is None or dfm_ct.shape[0] < 12: 
            
            return None

        fa_vals = factor_weekly_values
        
        moms_vals = moments_by_ticker.get(t, None)
        
        fdw_t = fd_weekly_by_ticker.get(t, None) 

        idx_m = np.asarray(align["idx_m"], dtype = np.int64)
        
        idx_fa = np.asarray(align["idx_fa"], dtype = np.int64)
        
        idx_keep = np.asarray(align["idx_keep"], dtype = np.int64)
        
        idx_fd = align["idx_fd"]
        
        if idx_fd is not None:
        
            idx_fd = np.asarray(idx_fd, dtype = np.int64)

        sel_m = idx_m[idx_keep]
       
        sel_fa = idx_fa[idx_keep]
       
        if idx_fd is not None:
       
            sel_fd = idx_fd[idx_keep]

        n_m = len(dfm_ct)
     
        if fa_vals.shape[0] > 0:
     
            sel_fa = np.clip(sel_fa, 0, fa_vals.shape[0] - 1)
     
            upper_ok = (sel_m < n_m) & (sel_fa < fa_vals.shape[0])
      
        else:
      
            sel_fa = np.zeros_like(sel_m)
      
            upper_ok = (sel_m < n_m)

        if idx_fd is not None and fdw_t is not None:
       
            sel_fd = np.clip(sel_fd, 0, len(fdw_t["index"]) - 1)
       
        else:
       
            sel_fd = None

        idx_keep = idx_keep[upper_ok]
       
        sel_m = sel_m[upper_ok]
       
        sel_fa = sel_fa[upper_ok]
       
        if sel_fd is not None:
       
            sel_fd = sel_fd[upper_ok]

        if len(idx_keep) < self.SEQUENCE_LENGTH + 1:
           
            return None

        regs_list = list(regs)
       
        n_reg = len(regs_list)
       
        reg_mat = np.zeros((len(idx_keep), n_reg), dtype = np.float32)

        macro_idx = [reg_pos[m] for m in self.MACRO_REGRESSORS if m in reg_pos]
       
        macro_vals = dfm_ct[self.MACRO_REGRESSORS].to_numpy(np.float32, copy = False)
       
        reg_mat[:, macro_idx] = macro_vals[sel_m]

        if flags.get("has_factors", False) and factor_weekly_values.shape[0] > 0:
         
            factor_idx = [reg_pos[f] for f in self.FACTORS if f in reg_pos]
         
            reg_mat[:, factor_idx] = fa_vals[sel_fa, :]

        if flags.get("has_fin", False) and (fdw_t is not None) and (sel_fd is not None):
         
            vals = fdw_t["values"][sel_fd, :]
         
            if "Revenue" in reg_pos and "EPS (Basic)" in reg_pos:
         
                reg_mat[:, [reg_pos["Revenue"], reg_pos["EPS (Basic)"]]] = vals

        if flags.get("has_moms", False) and (moms_vals is not None):
         
            if "skew_104w_lag52" in reg_pos and "kurt_104w_lag52" in reg_pos:
         
                reg_mat[:, [reg_pos["skew_104w_lag52"], reg_pos["kurt_104w_lag52"]]] = moms_vals[idx_keep, :]

        vol_cols = [n for n in ("rv26","rv52","rq26","rq52","bpv26","bpv52") if n in reg_pos]
        
        if vol_cols:
        
            V = np.stack([
                np.nan_to_num(weekly_price_by_ticker[t][n][idx_keep], nan = 0.0, posinf = 0.0, neginf = 0.0)
                for n in vol_cols
            ], axis = 1)
            
            reg_mat[:, [reg_pos[n] for n in vol_cols]] = V

        if self.USE_GRAPH and graph_feats_by_ticker:
          
            GF = graph_feats_by_ticker.get(t)
          
            if GF is not None and GF.shape[0] >= len(idx_keep):
          
                for k, name in enumerate(self.GFEATS):
          
                    if name in reg_pos:
          
                        reg_mat[:, reg_pos[name]] = np.nan_to_num(GF[idx_keep, k], nan = 0.0, posinf = 0.0, neginf = 0.0)

        DEL_full = self.build_delta_matrix(
            reg_mat = reg_mat, 
            regs = regs_list
        )
      
        macro_cols = [reg_pos[m] for m in self.MACRO_REGRESSORS if m in reg_pos]
       
        if macro_cols:
            
            DEL_full[1:, macro_cols] = DEL_full[:-1, macro_cols]
            
            DEL_full[0,  macro_cols] = 0.0

        lr_vec = weekly_price_by_ticker[t]["lr"][idx_keep].astype(np.float32)
        
        lr_core = lr_vec[1:]

        return {
            "reg_mat": reg_mat, 
            "DEL_full": DEL_full, 
            "lr_core": lr_core, 
            "idx_keep": idx_keep
        }


    def _compute_graph_features(
        self,
        grouped_tickers, 
        weekly_price_by_ticker, 
        master_index
    ):
        """
        Compute graph-propagated features from rolling correlations with sparsification.

        Procedure
        ---------
        1) Rolling window (size = GRAPH_ROLL) computes the sample correlation matrix C_t
        on weekly returns. Diagonals are zeroed.

        2) Early sparsification:
        
        • Threshold: retain entries with |C_t(i, j)| ≥ τ (GRAPH_TAU).
        
        • Top-K: optionally retain the K largest |C_t(i, ·)| per row.

        3) Symmetric normalisation:
        
        Construct W_t with entries of the sparsified correlations and scale by

            Ŵ_t = D^{-1/2} W_t D^{-1/2},

        where D is the diagonal degree matrix D_{ii} = ∑_j |W_t(i, j)|.

        4) Graph features per ticker j at time t are neighbour-weighted sums:

            g_lr[t, j]   = ∑_i Ŵ_t(j, i) · lr[t, i],
            
            g_rv26[t, j] = ∑_i Ŵ_t(j, i) · rv26[t, i],
            
            g_rv52[t, j] = ∑_i Ŵ_t(j, i) · rv52[t, i].

        Monthly reuse optionally reuses Ŵ_t within a calendar month to reduce churn.

        Returns
        -------
        dict[str, np.ndarray]
            For each ticker, an array of shape (T_j, 3) with features [g_lr, g_rv26, g_rv52]
            aligned to the ticker’s own weekly index.

        Rationale
        ---------
        Graph propagation captures sectoral and cross-asset spillovers whilst constraining
        estimation variance via sparsification and degree normalisation.
        """

        idx = pd.DatetimeIndex(master_index)

        def _series(
            tkr, 
            key
        ):
    
            return pd.Series(weekly_price_by_ticker[tkr][key], index = pd.DatetimeIndex(weekly_price_by_ticker[tkr]["index"])).reindex(idx)

        lr_df = pd.DataFrame({
            t: _series(
                tkr = t, 
                key = "lr"
            ) for t in grouped_tickers
        }, index = idx).fillna(0.0)
        
        rv26_df = pd.DataFrame({
            t: _series(
                tkr = t,
                key = "rv26"
            ) for t in grouped_tickers
        }, index=idx).fillna(0.0)
        
        rv52_df = pd.DataFrame({
            t: _series(
                tkr = t, 
                key = "rv52"
            ) for t in grouped_tickers
        }, index = idx).fillna(0.0)

        X = lr_df.to_numpy(dtype = np.float32, copy = False) 
        
        R26 = rv26_df.to_numpy(dtype = np.float32, copy = False)
        
        R52 = rv52_df.to_numpy(dtype = np.float32, copy = False)
        
        T, M = X.shape

        win = int(self.GRAPH_ROLL)
        
        step = max(1, int(getattr(self, "GRAPH_STEP", 1)))
        
        tau = float(self.GRAPH_TAU)
        
        topK = int(self.GRAPH_TOPK) if self.GRAPH_TOPK else M

        g_lr = np.zeros((T, M), np.float32)
        
        g_rv26 = np.zeros((T, M), np.float32)
        
        g_rv52 = np.zeros((T, M), np.float32)

        Xd = X.astype(np.float64)
        
        s1 = np.zeros(M, np.float64)     
        
        s2 = np.zeros(M, np.float64)     
        
        S = np.zeros((M, M), np.float64)  

        for t0 in range(win):
          
            x = Xd[t0]
          
            s1 += x
          
            s2 += x * x
          
            S += np.outer(x, x)

        W_idx = np.empty((M, min(topK, M - 1)), dtype = np.int64) 
       
        W_val = np.zeros((M, min(topK, M - 1)), dtype = np.float32)

        last_month = None
        
        reuse_left = 0

        for t in range(win - 1, T):
            
            recompute = False
           
            if self.GRAPH_MONTHLY_REUSE:
           
                cur_month = idx[t].month
           
                if (last_month is None) or (cur_month != last_month) or (reuse_left <= 0):
           
                    recompute = True
           
                    last_month = cur_month
           
                    reuse_left = int(self.GRAPH_MONTHLY_REUSE_WEEKS)
           
                else:
           
                    reuse_left -= 1
           
            else:
           
                recompute = ((t - (win - 1)) % step == 0)

            if recompute:
           
                mu  = s1 / win
           
                var = (s2 - win * mu * mu) / max(win - 1, 1)
           
                sd  = np.sqrt(np.maximum(var, 1e-12))

                denom = np.outer(sd, sd)
           
                with np.errstate(divide = 'ignore', invalid = 'ignore'):
           
                    C = (S - win * np.outer(mu, mu)) / max(win - 1, 1)
           
                    C = np.divide(C, denom, out = np.zeros_like(C), where = (denom > 0))
           
                np.fill_diagonal(C, 0.0)

                C = np.where(np.abs(C) >= tau, C, 0.0)

                if topK < M:
                
                    idx_top = np.argpartition(-np.abs(C), kth = min(topK, M - 1)-1, axis = 1)[:, :topK]
                
                else:
                
                    idx_top = np.argpartition(-np.abs(C), kth = M - 2, axis = 1)[:, :min(topK, M - 1)]

                rows = np.arange(M)[:, None]
               
                vals = C[rows, idx_top]   

                d = np.sum(np.abs(vals), axis = 1) 
                
                Dinv = 1.0 / np.sqrt(np.clip(d, 1e-6, None))
                
                vals = (Dinv[:, None] * vals) * Dinv[idx_top]

                W_idx = idx_top.astype(np.int64, copy = False)
                
                W_val = vals.astype(np.float32, copy = False)

            v_lr = X[t]  
            
            v_r26 = R26[t]
            
            v_r52 = R52[t] 

            g_lr[t, :] = np.sum(W_val * v_lr [W_idx], axis = 1)
            
            g_rv26[t, :] = np.sum(W_val * v_r26[W_idx], axis = 1)
            
            g_rv52[t, :] = np.sum(W_val * v_r52[W_idx], axis = 1)

            if t + 1 < T:
                
                x_out = Xd[t - win + 1]
                
                x_in = Xd[t + 1]
                
                s1 += (x_in - x_out)
                
                s2 += (x_in * x_in - x_out * x_out)
                
                S += (np.outer(x_in, x_in) - np.outer(x_out, x_out))

        out = {}
      
        T_idx = np.arange(T, dtype = np.int64)
       
        for j, tkr in enumerate(grouped_tickers):
       
            G_all = np.column_stack([g_lr[:, j], g_rv26[:, j], g_rv52[:, j]])
       
            t_idx = pd.DatetimeIndex(weekly_price_by_ticker[tkr]["index"])
       
            pos = (pd.Series(T_idx, index = idx).reindex(t_idx).fillna(-1).to_numpy(dtype = np.int64))
       
            valid = pos >= 0
          
            arr = np.zeros((len(t_idx), 3), dtype = np.float32)
          
            if valid.any():
          
                arr[valid] = G_all[pos[valid]]
          
            out[tkr] = arr
        
        return out


    def _ensemble_calibration(
        self, 
        trained_models,
        X_cal,
        y_seq_cal, 
        mu_base_seq_cal, 
        tick_id_cal = None
    ):
        """
        Compute simple ensemble post-hoc calibration for per-step dispersion and horizon bias.

        Let μ_k,h and σ_k,h be per-model step means and scales. Define ensemble step
        means and variances:

            μ̄_h   = mean_k μ_k,h,
          
            σ̄_h^2 = mean_k σ_k,h^2,
           
            v_μ,h  = var_k  μ_k,h.

        Use σ_step(h) = sqrt(σ̄_h^2 + v_μ,h) as a conservative dispersion estimate.

        Horizon aggregation
        -------------------
        Total mean and standard deviation over H steps:

            μ_tot = ∑_{h=1}^H μ̄_h,
           
            σ_tot = sqrt( ∑_{h=1}^H σ_step(h)^2 ).

        Empirical residuals are formed against the calibration set’s total residual target
        (∑ y − ∑ μ_base). The robust scale is estimated via MAD:

            mad = median( |resid − median(resid)| ) / 0.6745.

        The multiplicative scale factor is sigma_scale = clip(mad / median(σ_tot), 0.8, 1.6).

        A horizon-total bias μ_bias is set to a shrunken median residual and later split
        equally per step (bias_step = μ_bias / H). A central 1 − α empirical band
        (conf_eps) is also recorded.

        Returns
        -------
        (sigma_scale, mu_bias, (eps_lo, eps_hi)) : tuple[float, float, tuple[float, float]]
        """

        mu_list = []
        
        sig2_list = []
        
        for m in trained_models:
        
            if tick_id_cal is not None:
        
                out = m({"seq": X_cal, "tick_id": tick_id_cal}, training = False)
        
            else:
        
                out = m(X_cal, training = False)

            params = out[1] if isinstance(out, (list, tuple)) else out["dist_head"]
            
            sig = (tf.nn.softplus(params[..., 1:2]) + self.SIGMA_FLOOR).numpy() 
            
            mu = (self.MU_MAX * tf.tanh(params[..., 0:1] * sig)).numpy()               

            mu_list.append(mu)
            
            sig2_list.append(sig ** 2)

        mu_step = np.mean(np.stack(mu_list,  axis = 0), axis = 0)                  
        
        sig2_step = np.mean(np.stack(sig2_list, axis = 0), axis = 0)                

        mu_tot = np.sum(mu_step, axis = 1, keepdims = True)[..., 0]              
        
        sd_tot = np.sqrt(np.sum(sig2_step, axis = 1, keepdims = True))[..., 0]       

        y_tot_res_cal = np.sum(y_seq_cal - mu_base_seq_cal, axis = 1, keepdims = True)[..., 0]  

        resid = (y_tot_res_cal - mu_tot).ravel()
        
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid))) / 0.6745
        
        pred = np.nanmedian(sd_tot.ravel())
        
        sigma_scale = float(np.clip((mad / max(pred, 1e-6)) if np.isfinite(mad) else 1.0, 0.8, 1.6))

        raw_bias = float(np.nanmedian(y_tot_res_cal - mu_tot))    
        
        shrink = 0.25                                           
        
        hard_cap = 0.3                                           

        mu_bias = float(np.clip(raw_bias * shrink, -hard_cap, hard_cap))

        alpha = 0.10
        
        eps_lo = float(np.nanquantile(resid, alpha / 2))
        
        eps_hi = float(np.nanquantile(resid, 1.0 - alpha / 2))

        return sigma_scale, mu_bias, (eps_lo, eps_hi)


    @staticmethod
    def transform_deltas(
        DEL: np.ndarray, 
        sc: RobustScaler, 
        q_low: np.ndarray, 
        q_high: np.ndarray
    ):
        """
        Clip and robust-scale delta features using a pre-fitted RobustScaler.

        Parameters
        ----------
        DEL : np.ndarray, shape (T, K)
            Raw delta features.
        sc : RobustScaler
            Fitted scaler; only `center_` and `scale_` are used.
        q_low, q_high : np.ndarray, shape (K,)
            Feature-wise clipping bounds (e.g., 1st and 99th percentiles).

        Returns
        -------
        np.ndarray, shape (T, K)
            Scaled features: (clip(DEL) − center) / scale.
        """

        DEL_clip = np.clip(DEL, q_low, q_high)
    
        DEL_scaled = (DEL_clip - sc.center_) / sc.scale_
    
        return DEL_scaled.astype(np.float32)


    @staticmethod
    def ensure_spd_for_cholesky(
        Sigma: np.ndarray, 
        min_eig: float = 1e-8, 
        shrink: float = 0.05, 
        max_tries: int = 6
    ) -> np.ndarray:
        """
        Project a symmetric matrix to the cone of symmetric positive-definite (SPD) matrices.

        Procedure
        ---------
        1) Symmetrise S = (S + Sᵀ)/2 and apply convex shrinkage:

            S ← (1 − γ) S + γ τ I,  where τ = tr(S)/p and γ = `shrink`.

        2) Eigen-decompose S = V diag(w) Vᵀ, clip eigenvalues: w ← max(w, min_eig).

        3) Reconstruct and, if necessary, add increasing diagonal jitter until the Cholesky
        factorisation succeeds.

        Parameters
        ----------
        Sigma : np.ndarray, shape (p, p)
        min_eig : float
            Minimum eigenvalue after clipping.
        shrink : float
            Ridge shrinkage weight towards τ I.
        max_tries : int
            Maximum jitter escalation steps.

        Returns
        -------
        np.ndarray, shape (p, p)
            SPD matrix suitable for Cholesky.
        """

        S = np.asarray(Sigma, dtype = np.float64)
    
        if not np.isfinite(S).all():
           
            S = np.where(np.isfinite(S), S, 0.0)
        
        S = 0.5 * (S + S.T)
        
        p = S.shape[0]
        
        I = np.eye(p, dtype = np.float64)
        
        tau = (np.trace(S) / p) if p > 0 else 1.0
       
        S = (1.0 - shrink) * S + shrink * tau * I
       
        w, V = np.linalg.eigh(S)
       
        w_clipped = np.maximum(w, min_eig)
       
        S = (V * w_clipped) @ V.T
       
        S = 0.5 * (S + S.T)
       
        jitter = 0.0
       
        for k in range(max_tries):
       
            try:
       
                np.linalg.cholesky(S + (jitter * I))
       
                if jitter > 0:
       
                    S = S + jitter * I
       
                return S.astype(np.float32)
       
            except np.linalg.LinAlgError:
       
                jitter = max(min_eig, (10.0 ** k) * min_eig)
       
        S = 0.5 * (S + S.T)
       
        S = S + (10.0 ** max_tries) * min_eig * I
       
        np.linalg.cholesky(S)
       
        return S.astype(np.float32)


    @staticmethod
    def _hansen_ab_constants(
        nu: np.ndarray, 
        lam: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Hansen (1994) skewed-t centring constants a(ν, λ) and b(ν, λ).

        Definitions
        -----------
        Let

            log c(ν) = lgamma((ν + 1)/2) − lgamma(ν/2) − 0.5 [log(π) + log(ν − 2)],
            
            c(ν)     = exp(log c(ν)).

        Then

            a(ν, λ) = 4 λ c(ν) (ν − 2) / (ν − 1),
            
            b(ν, λ) = sqrt( 1 + 3 λ^2 − a(ν, λ)^2 ).

        These constants ensure zero mean and unit variance for the transformed latent
        z via y = b z + a in the Hansen skewed-t construction.

        Parameters
        ----------
        nu : np.ndarray
        lam : np.ndarray

        Returns
        -------
        (a, b) : tuple[np.ndarray, np.ndarray]
            Broadcast-compatible arrays.
        """
        
        nu = np.asarray(nu, dtype = np.float64)
       
        lam = np.asarray(lam, dtype = np.float64)
       
        lam = np.clip(lam, -0.995, 0.995)

        nu_safe = np.maximum(nu, 2.000001)

        logc = gammaln((nu_safe + 1.0) / 2.0) - gammaln(nu_safe / 2.0) - 0.5 * (np.log(np.pi) + np.log(nu_safe - 2.0))
       
        c = np.exp(logc)

        a = 4.0 * lam * c * (nu_safe - 2.0) / (nu_safe - 1.0)
        
        b2 = 1.0 + 3.0 * lam ** 2 - a ** 2
        
        b = np.sqrt(np.maximum(b2, 1e-12))  

        return a.astype(np.float32), b.astype(np.float32)


    @staticmethod
    def fit_ret_scaler_from_logret(
        log_ret: np.ndarray
    ) -> RobustScaler:
        """
        Fit a robust scaler on log-returns (excluding the initial zero).

        Parameters
        ----------
        log_ret : np.ndarray, shape (T,)
            Log-returns where the first element may be zero due to differencing.

        Returns
        -------
        RobustScaler
            Fitted with a floor on `scale_` to avoid division by very small numbers.
        """

        sc = RobustScaler().fit(log_ret[1:].reshape(-1, 1))
    
        sc.scale_[sc.scale_ < 1e-6] = 1e-6
    
        return sc


    @staticmethod
    def slog1p_signed(
        x: np.ndarray
    ) -> np.ndarray:
        """
        Signed-log transform: 
        
            s(x) = sign(x) · log(1 + |x|).

        This transform is monotone, odd, and reduces the influence of large-magnitude
        values whilst preserving sign information.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        np.ndarray
        """

        x = np.asarray(x, float)
    
        return np.sign(x) * np.log1p(np.abs(x))


    def build_directH_model(
        self, 
        n_reg: int, 
        seed: int = SEED, 
        n_tickers: Optional[int] = None
    ):
        """
        Construct the direct-H sequence-to-sequence model with an LSTM backbone, a gated
        self-attention mixer, and two heads:
        
        (1) quantile head (q10, q50, q90) with monotonicity by construction; and
        
        (2) distribution head outputting raw (μ, σ, ν, λ) per horizon step.

        Architecture
        ------------
        • Input: sequence length SEQ_LEN, channels = 1 (scaled returns) + K regressors.
        • LSTM → LayerNorm.
        • Optional ticker embedding (if enabled) broadcast across time.
        • Dense projection to d_model and Multi-Head Attention (self-attention), then a
        gating blend between attention output and skip-connected projection:

            mix = sigmoid(g) ⊙ att + (1 − sigmoid(g)) ⊙ h.

        • Slice last H steps → heads:
        
        – Quantiles: parameterise with non-negative “gaps” to enforce q10 ≤ q50 ≤ q90:
            
            q10 = a,
            
            q50 = a + softplus(b) + ε,
            
            q90 = q50 + softplus(c) + ε.
        
        – Distribution: linear layer with 4 channels, transformed downstream into the
            Hansen skewed-t parameters.

        • Optional auxiliary head predicts realised volatility proxy over the horizon.

        Returns
        -------
        tf.keras.Model
            Keras model ready for compilation.

        Rationale
        ---------
        Direct H-step modelling avoids teacher forcing; the monotone quantile head
        improves coverage calibration; the skewed-t head models heavy tails and skew.
        """
    
        inp = Input((self.SEQUENCE_LENGTH, 1 + n_reg), dtype = "float32", name = "seq")

        x = LSTM(
            self._LSTM1, 
            return_sequences = True,
            kernel_regularizer = l2(self.L2_LAMBDA),
            recurrent_regularizer = l2(self.L2_LAMBDA),
            kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed),
            recurrent_initializer = tf.keras.initializers.Orthogonal(seed = seed),
            dropout = self._DROPOUT, 
            recurrent_dropout = 0.0
        )(inp)
        
        if self.USE_EMBEDDING and n_tickers:
          
            tick_id_inp = Input((), dtype = "int32", name = "tick_id")
          
            emb = tf.keras.layers.Embedding(
                n_tickers, 
                self.EMB_DIM,
                embeddings_initializer = tf.keras.initializers.GlorotUniform(seed = seed)
            )(tick_id_inp)
            
            emb_seq = Lambda(lambda e: tf.repeat(tf.expand_dims(e, axis = 1), repeats = self.SEQUENCE_LENGTH, axis = 1))(emb)
            
            x_in = Concatenate(axis = -1)([x, emb_seq])
            
            second_input = tick_id_inp
        
        else:
        
            x_in = x
        
            second_input = None

        d_model = self._LSTM2
        
        h = Dense(d_model)(x_in) 
        
        att = MultiHeadAttention(num_heads = 4, key_dim = max(1, d_model // 4))(h, h)
        
        att = Add()([att, h])
        
        att = LayerNormalization()(att)
        
        g = Dense(d_model, activation = "relu")(h)
        
        g = Dense(d_model, activation = "sigmoid")(g)
        
        mix = Add()([tf.keras.layers.Multiply()([g, att]), tf.keras.layers.Multiply()([Lambda(lambda z: 1.0 - z)(g), h])])
        
        mix = LayerNormalization()(mix)

        xH = Lambda(lambda t: t[:, -self.HORIZON:, :], name = "last_H")(mix)  

        q_raw = Dense(3, name = "q_raw")(xH)
        
        
        def monotone_quantiles_time(
            z
        ):
        
            a = z[..., 0:1]
        
            b_gap = tf.nn.softplus(z[..., 1:2]) + 1e-6
        
            c_gap = tf.nn.softplus(z[..., 2:3]) + 1e-6
        
            q10 = a
        
            q50 = a + b_gap
        
            q90 = a + b_gap + c_gap
        
            return tf.concat([q10, q50, q90], axis = -1)
        
        
        q_out = Lambda(monotone_quantiles_time, name = "q_head")(q_raw)  

        d_params = Dense(4, name = "dist_head")(xH) 

        outputs = [q_out, d_params]

        if self.USE_AUX_RV_TARGET:
            
            pooled = Lambda(lambda t: tf.reduce_mean(t, axis = 1), name = "pool_H")(xH)  
            
            rv_head = Dense(1, name = "rv_head")(pooled)
            
            outputs.append(rv_head)

        model = Model(inp if second_input is None else [inp, second_input], outputs)
        
        return model


    def pinball_loss_seq(
        self,
        y_true,
        y_pred
    ):
        """
        Wrapper around the compiled TensorFlow pinball loss for Keras integration.
        """

        return _pinball_loss_seq_tf(
            y_true = y_true, 
            y_pred = y_pred
        )


    def skewt_nll_seq(
        self, 
        y_true_res,
        params
    ):
        """
        Wrapper around the compiled TensorFlow skewed-t negative log-likelihood with
        regularisation, passing in class hyper-parameters as tf constants.
        """
    
        return _skewt_nll_seq_tf(
            y_true_res = y_true_res, 
            params = params,
            sigma_floor = tf.constant(self.SIGMA_FLOOR, tf.float32),
            mu_max = tf.constant(self.MU_MAX, tf.float32),
            lambda_sigma = tf.constant(self.LAMBDA_SIGMA, tf.float32),
            lambda_invnu = tf.constant(self.LAMBDA_INVNU, tf.float32),
            mu_sum_penalty = tf.constant(self.MU_SUM_PENALTY, tf.float32),
            nu_floor = tf.constant(self.NU_FLOOR, tf.float32),
        )


    def _residuals_for_corr(
        self, 
        lr_df,
        factors_df = None, 
        window = None,
        mode = "ar1"
    ):
        """
        Estimate de-noised residuals for correlation modelling.

        Two modes:
        • 'factors': regress each asset’s returns on provided factor returns via OLS over
        the window; residuals are y − F β̂.
        • 'ar1' (default): use raw returns.

        Then estimate per-asset AR(1) within the window via closed-form OLS and compute
        one-step residuals 
        
            ε_t = r_t − [m̂ + (r_{t−1} − m̂) φ̂].

        Parameters
        ----------
        lr_df : pd.DataFrame
            Aligned returns; the last `window` rows are used.
        factors_df : Optional[pd.DataFrame]
            Factor returns aligned to `lr_df` for 'factors' mode.
        window : Optional[int]
            Lookback window length; defaults to `ROLL_R`.
        mode : {"ar1", "factors"}

        Returns
        -------
        pd.DataFrame
            Residuals ε_t with the same columns as `lr_df`, NaN for the first row.

        Rationale
        ---------
        Removing common components and weak AR dynamics stabilises cross-sectional
        correlation estimates that feed into the copula simulation.
        """
    
        W = int(window or self.ROLL_R)
    
        Y = lr_df.tail(W).to_numpy(np.float64)    
    
        maskY = np.isfinite(Y)

        if mode == "factors" and factors_df is not None and not factors_df.empty:
    
            idx = lr_df.tail(W).index
    
            F = factors_df.reindex(idx).ffill().to_numpy(np.float64, copy = False)  

            maskF = np.isfinite(F).all(axis = 1, keepdims = True)
           
            mask = maskY & maskF
           
            Yz = np.where(mask, Y, 0.0)
           
            Fz = np.where(maskF, F, 0.0)
           
            XtX = Fz.T @ Fz + 1e-4 * np.eye(F.shape[1])
           
            Beta = np.linalg.solve(XtX, Fz.T @ Yz)           
           
            R = Y - F @ Beta                           
        
        else:
        
            R = Y

        r0 = R[:-1, :]
        
        r1 = R[1:, :]
        
        mk = np.isfinite(r0) & np.isfinite(r1)
        
        n = mk.sum(axis=0).clip(min = 1)
        
        s1 = np.where(mk, r0, 0.0).sum(axis = 0)               
        
        s2 = np.where(mk, r0 * r0, 0.0).sum(axis = 0)           
        
        t1 = np.where(mk, r1, 0.0).sum(axis = 0)               
        
        t2 = np.where(mk, r0 * r1, 0.0).sum(axis = 0)        

        det = n*s2 - s1 * s1
       
        b0 = ( s2 * t1 - s1 * t2) / np.where(np.abs(det) < 1e-12, 1.0, det)
       
        b1 = (-s1 * t1 + n * t2) / np.where(np.abs(det) < 1e-12, 1.0, det)

        m = np.where(np.abs(1 - b1) > 1e-8, b0 / (1 - b1), 0.0)
        
        pred = m + (R[:-1, :] - m) * b1
        
        eps = np.full_like(R, np.nan)
        
        eps[1:, :] = R[1:, :] - pred
        
        return pd.DataFrame(eps, index = lr_df.tail(W).index, columns = lr_df.columns)


    def _estimate_nu_t(
        self, 
        x: np.ndarray, 
        grid = tuple(range(5, 31))
    ):
        """
        Grid-search MLE for Student-t degrees of freedom on standardised residuals.

        Method
        ------
        Standardise pooled residuals to zero mean and unit variance, then maximise

            ℓ(ν) = −0.5 (ν + 1) · mean[ log(1 + x^2 / (ν − 2)) ],

        over ν in a small grid (default 5…30). Returns ν clipped to [4, 40].

        Parameters
        ----------
        x : np.ndarray
            Residuals.
        grid : Iterable[int]
            Candidate ν values.

        Returns
        -------
        float
            Estimated ν.
        """

        x = x[np.isfinite(x)]
        
        if x.size < 100:
        
            return 8.0
        
        x = (x - np.mean(x)) / (np.std(x) + 1e-9)
        
        ll_best, nu_best = -np.inf, 8.0
        
        for nu in grid:
        
            v = float(nu)
        
            ll = -0.5 * (v + 1.0) * np.mean(np.log1p((x * x) / (v - 2.0)))
        
            if ll > ll_best:
        
                ll_best, nu_best = ll, v
        
        return float(max(4.0, min(40.0, nu_best)))


    def _sample_macro_future_deltas_release_cadence(
        self,
        df_levels: pd.DataFrame,          
        regs: list[str],
        reg_pos: dict[str, int],
        H: int,
        S: int,
        rng: np.random.Generator,
        p_boot: float = None,
    ) -> np.ndarray:
        """
        Sample future macro deltas on realistic release cadences using a stationary bootstrap.

        Construction
        ------------
        • Monthly indicators (“Interest”, “Unemp”, “Cpi”, “Balance Of Trade”) move every
        4 weeks; quarterly (“Gdp”, “Corporate Profits”, “Balance On Current Account”)
        every 13 weeks. Positions {3, 7, 11, …} or {12, 25, …} within the H-week
        horizon receive non-zero innovations; other weeks are zeros.

        • For each indicator, compute historical release-to-release innovations from
        level data:
            – log-differences for strictly positive series (“Cpi”, “Gdp”, “Corporate Profits”),
            – first differences otherwise.

        • Draw innovations using the stationary bootstrap (Politis–Romano):
        contiguous blocks with geometric length with parameter p_boot.

        Parameters
        ----------
        df_levels : pd.DataFrame
            Macro levels indexed by calendar date.
        regs : list[str]
        reg_pos : dict[str, int]
        H : int
            Horizon in weeks.
        S : int
            Number of simulation paths.
        rng : np.random.Generator
        p_boot : float
            Bootstrap restart probability.

        Returns
        -------
        np.ndarray, shape (S, H, K)
            Future delta tensor; zeros outside release weeks.

        Rationale
        ---------
        This design preserves the intermittency of macro updates and avoids smearing
        macro shocks across non-release weeks.
        """

        p_boot = float(self._BOOT_P if p_boot is None else p_boot)

        K = len(regs)
        
        out = np.zeros((S, H, K), dtype = np.float32)

        monthly_regs = {"Interest", "Unemp", "Cpi", "Balance Of Trade"}
        
        quarterly_regs = {"Gdp", "Corporate Profits", "Balance On Current Account"}

        diff_regs = {"Interest", "Unemp", "Balance Of Trade", "Balance On Current Account"}
        
        logdiff_regs= {"Cpi", "Gdp", "Corporate Profits"}

        dfM = df_levels.sort_index().resample("ME").last().dropna(how = "all")
        
        dfQ = df_levels.sort_index().resample("QE").last().dropna(how = "all")

        eps_local = float(self.SMALL_FLOOR)
        
        weeks_per_month = 4
        
        weeks_per_quarter = 13
        
        rel_pos_M = np.arange(weeks_per_month - 1, H, weeks_per_month, dtype = np.int64)
        
        rel_pos_Q = np.arange(weeks_per_quarter - 1, H, weeks_per_quarter, dtype = np.int64)


        def _innovations(
            series: pd.Series, 
            is_log: bool, 
            _eps: float = eps_local
        ) -> np.ndarray:
           
            x = series.astype(np.float64)
           
            if is_log:
           
                x = np.log(np.maximum(x, _eps))
           
            v = x.diff().dropna().to_numpy()
           
            v = v[np.isfinite(v)]
           
            return v.astype(np.float32, copy=False)


        O_M = self._bootstrap_plan(
            S = S,
            H = len(rel_pos_M),
            p = p_boot, 
            rng = rng
        ) if len(rel_pos_M) > 0 else None
        
        O_Q = self._bootstrap_plan(
            S = S, 
            H = len(rel_pos_Q),
            p = p_boot, 
            rng = rng
        ) if len(rel_pos_Q) > 0 else None

        for name in regs:
          
            if name not in reg_pos:
          
                continue
          
            j = reg_pos[name]

            is_monthly = name in monthly_regs
          
            is_quarter = name in quarterly_regs
          
            if not (is_monthly or is_quarter):
          
                is_monthly = True

            is_log = name in logdiff_regs
          
            panel = dfM if is_monthly else dfQ

            if name not in panel.columns:
              
                continue

            innov = _innovations(
                series = panel[name], 
                is_log = is_log
            )  
            
            L = len(innov)
           
            if L < 4:

                continue

            R = len(rel_pos_M if is_monthly else rel_pos_Q)
            
            if R == 0:
            
                continue

            O = (O_M if is_monthly else O_Q)
            
            if O is None or O.shape[1] != R:
            
                O = self._bootstrap_plan(
                    S = S,
                    H = R,
                    p = p_boot,
                    rng = rng
                )
            
            group_starts = rng.integers(0, L, size = (S, R), dtype = np.int64)
            
            idx = (group_starts + O) % L
            
            draws = innov[idx]

            pos = (rel_pos_M if is_monthly else rel_pos_Q)

            for r_i, wpos in enumerate(pos):
              
                if wpos >= H:
              
                    break
              
                out[:, wpos, j] = draws[:, r_i]

        return out  


    def _pairwise_corr_shrunk(
        self, 
        df: pd.DataFrame,
        min_pairs: int = 10, 
        alpha: float = 0.10
    ) -> np.ndarray:
        """
        Compute a pairwise correlation matrix under missing data with shrinkage.

        Procedure
        ---------
        1) For each pair (i, j), compute covariance using only rows with both finite.
        
        2) Convert to correlation by dividing by √(var_i var_j).
        
        3) Enforce a minimum pair count; otherwise set correlation to zero.
        
        4) Convex shrinkage towards the identity: 
        
            R ← (1 − α) R + α I.
        
        5) Enforce SPD via `ensure_spd_for_cholesky`.

        Parameters
        ----------
        df : pd.DataFrame
        min_pairs : int
            Minimum overlapping observations.
        alpha : float
            Shrinkage weight.

        Returns
        -------
        np.ndarray, shape (M, M)
            SPD correlation matrix.
        """

        X = df.to_numpy(np.float64, copy = False)  
    
        Tn, M = X.shape
       
        if M == 0:
       
            return np.zeros((0, 0), dtype = np.float32)
       
        if Tn < 2:
       
            return np.eye(M, dtype = np.float32)

        mask = np.isfinite(X)
       
        if mask.sum() == 0 or (mask.sum(axis = 0) == 0).all():
          
            return np.eye(M, dtype = np.float32)

        cnt = mask.sum(axis = 0, keepdims = True)     
                    
        sumv = np.where(mask, X, 0.0).sum(axis = 0, keepdims = True)   
        
        mu  = np.divide(sumv, np.maximum(cnt, 1), where = (cnt > 0))

        Xz = np.where(mask, X - mu, 0.0)

        n = mask.T @ mask       
                      
        S = Xz.T @ Xz                         
        
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            
            cov = S / np.maximum(n - 1, 1)

        sd = np.sqrt(np.diag(cov))
      
        denom = np.outer(sd, sd)

        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            
            C = np.divide(cov, denom, where = (denom > 0))

        C = np.where(n >= min_pairs, C, 0.0)
       
        C = np.clip(np.nan_to_num(C, nan = 0.0, posinf = 0.0, neginf = 0.0), -0.99, 0.99)
        
        np.fill_diagonal(C, 1.0)

        R = (1.0 - alpha) * C + alpha * np.eye(M)
        
        R = 0.5 * (R + R.T)
        
        R = self.ensure_spd_for_cholesky(
            Sigma = R, 
            min_eig = 1e-6, 
            shrink = 0.0
        )
        
        return R.astype(np.float32)


    def _kronecker_shocks(
        self,
        R_assets,
        H,
        S, 
        rho,
        seed
    ):
        """
        Generate Gaussian shocks with Kronecker-like dependence: AR(1) in time and a
        fixed cross-sectional correlation.

        Construction
        ------------
        Let ε_{s,t} ~ N(0, R_assets) i.i.d. over s (simulation) and t (time). Form

            Z_{s,0} = L ε_{s,0},
           
            Z_{s,t} = ρ Z_{s,t−1} + √(1 − ρ²) L ε_{s,t}   for t ≥ 1,

        where R_assets = L Lᵀ is the cross-sectional correlation Cholesky factor.

        Parameters
        ----------
        R_assets : np.ndarray, shape (M, M)
        H : int
        S : int
        rho : float
            Temporal AR(1) coefficient.
        seed : int

        Returns
        -------
        np.ndarray, shape (S, H, M)
            Correlated Gaussian innovations.
        """
       
        M = R_assets.shape[0]
       
        L = np.linalg.cholesky(self.ensure_spd_for_cholesky(
            Sigma = R_assets, 
            min_eig = 1e-6, 
            shrink = 0.0
        )).astype(np.float32)
       
        rng = np.random.default_rng(seed)
       
        eps = rng.standard_normal((S, H, M)).astype(np.float32)
       
        eps = np.einsum('ij,shj->shi', L, eps, optimize = True)   
       
        Z = np.empty_like(eps)
       
        Z[:, 0, :] = eps[:, 0, :]
       
        alpha = np.float32(np.sqrt(max(1e-9, 1.0 - rho * rho)))
       
        for t in range(1, H):
       
            Z[:, t, :] = rho * Z[:, t-1, :] + alpha * eps[:, t, :]
       
        return Z


    def _targeted_shrink_with_graph(
        self,
        R_emp: np.ndarray,
        lr_df: pd.DataFrame,
        tau: float,
        topk: int,
        beta_on: float = 0.15,
        beta_off: float = 0.60
    ) -> np.ndarray:
        """
        Apply targeted shrinkage to an empirical correlation matrix guided by a sparse graph.

        Given a binary adjacency A from thresholded/top-K correlations, shrink on-graph
        entries less than off-graph entries:

            R_on  ← (1 − β_on)  · R_emp_on,
            R_off ← (1 − β_off) · R_emp_off,

        with β_off ≥ β_on. Diagonals set to 1, symmetrise, and project to SPD.

        Parameters
        ----------
        R_emp : np.ndarray, shape (M, M)
        lr_df : pd.DataFrame
        tau : float
        topk : int
        beta_on : float
        beta_off : float

        Returns
        -------
        np.ndarray, shape (M, M)
            SPD correlation matrix.
        """

        X = lr_df.tail(self.ROLL_R)
        
        if X.shape[0] < 3 or X.shape[1] < 2:
        
            M = lr_df.shape[1]
        
            return self.ensure_spd_for_cholesky(
                Sigma = np.eye(M, dtype = np.float32), 
                min_eig = 1e-6, 
                shrink = 0.02
            )

        C = self._pairwise_corr_shrunk(
            df = X, 
            min_pairs = 10, 
            alpha = 0.10
        ).astype(np.float32)
        
        np.fill_diagonal(C, 0.0)

        A = np.where(np.abs(C) >= tau, 1.0, 0.0)
        
        if topk and topk < A.shape[0]:
        
            idx_top = np.argpartition(-np.abs(C), kth = topk - 1, axis = 1)[:, :topk]
        
            rows = np.arange(A.shape[0])[:, None]
        
            A2 = np.zeros_like(A)
            
            A2[rows, idx_top] = 1.0
           
            A = np.maximum(A, A2)
       
        on = A.astype(bool)
       
        off = ~on
       
        R = R_emp.copy()
       
        R[on] *= (1.0 - beta_on)
        
        R[off] *= (1.0 - beta_off)
        
        np.fill_diagonal(R, 1.0)
        
        R = 0.5 * (R + R.T)
        
        R = self.ensure_spd_for_cholesky(
            Sigma = R,
            min_eig = 1e-6,
            shrink = 0.02
        )
        
        return R.astype(np.float32)


    def _bootstrap_plan(
        self, 
        S: int,
        H: int, 
        p: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Precompute offset matrix O for the stationary bootstrap.

        Stationary bootstrap (Politis–Romano):
        • At each step h, with probability p start a new block; otherwise continue the
        previous block.
        • The offset O_{·,h} counts steps since the last restart within each path.

        Given per-path random starts S0 ∈ {0, …, L − 1}, the bootstrapped indices are

            idx_{·,h} = (S0_{·,G_h} + O_{·,h}) mod L,

        where G_h is the cumulative restart counter.

        Parameters
        ----------
        S : int
        H : int
        p : float
        rng : np.random.Generator

        Returns
        -------
        np.ndarray, shape (S, H)
            Offsets since last restart.
        """

        if H <= 0 or S <= 0:
        
            return np.zeros((S, H), dtype = np.int64)
        
        Rm = (rng.random((S, H)) < p)
        
        Rm[:, 0] = True
        
        G = np.cumsum(Rm, axis = 1) - 1
        
        t_idx = np.arange(H, dtype = np.int64)[None, :]
        
        last_restart_idx = np.maximum.accumulate(np.where(Rm, t_idx, -1), axis = 1)
        
        O = t_idx - last_restart_idx
        
        return O.astype(np.int64, copy = False)


    def _make_windows_for_ticker(
        self,
        t: str,
        ctry: str,
        regs: list[str],
        reg_pos: dict[str, int],
        HIST: int,
        HOR: int,
        SEQ_LEN: int,
        sc_reg_full,
        q_low: np.ndarray,
        q_high: np.ndarray,
        ret_scaler_full,
        macro_idx,
        factors_idx,
        reg_mat_pre: Optional[np.ndarray] = None,
        DEL_full_pre: Optional[np.ndarray] = None,
        lr_core_pre: Optional[np.ndarray] = None,
    ):
        """
        Construct training and calibration windows for a single ticker using global scalers.

        Outputs
        -------
        • X_tr, y_seq_res_tr : training inputs and residual targets (y − μ_base).
        
        • X_cal, y_seq_res_cal : calibration inputs and residual targets.
        
        • tick_id_{tr,cal} : integer ticker IDs (for embedding).
        
        • Optionally rv_{tr,cal} : auxiliary realised-volatility target per window.

        Key steps
        ---------
        1) Alignment and feature assembly identical to forecasting.
        
        2) Delta construction and robust scaling using global percentiles and RobustScaler.
        
        3) Sliding windows:
        
        – Inputs X: past HIST scaled returns in channel 0; deltas for SEQ_LEN steps in
            channels 1…K with future H steps set to zero for returns and to future
            deltas for regressors (future graph features optionally zeroed).
        
        – Targets y_seq: the next H raw log-returns (framed per step).
        
        4) AR(1) baseline μ_base_seq is computed from past returns and subtracted from y.
        
        5) Optional RV auxiliary: standard deviation of the H-step raw returns.

        Splitting
        ---------
        If enough windows exist, use `TimeSeriesSplit` with 1–2 splits; otherwise use an
        80/20 chronological split.

        Returns
        -------
        tuple
            Either 6 or 8 tensors depending on auxiliary RV configuration.

        Rationale
        ---------
        Subtracting an AR(1) baseline reduces the burden on the distributional head and
        improves stability of horizon-wise learning under changing unconditional means.
        """

        flags = STATE["presence_flags"].get(t, {})
        
        align = STATE["align_cache"].get(t)
        
        if align is None:
        
            return None

        dfm_ct = STATE["macro_weekly_by_country"].get(ctry)
        
        if dfm_ct is None or dfm_ct.shape[0] < 12:
        
            return None

        fa_vals = STATE["factor_weekly_values"]
       
        moms_vals = STATE["moments_by_ticker"].get(t)
       
        fdw = STATE["fd_weekly_by_ticker"].get(t, None) if t in STATE["fd_rec_dict"] else None

        idx_m = np.asarray(align["idx_m"], dtype = np.int64)
       
        idx_fa = np.asarray(align["idx_fa"], dtype = np.int64)
       
        idx_keep= np.asarray(align["idx_keep"], dtype = np.int64)
       
        idx_fd = align["idx_fd"]
       
        if idx_fd is not None:
       
            idx_fd = np.asarray(idx_fd, dtype = np.int64)

        sel_m = idx_m[idx_keep]
       
        sel_fa = idx_fa[idx_keep]
       
        if idx_fd is not None:
       
            sel_fd = idx_fd[idx_keep]
       
        else:
       
            sel_fd = None

        n_m = len(dfm_ct)
       
        if fa_vals.shape[0] > 0:
       
            sel_fa = np.clip(sel_fa, 0, fa_vals.shape[0] - 1)
       
            upper_ok = (sel_m < n_m) & (sel_fa < fa_vals.shape[0])
       
        else:
       
            sel_fa = np.zeros_like(sel_m)
       
            upper_ok = (sel_m < n_m)

        if sel_fd is not None and fdw is not None:
       
            sel_fd = np.clip(sel_fd, 0, len(fdw["index"]) - 1)
       
        idx_keep = idx_keep[upper_ok]
        
        sel_m = sel_m[upper_ok]
       
        sel_fa = sel_fa[upper_ok]
       
        if sel_fd is not None:
       
            sel_fd = sel_fd[upper_ok]

        if len(idx_keep) < SEQ_LEN + 1:
       
            return None

        if (reg_mat_pre is not None) and (DEL_full_pre is not None) and (lr_core_pre is not None):
       
            reg_mat = reg_mat_pre
       
            DEL_full = DEL_full_pre
       
            lr_core = lr_core_pre
       
            goto_scaling = True
       
        else:
       
            goto_scaling = False
            
        regs_list = list(regs)
       
        n_reg = len(regs_list)
        
        if not goto_scaling:
            
            reg_mat = np.zeros((len(idx_keep), n_reg), dtype = np.float32)

            macro_vals = dfm_ct[self.MACRO_REGRESSORS].to_numpy(np.float32, copy = False)
       
            reg_mat[:, macro_idx] = macro_vals[sel_m]

            if flags.get("has_factors", False) and fa_vals.shape[0] > 0:
                
                reg_mat[:, factors_idx] = fa_vals[sel_fa, :]


            if flags.get("has_fin", False) and (fdw is not None) and (sel_fd is not None):
               
                vals = fdw["values"][sel_fd, :]  
               
                reg_mat[:, [reg_pos["Revenue"], reg_pos["EPS (Basic)"]]] = vals


            if flags.get("has_moms", False) and (moms_vals is not None):
               
                reg_mat[:, [reg_pos["skew_104w_lag52"], reg_pos["kurt_104w_lag52"]]] = moms_vals[idx_keep, :]

            vol_cols = [n for n in ("rv26", "rv52", "rq26", "rq52", "bpv26", "bpv52") if n in reg_pos]
            
            if vol_cols:
            
                V = np.stack([np.nan_to_num(STATE["weekly_price_by_ticker"][t][n][idx_keep], nan = 0.0, posinf = 0.0, neginf = 0.0) for n in vol_cols], axis = 1)
                
                reg_mat[:, [reg_pos[n] for n in vol_cols]] = V

            if self.USE_GRAPH and "graph_feats_by_ticker" in STATE:
              
                GF = STATE["graph_feats_by_ticker"].get(t)
              
                if GF is not None and GF.shape[0] >= len(idx_keep):
              
                    vals = GF[idx_keep, :]  
              
                    for k, name in enumerate(self.GFEATS):
              
                        if name in reg_pos:
              
                            reg_mat[:, reg_pos[name]] = np.nan_to_num(vals[:, k], nan = 0.0, posinf = 0.0, neginf = 0.0)


            DEL_full = self.build_delta_matrix(
                reg_mat = reg_mat, 
                regs = regs_list
            )
        
            macro_cols = [reg_pos[m] for m in self.MACRO_REGRESSORS if m in reg_pos]
           
            if macro_cols:
           
                DEL_full[1:, macro_cols] = DEL_full[:-1, macro_cols]
           
                DEL_full[0, macro_cols] = 0.0
                
            lr_vec = STATE["weekly_price_by_ticker"][t]["lr"][idx_keep].astype(np.float32)
            
            lr_core = lr_vec[1:]
       
        scaled_reg_full = self.transform_deltas(
            DEL = DEL_full,
            sc = sc_reg_full, 
            q_low = q_low, 
            q_high = q_high
        )
        
        scaled_ret_core = ((lr_core - ret_scaler_full.center_) / ret_scaler_full.scale_).astype(np.float32)

        T_reg = DEL_full.shape[0]
        
        n_all = T_reg - SEQ_LEN + 1
        
        if n_all <= 1:
        
            return None

        n_splits = self.choose_splits(
            N = n_all
        )
        
        if n_splits < 2:
         
            cut = max(1, int(0.8 * n_all))
         
            train_idx = np.arange(cut)
         
            cal_idx = np.arange(cut, n_all)
        
        else:
        
            tscv = TimeSeriesSplit(n_splits = n_splits)
        
            train_idx, cal_idx = list(tscv.split(np.arange(n_all)))[-1]

        pr_view = sliding_window_view(scaled_ret_core, HIST)                     
        
        scaled_reg_full = np.ascontiguousarray(scaled_reg_full, dtype = np.float32)
        
        reg_view = sliding_window_view(scaled_reg_full, window_shape = SEQ_LEN, axis = 0).transpose(0, 2, 1)

        fr_view = sliding_window_view(lr_core, HOR)                             


        def _build_X_for(
            idx_arr: np.ndarray
        ) -> np.ndarray:
            
            X = np.empty((len(idx_arr), SEQ_LEN, 1 + n_reg), np.float32)
            
            X[:, :HIST, 0] = pr_view[idx_arr]
            
            X[:, HIST:, 0] = 0.0
            
            X[:, :, 1:] = reg_view[idx_arr]
            
            if self.GFEATS_BLOCK_FUTURE:
            
                blocked = [reg_pos[n] for n in self.GFEATS if n in reg_pos]
            
                if blocked:
            
                    X[:, self.HIST_WINDOW:, 1 + np.array(blocked, dtype = np.int64)] = 0.0
                    
            return X


        def _build_y_seq_for(
            idx_arr: np.ndarray
        ) -> np.ndarray:
        
            return fr_view[idx_arr + HIST].astype(np.float32)[..., None]


        X_tr = _build_X_for(
            idx_arr = train_idx
        )
        
        y_seq_tr = _build_y_seq_for(
            idx_arr = train_idx
        ) 
        
        X_cal = _build_X_for(
            idx_arr = cal_idx
        )
        
        y_seq_cal = _build_y_seq_for(
            idx_arr = cal_idx
        )  

        tick_idx = np.int32(STATE["tickers"].index(t))
        
        tick_id_tr = np.full((len(X_tr),), tick_idx, dtype = np.int32)
        
        tick_id_cal = np.full((len(X_cal),), tick_idx, dtype = np.int32)

        last_r_tr  = lr_core[train_idx + HIST - 1]
        
        last_r_cal = lr_core[cal_idx  + HIST - 1]

        m_hat, phi_hat = self.fit_ar1_baseline(
            log_ret = lr_core[: (train_idx[-1] + HIST)]
        )

        mu_base_seq_tr = self.ar1_seq(
            r_t = last_r_tr, 
            H = HOR, 
            m_hat = m_hat,
            phi_hat = phi_hat
        )[..., None]
        
        mu_base_seq_cal = self.ar1_seq(
            r_t = last_r_cal, 
            H = HOR,
            m_hat = m_hat, 
            phi_hat = phi_hat
        )[..., None]

        y_seq_res_tr = y_seq_tr  - mu_base_seq_tr
        
        y_seq_res_cal = y_seq_cal - mu_base_seq_cal

        if self.USE_AUX_RV_TARGET:
          
            rv_tr = np.std(fr_view[train_idx + HIST], axis = 1, keepdims = True).astype(np.float32)
          
            rv_cal = np.std(fr_view[cal_idx + HIST], axis = 1, keepdims = True).astype(np.float32)
          
            return X_tr, y_seq_res_tr, X_cal, y_seq_res_cal, tick_id_tr, tick_id_cal, rv_tr, rv_cal

        else:
            
            return X_tr, y_seq_res_tr, X_cal, y_seq_res_cal, tick_id_tr, tick_id_cal


    @staticmethod
    def _rolling_std_np(
        x: np.ndarray,
        w: int
    ) -> np.ndarray:
        """
        Rolling sample standard deviation with numerically stable cumulative sums.

        Parameters
        ----------
        x : np.ndarray
        w : int
            Window length.

        Returns
        -------
        np.ndarray
            Rolling σ_t over x with unbiased denominator (w − 1).
        """
    
        x = np.asarray(x, np.float64)
    
        n = x.size
    
        out = np.full(n, np.nan, np.float32)
    
        if n < w or w <= 1:
    
            return out
    
        s1 = np.r_[0.0, np.cumsum(x)]
    
        s2 = np.r_[0.0, np.cumsum(x * x)]
    
        mu_w = (s1[w:] - s1[:-w]) / w
    
        var = (s2[w:] - s2[:-w]) / max(w - 1, 1) - (w / max(w - 1, 1)) * mu_w * mu_w
    
        var = np.maximum(var, 0.0)
    
        out[w-1:] = np.sqrt(var).astype(np.float32)
    
        return out


    @staticmethod
    def _rolling_rq_np(
        x: np.ndarray, 
        w: int
    ) -> np.ndarray:
        """
        Rolling realised quarticity proxy.

        Definition
        ----------
        For window w, define m_t = mean(|x|^4) over the window, then

            rq_t = m_t^{1/4}.

        This proxy approximates the fourth moment scale used in volatility-of-volatility
        contexts.

        Parameters
        ----------
        x : np.ndarray
        w : int

        Returns
        -------
        np.ndarray
        """
   
        x4 = np.abs(np.asarray(x, np.float64)) ** 4
    
        n = x4.size
    
        out = np.full(n, np.nan, np.float32)
    
        if n < w or w <= 0:
    
            return out
    
        s = np.r_[0.0, np.cumsum(x4)]
    
        m = (s[w:] - s[:-w]) / w
    
        out[w-1:] = np.power(np.maximum(m, 0.0), 0.25).astype(np.float32)
    
        return out


    @staticmethod
    def _rolling_bpv_np(
        x: np.ndarray,
        w: int
    ) -> np.ndarray:
        """
        Rolling bipower variation proxy.

        Definition
        ----------
        Let a_t = |x_t| and consider the lag-one product series p_t = a_t a_{t−1}. The
        bipower variation proxy over a window is

            bpv_t = (π / 2) · mean( p_t ).

        Parameters
        ----------
        x : np.ndarray
        w : int

        Returns
        -------
        np.ndarray
        """

        a = np.abs(np.asarray(x, np.float64))
    
        prod = a * np.r_[np.nan, a[:-1]]
    
        prod = np.nan_to_num(prod, nan = 0.0)
    
        n = prod.size
        
        out = np.full(n, np.nan, np.float32)
        
        if n < w or w <= 0:
        
            return out
        
        s = np.r_[0.0, np.cumsum(prod)]
        
        m = (s[w:] - s[:-w]) / w
        
        out[w-1:] = (np.pi / 2.0 * np.maximum(m, 0.0)).astype(np.float32)
        
        return out


    class _SnapshotAverager(Callback):
        """
        Keras callback that stores the last N epoch snapshots and averages them at the end.

        Averaging the last snapshots approximates a cosine-annealed SWA-like procedure
        that can reduce generalisation error without extending training time.
        """

        def __init__(
            self, 
            keep_last = 3
        ):
        
            super().__init__()
        
            self.keep_last = keep_last
        
            self.snaps = []
        
        
        def on_epoch_end(
            self,
            epoch, 
            logs = None
        ):
            """
            Store a deep copy of the model weights at the end of each epoch and keep at most
            `keep_last` snapshots.
            """

            w = self.model.get_weights()
            
            self.snaps.append([np.copy(x) for x in w])
            
            if len(self.snaps) > self.keep_last:
            
                self.snaps.pop(0)
        
        
        def average_into_model(
            self, 
            model
        ):
            """
            Replace the model weights with the arithmetic mean of stored snapshots if any
            exist. Layers are averaged parameter-wise.
            """

            if not self.snaps:
        
                return
        
            avg = []
        
            K = len(self.snaps)
        
            for tensors in zip(*self.snaps):
        
                s = np.zeros_like(tensors[0])
        
                for t in tensors:
        
                    s += t
        
                s /= float(K)
        
                avg.append(s)
        
            model.set_weights(avg)


    def build_state(
        self
    ) -> Dict[str, Any]:
        """
        Assemble the global immutable state required for training and forecasting.

        Main artefacts
        --------------
        • Ticker universe and per-ticker weekly series (price, returns, realised proxies).
        
        • Country-level macro weekly panels and original monthly/quarterly macro levels.
        
        • Analyst fundamentals (Revenue, EPS) resampled to weekly.
        
        • Factor returns (weekly) and bootstrapped factor futures stored in SharedMemory.
        
        • Graph features computed from rolling sparse correlations.
        
        • Cross-asset correlation/covariance structures and copula mixture weights:
        
        – Residuals via factor or AR1 de-noising.
        
        – Empirical correlation with shrinkage and targeted graph-guided shrinkage.
        
        – Temporal AR coefficient ρ (median AR(1) in residuals).
        
        – Gaussian shocks Z_{S,H,M} and, if used, t-copula scale mixture w_{S,H}.

        • Global robust scalers and clipping quantiles for deltas and returns.
        
        • Per-ticker unscaled caches (DEL, REGMAT, LRCORE) to avoid recomputation.

        Returns
        -------
        dict
            The `state_pack` containing all above structures, plus metadata for
            SharedMemory blocks for later access and cleanup.

        Notes
        -----
        All arrays are prepared in `float32` where appropriate to reduce memory footprint.
        """

        self.logger.info("Building global state …")
    
        if self.SAVE_ARTIFACTS and not _os.path.exists(self.ARTIFACT_DIR):
    
            _os.makedirs(self.ARTIFACT_DIR, exist_ok = True)

        fdata = FinancialForecastData()
    
        macro = fdata.macro
    
        r = macro.r

        if self.tickers_arg:
          
            tickers = list(self.tickers_arg)
       
        else:
         
            tickers = ['GOOG', 'KO', 'NVDA', 'TJX', 'PLTR', 'TTWO', 'HOOD', 'MCD']

        close_df = r.weekly_close
      
        dates_all = close_df.index.values
      
        tick_list = close_df.columns.values
      
        price_arr = close_df.to_numpy(dtype = np.float32, copy = False)
      
        T_dates, M = price_arr.shape

        price_rec = np.empty(
            T_dates * M,
            dtype=[("ds", "datetime64[ns]"),
                   ("Ticker", f"U{max(1, max(len(t) for t in tick_list) if len(tick_list) > 0 else 1)}"),
                   ("y", "float32")]
        )
      
        price_rec["ds"] = np.repeat(dates_all, M)
      
        price_rec["Ticker"] = np.tile(tick_list, T_dates)
      
        price_rec["y"] = price_arr.reshape(-1)
      
        price_rec = price_rec[~np.isnan(price_rec["y"])]

        analyst = r.analyst
      
        country_map = {t: str(c) for t, c in zip(analyst.index, analyst["country"])}
      
        country_arr = np.array([country_map.get(t, "") for t in price_rec["Ticker"]])
      
        price_rec = rfn.append_fields(price_rec, "country", country_arr, usemask = False)
      
        price_rec = price_rec[np.lexsort((price_rec["country"], price_rec["ds"]))]

        raw_macro = macro.assign_macro_history_large_non_pct().reset_index()
       
        raw_macro = raw_macro.rename(columns = {"year": "ds"} if "year" in raw_macro else {raw_macro.columns[1]: "ds"})
       
        if isinstance(raw_macro.index, pd.PeriodIndex):
       
            raw_macro.index = raw_macro.index.to_timestamp(how = "end")
       
        ds = raw_macro["ds"]
       
        if is_period_dtype(ds):
       
            raw_macro["ds"] = ds.dt.to_timestamp(how = "end")
       
        elif is_datetime64_any_dtype(ds):
       
            pass
       
        elif ds.dtype == object and len(ds) and isinstance(ds.iloc[0], pd.Period):
       
            raw_macro["ds"] = pd.PeriodIndex(ds).to_timestamp(how = "end")
       
        else:
            raw_macro["ds"] = pd.to_datetime(ds, errors = "coerce")
       
        raw_macro["country"] = raw_macro["ticker"].map(country_map)
       
        macro_clean = raw_macro[["ds", "country"] + self.MACRO_REGRESSORS].dropna()
       
        len_macro_clean = len(macro_clean)

        macro_rec = np.empty(
            len_macro_clean,
            dtype = [("ds", "datetime64[ns]"),
                   ("country", f"U{max(1, max(len(c) for c in macro_clean['country']) if len_macro_clean > 0 else 1)}")]
                 + [(reg, "float32") for reg in self.MACRO_REGRESSORS]
        )
        
        macro_rec["ds"] = macro_clean["ds"].values
        
        macro_rec["country"] = macro_clean["country"].values
        
        for reg in self.MACRO_REGRESSORS:
        
            macro_rec[reg] = macro_clean[reg].to_numpy(dtype = np.float32, copy = False)
        
        macro_rec = macro_rec[np.lexsort((macro_rec["ds"], macro_rec["country"]))]

        unique_countries, first_idx = np.unique(macro_rec["country"], return_index = True)
        
        country_slices: Dict[str, Tuple[int, int]] = {}
        
        for i, ctry in enumerate(unique_countries):
        
            start = first_idx[i]
        
            end = first_idx[i + 1] if i + 1 < len(first_idx) else len(macro_rec)
        
            country_slices[ctry] = (start, end)

        rng_global = np.random.default_rng(self.SEED)
        
        hor = self.HORIZON

        macro_weekly_by_country = {}
        
        macro_weekly_idx_by_country = {}
        
        macro_levels_by_country = {}  

        for ctry, (s, e) in country_slices.items():
        
            rec = macro_rec[s:e]
        
            if len(rec) == 0:
        
                macro_weekly_by_country[ctry] = None
        
                macro_weekly_idx_by_country[ctry] = None
        
                macro_levels_by_country[ctry] = None 
        
                continue
        
            dfm = pd.DataFrame({reg: rec[reg] for reg in self.MACRO_REGRESSORS}, index = pd.DatetimeIndex(rec["ds"])).sort_index()
            
            dfw = (dfm[~dfm.index.duplicated(keep = "first")].resample("W-FRI").mean().ffill().dropna())
            
            macro_weekly_by_country[ctry] = dfw
            
            macro_weekly_idx_by_country[ctry] = dfw.index.values
            
            macro_levels_by_country[ctry] = dfm[~dfm.index.duplicated(keep = "first")]  

        fin_raw = fdata.prophet_data
       
        fd_rec_dict: Dict[str, np.ndarray] = {}
       
        for t in tickers:
       
            df_fd = (fin_raw.get(t, pd.DataFrame())
                     .reset_index()
                     .rename(columns = {"index": "ds", "rev": "Revenue", "eps": "EPS (Basic)"}))
            
            if df_fd.empty:
            
                continue
            
            df_fd["ds"] = pd.to_datetime(df_fd["ds"])
            
            df_fd = df_fd[["ds", "Revenue", "EPS (Basic)"]].dropna()
            
            if df_fd.empty:
            
                continue
            
            rec = np.empty(len(df_fd), dtype = [("ds", "datetime64[ns]"), ("Revenue", "float32"), ("EPS (Basic)", "float32")])
            
            rec["ds"] = df_fd["ds"].values
            
            rec["Revenue"] = df_fd["Revenue"].to_numpy(dtype = np.float32, copy = False)
            
            rec["EPS (Basic)"] = df_fd["EPS (Basic)"].to_numpy(dtype = np.float32, copy = False)
            
            fd_rec_dict[t] = np.sort(rec, order = "ds")

        fac_w: pd.DataFrame = macro.r.factor_weekly_rets()
       
        if set(self.FACTORS).issubset(fac_w.columns):
       
            fac_vals = fac_w[self.FACTORS].dropna().to_numpy(dtype = np.float32)
       
        else:
       
            fac_vals = np.zeros((0, len(self.FACTORS)), np.float32)

        if fac_vals.size == 0:
       
            factor_future_global = np.zeros((self.N_SIMS, hor, len(self.FACTORS)), np.float32)
       
        else:
       
            L = fac_vals.shape[0]
       
            idx = self.stationary_bootstrap_indices(
                L = L, 
                n_sims = self.N_SIMS,
                H = hor, 
                p = self._BOOT_P, 
                rng = rng_global
            )
       
            factor_future_global = fac_vals[idx, :]

        shm_factor = shared_memory.SharedMemory(create = True, size = factor_future_global.nbytes)
        
        np.ndarray(factor_future_global.shape, dtype = factor_future_global.dtype, buffer = shm_factor.buf)[:] = factor_future_global
        
        factor_future_meta = {
            "shm_name": shm_factor.name, 
            "shape": factor_future_global.shape,
            "dtype": str(factor_future_global.dtype)
        }
        
        self._created_shms.append(shm_factor)

        factor_weekly = (fac_w.sort_index().resample("W-FRI").mean().ffill())
        
        factor_weekly_index = factor_weekly.index.values
        
        if set(self.FACTORS).issubset(factor_weekly.columns):
        
            factor_weekly_values = factor_weekly[self.FACTORS].to_numpy(np.float32)
        
        else:
        
            factor_weekly_values = np.zeros((0, len(self.FACTORS)), np.float32)
   

        def _norm_country(
            x
        ) -> str:
        
            try:
        
                if pd.isna(x):
        
                    return "UNK"
        
            except Exception:
        
                pass
        
            s = str(x).strip()
        
            return s if s and s.lower() not in ("nan", "none") else "UNK"

        
        by_country: Dict[str, List[str]] = {}
        
        for t in tickers:
        
            c = _norm_country(
                x = analyst["country"].get(t, None)
            )
        
            by_country.setdefault(c, []).append(t)

        grouped_tickers: List[str] = []
        
        for c in sorted(by_country.keys(), key=str):
        
            grouped_tickers.extend(sorted(by_country[c]))

        moments_by_ticker = {}
        
        weekly_price_by_ticker: Dict[str, Dict[str, np.ndarray]] = {}
        
        fd_weekly_by_ticker: Dict[str, Dict[str, np.ndarray]] = {}

        seq_len = self.SEQUENCE_LENGTH
        
        for t in grouped_tickers:
        
            try:
        
                ctx = (_time_limit(self.TICKER_TIMEOUT_BUILDSTATE_SEC, f"build_state:ticker={t}")
                       if self.ENABLE_TICKER_TIMEOUTS else nullcontext())

                with ctx:
        
                    pr = price_rec[price_rec["Ticker"] == t]

                    if len(pr) == 0:
        
                        continue

                    s = (pd.DataFrame({"ds": pr["ds"], "y": pr["y"]})
                         .set_index("ds").sort_index()["y"]
                         .resample("W-FRI").last().ffill())

                    y = s.to_numpy(dtype=np.float32, copy=False)
        
                    ys = np.maximum(y, self.SMALL_FLOOR)
        
                    lr = np.zeros_like(ys, dtype=np.float32)
        
                    lr[1:] = np.log(ys[1:]) - np.log(ys[:-1])

                    weekly_price_by_ticker[t] = {
                        "index": s.index.values,
                        "y": y,
                        "lr": lr
                    }

                    rv26 = self._rolling_std_np(
                        x = lr,
                        w = 26
                    )
        
                    rv52 = self._rolling_std_np(
                        x = lr,
                        w = 52
                    )
        
                    weekly_price_by_ticker[t]["rv26"] = rv26
        
                    weekly_price_by_ticker[t]["rv52"] = rv52

                    rq26 = self._rolling_rq_np(
                        x = lr,
                        w = 26
                    )
                  
                    rq52 = self._rolling_rq_np(
                        x = lr,
                        w = 52
                    )
                  
                    weekly_price_by_ticker[t]["rq26"] = rq26
                  
                    weekly_price_by_ticker[t]["rq52"] = rq52

                    bpv26 = self._rolling_bpv_np(
                        x = lr,
                        w = 26
                    )
                    
                    bpv52 = self._rolling_bpv_np(
                        x = lr,
                        w = 52
                    )
                    
                    weekly_price_by_ticker[t]["bpv26"] = bpv26
                    
                    weekly_price_by_ticker[t]["bpv52"] = bpv52

                    s_lr = pd.Series(lr, index=s.index)
                   
                    skew = s_lr.rolling(seq_len, min_periods = seq_len).skew().shift(hor)
                   
                    kurt = s_lr.rolling(seq_len, min_periods = seq_len).kurt().shift(hor)

                    moments_by_ticker[t] = np.column_stack((
                        np.nan_to_num(skew.to_numpy(dtype = np.float32), nan = 0.0, posinf = 0.0, neginf = 0.0),
                        np.nan_to_num(kurt.to_numpy(dtype = np.float32), nan = 0.0, posinf = 0.0, neginf = 0.0),
                    )).astype(np.float32, copy = False)

                    if t in fd_rec_dict:
                    
                        df_fd = (pd.DataFrame({
                            "ds": fd_rec_dict[t]["ds"],
                            "Revenue": fd_rec_dict[t]["Revenue"],
                            "EPS (Basic)": fd_rec_dict[t]["EPS (Basic)"],
                        }).set_index("ds").sort_index())

                        fdw = df_fd.resample("W-FRI").last().ffill()

                        fd_weekly_by_ticker[t] = {
                            "index": fdw.index.values,
                            "values": fdw[["Revenue", "EPS (Basic)"]].to_numpy(dtype = np.float32, copy = False)
                        }

            except TimeoutError as e:

                weekly_price_by_ticker.pop(t, None)

                moments_by_ticker.pop(t, None)

                fd_weekly_by_ticker.pop(t, None)

                self.logger.warning("[TIMEOUT] %s — skipping ticker %s", str(e), t)

                continue

            except Exception as e:

                weekly_price_by_ticker.pop(t, None)

                moments_by_ticker.pop(t, None)

                fd_weekly_by_ticker.pop(t, None)

                self.logger.warning("[ERROR] build_state ticker=%s failed (%s) — skipping", t, repr(e), exc_info=True)

                continue

        graph_feats_by_ticker = {}
        
        if self.USE_GRAPH:
            
            master_index = dates_all 
            
            graph_feats_by_ticker = self._compute_graph_features(
                grouped_tickers = grouped_tickers,
                weekly_price_by_ticker = weekly_price_by_ticker,
                master_index = master_index
            )

        corr_meta = None
      
        mix_meta = None
      
        ticker_to_col = {
            t: i for i, t in enumerate(grouped_tickers)
        }

        if self.USE_CORR_SIM:
           
            master_index = r.weekly_close.index.values
           
            lr_df_all = self._aligned_lr_matrix(
                grouped_tickers = grouped_tickers, 
                weekly_price_by_ticker = weekly_price_by_ticker, 
                master_index = master_index
            )

            try:
                
                factors_ref = factor_weekly[self.FACTORS] if set(self.FACTORS).issubset(factor_weekly.columns) else None
            
            except Exception:
            
                factors_ref = None
            
            eps_df = self._residuals_for_corr(
                lr_df = lr_df_all, 
                factors_df = factors_ref,
                mode = ("factors" if factors_ref is not None else "ar1")
            )

            R_emp = self._pairwise_corr_shrunk(
                df = eps_df, 
                min_pairs = 10, 
                alpha = 0.10
            )  

            R_assets = self._targeted_shrink_with_graph(
                R_emp = R_emp, 
                lr_df = lr_df_all, 
                tau = self.GRAPH_TAU, 
                topk = self.GRAPH_TOPK,
                beta_on = 0.15, 
                beta_off = 0.60
            )  

            rho_list = []
            
            for t in eps_df.columns:
            
                e = eps_df[t].to_numpy()
            
                e0 = e[:-1]
                
                e1 = e[1:]
                
                m = np.isfinite(e0) & np.isfinite(e1)
               
                if m.sum() >= 10:
               
                    rho_list.append(np.corrcoef(e0[m], e1[m])[0, 1])
           
            rho = float(np.nanmedian(rho_list)) if len(rho_list) else 0.0
           
            rho = float(np.clip(np.nan_to_num(rho, nan=0.0), 0.0, 0.9))

            S, M = self.N_SIMS, R_assets.shape[0]
           
            Z = self._kronecker_shocks(
                R_assets = R_assets, 
                H = hor,
                S = S, 
                rho = rho, 
                seed = self.SEED ^ 0xC0FFEE
            )

            try:
            
                nu_use = float(self.NU_COPULA) if (self.NU_COPULA is not None) else self._estimate_nu_t(eps_df.to_numpy().ravel())
            
            except Exception:
            
                nu_use = 8.0

            if np.isfinite(nu_use) and nu_use > 2.0:
                
                w = np.random.default_rng(self.SEED ^ 0xBEEF).chisquare(df = nu_use, size = (S, hor)).astype(np.float32)
               
                inv_sqrt = (nu_use / np.clip(w, 1e-6, None)) ** 0.5  
         
            else:
         
                inv_sqrt = np.ones((S, hor), dtype = np.float32)

            shm_Z = shared_memory.SharedMemory(create = True, size = Z.nbytes)
            
            np.ndarray(Z.shape, dtype = Z.dtype, buffer = shm_Z.buf)[:] = Z
            
            corr_meta = {
                "shm_name": shm_Z.name, 
                "shape": Z.shape, 
                "dtype": str(Z.dtype)
            }
            
            self._created_shms.append(shm_Z)

            shm_mix = shared_memory.SharedMemory(create = True, size = inv_sqrt.nbytes)
           
            np.ndarray(inv_sqrt.shape, dtype = inv_sqrt.dtype, buffer = shm_mix.buf)[:] = inv_sqrt
            
            mix_meta = {
                "shm_name": shm_mix.name,
                "shape": inv_sqrt.shape, 
                "dtype": str(inv_sqrt.dtype)
            }
            
            self._created_shms.append(shm_mix)

        align_cache = {}
        
        for t in grouped_tickers:
        
            wp = weekly_price_by_ticker.get(t)
        
            if wp is None:
        
                continue
        
            dates = wp["index"]
        
            ctry_t = analyst["country"].get(t, None)
        
            dfw = macro_weekly_by_country.get(ctry_t)
        
            mw_idx = macro_weekly_idx_by_country.get(ctry_t)
        
            if dfw is None or mw_idx is None:
        
                continue
        
            idx_m = np.searchsorted(mw_idx, dates, side = "right") - 1
        
            valid_m = idx_m >= 0
            
            if factor_weekly_values.shape[0] > 0:
                
                idx_fa = np.searchsorted(factor_weekly_index, dates, side = "right") - 1
                
                valid_fa = idx_fa >= 0
                
            else:
                
                idx_fa = np.zeros_like(idx_m)
                
                valid_fa = np.ones_like(idx_m, dtype = bool)
           
            if t in fd_weekly_by_ticker:
               
                fd_idx = fd_weekly_by_ticker[t]["index"]
               
                idx_fd = np.searchsorted(fd_idx, dates, side = "right") - 1
               
                valid_fd = idx_fd >= 0
            
            else:
            
                idx_fd = None
            
                valid_fd = np.ones(len(dates), bool)
           
            keep = valid_m & valid_fa & (valid_fd if idx_fd is not None else True)
           
            idx_keep = np.nonzero(keep)[0]
           
            align_cache[t] = {
                "idx_m": idx_m, 
                "valid_m": valid_m,
                "idx_fa": idx_fa,
                "valid_fa": valid_fa,
                "idx_fd": idx_fd, 
                "valid_fd": valid_fd,
                "idx_keep": idx_keep
            }


        regs = list(self.ALL_REGRESSORS)
        
        reg_pos = {name: i for i, name in enumerate(regs)}
        
        DEL_by_ticker: Dict[str, np.ndarray] = {}
        
        REGMAT_by_ticker: Dict[str, np.ndarray] = {}
        
        LRCORE_by_ticker: Dict[str, np.ndarray] = {}

        tail_for_scaler = []

        for t in grouped_tickers:
         
            try:
         
                ctx = (_time_limit(self.TICKER_TIMEOUT_CACHEBUILD_SEC, f"unscaled_cache:ticker={t}")
                       if self.ENABLE_TICKER_TIMEOUTS else nullcontext())

                with ctx:
                
                    built = self._build_unscaled_cache_for_ticker(
                        t = t,
                        regs = regs,
                        reg_pos = reg_pos,
                        align_cache = align_cache,
                        macro_weekly_by_country = macro_weekly_by_country,
                        factor_weekly_values = factor_weekly_values,
                        moments_by_ticker = moments_by_ticker,
                        fd_weekly_by_ticker = fd_weekly_by_ticker,
                        fd_rec_dict = fd_rec_dict,
                        weekly_price_by_ticker = weekly_price_by_ticker,
                        graph_feats_by_ticker = graph_feats_by_ticker,
                        analyst = analyst
                    )

                if built is None:
                   
                    continue

                DEL_by_ticker[t] = built["DEL_full"]
            
                REGMAT_by_ticker[t] = built["reg_mat"]
            
                LRCORE_by_ticker[t] = built["lr_core"]

                if built["DEL_full"].shape[0] > 0:
            
                    tail_for_scaler.append(built["DEL_full"][-min(260, built["DEL_full"].shape[0]):])

            except TimeoutError as e:
            
                self.logger.warning("[TIMEOUT] %s — skipping ticker %s", str(e), t)
            
                continue
            
            except Exception as e:
            
                self.logger.warning("[ERROR] cache build ticker=%s failed (%s) — skipping", t, repr(e), exc_info=True)
            
                continue

        if tail_for_scaler:
        
            pool_DEL = np.concatenate(tail_for_scaler, axis = 0)
        
        else:
        
            pool_DEL = np.zeros((1, len(regs)), dtype = np.float32)

        sc_reg_global = RobustScaler().fit(pool_DEL)
       
        sc_reg_global.scale_ = np.maximum(sc_reg_global.scale_, 1e-6)
       
        q_low_global = np.nanpercentile(pool_DEL, 1.0, axis = 0).astype(np.float32)
       
        q_high_global = np.nanpercentile(pool_DEL, 99.0, axis = 0).astype(np.float32)
       
        near_const = (q_high_global - q_low_global) < 1e-8
       
        if np.any(near_const):
       
            q_low_global[near_const]  = (sc_reg_global.center_[near_const] - 3.0 * sc_reg_global.scale_[near_const]).astype(np.float32)
       
            q_high_global[near_const] = (sc_reg_global.center_[near_const] + 3.0 * sc_reg_global.scale_[near_const]).astype(np.float32)

        pool_lr = []
       
        for t, lr_core in LRCORE_by_ticker.items():
       
            if lr_core.size:
       
                pool_lr.append(lr_core[-min(260, lr_core.shape[0]):])
       
        if pool_lr:
       
            pool_lr_arr = np.concatenate(pool_lr).reshape(-1, 1)
       
            ret_scaler_global = RobustScaler().fit(pool_lr_arr)
       
            ret_scaler_global.scale_[ret_scaler_global.scale_ < 1e-6] = 1e-6
       
        else:
       
            ret_scaler_global = self.fit_ret_scaler_from_logret(
                log_ret = np.array([0.0, 0.0], dtype = np.float32)
            )

        global_scalers_meta = {
            "regs": regs,
            "reg_pos": reg_pos,
            "sc_reg_center": sc_reg_global.center_.astype(np.float32),
            "sc_reg_scale": sc_reg_global.scale_.astype(np.float32),
            "q_low": q_low_global.astype(np.float32),
            "q_high": q_high_global.astype(np.float32),
            "ret_center": float(ret_scaler_global.center_.reshape(-1)[0]),
            "ret_scale": float(ret_scaler_global.scale_.reshape(-1)[0]),
        }

        state_pack: Dict[str, Any] = {
            "tickers": grouped_tickers,
            "country_slices": country_slices,
            "fd_rec_dict": fd_rec_dict,
            "next_fc": fdata.next_period_forecast(),
            "latest_price": r.last_price,
            "analyst": analyst,
            "ticker_country": analyst["country"],
            "macro_weekly_by_country": macro_weekly_by_country,
            "moments_by_ticker": moments_by_ticker,
            "macro_weekly_idx_by_country": macro_weekly_idx_by_country,
            "factor_weekly_index": factor_weekly_index,
            "factor_weekly_values": factor_weekly_values,
            "align_cache": align_cache,
            "weekly_price_by_ticker": weekly_price_by_ticker,
            "graph_feats_by_ticker": graph_feats_by_ticker, 
            "fd_weekly_by_ticker": fd_weekly_by_ticker,
            "factor_future_meta": factor_future_meta,
            "corr_shocks_meta": corr_meta,           
            "t_mix_meta": mix_meta,                 
            "ticker_to_col": ticker_to_col,         
            "macro_levels_by_country": macro_levels_by_country, 
            "global_scalers_meta": global_scalers_meta,
            "DEL_by_ticker": DEL_by_ticker,
            "REGMAT_by_ticker": REGMAT_by_ticker,
            "LRCORE_by_ticker": LRCORE_by_ticker,
        }
        
        presence_flags = {
            t: {
                "has_factors": (factor_weekly_values.shape[0] > 0),
                "has_fin": (t in fd_weekly_by_ticker),
                "has_moms": (t in moments_by_ticker),
            } for t in grouped_tickers
        }
        
        state_pack["presence_flags"] = presence_flags

        self.logger.info("Global state built (%d tickers).", len(grouped_tickers))
        
        return state_pack


    def _train_global_and_serialise(
        self,
        state_pack: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train a pooled global direct-H model (or ensemble) using global scalers and
        serialise weights to disk; compute ensemble calibration statistics.

        Procedure
        ---------
        1) Build pooled windows across all eligible tickers using global scalers.
        
        2) Compile the model with mixed loss:
        
        – Quantile pinball loss on per-step quantiles,
        
        – Skewed-t NLL on residuals,
        
        – Optional Huber loss on auxiliary RV target.
        
        Loss weights are modestly tilted towards quantiles.

        3) Optimisation with Adam, cosine-restart schedule, gradient clipnorm=1.0, early
        stopping and last-epoch snapshot averaging.

        4) Save weights per seed and compute calibration terms on a held-out pool:
        sigma_scale, mu_bias, empirical conf_eps.

        Side effects
        ------------
        Adds to `state_pack`:
        
        • "global_model_weight_paths", "global_model_n_reg", "global_model_n_tickers",
        
        • "calibration_meta" with {'sigma_scale', 'mu_bias', 'conf_eps'}.

        Returns
        -------
        dict
            Updated `state_pack`.

        Rationale
        ---------
        A single global model exploits cross-sectional sharing; the ensemble captures
        parameter uncertainty; the calibration aligns model dispersion with empirical
        forecast errors at the horizon level.
        """

        global STATE
        
        _prev_STATE = STATE
        
        STATE = state_pack
        
        try:
        
            if not self.GLOBAL_MODEL_DIR.exists():
        
                self.GLOBAL_MODEL_DIR.mkdir(parents = True, exist_ok = True)
            
            regs = state_pack["global_scalers_meta"]["regs"]
            
            reg_pos = state_pack["global_scalers_meta"]["reg_pos"]
            
            sc_center= state_pack["global_scalers_meta"]["sc_reg_center"]
            
            sc_scale = state_pack["global_scalers_meta"]["sc_reg_scale"]
            
            q_low = state_pack["global_scalers_meta"]["q_low"]
            
            q_high = state_pack["global_scalers_meta"]["q_high"]
            
            ret_center = state_pack["global_scalers_meta"]["ret_center"]
            
            ret_scale = state_pack["global_scalers_meta"]["ret_scale"]

            class _RS:
           
                pass
           
            sc_reg_full = _RS()
            
            sc_reg_full.center_ = sc_center
            
            sc_reg_full.scale_ = sc_scale
            
            ret_scaler_full = _RS()
            
            ret_scaler_full.center_ = np.array([ret_center], np.float32)
            
            ret_scaler_full.scale_ = np.array([ret_scale], np.float32)

            HIST = self.HIST_WINDOW
            
            HOR = self.HORIZON
            
            SEQ_LEN = self.SEQUENCE_LENGTH
           
            macro_idx = [reg_pos[m] for m in self.MACRO_REGRESSORS if m in reg_pos]
           
            factor_idx = [reg_pos[f] for f in self.FACTORS if f in reg_pos]

            pool_X_tr = []
            
            pool_y_tr = []
            
            pool_X_cal = []
            
            pool_y_cal = []
            
            pool_id_tr =[]
            
            pool_id_cal = []
            
            pool_rv_tr = []
            
            pool_rv_cal = []

            for t in state_pack["tickers"]:
          
                analyst = state_pack["analyst"]
          
                ctry = analyst["country"].get(t, None)
          
                if ctry not in state_pack["country_slices"]:
          
                    continue

                DEL = state_pack["DEL_by_ticker"].get(t)
          
                RM = state_pack["REGMAT_by_ticker"].get(t)
          
                LR = state_pack["LRCORE_by_ticker"].get(t)
          
                if DEL is None or RM is None or LR is None:
          
                    continue

                try:
          
                    ctx = (_time_limit(self.TICKER_TIMEOUT_WINDOWS_SEC, f"make_windows:ticker={t}")
                           if self.ENABLE_TICKER_TIMEOUTS else nullcontext())

                    with ctx:
          
                        built = self._make_windows_for_ticker(
                            t = t,
                            ctry = ctry,
                            regs = regs,
                            reg_pos = reg_pos,
                            HIST = HIST,
                            HOR = HOR,
                            SEQ_LEN = SEQ_LEN,
                            sc_reg_full = sc_reg_full,
                            q_low = q_low,
                            q_high = q_high,
                            ret_scaler_full = ret_scaler_full,
                            macro_idx = macro_idx,
                            factors_idx = factor_idx,
                            reg_mat_pre = RM,
                            DEL_full_pre = DEL,
                            lr_core_pre = LR
                        )

                except TimeoutError as e:
          
                    self.logger.warning("[TIMEOUT] %s — skipping ticker %s", str(e), t)
          
                    continue
          
                except Exception as e:
          
                    self.logger.warning("[ERROR] make_windows ticker=%s failed (%s) — skipping", t, repr(e), exc_info=True)
          
                    continue

                if built is None:
          
                    continue
                
                if self.USE_AUX_RV_TARGET:
                
                    X_tr, y_tr, X_cal, y_cal, id_tr, id_cal, rv_tr, rv_cal = built
                
                else:
                
                    X_tr, y_tr, X_cal, y_cal, id_tr, id_cal = built

                if len(X_tr) < 8 or len(X_cal) < 4:
                
                    continue

                pool_X_tr.append(X_tr)
                
                pool_y_tr.append(y_tr)
                
                pool_X_cal.append(X_cal)
                
                pool_y_cal.append(y_cal)
                
                pool_id_tr.append(id_tr)
                
                pool_id_cal.append(id_cal)
                
                if self.USE_AUX_RV_TARGET:
                
                    pool_rv_tr.append(rv_tr)
                    
                    pool_rv_cal.append(rv_cal)


            if not pool_X_tr:
                
                raise RuntimeError("Global training pool is empty; check caches.")

            X_tr_pool = np.concatenate(pool_X_tr, axis = 0).astype(np.float32, copy = False)
         
            y_tr_pool = np.concatenate(pool_y_tr, axis = 0).astype(np.float32, copy = False)
         
            id_tr_pool = np.concatenate(pool_id_tr, axis = 0).astype(np.int32, copy = False)

            if self.USE_AUX_RV_TARGET and len(pool_rv_tr):
         
                rv_tr_pool = np.concatenate(pool_rv_tr, axis = 0).astype(np.float32, copy = False)
         
            else:
                
                rv_tr_pool = None

            X_cal_pool = np.concatenate(pool_X_cal, axis = 0).astype(np.float32, copy = False)
            
            y_cal_pool = np.concatenate(pool_y_cal, axis = 0).astype(np.float32, copy = False)
            
            id_cal_pool = np.concatenate(pool_id_cal, axis = 0).astype(np.int32, copy = False)

            if self.USE_AUX_RV_TARGET and len(pool_rv_cal):
            
                rv_cal_pool = np.concatenate(pool_rv_cal, axis = 0).astype(np.float32, copy = False)
            
            else:
                rv_cal_pool = None

            if self.USE_EMBEDDING:
              
                inputs_tr = {
                    "seq": X_tr_pool,  
                    "tick_id": id_tr_pool
                }
             
                inputs_cal = {
                    "seq": X_cal_pool,
                    "tick_id": id_cal_pool
                }
           
            else:
              
                inputs_tr = X_tr_pool
              
                inputs_cal = X_cal_pool

            targets_tr = (y_tr_pool, y_tr_pool) if not self.USE_AUX_RV_TARGET else (y_tr_pool, y_tr_pool, rv_tr_pool)
            
            targets_cal= (y_cal_pool, y_cal_pool) if not self.USE_AUX_RV_TARGET else (y_cal_pool, y_cal_pool, rv_cal_pool)

            if self.USE_TIME_DECAY_WEIGHTS:
            
                w_chunks = []
            
                for X in pool_X_tr:
            
                    n = len(X)
            
                    w = (0.2 + 0.8 * (np.arange(n, dtype = np.float32) / max(n - 1, 1))).astype(np.float32)
            
                    w_chunks.append(w)
            
                w_all = np.concatenate(w_chunks, axis = 0).astype(np.float32, copy = False)

                if self.USE_AUX_RV_TARGET:
                 
                    sw_tr = (w_all, w_all, w_all)
              
                else:
                
                    sw_tr = (w_all, w_all)
           
            else:
            
                sw_tr = None

            ds_tr = tf.data.Dataset.from_tensor_slices(
                (inputs_tr, targets_tr) if sw_tr is None else (inputs_tr, targets_tr, sw_tr)
            ).shuffle(min(50_000, 5 * self.BATCH), seed = self.SEED, reshuffle_each_iteration = True).batch(self.BATCH, drop_remainder=False).repeat().prefetch(tf.data.AUTOTUNE)

            ds_cal = tf.data.Dataset.from_tensor_slices((inputs_cal, targets_cal)).batch(self.BATCH).cache().prefetch(tf.data.AUTOTUNE)

            n_tickers_total = len(state_pack["tickers"])
          
            n_reg = len(regs)
            
            state_pack["global_model_n_reg"] = n_reg
          
            n_train = int(X_tr_pool.shape[0])
          
            steps_per_epoch = max(1, min(int(np.ceil(n_train / self.BATCH)), 5_000))
          
            first_steps = max(50, 2 * steps_per_epoch)                
            
            seeds = (self.ENSEMBLE_SEEDS if self.USE_ENSEMBLE else [self.SEED])
          
            weight_paths = []

            for s in seeds:
          
                model = self.build_directH_model(
                    n_reg = n_reg,
                    seed = s,
                    n_tickers = (n_tickers_total if self.USE_EMBEDDING else None)
                )
                
                lr_sched = CosineDecayRestarts(
                    initial_learning_rate = self._LR, 
                    first_decay_steps = first_steps, 
                    t_mul = 2.0, 
                    m_mul = 0.8, 
                    alpha = 1e-5
                )
                
                opt = Adam(
                    learning_rate = lr_sched, 
                    clipnorm = 1.0
                )

                if self.USE_AUX_RV_TARGET:
                
                    model.compile(
                        optimizer = opt,
                        loss = [self.pinball_loss_seq, self.skewt_nll_seq, tf.keras.losses.Huber()],
                        loss_weights = [0.7, 0.2, 0.1],
                        jit_compile = False
                    )
                    
                else:
                    
                    model.compile(
                        optimizer = opt,
                        loss = [self.pinball_loss_seq, self.skewt_nll_seq],
                        loss_weights = [0.8, 0.2], 
                        jit_compile = False
                    )

                snap_cb = self._SnapshotAverager(
                    keep_last = 5
                )
               
                callbacks = [EarlyStopping(
                    monitor = "val_loss", 
                    patience = self.PATIENCE, 
                    restore_best_weights = True
                ), snap_cb]

                model.fit(
                    ds_tr, 
                    validation_data = ds_cal,
                    epochs = self.EPOCHS, 
                    callbacks = callbacks, 
                    verbose = 0, 
                    steps_per_epoch = steps_per_epoch
                )
                
                snap_cb.average_into_model(
                    model = model
                )

                wpath = str(self.GLOBAL_MODEL_DIR / f"{self.GLOBAL_MODEL_BASENAME}_{s}.weights.h5")
               
                model.save_weights(wpath)
               
                weight_paths.append(wpath)

            models_for_cal = []
            
            for s, wpath in zip(seeds, weight_paths):
            
                m = self.build_directH_model(
                    n_reg = n_reg,
                    seed = s,
                    n_tickers = (n_tickers_total if self.USE_EMBEDDING else None)
                )
               
                m.load_weights(wpath)
               
                models_for_cal.append(m)

            zero_base_cal = np.zeros_like(y_cal_pool, dtype = np.float32)

            sigma_scale, mu_bias, conf_eps = self._ensemble_calibration(
                trained_models = models_for_cal,
                X_cal = X_cal_pool,                 
                y_seq_cal = y_cal_pool,
                mu_base_seq_cal = zero_base_cal,
                tick_id_cal = (id_cal_pool if self.USE_EMBEDDING else None)
            )

            self._SIGMA_CAL_SCALE = float(sigma_scale)
           
            self._MU_CAL_BIAS = float(mu_bias)
           
            self._CONF_EPS = (float(conf_eps[0]), float(conf_eps[1]))

            state_pack["calibration_meta"] = {
                "sigma_scale": float(sigma_scale),
                "mu_bias": float(mu_bias),
                "conf_eps": (float(conf_eps[0]), float(conf_eps[1])),
            }
            
            state_pack["global_model_weight_paths"] = weight_paths
            
            state_pack["global_model_n_tickers"] = len(state_pack["tickers"])
            
            return state_pack
        
        finally:

            STATE = _prev_STATE        


    def _prepare_Xin_for_all(
        self, 
        state_pack
    ):
        """
        Build one inference window per ticker for batched forecasting.

        For each ticker:
        
        • Assemble and scale the last HIST deltas and returns with global scalers.
        
        • Construct future deltas: macro (bootstrapped release cadence), factor (mean of
        S simulated paths), fundamentals (analyst-driven soft targets distributed over
        H with a cosine weight), and zeroed graph features if configured.
       
        • Assemble X_in of shape (SEQ_LEN, 1 + K): past returns in channel 0, past
        deltas for HIST rows, future deltas for H rows.
       
        • Compute an AR(1) baseline μ_base_seq with clipped φ and m; store current price.

        Returns
        -------
        X_all : np.ndarray, shape (N, SEQ_LEN, 1 + K)
            Batched input sequences for all kept tickers.
        id_all : np.ndarray, shape (N,)
            Integer ticker IDs (for embedding).
        kept : list[str]
            Tickers for which a valid window could be built.
        mu_base : np.ndarray, shape (N, H, 1)
            Per-ticker baseline AR(1) paths.
        cur_price : np.ndarray, shape (N,)
            Latest prices for level conversion.
        """

        gs = state_pack["global_scalers_meta"]
        
        regs = list(gs["regs"])
        
        reg_pos = dict(gs["reg_pos"])
        
        n_reg = len(regs)
        
        HIST = self.HIST_WINDOW
        
        HOR = self.HORIZON
        
        SEQ_LEN = self.SEQUENCE_LENGTH

        class _RS: 
            
            pass
        
        sc_reg_full = _RS()
        
        sc_reg_full.center_ = gs["sc_reg_center"]
        
        sc_reg_full.scale_ = gs["sc_reg_scale"]
        
        ql_full = gs["q_low"]
        
        qh_full = gs["q_high"]
        
        ret_sc  = _RS()
        
        ret_sc.center_ = np.array([gs["ret_center"]], np.float32)
        
        ret_sc.scale_ = np.array([gs["ret_scale"]], np.float32)

        macro_idx = [reg_pos[m] for m in self.MACRO_REGRESSORS if m in reg_pos]
      
        factor_idx = [reg_pos[f] for f in self.FACTORS if f in reg_pos]
      
        fin_rev_idx = reg_pos.get("Revenue", None)
      
        fin_eps_idx = reg_pos.get("EPS (Basic)", None)
       
        g_idx = [reg_pos[g] for g in self.GFEATS if g in reg_pos]

        tickers = state_pack["tickers"]
        
        analyst = state_pack["analyst"]
        
        df_fa = state_pack["factor_weekly_values"]
        
        latest_price = state_pack["latest_price"]

        factor_future = None
        
        fac_meta_global = state_pack.get("factor_future_meta")
        
        shm_f = None
        
        try:
        
            if fac_meta_global:

                shm_f = shared_memory.SharedMemory(name = fac_meta_global["shm_name"])
                
                factor_future = np.ndarray(
                    shape = tuple(fac_meta_global["shape"]),
                    dtype = np.dtype(fac_meta_global["dtype"]),
                    buffer = shm_f.buf
                )
                
        except Exception:
       
            factor_future = None
       
        if factor_future is None:
       
            factor_future = np.zeros((self.N_SIMS, HOR, len(self.FACTORS)), np.float32)
       
        fa_mean_future = factor_future.mean(axis = 0) if factor_idx else None  

        fut_macro_by_country: Dict[str, np.ndarray] = {}
        
        for ctry, df_levels in state_pack.get("macro_levels_by_country", {}).items():
        
            if df_levels is None:
        
                continue
        
            fut_macro_by_country[ctry] = self._sample_macro_future_deltas_release_cadence(
                df_levels = df_levels,
                regs = regs, 
                reg_pos = reg_pos,
                H = HOR, 
                S = 1, 
                rng = np.random.default_rng(self._seed_from_str(ctry + ":macro", base = self.SEED)),
                p_boot = self._BOOT_P
            )[0] 

        X_list = []
        
        id_list = []
        
        kept = []
        
        mu_base_list = []
        
        price_list = []
        
        for t in tickers:
        
            align = state_pack["align_cache"].get(t)
        
            if align is None:
        
                continue

            ctry = analyst["country"].get(t, None)
        
            dfm_ct = state_pack["macro_weekly_by_country"].get(ctry)
        
            if dfm_ct is None or dfm_ct.shape[0] < 12:
        
                continue

            idx_m = np.asarray(align["idx_m"], np.int64)
            
            idx_fa = np.asarray(align["idx_fa"], np.int64)
            
            idx_keep= np.asarray(align["idx_keep"], np.int64)
            
            idx_fd = align["idx_fd"]
            
            if idx_fd is not None:
             
                idx_fd = np.asarray(idx_fd, np.int64)

            sel_m = idx_m[idx_keep]
          
            sel_fa = idx_fa[idx_keep]
            
            sel_fd = idx_fd[idx_keep] if idx_fd is not None else None

            n_m = len(dfm_ct)
           
            n_fa = df_fa.shape[0]
           
            if n_fa > 0:
           
                sel_fa = np.clip(sel_fa, 0, n_fa - 1)
           
                upper_ok = (sel_m < n_m) & (sel_fa < n_fa)
           
            else:
           
                sel_fa = np.zeros_like(sel_m)
           
                upper_ok = (sel_m < n_m)

            idx_keep = idx_keep[upper_ok]
           
            sel_m = sel_m[upper_ok]
           
            sel_fa = sel_fa[upper_ok]
           
            if sel_fd is not None:
           
                sel_fd = sel_fd[upper_ok]

            if len(idx_keep) < SEQ_LEN + 1:
           
                continue

            n_rows = len(idx_keep)
           
            reg_mat = np.zeros((n_rows, n_reg), np.float32)
           
            macro_vals = dfm_ct[self.MACRO_REGRESSORS].to_numpy(np.float32, copy = False)
            
            reg_mat[:, macro_idx] = macro_vals[sel_m]
            
            if n_fa > 0 and factor_idx:
            
                reg_mat[:, factor_idx] = df_fa[sel_fa, :]
            
            if (t in state_pack["fd_weekly_by_ticker"]) and (sel_fd is not None) and (fin_rev_idx is not None) and (fin_eps_idx is not None):
            
                fdw = state_pack["fd_weekly_by_ticker"][t]
            
                vals = fdw["values"][sel_fd, :]
            
                reg_mat[:, [fin_rev_idx, fin_eps_idx]] = vals
            
            moms = state_pack["moments_by_ticker"].get(t)
            
            if moms is not None and ("skew_104w_lag52" in reg_pos) and ("kurt_104w_lag52" in reg_pos):
            
                reg_mat[:, [reg_pos["skew_104w_lag52"], reg_pos["kurt_104w_lag52"]]] = moms[idx_keep, :]

            vol_cols = [n for n in ("rv26", "rv52", "rq26", "rq52", "bpv26", "bpv52") if n in reg_pos]
           
            if vol_cols:
           
                WP = state_pack["weekly_price_by_ticker"][t]
           
                V = np.stack([np.nan_to_num(WP[n][idx_keep], nan = 0.0, posinf = 0.0, neginf = 0.0) for n in vol_cols], axis = 1)
           
                reg_mat[:, [reg_pos[n] for n in vol_cols]] = V

            if self.USE_GRAPH and ("graph_feats_by_ticker" in state_pack):
            
                G = state_pack["graph_feats_by_ticker"].get(t)
            
                if G is not None and G.shape[0] >= len(idx_keep):
            
                    vals = G[idx_keep, :]
            
                    for k, name in enumerate(self.GFEATS):
            
                        if name in reg_pos:
            
                            reg_mat[:, reg_pos[name]] = np.nan_to_num(vals[:, k], nan = 0.0, posinf = 0.0, neginf = 0.0)

            DEL_hist = self.build_delta_matrix(
                reg_mat = reg_mat,
                regs = regs
            )
           
            if macro_idx:
           
                DEL_hist[1:, macro_idx] = DEL_hist[:-1, macro_idx]
           
                DEL_hist[0,  macro_idx] = 0.0
           
            DEL_hist_scaled = self.transform_deltas(
                DEL = DEL_hist, 
                sc = sc_reg_full, 
                q_low = ql_full, 
                q_high = qh_full
            )

            lr_vec = state_pack["weekly_price_by_ticker"][t]["lr"][idx_keep].astype(np.float32, copy = False)
            
            lr_core = lr_vec[1:]  

            r_hist_scaled = ((lr_core - ret_sc.center_[0]) / ret_sc.scale_[0]).astype(np.float32)

            if DEL_hist_scaled.shape[0] < (SEQ_LEN + 4):
                
                continue

            hist_reg_scaled_tail = DEL_hist_scaled[-HIST:, :]
            
            hist_reg_scaled_tail = np.ascontiguousarray(hist_reg_scaled_tail, dtype = np.float32)

            fut_DEL = np.zeros((HOR, n_reg), np.float32)
            
            fm = fut_macro_by_country.get(ctry)
            
            if fm is not None:
            
                fut_DEL += fm

            if factor_idx and fa_mean_future is not None and fa_mean_future.size:
            
                fut_DEL[:, factor_idx] = fa_mean_future

            last_rev = last_eps = self.SMALL_FLOOR
            
            if (t in state_pack["fd_weekly_by_ticker"]) and (sel_fd is not None):
            
                fdw = state_pack["fd_weekly_by_ticker"][t]
            
                last_rev = float(fdw["values"][sel_fd][-1, 0])
            
                last_eps = float(fdw["values"][sel_fd][-1, 1])

            row_fore = state_pack["next_fc"].loc[t] if (t in state_pack["next_fc"].index) else pd.Series()
            
            n_yahoo = float(row_fore.get("num_analysts_y", np.nan))
            
            n_sa = float(row_fore.get("num_analysts",   np.nan))
            
            comb = _analyst_sigmas_and_targets_combined(
                n_yahoo = n_yahoo,
                n_sa = n_sa, 
                row_fore = row_fore
            )
            
            rng_scn = np.random.default_rng(self._seed_from_str(
                s = t + ":scn", 
                base = self.SEED
            ))

            if np.isfinite(comb["targ_rev"]) and last_rev > 0 and (fin_rev_idx is not None):
            
                mu_r = np.log(max(comb["targ_rev"], 1e-12)) - np.log(max(last_rev, 1e-12))
            
                dT_r = rng_scn.normal(loc = mu_r, scale = comb["rev_sigma"], size = self.N_SIMS).astype(np.float32)
            
                wH = 1.0 - np.cos(np.linspace(0, np.pi, HOR, dtype = np.float32))
                
                wH /= wH.sum()
            
                fut_DEL[:, fin_rev_idx] = (dT_r[:, None] * wH[None, :]).mean(axis = 0)

            if np.isfinite(comb["targ_eps"]) and (fin_eps_idx is not None):
               
                mu_e = self.slog1p_signed(
                    x = np.array([comb["targ_eps"]])
                )[0] - self.slog1p_signed(
                    x = np.array([last_eps])
                )[0]
               
                dT_e = rng_scn.normal(loc = mu_e, scale = comb["eps_sigma"], size = self.N_SIMS).astype(np.float32)
               
                wH = 1.0 - np.cos(np.linspace(0, np.pi, HOR, dtype=np.float32))
                
                wH /= wH.sum()
               
                fut_DEL[:, fin_eps_idx] = (dT_e[:, None] * wH[None, :]).mean(axis = 0)

            if self.GFEATS_BLOCK_FUTURE and g_idx:
               
                fut_DEL[:, g_idx] = 0.0

            fut_DEL_scaled = self.transform_deltas(
                DEL = fut_DEL, 
                sc = sc_reg_full, 
                q_low = ql_full, 
                q_high = qh_full
            )

            X_in = np.zeros((self.SEQUENCE_LENGTH, 1 + n_reg), np.float32)
           
            X_in[:HIST, 0] = r_hist_scaled[-HIST:]
           
            X_in[HIST:, 0] = 0.0
           
            X_in[:HIST, 1:] = hist_reg_scaled_tail
           
            X_in[HIST:, 1:] = fut_DEL_scaled

            m_hat, phi_hat = self.fit_ar1_baseline(
                log_ret = lr_core
            )
            
            phi_hat = float(np.clip(phi_hat, -0.3, 0.3))
            
            m_hat = float(np.clip(m_hat, -0.005, 0.005))
            
            last_r = lr_core[-1] if lr_core.size else 0.0
            
            mb = self.ar1_seq(last_r, HOR, m_hat, phi_hat)
            
            if mb.ndim == 2 and mb.shape[0] == 1:  
            
                mb = mb[0]
            
            mu_base_seq = mb.reshape(HOR, 1).astype(np.float32)
            
            cur_p = float(latest_price.get(t, np.nan))
            
            if not np.isfinite(cur_p):
            
                continue

            X_list.append(X_in)
            
            id_list.append(np.int32(tickers.index(t)))
            
            kept.append(t)
            
            mu_base_list.append(mu_base_seq)
            
            price_list.append(cur_p)

        try:
           
            if shm_f is not None:
           
                shm_f.close()
        
        except Exception:
        
            pass

        if not X_list:
        
            return (np.zeros((0, self.SEQUENCE_LENGTH, 1+n_reg), np.float32),
                    np.zeros((0,), np.int32), [],
                    np.zeros((0, HOR, 1), np.float32),
                    np.zeros((0,), np.float32))

        X_all = np.stack(X_list, axis = 0).astype(np.float32, copy = False)
       
        id_all = np.asarray(id_list, np.int32)
       
        mu_base = np.stack(mu_base_list, axis = 0).astype(np.float32, copy = False)
       
        cur_price = np.asarray(price_list, np.float32)
       
        return X_all, id_all, kept, mu_base, cur_price


    def _predict_params_batched(
        self, 
        models,
        X_all, 
        id_all,
        return_per_model = False
    ):
        """
        Run all loaded models to obtain per-step distribution parameters and ensemble
        aggregates.

        For each model k, extract:
        
        • μ_k = μ_max · tanh(raw_μ · σ_k), with σ_k = softplus(raw_σ) + sigma_floor,
        
        • ν_k, λ_k via softplus/tanh transforms.

        Aggregate
        ---------
        • μ_step = mean_k μ_k,
        
        • σ_step^2 = mean_k σ_k^2,
        
        • var_μ = var_k μ_k (ddof=1), then
        sig_step = sqrt(σ_step^2 + var_μ) to reflect both aleatoric and parameter
        uncertainty.

        • ν̄, λ̄ are robust means: component-wise median for ν (lower bounded),
        Fisher-z mean for λ (via arctanh/mean/tanh).

        Parameters
        ----------
        models : list[tf.keras.Model]
        X_all : np.ndarray
        id_all : np.ndarray
        return_per_model : bool

        Returns
        -------
        If `return_per_model` is True:
            (mu_step, sig_step, nu_bar, lam_bar, mus_list, sig2_list)
        Else:
            (mu_step, sig_step, nu_bar, lam_bar)
        """

        mus = []
        
        sig2s = []
        
        nus = []
        
        lams = []
        
        inp = {"seq": X_all, "tick_id": id_all} if (self.USE_EMBEDDING and X_all.shape[0] > 0) else X_all
        
        for m in models:
        
            out = m(inp, training = False)
           
            params = out[1] if isinstance(out, (list, tuple)) else out["dist_head"]
           
            sig = (tf.nn.softplus(params[..., 1:2]) + self.SIGMA_FLOOR).numpy().astype(np.float32)
           
            mu = (self.MU_MAX * tf.tanh(params[..., 0:1] * sig)).numpy().astype(np.float32)
           
            nu = (self.NU_FLOOR + tf.nn.softplus(params[..., 2:3])).numpy().astype(np.float32)
           
            lam = (tf.tanh(params[..., 3:4])).numpy().astype(np.float32)
           
            mus.append(mu)
            
            sig2s.append(sig ** 2)
            
            nus.append(nu)
            
            lams.append(lam)

        mu_step = np.mean(np.stack(mus, axis = 0), axis = 0)
        
        sig2_step = np.mean(np.stack(sig2s, axis = 0), axis = 0)

        var_mu_step = np.var(np.stack(mus, axis = 0), axis = 0, ddof = 1)  
        
        sig_step = np.sqrt(np.maximum(sig2_step + var_mu_step, 1e-12)).astype(np.float32)

        nu_bar = np.maximum(np.median(np.stack(nus, axis = 0), axis = 0), 5.0).astype(np.float32)
       
        lam_bar = np.tanh(np.mean(np.arctanh(np.clip(np.stack(lams, axis = 0), -0.999, 0.999)), axis = 0)).astype(np.float32)

        if return_per_model:
       
            return mu_step, sig_step, nu_bar, lam_bar, mus, sig2s
       
        return mu_step, sig_step, nu_bar, lam_bar


    def forecast_many_batched(
        self,
        state_pack
    ):
        """
        End-to-end batched simulation for all tickers and all ensemble members.

        Simulation model
        ----------------
        Per step h and ticker n, draw the latent standardised innovation x_{s,h,n} from
        a Hansen skewed-t derived transform of a Student-t variate:

        1) Compute constants for ν_h and λ_h (broadcast over h, n):

            scale_const = sqrt( max(ν_h − 2, ε) / max(ν_h, 2 + ε) ),
            (a, b) = Hansen constants; threshold thr = −a / b.

        2) Uniforms → t:
       
        If correlated shocks Z are provided:
        
            – Convert correlated normals z via standard normal CDF to u = Φ(z),
            or use a t-copula: u = T_{ν_cop}( z · t_mix ), then
            z_std = T_{ν_h}^{−1}(u).
        
        Else:
        
            u ~ U(0,1), z_std = T_{ν_h}^{−1}(u).
        
        Standardise: z_unit = z_std · scale_const.

        3) Skew mapping:
        
            y = b · z_unit + a,
        
            x = y / (1 − λ_h) if z_unit < thr, else y / (1 + λ_h).

        4) Per-step mean/scale:
        
            step_draw = μ_h + σ_h · x,
        
            accumulate log-returns over h: logR = ∑_h step_draw.

        Per-model parameters:
        
            μ_h = μ_k,h + μ_base_h + bias_step,
        
            σ_h = σ_k,h · sigma_scale ⊕ macro_sigma_h,  element-wise √(σ_h² + σ_macro(h)²).

        Bounds and conversion:
        
            clip logR to [log(lbp), log(ubp)] and convert to simple returns via expm1.

        Outputs
        -------
        Returns a DataFrame with index = tickers and columns:
        
            • Min/Avg/Max Price from 5th/50th/95th percentiles of simulated returns
            times current price,
        
            • Returns : mean simulated simple return,
        
            • SE      : standard deviation of simulated simple return.

        Also returns an empty 'df_bad' frame placeholder for compatibility.

        Rationale
        ---------
        The skewed-t transform captures asymmetric heavy tails; the t-copula and
        Kronecker shocks supply cross-sectional and temporal dependence. Calibration
        terms align dispersion and bias at the horizon level.
        """
       
        X_all, id_all, kept, mu_base, cur_price = self._prepare_Xin_for_all(
            state_pack = state_pack
        )
        
        if X_all.shape[0] == 0:
        
            return pd.DataFrame(columns = ["Min Price", "Avg Price", "Max Price", "Returns", "SE"]), pd.DataFrame(columns = ["status", "reason"])
        
        if not np.isfinite(X_all).all():
          
            self.logger.warning("X_all contains non-finite values; fixing with nan_to_num.")
          
            X_all = np.nan_to_num(X_all, nan = 0.0, posinf = 0.0, neginf = 0.0)

        N = X_all.shape[0]
        
        H = self.HORIZON

        models = []
      
        n_reg_glob = int(state_pack.get("global_model_n_reg", 0))
      
        n_tickers_glob = int(state_pack.get("global_model_n_tickers", 0))
      
        for wpath in state_pack.get("global_model_weight_paths", []):
      
            m = self.build_directH_model(
                n_reg = n_reg_glob, 
                seed = self.SEED,
                n_tickers = (n_tickers_glob if self.USE_EMBEDDING else None)
            )
            
            m.load_weights(wpath)
            
            models.append(m)
        
        if not models:
        
            raise RuntimeError("No global models loaded")

        mu_step, sig_step, nu_bar, lam_bar, mus_list, sig2_list = self._predict_params_batched(
            models = models, 
            X_all = X_all,
            id_all = id_all, 
            return_per_model = True
        )

        cal = state_pack.get("calibration_meta", {"sigma_scale": 1.0, "mu_bias": 0.0, "conf_eps": (0.0, 0.0)})
       
        sigma_scale = float(cal.get("sigma_scale", 1.0))
       
        bias_step = float(cal.get("mu_bias", 0.0)) / float(H)

        sig_macro_vec = np.zeros((H,), np.float32)
        
        shmF = None
        
        try:
        
            fac_meta = state_pack.get("factor_future_meta")
        
            if fac_meta and (self.SIGMA_MACRO_ALPHA > 0.0):
        
                shmF = shared_memory.SharedMemory(name = fac_meta["shm_name"])
              
                FF = np.ndarray(shape = tuple(fac_meta["shape"]), dtype = np.dtype(fac_meta["dtype"]), buffer = shmF.buf) 
              
                fa_std_t = np.sqrt(np.mean(np.var(FF, axis = 0), axis = -1)).astype(np.float32)
              
                sig_macro_vec = np.clip(self.SIGMA_MACRO_ALPHA * fa_std_t, 0.0, 0.25)
        
        except Exception:
        
            pass
        
        finally:
        
            try:
        
                if shmF is not None:
        
                    shmF.close()
        
            except Exception:
        
                pass
        
        if self.SIGMA_MACRO_EXTRA > 0.0:
        
            sig_macro_vec = np.sqrt(sig_macro_vec ** 2 + (self.SIGMA_MACRO_EXTRA**2)).astype(np.float32)  

        if not np.isfinite(sig_macro_vec).all():
          
            sig_macro_vec = np.nan_to_num(sig_macro_vec, nan = 0.0, neginf = 0.0, posinf = 0.25).astype(np.float32)

        S_total = int(self.N_SIMS)
       
        Z = None
        
        t_mix = None
        
        shmZ = None
        
        shmM = None
      
        try:
      
            metaZ = state_pack.get("corr_shocks_meta")
      
            if metaZ:

                shmZ = shared_memory.SharedMemory(name = metaZ["shm_name"])
                
                Z = np.ndarray(shape = tuple(metaZ["shape"]), dtype = np.dtype(metaZ["dtype"]), buffer = shmZ.buf)  
        
            metaM = state_pack.get("t_mix_meta")
        
            if metaM:
        
                shmM = shared_memory.SharedMemory(name = metaM["shm_name"])
                
                t_mix = np.ndarray(shape=tuple(metaM["shape"]), dtype = np.dtype(metaM["dtype"]), buffer = shmM.buf)  
       
        except Exception:
       
            Z = None
            
            t_mix = None

        if Z is not None:
         
            cols = np.array([state_pack["ticker_to_col"][t] for t in kept], dtype = np.int64)
         
            z_whn_all = Z[:, :, cols]  

        K = max(1, len(models))
      
        base_each = S_total // K
      
        rem = S_total - base_each * K

        macro2 = (np.asarray(sig_macro_vec, np.float32).reshape(1, H) ** 2) if (sig_macro_vec.size == H) else None

        nu_k = nu_bar[..., 0].swapaxes(0, 1)[None, :, :]
       
        lam_k = lam_bar[..., 0].swapaxes(0, 1)[None, :, :]

        a_const, b_const = self._hansen_ab_constants(
            nu = nu_k[0], 
            lam = lam_k[0]
        )
       
        a_const = a_const[None, :, :]
       
        b_const = b_const[None, :, :]
       
        thr_const = (-a_const / b_const)
       
        scale_const = np.sqrt(np.maximum(nu_k - 2.0, 1e-12) / np.maximum(nu_k, 2.0001)).astype(np.float32)
        
        rets_models = []
        
        for k in range(K):

            S_each = base_each + (1 if k < rem else 0)
        
            if S_each <= 0:
        
                continue
            
            mu_k = (mus_list[k] + mu_base + bias_step).astype(np.float32)         
            
            sig_k = (np.sqrt(sig2_list[k]) * sigma_scale).astype(np.float32)       

            mu_k_ = mu_k[...,  0].swapaxes(0, 1)[None, :, :]
            
            sig_k_ = sig_k[..., 0].swapaxes(0, 1)[None, :, :]
            
            if macro2 is not None:
              
                sig_k[..., 0] = np.sqrt(np.maximum(sig_k[..., 0] ** 2 + macro2, 1e-12))

            if Z is not None:
              
                s0 = int(np.sum([base_each + (1 if i < rem else 0) for i in range(k)]))
              
                s1 = s0 + S_each
              
                z_k = z_whn_all[s0:s1, :, :]  
              
                if (t_mix is not None) and (self.NU_COPULA is not None):
              
                    u_k = t_cdf(float(self.NU_COPULA), z_k * t_mix[s0:s1, :, None])
              
                    u_k = np.where(np.isfinite(u_k), u_k, 0.5)
              
                else:
              
                    u_k = n_cdf(z_k)
            
            else:
            
                u_k = np.random.default_rng(self.SEED ^ (0x5151 + k)).uniform(size = (S_each, H, N)).astype(np.float32)

            u_k = np.clip(u_k, 1e-12, 1 - 1e-12).astype(np.float32)

            z_std = t_ppf(np.maximum(nu_k, 2.1001), u_k).astype(np.float32)
            
            z_unit = z_std * scale_const

            y = b_const * z_unit + a_const
           
            x = np.where(z_unit < thr_const, y / (1.0 - lam_k), y / (1.0 + lam_k))
           
            x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
            
            step_draws = mu_k_ + sig_k_ * x
           
            step_draws = np.where(np.isfinite(step_draws), step_draws, 0.0)
           
            logR_k = np.sum(step_draws, axis = 1)  

            rets_models.append(np.expm1(np.clip(logR_k, np.log(config.lbp), np.log(config.ubp)) ))

        rets = np.concatenate(rets_models, axis = 0) if len(rets_models) else np.zeros((S_total, N), np.float32)

        for h in (shmZ, shmM):
           
            try:
           
                if h is not None: h.close()
           
            except Exception:
           
                pass

        q05, q50, q95 = np.nanquantile(rets, [0.05, 0.5, 0.95], axis = 0)
        
        out = pd.DataFrame({
            "Ticker": kept,
            "Min Price": (1.0 + q05) * cur_price,
            "Avg Price": (1.0 + q50) * cur_price,
            "Max Price": (1.0 + q95) * cur_price,
            "Returns": np.nanmean(rets, axis = 0),
            "SE": np.nanstd(rets, axis = 0, ddof = 0),
        }).set_index("Ticker")
        
        print(out)
        
        return out, pd.DataFrame(columns = ["status", "reason"])


    @staticmethod
    def stationary_bootstrap_indices(
        L: int, 
        n_sims: int,
        H: int, 
        p: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Indices for the stationary bootstrap (Politis–Romano) over a length-L series.

        Mechanism
        ---------
        At each horizon step h and for each simulation path:
        • With probability p, start a new block with a fresh random start S0 ∈ {0,…,L−1}.
        • Else continue the previous block, advancing by one.
        The index at h is (current block start + offset since last restart) mod L.

        Parameters
        ----------
        L : int
        n_sims : int
        H : int
        p : float
        rng : np.random.Generator

        Returns
        -------
        np.ndarray, shape (n_sims, H)
            Bootstrapped indices into the historical series.
        """
       
        if L <= 0 or H <= 0 or n_sims <= 0:
       
            return np.zeros((n_sims, H), dtype = np.int64)
        
        R = (rng.random((n_sims, H)) < p)
      
        R[:, 0] = True
      
        G = np.cumsum(R, axis=1) - 1
      
        group_starts = rng.integers(0, L, size = (n_sims, H), dtype = np.int64)
      
        S0 = np.take_along_axis(group_starts, G, axis = 1)
      
        t_idx = np.arange(H, dtype = np.int64)[None, :]
      
        last_restart_idx = np.maximum.accumulate(np.where(R, t_idx, -1), axis = 1)
      
        O = t_idx - last_restart_idx
      
        pos = (S0 + O) % L
      
        return pos.astype(np.int64, copy = False)
    
    
    def _cleanup_shared_mem(
        self
    ):
        """
        Close and unlink all SharedMemory segments created by this instance.

        Safe to call multiple times. Exceptions during close/unlink are caught and logged.
        """
        
        for shm in getattr(self, "_created_shms", []):
        
            try:
        
                shm.close()
        
            except Exception:
        
                pass
        
            try:
        
                shm.unlink()
        
            except Exception:
        
                pass
        
        self._created_shms.clear()


    def run(
        self
    ):
        """
        Orchestrate the full pipeline: state build → global train → batched forecast.

        The function ensures SharedMemory cleanup even on exceptions and attempts a best-
        effort Excel export of results when enabled.

        Returns
        -------
        (df_ok, df_bad) : tuple[pd.DataFrame, pd.DataFrame]
            Forecast summary and a placeholder frame of skips/errors.

        Robustness
        ----------
        • KeyboardInterrupt is handled to allow a clean shutdown.
       
        • SharedMemory is always closed/unlinked in a finally block.
        """

        faulthandler.enable()

        df_ok = pd.DataFrame(columns = ["Min Price", "Avg Price", "Max Price", "Returns", "SE"])
       
        df_bad = pd.DataFrame(columns = ["status", "reason"])

        try:
            
            state_pack = self.build_state()
            
            state_pack = self._train_global_and_serialise(
                state_pack = state_pack
            )
    
            try:
                
                df_ok, df_bad = self.forecast_many_batched(
                    state_pack = state_pack
                )

                if SAVE_TO_EXCEL:
                   
                    try:
                   
                        if _os.path.exists(self.EXCEL_PATH):
                   
                            with pd.ExcelWriter(self.EXCEL_PATH, mode = "a", engine = "openpyxl", if_sheet_exists = "replace") as writer:
                   
                                df_ok.to_excel(writer, sheet_name = "LSTM_Cross_Asset")
                   
                                df_bad.to_excel(writer, sheet_name = "LSTM_Cross_Asset_skips")
                   
                        else:
                   
                            with pd.ExcelWriter(self.EXCEL_PATH, engine = "openpyxl") as writer:
                   
                                df_ok.to_excel(writer, sheet_name = "LSTM_Cross_Asset")
                   
                                df_bad.to_excel(writer, sheet_name = "LSTM_Cross_Asset_skips")
                   
                        self.logger.info("Saved results to %s", self.EXCEL_PATH)
                   
                    except Exception as ex:
                   
                        self.logger.error("Failed to write Excel: %s", ex)

                self.logger.info("Forecasting complete. ok=%d, skipped/error=%d", len(df_ok), len(df_bad))
              
                return df_ok, df_bad

            except KeyboardInterrupt:
             
                self.logger.warning("Interrupted by user; attempting clean shutdown…")
              
                return df_ok, df_bad

        finally:
          
            try:
          
                self._cleanup_shared_mem()
          
            except Exception as ex:
          
                self.logger.error("Shared memory cleanup failed: %s", ex)

    
    
if __name__ == "__main__":
  
    try:
  
        mp.set_start_method("spawn", force = True)
  
    except RuntimeError:
  
        pass

    profiler = cProfile.Profile()

    profiler.enable()

    try:

        forecaster = LSTM_Cross_Asset_Forecaster(
            tickers = config.tickers
        )

        forecaster.run()

    finally:

        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats("cumtime")

        stats.print_stats(20)
