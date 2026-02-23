import numpy as np
import config


def gordon_growth_model(
    ticker,
    dps,
    coe,
    g,
    cp,
    min_spread=0.01
):
    """
    Compute an implied equity value and return distribution using the Gordon Growth Model (GGM).

    This function applies the Gordon Growth Model to estimate an implied price per share from
    dividends per share (DPS), the cost of equity (COE), and a perpetual dividend growth rate (g).
    It then converts the implied price(s) into implied return(s) versus the current price (cp),
    summarising the distribution via its minimum, mean, maximum, and standard deviation.

    Mathematical definition
    -----------------------
    The Gordon Growth Model values an equity security under the assumption that dividends grow
    at a constant perpetual rate. In its standard form:

        P_0 = D_1 / (r - g)

    where:
 
    - P_0 is the intrinsic value (price) at time 0;
 
    - D_1 is the dividend expected in the next period (time 1);
 
    - r is the required return on equity (here, the cost of equity, ``coe``);
 
    - g is the perpetual dividend growth rate.

    In this implementation, ``dps`` is treated as the next-period dividend input (i.e. D_1),
    and the implied price for each observation is computed as:

        p_i = dps_i / (coe - g_i)

    provided that the denominator is well-defined and sufficiently positive.

    Implied return calculation
    --------------------------
    For each valid observation i, the function computes the implied simple return relative to
    the current price ``cp`` as:

        ret_i = (p_i / cp) - 1

    The summary statistics returned are:

        low_ret  = min_i ret_i
 
        avg_ret  = mean_i ret_i
 
        high_ret = max_i ret_i
 
        se       = std_i ret_i

    where the statistics are taken over valid (finite) observations only.

    Safeguards, clipping, and validity conditions
    ---------------------------------------------
    1) Dividend yield screen:
       The function first computes the mean dividend per share:

           dps_mean = mean(dps)

       and the average dividend yield:

           avg_y = dps_mean / cp

       If ``avg_y`` is below 0.01 (i.e. below 1%), the model is skipped because the implied
       valuation becomes highly sensitive to small changes in inputs when dividends are very low.
       In that case the function prints a message and returns five zeros.

    2) Growth-rate clipping:
       The growth rate is constrained to a plausible range:

       - Lower bound: -1.0 (allowing for large negative growth in pathological cases);
      
       - Upper bound: ``coe - 0.02``.

       This ensures that ``coe - g`` is not trivially close to zero by construction (the
       0.02 margin is an additional safety buffer), although the function still enforces an
       explicit minimum spread as described below.

    3) Minimum spread between discount rate and growth rate:
       The denominator is:

           denom_i = coe - g_i

       An observation is treated as valid only if:

       - ``dps_i`` is finite;
    
       - ``denom_i`` is finite;
    
       - ``denom_i > min_spread`` (default 0.01, i.e. at least a 1% gap);
    
       These conditions are enforced through a boolean mask.

    4) Price bounds (winsorisation via clipping):
       The implied price for each valid observation is clipped to remain within a band
       proportional to the current price:

           lb = config.lbp * cp
           ub = config.ubp * cp
           p_i = clip(dps_i / denom_i, lb, ub)

       This limits extreme valuations that can arise from noisy DPS values or a small
       ``coe - g`` spread.

    Parameters
    ----------
    ticker : str
        Ticker symbol used only for informative logging when the model is skipped.
    dps : array-like
        Dividends per share (treated as the next-period dividend D_1 for the GGM calculation).
        May contain NaNs; invalid entries are ignored via masking.
    coe : float
        Cost of equity (required return on equity), denoted r in the GGM formula.
    g : float or array-like
        Perpetual dividend growth rate. May be a scalar or array-like aligned with ``dps``.
        Values are clipped into [-1.0, coe - 0.02] before use.
    cp : float
        Current price per share used to compute dividend yield screening and implied returns.
    min_spread : float, optional
        Minimum required spread between ``coe`` and ``g`` (i.e. minimum allowed ``coe - g``)
        for an observation to be considered valid. Default is 0.01.

    Returns
    -------
    avg_p : float
        Mean implied price across valid observations after applying bounds.
    low_ret : float
        Minimum implied simple return across valid observations.
    avg_ret : float
        Mean implied simple return across valid observations.
    high_ret : float
        Maximum implied simple return across valid observations.
    se : float
        Standard deviation of implied returns across valid observations (a dispersion measure,
        not a statistical standard error of the mean).

    Notes
    -----
    - The function assumes constant perpetual growth and a stable required return, which are
      strong assumptions and may be inappropriate for firms with unstable payout policies.
  
    - The returned dispersion ``se`` is computed as the (population) standard deviation via
      ``numpy.nanstd`` over valid returns; it is not the standard error of the mean unless
      further scaled by the square root of the sample size.
   
    - If no valid observations remain after filtering (e.g. all NaN DPS values), the summary
      statistics may evaluate to NaN depending on NumPy’s behaviour for empty reductions.

    """
    
    dps_mean = np.nanmean(dps)
    
    avg_y = dps_mean / cp
    
    if avg_y < 0.01:
        
        print(f"Skipping GGM for {ticker} as avg dividend yield {avg_y:.4f} < 0.01")
        
        return 0, 0, 0, 0, 0
    
    dps = np.asarray(dps, float)
    
    g = np.clip(g, -1.0, coe - 0.02)   
   
    g = np.asarray(g, float)

    denom = coe - g
    
    mask = np.isfinite(dps) & np.isfinite(denom) & (denom > min_spread) & (cp > 1e-6)
    
    lb = config.lbp * cp
    
    ub = config.ubp * cp

    p = np.full_like(dps, np.nan, dtype = float)
    
    p[mask] = np.clip(dps[mask] / denom[mask], a_min = lb, a_max = ub)

    ret = np.full_like(dps, np.nan, dtype = float)
   
    ret[mask] = (p[mask] / cp) - 1.0
   
    low_ret = float(np.nanmin(ret))
   
    avg_ret = float(np.nanmean(ret))
   
    high_ret = float(np.nanmax(ret))
   
    avg_p = float(np.nanmean(p))
   
    se = float(np.nanstd(ret))

    return avg_p, low_ret, avg_ret, high_ret, se
