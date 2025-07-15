"""
Uses META Prophet with macro and financial regressors to forecast prices under different revenue/EPS scenarios.
"""

import logging
import datetime as dt
from typing import List, Tuple, Union, Dict
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from export_forecast import export_results
from data_processing.financial_forecast_data import FinancialForecastData
from itertools import product
import config



REV_KEYS = ['low_rev_y', 'avg_rev_y', 'high_rev_y',
            'low_rev', 'avg_rev', 'high_rev']

EPS_KEYS = ['low_eps_y', 'avg_eps_y', 'high_eps_y',
            'low_eps', 'avg_eps', 'high_eps']

SCENARIOS = list(product(REV_KEYS, EPS_KEYS))

MACRO_REGRESSORS = ['Interest', 'Cpi', 'Gdp', 'Unemp']

FIN_REGRESSORS = ['Revenue', 'EPS (Basic)']

ALL_REGRESSORS = MACRO_REGRESSORS + FIN_REGRESSORS

def configure_logger() -> logging.Logger:

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        ch = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger

logger = configure_logger()


def prepare_prophet_model(
    df_model: pd.DataFrame, 
    regressors: List[str]
) -> Prophet:
    """
    Prepare and fit a Prophet model using the given DataFrame and list of regressors.
    """
   
    model = Prophet(
        changepoint_prior_scale = 0.05,
        changepoint_range = 0.9,
        daily_seasonality = False,
        weekly_seasonality = True,
        yearly_seasonality = False
    )
   
    for reg in regressors:
        
        model.add_regressor(reg, standardize = True, prior_scale = 0.01)
   
    model.fit(df_model)
   
    return model


def add_financials(
    daily_df: pd.DataFrame, 
    fd: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach the latest available quarterly numbers to each trading day.
    Assumes both frames have a 'ds' column sorted ascending.
    """

    return pd.merge_asof(
        daily_df.sort_values('ds'),
        fd.sort_values('ds'),
        on = 'ds',
        direction = 'backward'
    )


def clip_to_bounds(
    df: pd.DataFrame, 
    price: float
) -> pd.DataFrame:
    """
    Limit yhat, yhat_lower, yhat_upper to [0.2⋅price, 5⋅price].
    Operates in-place and returns the same DataFrame for chaining.
    """

    lower = 0.2 * price
    upper = 5.0 * price

    df[['yhat', 'yhat_lower', 'yhat_upper']] = df[
        ['yhat', 'yhat_lower', 'yhat_upper']
    ].clip(lower = lower, upper = upper)

    return df


def evaluate_forecast(
    model: Prophet, 
    initial: str, 
    period: str, 
    horizon: str
) -> pd.DataFrame:
    """
    Evaluate forecast performance using Prophet's built-in cross-validation and performance metrics.
    """

    try:
        
        cv_results = cross_validation(
            model = model, 
            initial = initial, 
            period = period, 
            horizon = horizon
        )
        
        metrics = performance_metrics(
            df = cv_results
        )
        
        return metrics
    
    except Exception as e:
        
        logger.error("Error during cross-validation: %s", e)
        
        return pd.DataFrame()


def _linear(
    series: pd.Series, 
    end_value: Union[float, np.nan]
) -> pd.Series:
    """
    Create a linear ramp from series.iloc[0] up to end_value over len(series) points.
    If end_value is NaN, return the original series unchanged.
    """

    if pd.isna(end_value):
        
        return series

    start = series.iloc[0]

    length = len(series)

    return pd.Series(
        np.linspace(start, end_value, length),
        index = series.index,
        dtype = series.dtype
    )


def build_base_future(
    model: Prophet,
    forecast_period: int,
    macro_df: pd.DataFrame,
    fin_df: pd.DataFrame,
    last_vals: pd.Series,
    regressors: List[str],
    int_array: Union[np.ndarray, None],
    inf_array: Union[np.ndarray, None],
    gdp_array: Union[np.ndarray, None],
    unemp_array: Union[np.ndarray, None]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a single 'base' future DataFrame containing:
      - ds (history + forecast dates)
      - financial regressors forward‐filled through history
      - macro regressors forward‐filled through history
      - arrays of interpolated macro values for the forecast horizon
    Returns:
      future_base: DataFrame with all regressors filled up to last historical date;
                   new rows for forecast period have placeholder (will be overwritten).
      interp_int_allH, interp_inf_allH, interp_gdp_allH, interp_unemp_allH: 
         numpy arrays of length H = forecast_period, ready to assign into future_base.
    """

    future_base = model.make_future_dataframe(
        periods = forecast_period,
        freq = 'W',
        include_history = True
    )

    future_base = pd.merge_asof(
        future_base,
        fin_df,       
        on = 'ds',
        direction = 'backward'
    )

    future_base = pd.merge_asof(
        future_base,
        macro_df,     
        on = 'ds',
        direction = 'backward'
    )

    future_base[regressors] = future_base[regressors].ffill().bfill()

    horizon_mask = future_base['ds'] > last_vals['ds']
   
    H = horizon_mask.sum()
   
    h_idx = np.arange(1, H + 1)

    if 'Interest' in regressors:
   
        if int_array is not None and len(int_array) > 1 and not np.all(np.isnan(int_array)):
   
            L_int = len(int_array)
   
            seg_int = forecast_period / (L_int - 1)
   
            x_int = np.arange(L_int) * seg_int
   
            interp_int_allH = np.interp(h_idx, x_int, int_array)
   
        else:
   
            interp_int_allH = None
   
    else:
   
        interp_int_allH = None

    if 'Cpi' in regressors:
   
        if inf_array is not None and len(inf_array) > 1 and not np.all(np.isnan(inf_array)):
   
            L_inf = len(inf_array)
   
            seg_inf = forecast_period / (L_inf - 1)
   
            x_inf = np.arange(L_inf) * seg_inf
   
            interp_inf_allH = np.interp(h_idx, x_inf, inf_array)
   
        else:
   
            interp_inf_allH = None
   
    else:
   
        interp_inf_allH = None

    if 'Gdp' in regressors:
   
        if gdp_array is not None and len(gdp_array) > 2 and not np.all(np.isnan(gdp_array)):
   
            start_gdp = gdp_array[1]
            end_gdp = gdp_array[2]
   
            interp_gdp_allH = np.linspace(start_gdp, end_gdp, H)
   
        else:
   
            interp_gdp_allH = None
   
    else:
   
        interp_gdp_allH = None

    if 'Unemp' in regressors:
        
        if unemp_array is not None and len(unemp_array) > 1 and not np.all(np.isnan(unemp_array)):
        
            L_unemp = len(unemp_array)
        
            seg_unemp = forecast_period / (L_unemp - 1)
        
            x_unemp = np.arange(L_unemp) * seg_unemp
        
            interp_unemp_allH = np.interp(h_idx, x_unemp, unemp_array)
        
        else:
        
            interp_unemp_allH = None
    else:
    
        interp_unemp_allH = None

    return (
        future_base,
        horizon_mask.values,      
        interp_int_allH,
        interp_inf_allH,
        interp_gdp_allH,
        interp_unemp_allH
    )
    
def forecast_with_prophet(
    model: Prophet,
    current_price: float,
    last_vals: pd.Series,
    regressors: List[str],
    future_base: pd.DataFrame,
    horizon_mask: np.ndarray,
    interp_int_allH: np.ndarray,
    interp_inf_allH: np.ndarray,
    interp_gdp_allH: np.ndarray,
    interp_unemp_allH: np.ndarray,
    rev_target: Union[float, np.nan] = None,
    eps_target: Union[float, np.nan] = None
) -> pd.DataFrame:
    """
    Generate a forecast using a single caller‐provided 'future_base' DataFrame
    plus precomputed macro interpolation arrays. Only 'Revenue' and 'EPS (Basic)'
    vary by scenario (via rev_target, eps_target). Everything else is in future_base.

    Finally, calls model.predict() and clips to [0.2*price, 5*price].
    """
   
    future = future_base.copy()
   
    if 'Revenue' in regressors:
       
        if not pd.isna(rev_target):
       
            future.loc[horizon_mask, 'Revenue'] = _linear(
                series = future.loc[horizon_mask, 'Revenue'],
                end_value = rev_target
            )
   
    if 'EPS (Basic)' in regressors:
        
        if not pd.isna(eps_target):
        
            future.loc[horizon_mask, 'EPS (Basic)'] = _linear(
                series = future.loc[horizon_mask, 'EPS (Basic)'],
                end_value = eps_target
            )

    if interp_int_allH is not None and 'Interest' in regressors:
        
        future.loc[horizon_mask, 'Interest'] = interp_int_allH
   
    elif 'Interest' in regressors and not pd.isna(interp_int_allH):
       
        future.loc[horizon_mask, 'Interest'] = _linear(
            series = future.loc[horizon_mask, 'Interest'],
            end_value = float(interp_int_allH) if np.isscalar(interp_int_allH) else None
        )

    if interp_inf_allH is not None and 'Cpi' in regressors:
       
        future.loc[horizon_mask, 'Cpi'] = interp_inf_allH
   
    elif 'Cpi' in regressors and not pd.isna(interp_inf_allH):
        
        future.loc[horizon_mask, 'Cpi'] = _linear(
            
            series = future.loc[horizon_mask, 'Cpi'],
            end_value = float(interp_inf_allH) if np.isscalar(interp_inf_allH) else None
        )

    if interp_gdp_allH is not None and 'Gdp' in regressors:
        
        future.loc[horizon_mask, 'Gdp'] = interp_gdp_allH
   
    elif 'Gdp' in regressors and not pd.isna(interp_gdp_allH):
        
        future.loc[horizon_mask, 'Gdp'] = _linear(
            series = future.loc[horizon_mask, 'Gdp'],
            end_value = float(interp_gdp_allH) if np.isscalar(interp_gdp_allH) else None
        )

    if interp_unemp_allH is not None and 'Unemp' in regressors:
        
        future.loc[horizon_mask, 'Unemp'] = interp_unemp_allH
   
    elif 'Unemp' in regressors and not pd.isna(interp_unemp_allH):
        
        future.loc[horizon_mask, 'Unemp'] = _linear(
            series = future.loc[horizon_mask, 'Unemp'],
            end_value = float(interp_unemp_allH) if np.isscalar(interp_unemp_allH) else None
        )

    if future[regressors].isna().any().any():
       
        missing = future[regressors].isna().sum()
       
        raise ValueError(f"NaNs remain in regressors after filling:\n{missing}")

    forecast = model.predict(future)
    
    forecast = clip_to_bounds(
        df = forecast, 
        price = current_price
    )

    return forecast


def forecast_with_prophet_without_fd(
    model: Prophet,
    forecast_period: int,
    macro_df: pd.DataFrame,
    last_vals: pd.Series,
    current_price: float,
    regressors: List[str]
) -> pd.DataFrame:
    """
    Generate a forecast using the fitted Prophet model when there is no financial data.
    Merges macro data, fills missing values, sets regressors to last observed values,
    then predicts. Finally, clip yhat to [0, ∞).
    """

    future = model.make_future_dataframe(
        periods = forecast_period, 
        freq = 'W'
    )
    
    future['ds'] = pd.to_datetime(future['ds'])

    macro_df['ds'] = pd.to_datetime(macro_df['ds'])

    future = future.merge(
        macro_df, 
        on = 'ds', 
        how = 'left'
    )
    
    future.ffill(inplace=True)

    for reg in regressors:
        
        future[reg] = last_vals[reg]

    forecast = model.predict(future)
   
    forecast = clip_to_bounds(
        df = forecast, 
        price = current_price
    )
    
    forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']]

    return forecast


def main() -> None:

    fdata = FinancialForecastData()
   
    macro = fdata.macro
   
    r = macro.r
   
    tickers = r.tickers

    forecast_period = 52  
   
    cv_initial = f"{forecast_period * 3} W"
   
    cv_period = f"{int(forecast_period * 0.5)} W"
   
    cv_horizon = f"{forecast_period} W"

    logger.info("Importing data from Excel ...")

    close = r.weekly_close

    next_fc = fdata.next_period_forecast()
   
    next_macro_dict = macro.macro_forecast_dict()

    latest_prices = r.last_price

    analyst = r.analyst
   
    num_analysts = analyst['numberOfAnalystOpinions']

    raw_macro = macro.assign_macro_history_non_pct()
   
    macro_history = (
        raw_macro
        .reset_index()
        .rename(columns = {'year': 'ds'})
        [['ticker', 'ds'] + MACRO_REGRESSORS]
    )

    if pd.api.types.is_period_dtype(macro_history['ds']):
       
        macro_history['ds'] = macro_history['ds'].dt.to_timestamp()
   
    else:
        
        macro_history['ds'] = pd.to_datetime(macro_history['ds'])

    macro_history.sort_values(['ticker', 'ds'], inplace = True)
    
    macro_groups = macro_history.groupby('ticker')

    fin_data_raw: Dict[str, pd.DataFrame] = fdata.prophet_data
    
    fin_data_processed: Dict[str, pd.DataFrame] = {}

    for tk in tickers:
       
        df_fd = fin_data_raw.get(tk, pd.DataFrame()).reset_index().rename(
            
            columns = {
                'index': 'ds',
                'rev': 'Revenue',
                'eps': 'EPS (Basic)'
            }
        )

        if 'ds' in df_fd.columns:
          
            df_fd['ds'] = pd.to_datetime(df_fd['ds'])

        df_fd.sort_values('ds', inplace = True)
      
        fin_data_processed[tk] = df_fd

    min_price = {}
    max_price = {}
    avg_price = {}
    avg_returns_dict = {}
    scenario_se = {}
    se = {}
    final_rmse = {}

    logger.info("Computing Prophet Forecasts ...")

    for ticker in tickers:
       
        logger.info("Processing ticker: %s", ticker)

        current_price = latest_prices.get(ticker, np.nan)

        if pd.isna(current_price):
           
            logger.warning("No current price for %s. Skipping.", ticker)
           
            continue

        macro_forecasts = next_macro_dict.get(ticker, {})

        int_array   = np.array(macro_forecasts.get('InterestRate', [np.nan]))
      
        inf_array   = np.array(macro_forecasts.get('Consumer_Price_Index_Cpi', [np.nan]))
      
        gdp_array   = np.array(macro_forecasts.get('GDP', [np.nan, np.nan, np.nan]))
      
        unemp_array = np.array(macro_forecasts.get('Unemployment', [np.nan]))

        df_price = pd.DataFrame({
            'ds': close.index,
            'y':  close[ticker]
        })

        df_price['ds'] = pd.to_datetime(df_price['ds'])
      
        df_price.sort_values('ds', inplace = True) 

        if ticker not in macro_groups.groups:
           
            logger.warning("No macro history for %s. Skipping.", ticker)
           
            min_price[ticker] = max_price[ticker] = avg_price[ticker] = scenario_se[ticker] = avg_returns_dict[ticker] = 0.0
           
            continue

        tm = macro_groups.get_group(ticker).drop(columns = 'ticker').copy()

        fd_ticker = fin_data_processed.get(ticker, pd.DataFrame()).copy()

        if fd_ticker.empty:
            
            df_model = df_price.merge(tm, on='ds', how='left')
            
            df_model.ffill(inplace=True)
            
            df_model.dropna(inplace=True)
            
            regressors = MACRO_REGRESSORS
       
        else:
       
            df_price_fd = add_financials(df_price, fd_ticker)  
       
            df_model = df_price_fd.merge(tm, on='ds', how='left')
       
            df_model.ffill(inplace=True)
       
            df_model.dropna(inplace=True)
       
            regressors = ALL_REGRESSORS

        if df_model.empty:
       
            logger.warning("Insufficient data for ticker %s. Skipping.", ticker)
       
            min_price[ticker] = max_price[ticker] = avg_price[ticker] = scenario_se[ticker] = avg_returns_dict[ticker] = 0.0
       
            continue

        try:
       
            m_prophet = prepare_prophet_model(
                df_model = df_model[['ds', 'y'] + regressors], 
                regressors = regressors
            )
       
        except Exception as e:
       
            logger.error("Failed to fit Prophet model for %s: %s", ticker, e)
       
            min_price[ticker] = max_price[ticker] = avg_price[ticker] = scenario_se[ticker] = avg_returns_dict[ticker] = 0.0
       
            continue

        cv_metrics = evaluate_forecast(
            model = m_prophet, 
            initial = cv_initial, 
            period = cv_period, 
            horizon = cv_horizon
        )

        if not cv_metrics.empty:
            
            final_rmse[ticker] = min(cv_metrics['rmse'].iat[-1] / current_price, 2)
        
        else:
           
            final_rmse[ticker] = np.nan

        last_vals = df_model.iloc[-1]

        if fd_ticker.empty:
            
            try:
                forecast = forecast_with_prophet_without_fd(
                    model = m_prophet,
                    forecast_period = forecast_period,
                    macro_df = tm,          
                    last_vals = last_vals,
                    current_price = current_price,
                    regressors = regressors
                )
            
                min_price[ticker] = forecast['yhat_lower'].iloc[-1]
                max_price[ticker] = forecast['yhat_upper'].iloc[-1]
                avg_price[ticker] = forecast['yhat'].iloc[-1]
            
                avg_returns_dict[ticker] = ((avg_price[ticker] / current_price) - 1 if current_price != 0 else np.nan)
            
                scenario_se[ticker] = ((max_price[ticker] - min_price[ticker]) / (2 * 1.96 * current_price) if current_price != 0 else np.nan)

            except Exception as e:
               
                logger.error("Forecasting without financial data failed for %s: %s", ticker, e)
               
                min_price[ticker] = max_price[ticker] = avg_price[ticker] = scenario_se[ticker] = avg_returns_dict[ticker] = 0.0

        else:
            (
                future_base,
                horizon_mask,
                interp_int_allH,
                interp_inf_allH,
                interp_gdp_allH,
                interp_unemp_allH
            ) = build_base_future(
                model = m_prophet,
                forecast_period = forecast_period,
                macro_df = tm,
                fin_df = fd_ticker,
                last_vals = last_vals,
                regressors = regressors,
                int_array = int_array,
                inf_array = inf_array,
                gdp_array = gdp_array,
                unemp_array = unemp_array
            )

            results = []
            
            for rev_key, eps_key in SCENARIOS:
                
                label = f"{rev_key}|{eps_key}"
                
                rev_target = next_fc.at[ticker, rev_key]
                eps_target = next_fc.at[ticker, eps_key]

                try:
                
                    fc = forecast_with_prophet(
                        model = m_prophet,
                        current_price = current_price,
                        last_vals = last_vals,
                        regressors = regressors,
                        future_base = future_base,
                        horizon_mask = horizon_mask,
                        interp_int_allH = interp_int_allH,
                        interp_inf_allH = interp_inf_allH,
                        interp_gdp_allH = interp_gdp_allH,
                        interp_unemp_allH = interp_unemp_allH,
                        rev_target = rev_target,
                        eps_target = eps_target
                    )
                    
                    yhat_val = fc['yhat'].iloc[-1]
                    
                    yhat_lower = fc['yhat_lower'].iloc[-1]
                    
                    yhat_upper = fc['yhat_upper'].iloc[-1]
               
                except Exception as e:
               
                    logger.error("Scenario %s failed for %s: %s", label, ticker, e)
               
                    yhat_val, yhat_lower, yhat_upper = 0.0, 0.0, 0.0

                results.append({
                    'Ticker':    ticker,
                    'Scenario':  label,
                    'RevTarget': rev_target,
                    'EpsTarget': eps_target,
                    'yhat':      yhat_val,
                    'yhat_lower': yhat_lower,
                    'yhat_upper': yhat_upper
                })

            if not results:
                
                logger.warning("No scenarios available for ticker %s. Skipping.", ticker)
                
                min_price[ticker] = max_price[ticker] = avg_price[ticker] = scenario_se[ticker] = avg_returns_dict[ticker] = 0.0
           
            else:
           
                scenario_df = (
                    pd.DataFrame(results)
                    .set_index('Ticker')
                    .sort_index()
                )

                min_price[ticker] = scenario_df['yhat_lower'].min()
                max_price[ticker] = scenario_df['yhat_upper'].max()

                all_y = scenario_df['yhat'].values
              
                all_low = scenario_df['yhat_lower'].values
              
                all_high= scenario_df['yhat_upper'].values
            
                scenario_array = np.concatenate([all_y, all_low, all_high])

                avg_price[ticker] = (all_y.mean() if all_y.size > 0 else 0.0)
              
                returns_arr = ((scenario_array / current_price) - 1 if (current_price != 0 and scenario_array.size > 0) else np.zeros_like(scenario_array))
              
                avg_returns_dict[ticker] = (returns_arr.mean() if returns_arr.size > 0 else 0.0)
            
                scenario_vol = (returns_arr.std() if returns_arr.size > 0 else 0.0)
              
                scenario_se[ticker] = (scenario_vol / np.sqrt(num_analysts[ticker]) if (num_analysts[ticker] > 0) else 0.0)

    max_rmse = max(pd.Series(final_rmse).dropna())
    
    for ticker in tickers:

        if ticker in final_rmse:

            if pd.isna(final_rmse[ticker]):
             
                se[ticker] = np.sqrt(scenario_se[ticker]**2 + (max_rmse**2))
            else:
               
                se[ticker] = np.sqrt((scenario_se[ticker]**2) + (final_rmse[ticker]**2))

        else:
          
            se[ticker] = 0

        logger.info(
            "Ticker: %s, Low: %.2f, Avg: %.2f, High: %.2f, Returns: %.4f, SE: %.4f", 
            ticker, 
            min_price[ticker], 
            avg_price[ticker],
            max_price[ticker],
            avg_returns_dict[ticker],
            se[ticker]
        )

    prophet_results = pd.DataFrame({
        'Ticker': tickers,
        'Current Price': [latest_prices.get(tk, np.nan) for tk in tickers],
        'Avg Price': [avg_price.get(tk, np.nan) for tk in tickers],
        'Low Price': [min_price.get(tk, np.nan) for tk in tickers],
        'High Price': [max_price.get(tk, np.nan) for tk in tickers],
        'Returns': [avg_returns_dict.get(tk, np.nan) for tk in tickers],
        'SE': [se.get(tk, np.nan) for tk in tickers]
    }).set_index('Ticker')

    sheets_to_write = {
        "Prophet Pred": prophet_results,
    }

    export_results(
        sheets = sheets_to_write, 
        output_excel_file = config.MODEL_FILE
    )
    
    logger.info("Prophet forecasting, cross-validation, and export completed.")
    
    
if __name__ == "__main__":
    main()
