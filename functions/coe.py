import pandas as pd

def calculate_cost_of_equity(
    tickers: list[str],
    rf: float,
    beta_series: pd.Series,
    spx_expected_return: float,
    crp_df: pd.DataFrame,
    currency_bl_df: pd.DataFrame,
    country_to_pair: pd.DataFrame,
    ticker_country_map: dict[str, str]
) -> pd.DataFrame:
    """
    Calculate cost of equity for each ticker:
        COE = rf + beta * ((spx_expected_return - rf) + country_risk + currency_risk)

    Args:
        tickers: List of ticker symbols.
        rf: Risk-free rate (e.g. 0.046 for 4.6%).
        beta_series: Series indexed by ticker, values are beta.
        spx_expected_return: Posterior expected return for S&P500 (^GSPC) from BL.
        crp_df: DataFrame from RatioData.crp(), index=country, column 'CRP'.
        currency_bl_df: DataFrame from run_black_litterman_on_currency(), index=currency code.
        ticker_country_map: Mapping from ticker to its country name.

    Returns:
        DataFrame indexed by ticker with columns ['Beta','Country','CRP','Currency Premium','COE'].
    """
    records = []
    def_cur = 'GBPUSD'
    crp_mean = crp_df['CRP'].mean()
    for t in tickers:
        beta = beta_series.loc[t]
        country = ticker_country_map.get(t, 'NA')
        crp_val = crp_df['CRP'].get(country, crp_mean)
        ccy = country_to_pair.get(country, def_cur)
        curr_prem = currency_bl_df.at[ccy]
        coe = rf + beta * max((max((spx_expected_return - rf), 0) + crp_val + curr_prem), 0)
        records.append({
            'Ticker': t,
            'Country': country,
            'Beta': beta,
            'CRP': crp_val,
            'Currency Premium': curr_prem,
            'COE': coe
        })
    return pd.DataFrame(records).set_index('Ticker')
