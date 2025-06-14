"""
Aggregates predictions from multiple models and derives a blended return forecast per ticker.
"""

import numpy as np
import pandas as pd
import datetime as dt
import logging
from typing import Dict, Tuple
from data_processing.ratio_data import RatioData
from functions.export_forecast import export_results

today = dt.date.today()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

EXCEL_OUT_FILE   = f'Portfolio_Optimisation_Forecast_{today}.xlsx'
EXCEL_IN_FILE    = f'Portfolio_Optimisation_Data_{today}.xlsx'

MIN_STD        = 1e-3
RET_CLIP_LOW   = -0.80
RET_CLIP_HIGH  =  5.00
MAX_MODEL_WT   = 0.25

def ensure_headers_are_strings(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col) if col is not None else '' for col in df.columns]
    if df.index.name is None:
        df.index.name = 'Index'
    else:
        df.index.name = str(df.index.name)
    return df


def fix_header_cells(ws):
    for cell in ws[1]:
        cell.value = str(cell.value) if cell.value is not None else ''


class PortfolioOptimizer:
    def __init__(self, excel_file: str, ratio_data: RatioData):
        self.excel_file_out = excel_file
        self.excel_file_in = EXCEL_IN_FILE
        self.ratio_data = ratio_data
        self.today = dt.date.today()
        self.LOWER_PERCENTILE = 25
        self.UPPER_PERCENTILE = 75
        self.NINETY_PERCENTILE = 90
        self._load_all_data()

    def _load_all_data(self) -> None:
        xls = pd.ExcelFile(self.excel_file_out)

        self.wsb = (
            xls.parse('Sentiment Findings',
                      usecols=['ticker', 'avg_sentiment', 'mentions'],
                      index_col='ticker')
            .sort_index()
        )
        self.wsb.index = self.wsb.index.str.upper()

        model_sheets = {
            'Prophet Pred':    'Prophet',
            'Analyst Target':  'AnalystTarget',
            'Exponential Returns': 'EMA',
            'Lin Reg Returns': 'LinReg',
            'DCF':             'DCF',
            'DCFE':            'DCFE',
            'Daily Returns':   'Daily',
            'RI':              'RI',
            'CAPM BL Pred':    'CAPM',
            'SARIMAX Monte Carlo': 'SARIMAX',
            'Rel Val Pred':    'RelVal'
        }
        self.models: Dict[str, pd.DataFrame] = {}
        for sheet_name, name in model_sheets.items():
            cols = ['Ticker', 'Returns', 'SE'] + (
                ['Current Price'] if sheet_name == 'Analyst Target' else []
            )
            df = (
                xls.parse(sheet_name, usecols=cols, index_col=0)
                .sort_index()
            )
            self.models[name] = df

        analyst_cols = [
            'Ticker', 'dividendYield', 'recommendationKey', 'sharesShort', 'sharesShortPriorMonth',
            'sharesOutstanding', 'beta', 'earningsGrowth', 'revenueGrowth', 'debtToEquity',
            'Return on Assets', 'returnOnEquity', 'priceToBook', 'trailingEps', 'forwardEps',
            'Gross Margin', 'Current Price', 'Low Price', 'numberOfAnalystOpinions',
            'Net Income', 'Operating Cash Flow', 'Previous Return on Assets',
            'Long Term Debt', 'Previous Long Term Debt', 'Current Ratio', 'Previous Current Ratio',
            'New Shares Issued', 'Previous Gross Margin', 'Asset Turnover',
            'Previous Asset Turnover', 'Insider Purchases', 'Avg EPS Estimate', 'marketCap'
        ]
        self.analyst_df = (
            xls.parse('Analyst Data', usecols=analyst_cols, index_col=0)
            .sort_index()
        )
        if self.analyst_df.index.dtype == object:
            self.analyst_df.index = self.analyst_df.index.str.upper()

        self.latest_prices = self.ratio_data.last_price
        self.tickers = self.latest_prices.index.tolist()

        self.signal_scores = (
            pd.read_excel(self.excel_file_in,
                          sheet_name="Signal Scores",
                          index_col=0)
            .iloc[-1]
        )
    def short_score(self,
                    shares_short: pd.Series,
                    shares_outstanding: pd.Series,
                    shares_short_prior: pd.Series
                ) -> pd.Series:
        """
        Compute a short interest score based on current and prior shares short.
        """

        common_idx = (
            shares_short.index
            .intersection(shares_outstanding.index)
            .intersection(shares_short_prior.index)
        )
        curr = shares_short.reindex(common_idx).fillna(0)
        out = shares_outstanding.reindex(common_idx)
        prior = shares_short_prior.reindex(common_idx).fillna(0)

        ratio = curr.div(out).fillna(0)

        ratio_nz = ratio[ratio != 0].dropna()
        if ratio_nz.empty:
            return pd.Series(0.0, index=common_idx)

        upper = np.percentile(ratio_nz, self.UPPER_PERCENTILE)

        scores = pd.Series(0.0, index=common_idx)

        scores -= (ratio >= upper).astype(int)

        prior_ratio = prior.div(out).fillna(0)
        mask_prior_nz = prior > 0
        scores -= ((ratio >= 1.05 * prior_ratio) & mask_prior_nz).astype(int)

        scores += (ratio <= 0.95 * prior_ratio).astype(int)

        return scores.astype(int)


    def wsb_score(self, base_score: pd.Series) -> pd.Series:
        """
        Adjust the base score based on sentiment from an external source.
        """

        common = self.wsb.index.intersection(base_score.index)

        adjusted = base_score.copy()

        wsb_slice = self.wsb.loc[common]

        pos = wsb_slice['avg_sentiment'] > 0

        neg = wsb_slice['avg_sentiment'] < 0

        high_mentions = wsb_slice['mentions'] > 4

        adjusted.loc[common] += pos.astype(int)

        adjusted.loc[common] += (pos & high_mentions).astype(int)

        adjusted.loc[common] -= neg.astype(int)

        adjusted.loc[common] -= (neg & high_mentions).astype(int)

        return adjusted
    
    
    def mcap_score(self, mcap: pd.Series, score: pd.Series) -> pd.Series:
        
        high_mcap = np.percentile(mcap[mcap != 0].dropna(), self.UPPER_PERCENTILE)
        
        score += (mcap > high_mcap).astype(int)
        
        return score
        

    def earnings_growth_score(self, earnings_growth: pd.Series, score: pd.Series, ind_earnings_growth: pd.Series) -> pd.Series:
        """
        Adjust the score based on earnings growth.
        """

        common_idx = earnings_growth.index.intersection(score.index)

        eg = earnings_growth.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)

        eg_high = np.percentile(eg[eg != 0].dropna(), self.UPPER_PERCENTILE)

        sc += (eg > 0).astype(int)

        sc -= (eg < 0).astype(int)

        sc += (eg > eg_high).astype(int)
        
        sc +=(eg > ind_earnings_growth).astype(int)

        return sc


    def operating_cash_flow_score(self, operating_cash_flow: pd.Series, score: pd.Series) -> pd.Series:
        """
        Adjust the score based on operating cash flow.
        """

        common_idx = operating_cash_flow.index.intersection(score.index)

        ocf = operating_cash_flow.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)

        sc += (ocf > 0).astype(int)

        return sc


    def revenue_growth_score(self, rev_growth: pd.Series, score: pd.Series, ind_rvg: pd.Series) -> pd.Series:
        """
        Compare revenue growth to an industry baseline.
        """

        common_idx = rev_growth.index.intersection(score.index).intersection(ind_rvg.index)

        rg = rev_growth.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)

        iv = ind_rvg.reindex(common_idx).fillna(0)

        nonzero_idx = rg[rg != 0].index

        sc.loc[nonzero_idx] += (rg.loc[nonzero_idx] > iv.loc[nonzero_idx]).astype(int)

        sc.loc[nonzero_idx] -= (rg.loc[nonzero_idx] < iv.loc[nonzero_idx]).astype(int)

        return sc


    def return_on_equity_score(self, return_on_equity: pd.Series, score: pd.Series, ind_roe: pd.Series) -> pd.Series:
        """
        Adjust the score based on return on equity compared to an industry baseline.
        """

        common_idx = return_on_equity.index.intersection(score.index).intersection(ind_roe.index)

        roe = return_on_equity.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)

        iro = ind_roe.reindex(common_idx).fillna(0)

        nonzero_idx = roe[roe != 0].index

        sc.loc[nonzero_idx] += (roe.loc[nonzero_idx] > iro.loc[nonzero_idx]).astype(int)

        sc.loc[nonzero_idx] -= (roe.loc[nonzero_idx] < iro.loc[nonzero_idx]).astype(int)
        
        sc.loc[nonzero_idx] += (roe.loc[nonzero_idx] > 0).astype(int)

        return sc
    
    def return_on_assets_score(self, return_on_assets: pd.Series, score: pd.Series, ind_roa: pd.Series) -> pd.Series:
        """
        Adjust the score based on return on assets.
        """

        common_idx = return_on_assets.index.intersection(score.index)

        roa = return_on_assets.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)
        
        nonzero_idx = roa[roa != 0].index
        
        sc.loc[nonzero_idx] += (roa.loc[nonzero_idx] > 0).astype(int)
        
        sc.loc[nonzero_idx] -= (roa.loc[nonzero_idx] < 0).astype(int)
        
        sc.loc[nonzero_idx] += (roa.loc[nonzero_idx] > ind_roa.loc[nonzero_idx]).astype(int)
        
        sc.loc[nonzero_idx] -= (roa.loc[nonzero_idx] < ind_roa.loc[nonzero_idx]).astype(int)

        return sc


    def price_to_book_score(self, price_to_book: pd.Series, score: pd.Series, ind_pb: pd.Series) -> pd.Series:
        """
        Adjust the score based on price-to-book ratio compared to an industry baseline.
        """

        common_idx = price_to_book.index.intersection(score.index).intersection(ind_pb.index)

        pb = price_to_book.reindex(common_idx).fillna(0)

        sc = score.reindex(common_idx, fill_value=0)

        ipb = ind_pb.reindex(common_idx).fillna(0)

        mask = ~pb.index.astype(str).to_series().str.endswith('.L')

        local_idx = pb.index[mask]

        sc.loc[local_idx] += (pb[mask] <= 1).astype(int)

        sc.loc[local_idx] += (pb[mask] < ipb[mask]).astype(int)

        sc.loc[local_idx] -= (pb[mask] > ipb[mask]).astype(int)

        return sc


    def ep_score(self, trailing_eps: pd.Series, forward_eps: pd.Series, price: pd.Series,
                 score: pd.Series, ind_pe: pd.Series,
                 eps_1y: pd.Series) -> pd.Series:
        """
        Adjust the score based on EPS ratios.
        """

        slist = [trailing_eps, forward_eps, price, score, ind_pe]

        common_idx = slist[0].index

        for s in slist[1:]:

            common_idx = common_idx.intersection(s.index)

        teps, feps, pr, sc, ipe, eps_1y = [s.reindex(common_idx).fillna(0) for s in (trailing_eps, forward_eps, price, score, ind_pe, eps_1y)]

        with np.errstate(divide='ignore', invalid='ignore'):

            trailing_ratio = np.where(teps != 0, pr / teps, np.inf)

            forward_ratio = np.where(feps != 0, pr / feps, np.inf)
            
            ratio_1y = np.where(eps_1y != 0, pr / eps_1y, np.inf)

        trailing_series = pd.Series(trailing_ratio, index=common_idx)

        forward_series = pd.Series(forward_ratio, index=common_idx)
        
        series_1y = pd.Series(ratio_1y, index=common_idx)

        mask_no_l = ~pd.Series(common_idx, index=common_idx).str.endswith('.L')

        valid_idx = common_idx[mask_no_l]

        sc.loc[valid_idx] += (forward_series[valid_idx] < ipe[valid_idx]).astype(int)

        sc.loc[valid_idx] -= (forward_series[valid_idx] > ipe[valid_idx]).astype(int)

        sc.loc[valid_idx] += (forward_series[valid_idx] < trailing_series[valid_idx]).astype(int)
        
        sc.loc[valid_idx] -= (forward_series[valid_idx] > trailing_series[valid_idx]).astype(int)
        
        sc.loc[valid_idx] += (series_1y[valid_idx] > trailing_series[valid_idx]).astype(int)
        
        sc.loc[valid_idx] -= (series_1y[valid_idx] < trailing_series[valid_idx]).astype(int)

        return sc

    def compute_combination_forecast(
        self,
        region_indicators: Dict[str, pd.Series]
    ) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, pd.Series], pd.DataFrame]:
        """
        Compute combination returns, volatility, final score, model weights, and breakdown.
        """
        names = list(self.models.keys())
        rets = [self.models[n]['Returns'] for n in names]
        ses  = [self.models[n]['SE']      for n in names]

        a = self.analyst_df
        div = a['dividendYield'] / 100
        recommendation = a['recommendationKey']
        shares_short = a['sharesShort']
        shares_outstanding = a['sharesOutstanding']
        shares_short_prior = a['sharesShortPriorMonth']
        beta = a['beta']
        earnings_growth = a['earningsGrowth']
        rev_growth = a['revenueGrowth']
        debt_to_equity = a['debtToEquity']
        roa = a['Return on Assets']
        roe = a['returnOnEquity']
        pb = a['priceToBook']
        teps = a['trailingEps']
        feps = a['forwardEps']
        gross_margin = a['Gross Margin']
        price = a['Current Price']
        nY = a['numberOfAnalystOpinions']
        lower_target = a['Low Price']
        insider_purchases = a['Insider Purchases']
        net_income = a['Net Income']
        operating_cashflow = a['Operating Cash Flow']
        prev_roa = a['Previous Return on Assets']
        long_debt = a['Long Term Debt']
        prev_long_debt = a['Previous Long Term Debt']
        current_ratio = a['Current Ratio']
        prev_current_ratio = a['Previous Current Ratio']
        shares_issued = a['New Shares Issued']
        prev_gm = a['Previous Gross Margin']
        at = a['Asset Turnover']
        prev_at = a['Previous Asset Turnover']
        eps_1y = a['Avg EPS Estimate']
        mcap = a['marketCap']

        ind_pe  = region_indicators['PE']
        ind_pb  = region_indicators['PB']
        ind_roe = region_indicators['ROE']
        ind_roa = region_indicators['ROA']
        ind_rvg = region_indicators['RevG']
        ind_eg  = region_indicators.get('EarningsG', pd.Series(0, index=price.index))

        all_series = rets + ses + [
            div, recommendation, shares_short, shares_outstanding,
            shares_short_prior, beta, earnings_growth, rev_growth,
            debt_to_equity, roa, roe, pb, teps, feps,
            gross_margin, price, nY, lower_target, insider_purchases,
            net_income, operating_cashflow, prev_roa, long_debt,
            prev_long_debt, current_ratio, prev_current_ratio,
            shares_issued, prev_gm, at, prev_at, eps_1y, mcap,
            ind_pe, ind_pb, ind_roe, ind_roa, ind_rvg, ind_eg
        ]
        common_idx = set(all_series[0].index)
        for s in all_series[1:]:
            common_idx &= set(s.index)
        common_idx = sorted(common_idx)

        rets = [r.reindex(common_idx).clip(RET_CLIP_LOW, RET_CLIP_HIGH) for r in rets]
        ses  = [s.reindex(common_idx).clip(lower=MIN_STD) for s in ses]

        n_models = len(names)
        fund_series = [x.reindex(common_idx).fillna(0)
                    for x in all_series[2 * n_models:]]
        (
            div, recommendation, shares_short, shares_outstanding,
            shares_short_prior, beta, earnings_growth, rev_growth,
            debt_to_equity, roa, roe, pb, teps, feps,
            gross_margin, price, nY, lower_target, insider_purchases,
            net_income, operating_cashflow, prev_roa, long_debt,
            prev_long_debt, current_ratio, prev_current_ratio,
            shares_issued, prev_gm, at, prev_at, eps_1y, mcap,
            ind_pe, ind_pb, ind_roe, ind_roa, ind_rvg, ind_eg
        ) = fund_series

        valid = [~((r.abs() > 10) | (s <= MIN_STD)) for r, s in zip(rets, ses)]

        inv_var = [np.where(v, np.sqrt(1 / s), 0.0) for v, s in zip(valid, ses)]
        tot_inv = sum(inv_var)
        raw_w = [iv / tot_inv for iv in inv_var]

        def cap_norm(w_arr, cap=MAX_MODEL_WT, mask=None):
            final = np.minimum(w_arr, cap)
            if mask is not None:
                final = np.where(mask, final, 0.)
            for _ in range(1000):
                deficit = 1 - final.sum(axis=0)
                if np.all(deficit <= 1e-8):
                    break
                room = np.maximum(cap - final, 0.)
                if mask is not None:
                    room = np.where(mask, room, 0.)
                for j, d in enumerate(deficit):
                    if d <= 0 or room[:, j].sum() == 0:
                        continue
                    alloc = room[:, j]
                    final[:, j] += d * alloc / alloc.sum()
                    final[:, j] = np.minimum(final[:, j], cap)
                    if mask is not None:
                        final[:, j] = np.where(mask[:, j], final[:, j], 0.)
            return final

        w_arr = cap_norm(np.vstack(raw_w), cap=MAX_MODEL_WT, mask=np.vstack(valid))
        weights = {names[i]: pd.Series(w_arr[i], index=common_idx, name=names[i])
                   for i in range(len(names))}

        comb_rets = div.copy()
        for n in names:
            comb_rets += weights[n] * self.models[n].loc[common_idx, 'Returns']
        comb_var = sum((weights[n] * ses[i]) ** 2 for i, n in enumerate(names))
        comb_stds = np.sqrt(comb_var)

        score_base = self.short_score(shares_short, shares_outstanding, shares_short_prior)
        score_ws   = self.wsb_score(score_base.copy())
        score_eg   = self.earnings_growth_score(earnings_growth, score_ws.copy(), ind_eg)
        score_rg   = self.revenue_growth_score(rev_growth, score_eg.copy(), ind_rvg)
        score_roe  = self.return_on_equity_score(roe, score_rg.copy(), ind_roe)
        score_roa  = self.return_on_assets_score(roa, score_roe.copy(), ind_roa)
        score_pb   = self.price_to_book_score(pb, score_roa.copy(), ind_pb)
        score_eps  = self.ep_score(teps, feps, price, score_pb.copy(), ind_pe, eps_1y)

        upper_std = np.percentile(comb_stds, self.UPPER_PERCENTILE)
        std_adj   = -(comb_stds > upper_std).astype(int)
        lower_target_adj  = (lower_target > price).astype(int)
        rec_strong_buy_adj = pd.Series(np.where(recommendation == 'strong_buy', 3, 0), index=common_idx)
        rec_sell_adj = pd.Series(np.where(recommendation.isin(['sell','strong_sell']), -5, 0), index=common_idx)
        insider_pos_adj = pd.Series(np.where(insider_purchases > 0, 2, 0), index=common_idx)
        insider_neg_adj = pd.Series(np.where(insider_purchases < 0,-1, 0), index=common_idx)
        net_income_adj   = pd.Series(np.where(net_income > 0, 1, 0), index=common_idx)
        ocf_adj          = pd.Series(np.where(operating_cashflow > net_income, 1, 0), index=common_idx)
        ld_adj           = pd.Series(np.where(prev_long_debt > long_debt, 1, 0), index=common_idx)
        cr_adj           = pd.Series(np.where(prev_current_ratio < current_ratio, 1, 0), index=common_idx)
        no_new_shares_adj= pd.Series(np.where(shares_issued <= 0,1, 0), index=common_idx)
        gm_adj           = pd.Series(np.where(gross_margin > prev_gm, 1, 0), index=common_idx)
        at_adj           = pd.Series(np.where(prev_at < at, 1, 0), index=common_idx)

        score_breakdown = pd.DataFrame({
            'Short Score': score_base,
            'WSB Adjustment': score_ws - score_base,
            'Earnings Growth Adjustment': score_eg - score_ws,
            'Revenue Growth Adjustment': score_rg - score_eg,
            'Return on Equity Adjustment': score_roe - score_rg,
            'Price-to-Book Adjustment': score_pb - score_roe,
            'EPS Adjustment': score_eps - score_pb,
            'Volatility Adjustment': std_adj,
            'Lower Target Price': lower_target_adj,
            'Recommendation Strong Buy': rec_strong_buy_adj,
            'Recommendation Sell/Strong Sell': rec_sell_adj,
            'Insider Purchases Positive': insider_pos_adj,
            'Insider Purchases Negative': insider_neg_adj,
            'Net Income Positive': net_income_adj,
            'OCF > Net Income': ocf_adj,
            'Long-Debt Improvement': ld_adj,
            'Current-Ratio Improvement': cr_adj,
            'No New Shares Issued': no_new_shares_adj,
            'Gross-Margin Improvement': gm_adj,
            'Asset-Turnover Improvement': at_adj
        })

        additional_total = score_breakdown.drop(columns=['Short Score']).sum(axis=1)
        final_scores = score_eps + additional_total

        sell_mask = recommendation.isin(['sell','strong_sell'])
        final_scores.loc[sell_mask] = -100

        final_scores += self.signal_scores

        final_scores = pd.Series(np.minimum(final_scores, nY), index=common_idx)

        bear_rets = comb_rets - 1.96 * comb_stds
        final_scores.loc[bear_rets < 0] = np.minimum(final_scores, 10)

        score_breakdown['Final Score'] = final_scores

        return comb_rets, comb_stds, final_scores, weights, score_breakdown


def main():
    logging.info('Loading data...')
    r = RatioData()
    optimizer = PortfolioOptimizer(EXCEL_OUT_FILE, r)

    metrics = r.dicts()
    region_ind = {
        'PE': pd.Series([metrics['PE'][t]['Region-Industry'] for t in optimizer.tickers], index=optimizer.tickers),
        'PB': pd.Series([metrics['PB'][t]['Region-Industry'] for t in optimizer.tickers], index=optimizer.tickers),
        'ROE': pd.Series([metrics['ROE'][t]['Region-Industry'] for t in optimizer.tickers], index=optimizer.tickers),
        'ROA': pd.Series([metrics['ROA'][t]['Region-Industry'] for t in optimizer.tickers], index=optimizer.tickers),
        'RevG': pd.Series([metrics['rev1y'][t]['Region-Industry'] for t in optimizer.tickers], index=optimizer.tickers),
        'EarningsG': pd.Series([metrics['eps1y'][t]['Region-Industry'] for t in optimizer.tickers], index=optimizer.tickers)
    }

    logging.info('Computing combination forecast...')
    comb_rets, comb_stds, final_scores, weights, score_breakdown = (
        optimizer.compute_combination_forecast(region_ind)
    )

    idx = optimizer.latest_prices.index.sort_values()
    price = optimizer.latest_prices.reindex(idx)
    bull = comb_rets + 1.96 * comb_stds
    bear = comb_rets - 1.96 * comb_stds

    df = pd.DataFrame({
            'Ticker': idx,
            'Current Price': price,
            'Avg Price': np.round(price * (comb_rets + 1), 2),
            'Low Price': np.round(price * (bear + 1), 2),
            'High Price': np.round(price * (bull + 1), 2),
            'Returns': comb_rets,
            'Low Returns': bear,
            'High Returns': bull,
            'SE': comb_stds
        }, index=idx).set_index('Ticker')
    for name, w in weights.items():
        df[f'{name} (%)'] = w.reindex(idx) * 100
    df['Score'] = final_scores.reindex(idx)

    df = ensure_headers_are_strings(df)
    score_breakdown = ensure_headers_are_strings(score_breakdown)
    sheets_to_upload = {
        'Combination Forecast': df,
        'Score Breakdown': score_breakdown
    }
    export_results(sheets_to_upload)
    logging.info('Done.')

if __name__ == '__main__':
    main()
