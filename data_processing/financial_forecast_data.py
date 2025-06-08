from __future__ import annotations
import datetime as dt
import re
from pathlib import Path
from typing import Final
import numpy as np
import pandas as pd
from macro_data import MacroData

macro = MacroData()
r = macro.r

_TODAY_TS: Final = pd.Timestamp(dt.date.today())
_HORIZON_YEARS = 5
_SCALES = {"T": 1e12, "B": 1e9, "M": 1e6}

def _parse_abbrev(val) -> float:
    if isinstance(val, (int, float)) and not pd.isna(val):
        return float(val)
    s = str(val).strip().upper()
    m = re.match(r"^([0-9.,]+)\s*([TMB])?$", s)
    if not m:
        return np.nan
    num = float(m.group(1).replace(",", ""))
    return num * _SCALES.get(m.group(2), 1)


def _safe_stat(series: pd.Series, fn, default: float = 0.0) -> float:
    sr = series.dropna()
    return float(fn(sr)) if not sr.empty else default


class FinancialForecastData:
    """
    Provides annuals, forecasts, and KPIs for a set of tickers.
    """

    _SHEET_INCOME     = "Income-TTM"
    _SHEET_CASHFLOW   = "Cash-Flow-TTM"
    _SHEET_BALANCE    = "Balance-Sheet-TTM"
    _SHEET_RATIOS     = "Ratios-TTM"
    _SHEET_RATIOS_ANN = "Ratios-Annual"
    _SHEET_INCOME_ANN = "Income-Annual"

    _INCOME_ROWS    = ["Revenue","EPS (Basic)","EBIT","Net Income","Income Tax","Pretax Income","Interest Expense / Income","EBITDA","Depreciation & Amortization"]
    _CASHFLOW_ROWS  = ["Operating Cash Flow","Capital Expenditures","Acquisitions","Share-Based Compensation","Free Cash Flow","Other Operating Activities","Dividends Paid"]
    _BALANCE_ROWS   = ["Working Capital","Total Debt","Cash & Cash Equivalents","Book Value Per Share"]
    _RATIOS_ROWS    = ["EV/Revenue","Enterprise Value","Market Capitalization","PE Ratio","PS Ratio","PB Ratio"]
    _RATIOS_ROWS_ANN= ["Return on Equity (ROE)","Payout Ratio"]
    _INCOME_ROWS_ANN= ["Revenue Growth","EPS Growth"]
    _FORECAST_COLS  = ["num_analysts","low_rev","avg_rev","high_rev","low_eps","avg_eps","high_eps"]
    _KPI_COLS       = ["mc_ev","market_value_debt","capex_rev_ratio","exp_evs","exp_pe","exp_ps","exp_ptb","bvps_0","roe","payout_ratio"]

    def __init__(
        self,
        tickers: list[str] = macro.r.tickers,
        quiet: bool = False,
    ):
        self.tickers      = list(tickers)
        self.root_dir     = Path("/Users/georgewright/modelling/stock_analysis_data")
        self.macro = macro
        self.analyst_df   = macro.r.analyst
        self.ind_dict     = macro.r.dicts()
        self.quiet        = quiet

        self._precompute_analyst_est()

        self.income       = {}
        self.cash         = {}
        self.bal          = {}
        self.ratios       = {}
        self.ratios_ann   = {}
        self.income_ann   = {}
        self.annuals      = {}
        self.forecast     = {}
        self.kpis         = {}
        self.prophet_data = {}

        self._load_all()
        self._cache_first_forecasts()

    def _precompute_analyst_est(self) -> None:
        adf = self.analyst_df
        self._analyst_est = pd.DataFrame({
            "low_rev_y":  adf["Low Revenue Estimate"].apply(_parse_abbrev),
            "avg_rev_y":  adf["Avg Revenue Estimate"].apply(_parse_abbrev),
            "high_rev_y": adf["High Revenue Estimate"].apply(_parse_abbrev),
            "low_eps_y":  pd.to_numeric(adf["Low EPS Estimate"], errors="coerce"),
            "avg_eps_y":  pd.to_numeric(adf["Avg EPS Estimate"], errors="coerce"),
            "high_eps_y": pd.to_numeric(adf["High EPS Estimate"], errors="coerce"),
        })

    def _empty_forecast(self) -> pd.DataFrame:
        dates = pd.date_range(
            start=_TODAY_TS + pd.offsets.YearEnd(1),
            periods=_HORIZON_YEARS + 1,
            freq="YE-DEC",
        )
        return pd.DataFrame(np.nan, index=dates, columns=self._FORECAST_COLS)

    def _empty_kpis(self) -> pd.DataFrame:
        return pd.DataFrame(np.nan, index=[_TODAY_TS], columns=self._KPI_COLS)
    
    def currency_adjustment(self, ticker, df):
        """
        Returns the currency adjustment factor for a given ticker.
        """
        exchange_rates = macro.currency
        
        if ticker in ['BABA','PDD']:    
        
            return df * exchange_rates.loc['USDCNY']
       
        if ticker in ['BP.L','HSBA.L']: 
         
            return df * 10 / exchange_rates.loc['GBPUSD']
        
        if ticker == 'ASX':             
            
            return df * 30
       
        return df
    
    def _parse_growth(self, x):
        if isinstance(x, str):
            s = x.strip()
            if s.endswith("%"):
                try:
                    return float(s.rstrip("%")) / 100.0
                except ValueError:
                    return np.nan
            # if it doesn’t end in “%”, try to parse numeric anyway:
            try:
                val = float(s.replace(",", ""))
            except ValueError:
                return np.nan
            # If that numeric is > 1, assume it was “percent” (e.g. 12.0 meaning 12%).
            return val/100.0 if val > 1.0 else val

        elif isinstance(x, (int, float)):
            # If the cell is a raw float > 1.0 (e.g. 10.0), treat as “percent”:
            return float(x)/100.0 if x > 1.0 else float(x)
        else:
            return np.nan

    def _load_all(self) -> None:
        for tkr in self.tickers:
            try:
                self._process_ticker(tkr)
            except Exception as e:
                if not self.quiet:
                    print(f"[WARN] {tkr}: {e}")
                self.annuals[tkr]  = pd.DataFrame()
                self.forecast[tkr]= self._empty_forecast()
                self.kpis[tkr]    = self._empty_kpis()
                self.prophet_data[tkr] = pd.DataFrame()

    def _process_ticker(self, tkr: str) -> None:
        folder = self.root_dir / tkr
        fin_xlsx = folder / f"{tkr}-financials.xlsx"
        if not fin_xlsx.exists():
            raise FileNotFoundError(fin_xlsx)

        xls = pd.ExcelFile(fin_xlsx, engine="openpyxl")
        sheets = pd.read_excel(
            xls,
            sheet_name=[
                self._SHEET_INCOME,
                self._SHEET_CASHFLOW,
                self._SHEET_BALANCE,
                self._SHEET_RATIOS,
                self._SHEET_RATIOS_ANN,
                self._SHEET_INCOME_ANN,
            ],
            index_col=0,
            na_values=["—","-",""],
        )

        def prep(df: pd.DataFrame, rows: list[str]) -> pd.DataFrame:
            df.columns = pd.to_datetime(df.columns, errors="coerce")
            return df.reindex(rows).reindex(sorted(df.columns), axis=1)

        income_df      = prep(sheets[self._SHEET_INCOME],     self._INCOME_ROWS)
        cash_df        = prep(sheets[self._SHEET_CASHFLOW],   self._CASHFLOW_ROWS)
        bal_df         = prep(sheets[self._SHEET_BALANCE],    self._BALANCE_ROWS)
        ratios_df      = prep(sheets[self._SHEET_RATIOS],     self._RATIOS_ROWS)
        ratios_ann_df  = prep(sheets[self._SHEET_RATIOS_ANN], self._RATIOS_ROWS_ANN)
        income_ann_df  = prep(sheets[self._SHEET_INCOME_ANN], self._INCOME_ROWS_ANN)

        self.income[tkr]     = income_df
        self.cash[tkr]       = cash_df
        self.bal[tkr]        = bal_df
        self.ratios[tkr]     = ratios_df
        self.ratios_ann[tkr] = ratios_ann_df
        self.income_ann[tkr] = income_ann_df

        income_num = income_df.map(_parse_abbrev).ffill(axis=1).fillna(0)
        cash_num   = cash_df.map(_parse_abbrev).ffill(axis=1).fillna(0)
        bal_num    = bal_df.map(_parse_abbrev).ffill(axis=1).fillna(0)

        rev_hist = self.currency_adjustment(tkr, income_num.loc["Revenue"])
        eps_hist = self.currency_adjustment(tkr, pd.to_numeric(income_df.loc["EPS (Basic)"], errors="coerce").ffill().fillna(0))

        ocf   = self.currency_adjustment(tkr, cash_num.loc["Operating Cash Flow"])
        capex = self.currency_adjustment(tkr, cash_num.loc["Capital Expenditures"])
        aq    = -self.currency_adjustment(tkr, cash_num.loc["Acquisitions"])
        fcf   = self.currency_adjustment(tkr, cash_num.loc["Free Cash Flow"])
        sbc   = self.currency_adjustment(tkr, cash_num.loc["Share-Based Compensation"])
        ooa   = self.currency_adjustment(tkr, cash_num.loc["Other Operating Activities"])
        div   = self.currency_adjustment(tkr, cash_num.loc["Dividends Paid"])

        wc      = self.currency_adjustment(tkr, bal_num.loc["Working Capital"])
        tot_debt= self.currency_adjustment(tkr, bal_num.loc["Total Debt"])
        bvps_0  = self.currency_adjustment(tkr, bal_num.loc["Book Value Per Share"].iat[-1])

        ebit   = self.currency_adjustment(tkr, income_num.loc["EBIT"])
        da     = self.currency_adjustment(tkr, income_num.loc["Depreciation & Amortization"])
        ni     = self.currency_adjustment(tkr, income_num.loc["Net Income"])
        ebitda = self.currency_adjustment(tkr, income_num.loc["EBITDA"])
        tax    = self.currency_adjustment(tkr, income_num.loc["Income Tax"])
        pretax = self.currency_adjustment(tkr, income_num.loc["Pretax Income"])
        i_exp  = self.currency_adjustment(tkr, income_num.loc["Interest Expense / Income"])

        tax_rate     = tax / pretax.replace(0, np.nan)
        int_aftertax = i_exp * (1 - tax_rate)
        d_wc         = wc.diff()
        net_borrow   = tot_debt.diff().reindex(ocf.index, fill_value=0)

        annual = pd.DataFrame({
            "Revenue": rev_hist,
            "EPS": eps_hist,
            "OCF": ocf,
            "PretaxIncome": pretax,
            "InterestAfterTax": int_aftertax,
            "NetBorrowing": net_borrow,
            "Capex": capex,
            "EBIT": ebit,
            "Depreciation & Amortization": da,
            "Share-Based Compensation": sbc,
            "Acquisitions": aq,
            "Net Income": ni,
            "EBITDA": ebitda,
            "Total Debt": tot_debt,
            "Change in Working Capital": d_wc,
            "Other Operating Activities": ooa,
            "FCF": fcf,
            "Div": div,
        }).T.dropna(how="all").T
        self.annuals[tkr] = annual

        fcst_file = folder / f"{tkr}-analyst-forecasts.xlsx"
        if fcst_file.exists():
            fcst_df = self._read_analyst_sheet(fcst_file, tkr)
        else:
            fcst_df = self._make_synthetic_forecast(tkr, rev_hist, eps_hist, income_ann_df)
        self.forecast[tkr] = fcst_df

        self.kpis[tkr] = self._compute_kpis(tkr, ratios_df, capex, rev_hist, annual.index, bvps_0)
        self.prophet_data[tkr] = self._compute_prophet_data(tkr)

    def _read_analyst_sheet(self, path: Path, tkr) -> pd.DataFrame:
        df = pd.read_excel(path, index_col=0, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        dates = pd.to_datetime(df.loc["Period Ending"].dropna())
        data = {
            "num_analysts": df.loc["Analysts"].astype(int).values,
            "low_rev":      self.currency_adjustment(tkr, df.loc["Revenue Low"]).apply(_parse_abbrev).values,
            "avg_rev":      self.currency_adjustment(tkr, df.loc["Revenue"]).apply(_parse_abbrev).values,
            "high_rev":     self.currency_adjustment(tkr, df.loc["Revenue High"]).apply(_parse_abbrev).values,
            "low_eps":      self.currency_adjustment(tkr, pd.to_numeric(df.loc["EPS Low"], errors="coerce")).values,
            "avg_eps":      self.currency_adjustment(tkr, pd.to_numeric(df.loc["EPS"], errors="coerce")).values,
            "high_eps":     self.currency_adjustment(tkr, pd.to_numeric(df.loc["EPS High"], errors="coerce")).values,
        }
        return pd.DataFrame(data, index=dates).sort_index()

    def _filter_outliers_iqr(self, series: pd.Series, m: float = 1.5) -> pd.Series:
        """
        Remove any points outside [Q1 - m*IQR, Q3 + m*IQR].
        Returns the filtered series (i.e. with outliers dropped).
        """
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - m * iqr, q3 + m * iqr
        return series[(series >= lower) & (series <= upper)]

    def _make_synthetic_forecast(
        self,
        tkr: str,
        rev_hist: pd.Series,
        eps_hist: pd.Series,
        income_ann_df: pd.DataFrame,
    ) -> pd.DataFrame:
       
        ann_num = (income_ann_df
                .map(self._parse_growth)
                .ffill(axis=1)
                .fillna(0))

        rev_g = ann_num.loc["Revenue Growth"].dropna()
        eps_g = ann_num.loc["EPS Growth"].dropna()

        rev_g_filt = self._filter_outliers_iqr(rev_g, m=1.5)
        eps_g_filt = self._filter_outliers_iqr(eps_g, m=1.5)

        g_rev = (
            rev_g_filt.min(),
            rev_g_filt.mean(),
            rev_g_filt.max()
        )
        g_eps = (
            eps_g_filt.min(),
            eps_g_filt.mean(),
            eps_g_filt.max()
        )
        if tkr in self._analyst_est.index:
            base = self._analyst_est.loc[tkr]
            br, ar, hr = base[['low_rev_y','avg_rev_y','high_rev_y']]
            be, ae, he = base[['low_eps_y','avg_eps_y','high_eps_y']]
        else:
            br = ar = hr = rev_hist.iloc[-1]
            be = ae = he = eps_hist.iloc[-1]

        low_rev_list  = [br]
        avg_rev_list  = [ar]
        high_rev_list = [hr]
        low_eps_list  = [be]
        avg_eps_list  = [ae]
        high_eps_list = [he]

        for period in range(1, _HORIZON_YEARS + 1):
            if period == 1:
                low_rev_list.append (low_rev_list[-1])
                avg_rev_list.append (avg_rev_list[-1])
                high_rev_list.append(high_rev_list[-1])
                low_eps_list.append (low_eps_list[-1])
                avg_eps_list.append (avg_eps_list[-1])
                high_eps_list.append(high_eps_list[-1])
            else:
                low_rev_list.append (avg_rev_list[-1] * (1 + g_rev[0]))
                avg_rev_list.append (avg_rev_list[-1] * (1 + g_rev[1]))
                high_rev_list.append(avg_rev_list[-1] * (1 + g_rev[2]))
                low_eps_list.append (avg_eps_list[-1] * (1 + g_eps[0]))
                avg_eps_list.append (avg_eps_list[-1] * (1 + g_eps[1]))
                high_eps_list.append(avg_eps_list[-1] * (1 + g_eps[2]))

        low_rev_arr  = np.array(low_rev_list)
        avg_rev_arr  = np.array(avg_rev_list)
        high_rev_arr = np.array(high_rev_list)
        low_eps_arr  = np.array(low_eps_list)
        avg_eps_arr  = np.array(avg_eps_list)
        high_eps_arr = np.array(high_eps_list)

        dates = pd.date_range(
            start=_TODAY_TS + pd.offsets.YearEnd(1),
            periods=_HORIZON_YEARS + 1,
            freq="YE-DEC",
        )
        return pd.DataFrame({
            "num_analysts": [1] * len(dates),
            "low_rev":      low_rev_arr,
            "avg_rev":      avg_rev_arr,
            "high_rev":     high_rev_arr,
            "low_eps":      low_eps_arr,
            "avg_eps":      avg_eps_arr,
            "high_eps":     high_eps_arr,
        }, index=dates)

    def _compute_kpis(
        self,
        tkr:      str,
        ratios_df: pd.DataFrame,
        capex:     pd.Series,
        rev:       pd.Series,
        annual_index: pd.DatetimeIndex,
        bvps_0:    float,
    ) -> pd.DataFrame:
        """
        Build the scalar KPIs and return a 1×10 DataFrame:
            mc_ev, market_value_debt, capex_rev_ratio,
            exp_evs, exp_pe, exp_ps, exp_ptb,
            bvps_0, roe, payout_ratio.
        We explicitly coerce each ratio row to numeric before EWM.
        """

        ev_revenue = pd.to_numeric(
            ratios_df.loc["EV/Revenue"], errors="coerce"
        )
        evs_ticker = ev_revenue.ewm(span=20, adjust=False).mean().iat[-1]

        pe_series = pd.to_numeric(
            ratios_df.loc["PE Ratio"], errors="coerce"
        )
        pe_avg = pe_series.ewm(span=20, adjust=False).mean().iat[-1]

        ps_series = pd.to_numeric(
            ratios_df.loc["PS Ratio"], errors="coerce"
        )
        ps_avg = ps_series.ewm(span=20, adjust=False).mean().iat[-1]

        ptb_series = pd.to_numeric(
            ratios_df.loc["PB Ratio"], errors="coerce"
        )
        ptb_avg = ptb_series.ewm(span=20, adjust=False).mean().iat[-1]

        mc_series = ratios_df.loc["Market Capitalization"].apply(_parse_abbrev)
        ev_series = ratios_df.loc["Enterprise Value"].apply(_parse_abbrev)

        mc_ev = (
            (mc_series / ev_series)
            .replace([np.inf, -np.inf], np.nan)
            .ewm(span=20, adjust=False)
            .mean()
            .iat[-1]
        )

        capex_rev_ratio = (capex / rev).tail(20).mean()

        md = mc_series.iat[-1] - ev_series.iat[-1]

        roe_list = (
            self.ratios_ann[tkr]
                .loc["Return on Equity (ROE)"]
                .map(_parse_abbrev)
                .dropna()
                .tail(5)
        )
        if not roe_list.empty:
            roe = (roe_list + 1).prod() ** (1 / len(roe_list)) - 1
        else:
            roe = self.ind_dict["ROE"][tkr]["Region-Industry"]
        if pd.isna(roe):
            roe = self.ind_dict["ROE"][tkr]["Region-Industry"]

        payout = abs(
            self.ratios_ann[tkr]
                .loc["Payout Ratio"]
                .map(_parse_abbrev)
                .dropna()
                .tail(5)
                .mean()
        )

        return pd.DataFrame({
            "mc_ev":             [mc_ev],
            "market_value_debt": [md],
            "capex_rev_ratio":   [capex_rev_ratio],
            "exp_evs":           [evs_ticker],
            "exp_pe":            [pe_avg],
            "exp_ps":            [ps_avg],
            "exp_ptb":           [ptb_avg],
            "bvps_0":            [bvps_0],
            "roe":               [roe],
            "payout_ratio":      [payout],
        }, index=[annual_index[-1]])

    def _compute_prophet_data(self, tkr: str) -> pd.DataFrame:
        
        inc_full = self.income.get(tkr, pd.DataFrame())
        if inc_full.empty: 
            return pd.DataFrame()
        
        inc_full.columns = pd.to_datetime(inc_full.columns)

        inc = inc_full.loc[:, inc_full.columns >= '2000-01-01']

        required_inc = {"Revenue", "EPS (Basic)"}

        if not required_inc.issubset(inc.index): 
            return pd.DataFrame()

        return pd.DataFrame({
            "rev": self.currency_adjustment(tkr, inc.loc["Revenue"].ffill().fillna(0)),
            "eps": self.currency_adjustment(tkr, inc.loc["EPS (Basic)"].ffill().fillna(0)),
        })

    def _cache_first_forecasts(self) -> None:
        records = {
            tkr: (
                {col: df.iloc[0].get(col, np.nan) for col in self._FORECAST_COLS[1:]}
                if not df.empty else dict.fromkeys(self._FORECAST_COLS[1:], np.nan)
            )
            for tkr, df in self.forecast.items()
        }
        self._first_fc_df = pd.DataFrame.from_dict(records, orient="index")

    def next_period_forecast(self) -> pd.DataFrame:
        return self._analyst_est.join(self._first_fc_df, how="outer")

    def get_annual_income_growth(self) -> dict[str, pd.DataFrame]:
        """
        Returns a mapping ticker → DataFrame of annual 'Revenue Growth', 'EPS Growth', and 'Return'.
        """
        growth: dict[str, pd.DataFrame] = {}
        window = pd.DateOffset(months=1)

        for tkr in self.tickers:
            income_ann_df = self.income_ann.get(tkr)
            if income_ann_df is None or income_ann_df.empty:
                growth[tkr] = pd.DataFrame() 
                continue

            ann_num = income_ann_df.map(_parse_abbrev).ffill(axis=1).fillna(0)
            rev_g = ann_num.loc["Revenue Growth"]
            eps_g = ann_num.loc["EPS Growth"]

            closes = macro.r.close[tkr].sort_index()
            def window_mean(dt: pd.Timestamp) -> float:
                return closes.loc[dt - window : dt + window].mean()

            mean_price = pd.Series(
                {dt: window_mean(dt) for dt in rev_g.index}
            ).sort_index()
            ret_g = mean_price.pct_change()

            df = pd.DataFrame({
                "Revenue Growth": rev_g,
                "EPS Growth":     eps_g,
                "Return":         ret_g,
            }).dropna()
            growth[tkr] = df

        return growth

    def regression_dict(self) -> dict[str, pd.DataFrame]:
        """
        Returns ticker → DataFrame combining annual growth/returns with macro history.
        """
        macro_hist = macro.assign_macro_history()
        growth_dict = self.get_annual_income_growth()
        combined: dict[str, pd.DataFrame] = {}

        for tkr, df_growth in growth_dict.items():
            if df_growth.empty:
                continue

            df = df_growth.copy()
            df.index = pd.to_datetime(df.index).to_period("Y")

            try:
                macro_tkr = macro_hist.xs(tkr, level="ticker")
            except KeyError:
                continue

            joined = df.join(macro_tkr, how="inner")
            combined[tkr] = joined[[
                "Revenue Growth", "EPS Growth", "Return",
                "InterestRate_pct_change", "Inflation_rate", "GDP_growth", "Unemployment_pct_change"
            ]]

        return combined

    def _zero_covs(self) -> dict[str, float]:
        return dict.fromkeys(
            ("low_y","avg_y","high_y","low_fs","avg_fs","high_fs"),
            0.0
        )

    def get_forecast_pct_changes(
        self,
        ticker: str,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute revenue- and EPS-forecast percentage changes for a given ticker.
        """
        if (
            ticker not in self.annuals
            or self.annuals[ticker].empty
            or ticker not in self._analyst_est.index
            or ticker not in self._first_fc_df.index
        ):
            return self._zero_covs(), self._zero_covs()

        hist = self.annuals[ticker]
        base_rev = hist["Revenue"].iloc[-1]
        base_eps = hist["EPS"].iloc[-1]

        est   = self._analyst_est.loc[ticker]
        first = self._first_fc_df.loc[ticker]

        def pct(new, old):
            return (new - old) / abs(old) if pd.notna(new) and old != 0 else 0.0

        rev_covs = {
            "low_y":  pct(est["low_rev_y"], base_rev),
            "avg_y":  pct(est["avg_rev_y"], base_rev),
            "high_y": pct(est["high_rev_y"], base_rev),
            "low_fs": pct(first["low_rev"],   base_rev),
            "avg_fs": pct(first["avg_rev"],   base_rev),
            "high_fs":pct(first["high_rev"],  base_rev),
        }
        eps_covs = {
            "low_y":  pct(est["low_eps_y"], base_eps),
            "avg_y":  pct(est["avg_eps_y"], base_eps),
            "high_y": pct(est["high_eps_y"], base_eps),
            "low_fs": pct(first["low_eps"],   base_eps),
            "avg_fs": pct(first["avg_eps"],   base_eps),
            "high_fs":pct(first["high_eps"],  base_eps),
        }

        return rev_covs, eps_covs
