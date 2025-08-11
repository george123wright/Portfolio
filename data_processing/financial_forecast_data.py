"""
Handles per‑ticker financial statements, analyst forecasts and derived KPIs; supplies data for forecasting models.
"""

from __future__ import annotations
import datetime as dt
import re
from pathlib import Path
from typing import Final
import numpy as np
import pandas as pd
from functools import cached_property
from macro_data import MacroData
import config


macro = MacroData()

_TODAY_TS: Final = pd.Timestamp(config.TODAY)

_HORIZON_YEARS = 5

_SCALES = {
    "T": 1e12, 
    "B": 1e9, 
    "M": 1e6
}


def _parse_abbrev(
    val
) -> float:
 
    if isinstance(val, (int, float)) and not pd.isna(val):
        
        return float(val)
 
    s = str(val).strip().upper()
 
    m = re.match(r"^([0-9.,]+)\s*([TMB])?$", s)
 
    if not m:
        
        return np.nan
 
    num = float(m.group(1).replace(",", ""))
 
    return num * _SCALES.get(m.group(2), 1)


def _safe_stat(
    series: pd.Series, 
    fn, 
    default: float = 0.0
) -> float:
   
    sr = series.dropna()
   
    return float(fn(sr)) if not sr.empty else default


class FinancialForecastData:
    """
    Provides annuals, forecasts, KPIs, and prophet-ready data for a set of tickers.
    All data loads lazily on first access.
    """

    _SHEET_INCOME = "Income-TTM"
   
    _SHEET_CASHFLOW = "Cash-Flow-TTM"
   
    _SHEET_BALANCE = "Balance-Sheet-TTM"
   
    _SHEET_RATIOS = "Ratios-TTM"
   
    _SHEET_RATIOS_ANN = "Ratios-Annual"
   
    _SHEET_INCOME_ANN = "Income-Annual"

    _INCOME_ROWS = ["Revenue", "EPS (Basic)", "EBIT", "Net Income", "Income Tax",
                        "Pretax Income", "Interest Expense / Income", "EBITDA", "Depreciation & Amortization"]
    
    _CASHFLOW_ROWS = ["Operating Cash Flow", "Capital Expenditures", "Acquisitions",
                        "Share-Based Compensation", "Free Cash Flow", "Other Operating Activities", "Dividends Paid"]
    
    _BALANCE_ROWS = ["Working Capital", "Total Debt", "Cash & Cash Equivalents", "Book Value Per Share"]
   
    _RATIOS_ROWS = ["EV/Revenue", "Enterprise Value", "Market Capitalization", "PE Ratio", "PS Ratio", "PB Ratio"]
   
    _RATIOS_ROWS_ANN = ["Return on Equity (ROE)", "Payout Ratio"]
   
    _INCOME_ROWS_ANN = ["Revenue Growth", "EPS Growth"]

    _FORECAST_COLS = ["num_analysts", "low_rev", "avg_rev", "high_rev", "low_eps", "avg_eps", "high_eps"]
   
    _KPI_COLS = ["mc_ev", "market_value_debt", "capex_rev_ratio", "exp_evs", "exp_pe",
                      "exp_ps", "exp_ptb", "bvps_0", "roe", "payout_ratio"]


    def __init__(
        self,
        tickers: list[str] = config.tickers,
        quiet: bool = False,
    ):
        self.tickers = list(tickers)
        self.root_dir = config.ROOT_FIN_DIR
        self.macro = macro
        self.analyst_df = macro.r.analyst
        self.quiet = quiet


    def currency_adjustment(
        self, 
        ticker: str, s: pd.Series | pd.DataFrame
    ):
       
        ex = self.macro.currency
        
        if ticker in ["BABA", "PDD", "XIACY"]:
           
            rate = float(ex.loc["USDCNY"])
           
            return s / rate
        
        if ticker in ["BP.L"]:
           
            rate = float(ex.loc["GBPUSD"])
           
            return s * rate / 10
        
        if ticker == "ASX":
           
            rate = 30
           
            return s * rate
        
        return s


    def _parse_growth(
        self, 
        x
    ):
        
        if isinstance(x, str):
        
            s = x.strip()
        
            if s.endswith("%"):
        
                try:
               
                    return float(s.rstrip("%"))/100.0
               
                except ValueError:
               
                    return np.nan
        
            try:
               
                v = float(s.replace(",",""))
           
            except ValueError:
               
                return np.nan
           
            return v / 100.0 if v > 1.0 else v
        
        if isinstance(x,(int,float)):
          
            return float(x)/100.0 if x>1.0 else float(x)
        
        return np.nan


    def _filter_outliers_iqr(
        self, 
        series: pd.Series, 
        m: float = 1.5
    ) -> pd.Series:
       
        q1, q3 = series.quantile([0.25, 0.75])
        
        iqr = q3 - q1
        
        lo,hi = q1 - (m * iqr), q3 + (m * iqr)
       
        return series[(series >= lo) & (series <= hi)]


    def _empty_forecast(
        self
    ) -> pd.DataFrame:
        
        dates = pd.date_range(
            start = _TODAY_TS + pd.offsets.YearEnd(1),
            periods = _HORIZON_YEARS+1,
            freq= " YE-DEC",
        )
       
        return pd.DataFrame(np.nan, index=dates, columns=self._FORECAST_COLS)


    def _empty_kpis(self) -> pd.DataFrame:
        
        return pd.DataFrame(np.nan, index=[_TODAY_TS], columns=self._KPI_COLS)


    def _read_sheet(
        self, 
        tkr: str, 
        sheet: str, 
        rows: list[str]
    ) -> pd.DataFrame:
        """
        Load exactly one sheet and reindex to the rows+sorted columns.
        """
        
        folder = self.root_dir / tkr
        
        xlsx = folder / f"{tkr}-financials.xlsx"
        xls = pd.ExcelFile(xlsx, engine="openpyxl")
        
        df = pd.read_excel(
            xls, 
            sheet_name = sheet, 
            index_col=0,
            na_values=["—","-",""]
        )
        
        df.columns = pd.to_datetime(df.columns, errors="coerce")
        
        return df.reindex(rows).reindex(sorted(df.columns), axis=1)


    @cached_property
    def income(self) -> dict[str,pd.DataFrame]:
       
        d = {}
       
        for t in self.tickers:
       
            try:
               
                d[t] = self._read_sheet(
                    tkr = t, 
                    sheet = self._SHEET_INCOME, 
                    rows = self._INCOME_ROWS
                )
            
            except Exception as e:
            
                if not self.quiet: 
            
                    print(f"[WARN] income {t}: {e}")
               
                d[t] = pd.DataFrame()
      
        return d


    @cached_property
    def cash(self) -> dict[str,pd.DataFrame]:

        d = {}

        for t in self.tickers:

            try:
        
                d[t] = self._read_sheet(t, self._SHEET_CASHFLOW, self._CASHFLOW_ROWS)
        
            except Exception as e:
        
                if not self.quiet: 
        
                    print(f"[WARN] cash {t}: {e}")
                
                d[t] = pd.DataFrame()

        return d


    @cached_property
    def bal(self) -> dict[str,pd.DataFrame]:

        d = {}

        for t in self.tickers:

            try:
        
                d[t] = self._read_sheet(t, self._SHEET_BALANCE, self._BALANCE_ROWS)
        
            except Exception as e:
        
                if not self.quiet: 
        
                    print(f"[WARN] bal {t}: {e}")

                d[t] = pd.DataFrame()

        return d


    @cached_property
    def ratios(self) -> dict[str,pd.DataFrame]:
       
        d = {}
       
        for t in self.tickers:
       
            try:
      
                d[t] = self._read_sheet(
                    tkr = t, 
                    sheet = self._SHEET_RATIOS, 
                    rows = self._RATIOS_ROWS
                )
      
            except Exception as e:
      
                if not self.quiet: 
      
                    print(f"[WARN] ratios {t}: {e}")
               
                d[t] = pd.DataFrame()
     
        return d


    @cached_property
    def ratios_ann(self) -> dict[str,pd.DataFrame]:
       
        d = {}
       
        for t in self.tickers:
       
            try:
        
                d[t] = self._read_sheet(
                    tkr = t, 
                    sheet = self._SHEET_RATIOS_ANN, 
                    rows = self._RATIOS_ROWS_ANN
                )
        
            except Exception as e:
        
                if not self.quiet: 
        
                    print(f"[WARN] ratios_ann {t}: {e}")
               
                d[t] = pd.DataFrame()
     
        return d


    @cached_property
    def income_ann(self) -> dict[str,pd.DataFrame]:

        d = {}

        for t in self.tickers:

            try:
       
                d[t] = self._read_sheet(
                    tkr = t, 
                    sheet = self._SHEET_INCOME_ANN, 
                    rows = self._INCOME_ROWS_ANN
                )
       
            except Exception as e:
       
                if not self.quiet: 
       
                    print(f"[WARN] income_ann {t}: {e}")
               
                d[t] = pd.DataFrame()
      
        return d


    @cached_property
    def annuals(self) -> dict[str,pd.DataFrame]:
       
        out = {}
       
        for t in self.tickers:
       
            try:
         
                inc = self.income[t].map(_parse_abbrev).ffill(axis = 1).fillna(0)
         
                cf = self.cash[t].map(_parse_abbrev).ffill(axis = 1).fillna(0)
         
                bal = self.bal[t].map(_parse_abbrev).ffill(axis = 1).fillna(0)

                rev_hist = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Revenue"]
                )
         
                eps_hist = self.currency_adjustment(
                    ticker = t,
                    s = pd.to_numeric(
                        self.income[t].loc["EPS (Basic)"], errors="coerce"
                    ).ffill().fillna(0)
                )

                ocf = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Operating Cash Flow"]
                )
         
                capex = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Capital Expenditures"]
                )
         
                aq = -self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Acquisitions"]
                )
         
                fcf = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Free Cash Flow"]
                )
         
                sbc = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Share-Based Compensation"]
                )
         
                ooa = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Other Operating Activities"]
                )
         
                div = self.currency_adjustment(
                    ticker = t, 
                    s = cf.loc["Dividends Paid"]
                )

                wc = self.currency_adjustment(
                    ticker = t, 
                    s = bal.loc["Working Capital"]
                )
         
                tot_d = self.currency_adjustment(
                    ticker = t, 
                    s = bal.loc["Total Debt"]
                )
         
                bvps_0 = self.currency_adjustment(
                    ticker = t, 
                    s = bal.loc["Book Value Per Share"].iat[-1]
                )

                ebit = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["EBIT"]
                )
         
                da = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Depreciation & Amortization"]
                )
         
                ni = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Net Income"]
                )
         
                ebitda = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["EBITDA"]
                )
         
                tax = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Income Tax"]
                )
         
                pretax = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Pretax Income"]
                )
         
                iexp = self.currency_adjustment(
                    ticker = t, 
                    s = inc.loc["Interest Expense / Income"]
                )

                tax_rate = tax / pretax.replace(0, np.nan)
         
                int_at = iexp * (1 - tax_rate)
         
                d_wc = wc.diff()
         
                net_borrow = tot_d.diff().reindex(ocf.index, fill_value = 0)

                df = pd.DataFrame({
                    "Revenue": rev_hist,
                    "EPS": eps_hist,
                    "OCF": ocf,
                    "PretaxIncome": pretax,
                    "InterestAfterTax": int_at,
                    "NetBorrowing": net_borrow,
                    "Capex": capex,
                    "EBIT": ebit,
                    "Depreciation & Amortization": da,
                    "Share-Based Compensation": sbc,
                    "Acquisitions": aq,
                    "Net Income": ni,
                    "EBITDA": ebitda,
                    "Total Debt": tot_d,
                    "Change in Working Capital": d_wc,
                    "Other Operating Activities": ooa,
                    "FCF": fcf,
                    "Div": div,
                }).T.dropna(how="all").T

                out[t] = df

            except Exception as e:
          
                if not self.quiet: 
          
                    print(f"[WARN] annuals {t}: {e}")
               
                out[t] = pd.DataFrame()
      
        return out


    def _read_analyst_sheet(
        self, 
        path: Path, 
        tkr: str
    ) -> pd.DataFrame:
     
        df = pd.read_excel(
            path, 
            index_col = 0, 
            engine = "openpyxl"
        )
     
        df.columns = [str(c).strip() for c in df.columns]
     
        dates = pd.to_datetime(df.loc["Period Ending"].dropna())
     
        data = {
            "num_analysts": df.loc["Analysts"].astype(int).values,
            "low_rev": self.currency_adjustment(tkr, df.loc["Revenue Low"].apply(_parse_abbrev)).values,
            "avg_rev": self.currency_adjustment(tkr, df.loc["Revenue"].apply(_parse_abbrev)).values,
            "high_rev": self.currency_adjustment(tkr, df.loc["Revenue High"].apply(_parse_abbrev)).values,
            "low_eps": self.currency_adjustment(tkr, pd.to_numeric(df.loc["EPS Low"], errors="coerce")).values,
            "avg_eps": self.currency_adjustment(tkr, pd.to_numeric(df.loc["EPS"], errors="coerce")).values,
            "high_eps": self.currency_adjustment(tkr, pd.to_numeric(df.loc["EPS High"], errors="coerce")).values,
        }
     
        return pd.DataFrame(data, index=dates).sort_index()


    def _make_synthetic_forecast(
        self,
        tkr: str,
        rev_hist: pd.Series,
        eps_hist: pd.Series,
        income_ann_df: pd.DataFrame,
    ) -> pd.DataFrame:

        ann_num = income_ann_df.map(self._parse_growth).ffill(axis=1).fillna(0)

        rev_g = self._filter_outliers_iqr(
            series = ann_num.loc["Revenue Growth"].dropna()
        )
        
        eps_g = self._filter_outliers_iqr(
            series = ann_num.loc["EPS Growth"].dropna()
        )

        g_rev = (rev_g.min(), rev_g.mean(), rev_g.max())
     
        g_eps = (eps_g.min(), eps_g.mean(), eps_g.max())

        if tkr in self.analyst_df.index:

            base = self.analyst_df.loc[tkr]

            br, ar, hr = (base[c] for c in ["Low Revenue Estimate", "Avg Revenue Estimate", "High Revenue Estimate"])
          
            br, ar, hr = map(_parse_abbrev, [br, ar ,hr])
          
            be, ae, he = (base[c] for c in ["Low EPS Estimate", "Avg EPS Estimate", "High EPS Estimate"])
          
            be, ae, he = map(lambda x: pd.to_numeric(x, errors="coerce"), [be, ae, he])

        else:

            br = ar = hr = rev_hist.iloc[-1]
          
            be = ae = he = eps_hist.iloc[-1]

        low_r, avg_r, high_r = [ [br], [ar], [hr] ]
        low_e, avg_e, high_e = [ [be], [ae], [he] ]

        for p in range(1, _HORIZON_YEARS+1):

            if p==1:

                for lst in (low_r, avg_r, high_r, low_e, avg_e, high_e):
                    lst.append(lst[-1])

            else:
               
                low_r.append(avg_r[-1] * (1 + g_rev[0]))
                avg_r.append(avg_r[-1] * (1 + g_rev[1]))
                high_r.append(avg_r[-1] * (1 + g_rev[2]))
               
                low_e.append(avg_e[-1] * (1 + g_eps[0]))
                avg_e.append(avg_e[-1] * (1 + g_eps[1]))
                high_e.append(avg_e[-1] * (1 + g_eps[2]))

        dates = pd.date_range(
            start = _TODAY_TS + pd.offsets.YearEnd(1),
            periods = _HORIZON_YEARS+1,
            freq = "YE-DEC",
        )

        return pd.DataFrame({
            "num_analysts": [1]*len(dates),
            "low_rev": np.array(low_r),
            "avg_rev": np.array(avg_r),
            "high_rev": np.array(high_r),
            "low_eps": np.array(low_e),
            "avg_eps": np.array(avg_e),
            "high_eps": np.array(high_e),
        }, index = dates)


    @cached_property
    def forecast(self) -> dict[str,pd.DataFrame]:

        out = {}

        for t in self.tickers:

            try:
                fpath = self.root_dir / t / f"{t}-analyst-forecasts.xlsx"

                if fpath.exists():
                   
                    df = self._read_analyst_sheet(
                        path = fpath, 
                        tkr = t
                    )

                else:
                    ann_df = self.income_ann[t]
                  
                    rev = self.annuals[t]["Revenue"]
                  
                    eps = self.annuals[t]["EPS"]
                  
                    df = self._make_synthetic_forecast(
                        tkr = t, 
                        rev_hist = rev, 
                        eps_hist = eps, 
                        income_ann_df = ann_df
                    )

                out[t] = df

            except Exception as e:

                if not self.quiet: 
                    
                    print(f"[WARN] forecast {t}: {e}")
                
                out[t] = self._empty_forecast()
        
        return out


    @cached_property
    def kpis(self) -> dict[str,pd.DataFrame]:
       
        out = {}
       
        for t in self.tickers:
       
            try:
               
                ratios_df = self.ratios[t]
               
                capex = self.cash[t].map(_parse_abbrev).loc["Capital Expenditures"]
               
                rev_hist = self.annuals[t]["Revenue"]
               
                last_idx = self.annuals[t].index[-1]
               
                bvps0 = self.bal[t].map(_parse_abbrev).loc["Book Value Per Share"].iat[-1]
               
                out[t] = self._compute_kpis(t, ratios_df, capex, rev_hist, last_idx, bvps0)
       
            except Exception as e:
               
                if not self.quiet: 
               
                    print(f"[WARN] kpis {t}: {e}")
                
                out[t] = self._empty_kpis()
     
        return out

    
    def _compute_kpis(
        self, 
        tkr, 
        ratios_df, 
        capex, rev, 
        annual_index, 
        bvps_0
    ) -> pd.DataFrame:
    
        ev_rev = pd.to_numeric(ratios_df.loc["EV/Revenue"], errors = "coerce")
       
        evs_t = ev_rev.ewm(span = 20, adjust=False).mean().iat[-1]
       
        pe = pd.to_numeric(ratios_df.loc["PE Ratio"], errors = "coerce")
       
        ps = pd.to_numeric(ratios_df.loc["PS Ratio"], errors = "coerce")
       
        ptb = pd.to_numeric(ratios_df.loc["PB Ratio"], errors = "coerce")
       
        pe_a = pe.ewm(
            span = 20, 
            adjust = False
        ).mean().iat[-1]
       
        ps_a = ps.ewm(
            span = 20,
            adjust = False
        ).mean().iat[-1]
       
        ptb_a = ptb.ewm(
            span = 20, 
            adjust = False
        ).mean().iat[-1]

        mc = ratios_df.loc["Market Capitalization"].apply(_parse_abbrev)
       
        ev = ratios_df.loc["Enterprise Value"].apply(_parse_abbrev)
       
        mc_ev = (mc / ev).replace([np.inf, -np.inf], np.nan).ewm(
            span = 20, 
            adjust = False
            ).mean().iat[-1]

        cap_rev = (capex / rev).tail(20).mean()
        md = mc.iat[-1] - ev.iat[-1]

        roe_list = (
            self.ratios_ann[tkr]
                .loc["Return on Equity (ROE)"]
                .map(_parse_abbrev)
                .dropna().tail(5)
        )
        
        if not roe_list.empty:
           
            roe = (roe_list+1).prod() ** (1 / len(roe_list)) - 1
       
        else:
       
            roe = self.ind_dict["ROE"][tkr]["Region-Industry"] or 0.0

        payout = abs(
            self.ratios_ann[tkr]
                .loc["Payout Ratio"]
                .map(_parse_abbrev)
                .dropna()
                .tail(5)
                .mean()
        )

        return pd.DataFrame({
            "mc_ev": [mc_ev],
            "market_value_debt": [md],
            "capex_rev_ratio": [cap_rev],
            "exp_evs": [evs_t],
            "exp_pe": [pe_a],
            "exp_ps": [ps_a],
            "exp_ptb": [ptb_a],
            "bvps_0": [bvps_0],
            "roe": [roe],
            "payout_ratio": [payout],
        }, index=[annual_index])


    @cached_property
    def prophet_data(self) -> dict[str,pd.DataFrame]:
       
        out = {}
       
        for t in self.tickers:
       
            try:
       
                inc = self.income.get(t, pd.DataFrame())
       
                if inc.empty:
                  
                    out[t] = pd.DataFrame()
                  
                    continue

                inc.columns = pd.to_datetime(inc.columns)
       
                inc2 = inc.loc[:, inc.columns >= "2000-01-01"]
       
                if not {"Revenue","EPS (Basic)"}.issubset(inc2.index):
                  
                    out[t] = pd.DataFrame()
                  
                    continue

                rev = self.currency_adjustment(
                    ticker = t, 
                    s = inc2.loc["Revenue"].ffill().fillna(0)
                )
              
                eps = self.currency_adjustment(
                    ticker = t, 
                    s = pd.to_numeric(inc2.loc["EPS (Basic)"], errors="coerce").ffill().fillna(0)
                )
       
                out[t] = pd.DataFrame({"rev":rev, "eps":eps})
       
            except Exception as e:
       
                if not self.quiet: 
                    print(f"[WARN] prophet_data {t}: {e}")
                
                out[t] = pd.DataFrame()
       
        return out


    def next_period_forecast(self) -> pd.DataFrame:

        est = pd.DataFrame({
            "low_rev_y": self.analyst_df["Low Revenue Estimate"].apply(_parse_abbrev),
            "avg_rev_y": self.analyst_df["Avg Revenue Estimate"].apply(_parse_abbrev),
            "high_rev_y": self.analyst_df["High Revenue Estimate"].apply(_parse_abbrev),
            "low_eps_y": pd.to_numeric(self.analyst_df["Low EPS Estimate"], errors="coerce"),
            "avg_eps_y": pd.to_numeric(self.analyst_df["Avg EPS Estimate"], errors="coerce"),
            "high_eps_y": pd.to_numeric(self.analyst_df["High EPS Estimate"], errors="coerce"),
        }, index=self.analyst_df.index)
       
        first_fc = {
            t: (
                {c: df.iloc[0].get(c, np.nan) for c in self._FORECAST_COLS[1:]}
                if not df.empty else dict.fromkeys(self._FORECAST_COLS[1:], np.nan)
            )
            for t,df in self.forecast.items()
        }
      
        first_df = pd.DataFrame.from_dict(first_fc, orient="index")
      
        return est.join(first_df, how="outer")
    
    
    def get_forecast_pct_changes(
        self, 
        ticker: str
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute revenue- and EPS-forecast percentage changes for a given ticker.
        Returns (rev_covs, eps_covs) where each is a dict with keys:
        'low_y','avg_y','high_y','low_fs','avg_fs','high_fs'.
        """

        def _zero_covs():
            
            return {k: 0.0 for k in ("low_y", "avg_y", "high_y", "low_fs", "avg_fs", "high_fs")}


        if ticker not in self.annuals or self.annuals[ticker].empty:
           
            return _zero_covs(), _zero_covs()

        hist = self.annuals[ticker]
       
        base_rev = hist["Revenue"].iloc[-1]
       
        base_eps = hist["EPS"].iloc[-1]

        if self.analyst_df is not None and ticker in self.analyst_df.index:
            
            est = pd.Series({
                "low_rev_y": _parse_abbrev(self.analyst_df.at[ticker, "Low Revenue Estimate"]),
                "avg_rev_y": _parse_abbrev(self.analyst_df.at[ticker, "Avg Revenue Estimate"]),
                "high_rev_y": _parse_abbrev(self.analyst_df.at[ticker, "High Revenue Estimate"]),
                "low_eps_y": pd.to_numeric(self.analyst_df.at[ticker, "Low EPS Estimate"], errors = "coerce"),
                "avg_eps_y": pd.to_numeric(self.analyst_df.at[ticker, "Avg EPS Estimate"], errors = "coerce"),
                "high_eps_y": pd.to_numeric(self.analyst_df.at[ticker, "High EPS Estimate"], errors = "coerce"),
            })
       
        else:
       
            est = pd.Series(index = ["low_rev_y", "avg_rev_y", "high_rev_y", "low_eps_y", "avg_eps_y", "high_eps_y"], dtype = float)

        first_fc = self.forecast.get(ticker, pd.DataFrame())

        if first_fc is not None and not first_fc.empty:

            first = first_fc.iloc[0]

        else:

            first = pd.Series(index = ["low_rev", "avg_rev", "high_rev", "low_eps", "avg_eps", "high_eps"], dtype = float)


        def pct(
            new,
            old
        ):
            
            return (new - old) / abs(old) if pd.notna(new) and old not in (0, 0.0) else 0.0


        rev_covs = {
            "low_y": pct(est.get("low_rev_y"), base_rev),
            "avg_y": pct(est.get("avg_rev_y"), base_rev),
            "high_y": pct(est.get("high_rev_y"), base_rev),
            "low_fs": pct(first.get("low_rev"), base_rev),
            "avg_fs": pct(first.get("avg_rev"), base_rev),
            "high_fs": pct(first.get("high_rev"), base_rev),
        }
        
        eps_covs = {
            "low_y": pct(est.get("low_eps_y"), base_eps),
            "avg_y": pct(est.get("avg_eps_y"), base_eps),
            "high_y": pct(est.get("high_eps_y"), base_eps),
            "low_fs": pct(first.get("low_eps"), base_eps),
            "avg_fs": pct(first.get("avg_eps"), base_eps),
            "high_fs":pct(first.get("high_eps"), base_eps),
        }
        
        return rev_covs, eps_covs
    
    
    def regression_dict(
        self
    ) -> dict[str, pd.DataFrame]:
        """
        Returns ticker → DataFrame combining annual growth/returns with macro history.
        """
    
        macro_hist = self.macro.assign_macro_history()
       
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
                "Revenue Growth", 
                "EPS Growth", 
                "Return",
                "InterestRate_pct_change",
                "Inflation_rate", 
                "GDP_growth", 
                "Unemployment_pct_change"
            ]]

        return combined
    
    
    def get_annual_income_growth(
        self
    ) -> dict[str, pd.DataFrame]:
        """
        Returns a mapping ticker → DataFrame of annual 'Revenue Growth', 'EPS Growth', and 'Return'.
        """
    
        growth: dict[str, pd.DataFrame] = {}
     
        window = pd.DateOffset(months = 1)

        for tkr in self.tickers:
    
            income_ann_df = self.income_ann.get(tkr)
    
            if income_ann_df is None or income_ann_df.empty:
     
                growth[tkr] = pd.DataFrame() 
     
                continue

            ann_num = income_ann_df.map(_parse_abbrev).ffill(axis = 1).fillna(0)
    
            rev_g = ann_num.loc["Revenue Growth"]
     
            eps_g = ann_num.loc["EPS Growth"]

            closes = macro.r.close[tkr].sort_index()
    
    
            def window_mean(
                dt: pd.Timestamp
            ) -> float:
     
                return closes.loc[dt - window : dt + window].mean()


            mean_price = pd.Series(
                {dt: window_mean(dt) for dt in rev_g.index}
            ).sort_index()
    
            ret_g = mean_price.pct_change()

            df = pd.DataFrame({
                "Revenue Growth": rev_g,
                "EPS Growth": eps_g,
                "Return": ret_g,
            }).dropna()
    
            growth[tkr] = df

        return growth    
    
