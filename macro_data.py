
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ratio_data import RatioData

COUNTRY_FALLBACK = {   
    "Guernsey": "United Kingdom",
    "Taiwan": "China",
}

_QVARS = ["Interest", "Cpi", "Gdp", "Unemp"]   
_VAR_KEY = {                                 
    "Interest": "Interest_Rate",
    "Cpi": "Inflation_Rate",
    "Gdp": "Gdp",
    "Unemp": "Unemployment_Rate",
}


class MacroData:
    SHEET_INDEXES = "Stock Indexes and Commodities"

    _FORECAST_SHEETS = {
        "Inflation_Rate": "Inflation_Rate",
        "Gdp": "Gdp",
        "Unemployment_Rate": "Unemployment_Rate",
        "Interest_Rate": "Interest_Rate",
    }

    def __init__(self) -> None:
        self.today = dt.date.today()
        self.hist_path = Path("")
        self.forecast_path = Path(
            f"Portfolio_Optimisation_Data_{self.today}.xlsx"
        )

        self.historical: dict[str, pd.DataFrame] = {}
        self.historical_non_pct: dict[str, pd.DataFrame] = {}
        self.prices: pd.DataFrame | None = None
        self.forecasts: dict[str, pd.DataFrame] = {}

        self._sheet_key: dict[str, str] = {}      
        self._forecast_key: dict[str, str] = {}   
        self._hist_fallback: dict[str, set[str]] = {} 

        self.r = RatioData()

        self._load_historical()
        self._load_prices()
        self._load_forecasts()
        self._load_historical_non_pct()

    @staticmethod
    def _canon(name: str) -> str:
        """strip/lowcase for dictionary look-ups"""
        return name.strip().lower()

    def _sheet_for(self, country: str | None) -> str:
        """
        Return the sheet-name that holds *country*’s historical data,
        after applying COUNTRY_FALLBACK and case-insensitive matching.
        Defaults to 'United States'.
        """
        if not country:
            country = "United States"
        country = COUNTRY_FALLBACK.get(country, country)
        return self._sheet_key.get(self._canon(country), "United States")

    def _forecast_for(self, country: str) -> str:
        """
        Return the row-label used in forecast DataFrames for *country*.
        Guaranteed to succeed (falls back to U.S.).
        """
        return self._forecast_key.get(self._canon(country), "United States")

    @staticmethod
    def _annualise(
        series: pd.Series,
        method: str,
        col_name: str,
        pct_name: str,
    ) -> pd.DataFrame:
        """Quarterly → annual series with pct-change column."""
        if isinstance(series.index, pd.PeriodIndex):
            series = series.to_timestamp(how="end")

        ann = series.resample("YE").mean() if method == "mean" else series.resample("YE").last() 
        ann.index = ann.index.to_period("Y")

        df = ann.to_frame(col_name)
        df[pct_name] = df[col_name].pct_change(fill_method=None)
        return df

    @staticmethod
    def _with_us_fallback(
        df_country: pd.DataFrame, df_us: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fill *missing columns* or *NaNs* in df_country with U.S. values.
        Index alignment is preserved; no copy returned to caller.
        """
        df_country = df_country.reindex(columns=df_us.columns)
        return df_country.combine_first(df_us.loc[df_country.index])

    def _load_historical(self) -> None:
        wb = pd.ExcelFile(self.hist_path, engine="openpyxl")

        us_q = pd.read_excel(
            self.hist_path,
            sheet_name="United States",
            index_col=0,
            engine="openpyxl",
            parse_dates=True,
        )
        us_q.index = us_q.index.to_period("Q")

        us_ir = self._annualise(
            us_q["Interest"], "mean", "Interest_Rate", "InterestRate_pct_change"
        )
        us_inf = self._annualise(
            us_q["Cpi"], "mean", "CPI", "Inflation_rate"
        )
        us_gdp = self._annualise(
            us_q["Gdp"], "last", "Gdp", "GDP_growth"
        )
        us_ue = self._annualise(
            us_q["Unemp"], "mean", "Unemployment_Rate", "Unemployment_pct_change"
        )
        us_hist = us_ir.join(
            [us_inf["Inflation_rate"], us_gdp["GDP_growth"], us_ue["Unemployment_pct_change"]],  # noqa: E501
            how="inner",
        )
        self.historical["United States"] = us_hist
        self._sheet_key[self._canon("United States")] = "United States"
        self._hist_fallback["United States"] = set()

        for sheet in wb.sheet_names:
            if sheet in (self.SHEET_INDEXES, "United States"):
                continue
            self._sheet_key[self._canon(sheet)] = sheet

            df_q = pd.read_excel(
                self.hist_path,
                sheet_name=sheet,
                index_col=0,
                engine="openpyxl",
                parse_dates=True,
            )
            df_q.index = df_q.index.to_period("Q")

            missing_cols = {
                col
                for col in _QVARS
                if (col not in df_q.columns) or df_q[col].isna().all()
            }

            for col in _QVARS:
                if col not in df_q.columns:
                    df_q[col] = us_q.loc[df_q.index, col].values
                else:
                    df_q[col] = df_q[col].fillna(us_q.loc[df_q.index, col])

            ir_y = self._annualise(
                df_q["Interest"], "mean", "Interest_Rate", "InterestRate_pct_change"
            )
            inf_y = self._annualise(df_q["Cpi"], "mean", "CPI", "Inflation_rate")
            gdp_y = self._annualise(df_q["Gdp"], "last", "Gdp", "GDP_growth")
            ue_y = self._annualise(
                df_q["Unemp"], "mean", "Unemployment_Rate", "Unemployment_pct_change"
            )

            df_y = ir_y.join(
                [inf_y["Inflation_rate"], gdp_y["GDP_growth"], ue_y["Unemployment_pct_change"]],  
                how="inner",
            )
            df_y = self._with_us_fallback(df_y, us_hist)
            self.historical[sheet] = df_y
            self._hist_fallback[sheet] = missing_cols

    def _load_historical_non_pct(self) -> None:
        wb = pd.ExcelFile(self.hist_path, engine="openpyxl")

        us_q = pd.read_excel(
            self.hist_path,
            sheet_name="United States",
            index_col=0,
            engine="openpyxl",
            parse_dates=True,
        )
        us_q.index = us_q.index.to_period("Q")
        us_q = us_q[us_q.index >= pd.Period("2010Q1")]

        self.historical_non_pct["United States"] = pd.DataFrame(
            {k: us_q[k].ffill().fillna(0) for k in _QVARS}
        )

        for sheet in wb.sheet_names:
            if sheet in (self.SHEET_INDEXES, "United States", "FX"):
                continue

            df_q = pd.read_excel(
                self.hist_path,
                sheet_name=sheet,
                index_col=0,
                engine="openpyxl",
                parse_dates=True,
            )
            df_q.index = df_q.index.to_period("Q")
            df_q = df_q[df_q.index >= pd.Period("2010Q1")]

            for col in _QVARS:
                if col not in df_q.columns:
                    df_q[col] = us_q.loc[df_q.index, col].values
                else:
                    df_q[col] = df_q[col].fillna(us_q.loc[df_q.index, col])

            gdp_series = df_q["Gdp"].ffill().fillna(0)
            if sheet in ["CHINA", "Spain", "Canada", "France", "Germany", "United Kingdom"]:
                gdp_series = gdp_series / 1e9

            self.historical_non_pct[sheet] = pd.DataFrame(
                {
                    "Interest": df_q["Interest"].ffill().fillna(0),
                    "Cpi": df_q["Cpi"].ffill().fillna(0),
                    "Gdp": gdp_series,
                    "Unemp": df_q["Unemp"].ffill().fillna(0),
                }
            )

    def _load_prices(self) -> None:
        df = pd.read_excel(
            self.hist_path,
            sheet_name=self.SHEET_INDEXES,
            index_col=0,
            engine="openpyxl",
            parse_dates=True,
        )
        df.index = pd.to_datetime(df.index)
        self.prices = df

    def _load_forecasts(self) -> None:
        for var, sheet in self._FORECAST_SHEETS.items():
            df = pd.read_excel(
                self.forecast_path,
                sheet_name=sheet,
                index_col=0,
                engine="openpyxl",
            )
            try:
                df.index = pd.to_datetime(df.index, format="%Y").to_period("Y")
            except (ValueError, TypeError):
                pass
            self.forecasts[var] = df

        self._forecast_key = {
            self._canon(idx): idx for idx in self.forecasts["Interest_Rate"].index
        }

    def _fc_row(self, var_short: str, sheet_name: str) -> pd.Series:
        """
        Return the forecast row for *var_short* (“Interest”, “Cpi”, …)
        corresponding to historical *sheet_name*.
        """
        if var_short in self._hist_fallback.get(sheet_name, set()):
            target_country = "United States"
        else:
            target_country = sheet_name
        return self.forecasts[_VAR_KEY[var_short]].loc[self._forecast_for(target_country)]

    def get_base_macro_fc(self, country: str | None) -> dict[str, float]:
        sheet = self._sheet_for(country)

        ir = self._fc_row("Interest", sheet)
        inf = self._fc_row("Cpi", sheet)
        gdp = self._fc_row("Gdp", sheet)
        ue = self._fc_row("Unemp", sheet)

        ir_last, gdp_last, ue_last = ir["Last"], gdp["Last"], ue["Last"]

        return {
            "InterestRate_pct_change": (ir["Q1/26"] - ir_last) / ir_last,
            "Inflation_rate": inf["Q1/26"],
            "GDP_growth": (gdp["2026"] - gdp_last) / gdp_last,
            "Unemployment_pct_change": (ue["Q1/26"] - ue_last) / ue_last,
        }

    def macro_forecast_dict(self) -> dict[str, dict[float]]:
        records: dict[str, dict[float]] = {}

        for ticker in self.r.tickers:
            country_raw = (
                self.r.analyst.loc[ticker, "country"]
                if "country" in self.r.analyst.columns
                else None
            )
            sheet = self._sheet_for(country_raw)

            records[ticker] = {
                "InterestRate": self._fc_row("Interest", sheet).values,
                "Consumer_Price_Index_Cpi": self._fc_row("Cpi", sheet).values,
                "GDP": self._fc_row("Gdp", sheet)[["2025", "2026"]].values,
                "Unemployment": self._fc_row("Unemp", sheet).values,
            }

        return records

    def assign_macro_history(self) -> pd.DataFrame:
        frames: dict[str, pd.DataFrame] = {}
        for ticker in self.r.tickers:
            sheet = self._sheet_for(
                self.r.analyst.loc[ticker, "country"]
                if "country" in self.r.analyst.columns
                else None
            )
            if sheet in self.historical:
                frames[ticker] = self.historical[sheet]
        return pd.concat(frames, names=["ticker", "year"])

    def assign_macro_history_non_pct(self) -> pd.DataFrame:
        frames: dict[str, pd.DataFrame] = {}
        for ticker in self.r.tickers:
            sheet = self._sheet_for(
                self.r.analyst.loc[ticker, "country"]
                if "country" in self.r.analyst.columns
                else None
            )
            if sheet in self.historical_non_pct:
                frames[ticker] = self.historical_non_pct[sheet]
        return pd.concat(frames, names=["ticker", "year"])
    
    def assign_macro_forecasts(self) -> dict[str, dict[str, float]]:
        records: dict[str, dict[str, float]] = {}
        for ticker in self.r.tickers:
            country = (self.r.analyst.loc[ticker, 'country']
                       if 'country' in self.r.analyst.columns else None)
            if country in COUNTRY_FALLBACK:
                country = COUNTRY_FALLBACK[country]
            records[ticker] = self.get_base_macro_fc(country)
        return records
