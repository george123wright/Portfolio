from __future__ import annotations
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
from functools import cached_property
import os
from ratio_data import RatioData

r = RatioData()


COUNTRY_FALLBACK = {   
    "Guernsey": "United Kingdom",
    "Taiwan":   "China",
}

_QVARS = ["Interest", "Cpi", "Gdp", "Unemp"]
_VAR_KEY = {
    "Interest": "Interest_Rate",
    "Cpi":      "Inflation_Rate",
    "Gdp":      "Gdp",
    "Unemp":    "Unemployment_Rate",
}


class MacroData:
    SHEET_INDEXES = "Stock Indexes and Commodities"
    _FORECAST_SHEETS = {
        "Inflation_Rate":      "Inflation_Rate",
        "Gdp":                 "Gdp",
        "Unemployment_Rate":   "Unemployment_Rate",
        "Interest_Rate":       "Interest_Rate",
    }

    def __init__(
        self,
        hist_path: str | Path | None = None,
        forecast_path: str | Path | None = None,
    ) -> None:
        self.today         = dt.date.today()
        self.hist_path     = Path(
            hist_path
            or os.environ.get(
                "PORTFOLIO_HIST_PATH",
                "/Users/georgewright/macro_and_index_data.xlsx",
            )
        )
        self.forecast_path = Path(
            forecast_path
            or os.environ.get(
                "PORTFOLIO_FORECAST_PATH",
                f"/Users/georgewright/Portfolio_Optimisation_Data_{self.today}.xlsx",
            )
        )
        self.r = RatioData()

    @staticmethod
    def _canon(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _annualise(series: pd.Series, method: str, col: str, pct: str) -> pd.DataFrame:
        if isinstance(series.index, pd.PeriodIndex):
            series = series.to_timestamp(how="end")
        ann = series.resample("YE").mean() if method == "mean" else series.resample("YE").last()
        ann.index = ann.index.to_period("Y")
        df = ann.to_frame(col)
        df[pct] = df[col].pct_change(fill_method=None)
        return df

    @staticmethod
    def _with_us_fallback(df: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=us.columns)
        return df.combine_first(us.loc[df.index])


    @cached_property
    def _sheet_key(self) -> dict[str,str]:
        wb = pd.ExcelFile(self.hist_path, engine="openpyxl")
        key = {self._canon("United States"): "United States"}
        for s in wb.sheet_names:
            if s not in (self.SHEET_INDEXES, "United States"):
                key[self._canon(s)] = s
        return key

    def _sheet_for(self, country: str | None) -> str:
        if not country:
            country = "United States"
        country = COUNTRY_FALLBACK.get(country, country)
        return self._sheet_key.get(self._canon(country), "United States")


    @cached_property
    def historical(self) -> dict[str,pd.DataFrame]:
        wb = pd.ExcelFile(self.hist_path, engine="openpyxl")
        # first do US
        us_q = pd.read_excel(self.hist_path,
                             sheet_name="United States", index_col=0,
                             engine="openpyxl", parse_dates=True)
        us_q.index = us_q.index.to_period("Q")

        us_ir  = self._annualise(us_q["Interest"], "mean",
                                 "Interest_Rate", "InterestRate_pct_change")
        us_inf = self._annualise(us_q["Cpi"], "mean",
                                 "CPI", "Inflation_rate")
        us_gdp = self._annualise(us_q["Gdp"], "last",
                                 "Gdp", "GDP_growth")
        us_ue  = self._annualise(us_q["Unemp"], "mean",
                                 "Unemployment_Rate", "Unemployment_pct_change")

        us_hist = us_ir.join([
            us_inf["Inflation_rate"],
            us_gdp["GDP_growth"],
            us_ue["Unemployment_pct_change"]
        ], how="inner")

        data = {"United States": us_hist}

        for sheet in wb.sheet_names:
            if sheet in (self.SHEET_INDEXES, "United States"):
                continue

            df_q = pd.read_excel(self.hist_path,
                                 sheet_name=sheet, index_col=0,
                                 engine="openpyxl", parse_dates=True)
            df_q.index = df_q.index.to_period("Q")

            for v in _QVARS:
                if v not in df_q.columns:
                    df_q[v] = us_q[v].loc[df_q.index].values
                else:
                    df_q[v] = df_q[v].fillna(us_q[v].loc[df_q.index])

            ir  = self._annualise(df_q["Interest"], "mean",
                                  "Interest_Rate", "InterestRate_pct_change")
            inf = self._annualise(df_q["Cpi"], "mean",
                                  "CPI", "Inflation_rate")
            gdp = self._annualise(df_q["Gdp"], "last",
                                  "Gdp", "GDP_growth")
            ue  = self._annualise(df_q["Unemp"], "mean",
                                  "Unemployment_Rate", "Unemployment_pct_change")

            df_y = ir.join([
                inf["Inflation_rate"],
                gdp["GDP_growth"],
                ue["Unemployment_pct_change"]
            ], how="inner")
            df_y = self._with_us_fallback(df_y, us_hist)
            data[sheet] = df_y

        return data


    @cached_property
    def historical_non_pct(self) -> dict[str,pd.DataFrame]:
        wb = pd.ExcelFile(self.hist_path, engine="openpyxl")
        us_q = pd.read_excel(self.hist_path,
                             sheet_name="United States",
                             index_col=0, engine="openpyxl",
                             parse_dates=True)
        us_q.index = us_q.index.to_period("Q")
        us_q = us_q[us_q.index >= pd.Period("2010Q1")]

        data = {
            "United States": us_q[_QVARS].ffill().fillna(0)
        }

        for sheet in wb.sheet_names:
            if sheet in (self.SHEET_INDEXES, "United States", "FX"):
                continue

            df_q = pd.read_excel(self.hist_path,
                                 sheet_name=sheet,
                                 index_col=0, engine="openpyxl",
                                 parse_dates=True)
            df_q.index = df_q.index.to_period("Q")
            df_q = df_q[df_q.index >= pd.Period("2010Q1")]

            for v in _QVARS:
                df_q[v] = df_q.get(v, us_q[v].loc[df_q.index]).ffill().fillna(0)

            if sheet in ["CHINA","United Kingdom","Germany"]:
                df_q["Gdp"] = df_q["Gdp"] / 1e9

            data[sheet] = df_q[_QVARS]

        return data


    @cached_property
    def currency(self) -> pd.Series:
        df = pd.read_excel(self.forecast_path,
                           sheet_name="Currency",
                           index_col=0,
                           engine="openpyxl",
                           parse_dates=True)
        return df["Last"]


    @cached_property
    def prices(self) -> pd.DataFrame:
        df = pd.read_excel(self.hist_path,
                           sheet_name=self.SHEET_INDEXES,
                           index_col=0,
                           engine="openpyxl",
                           parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df


    @cached_property
    def forecasts(self) -> dict[str,pd.DataFrame]:
        out = {}
        wb = pd.ExcelFile(self.forecast_path, engine="openpyxl")
        for var, sheet in self._FORECAST_SHEETS.items():
            df = pd.read_excel(self.forecast_path,
                               sheet_name=sheet,
                               index_col=0,
                               engine="openpyxl")
            try:
                df.index = pd.to_datetime(df.index, format="%Y").to_period("Y")
            except (ValueError,TypeError):
                pass
            out[var] = df
        self._forecast_key = {
            self._canon(idx): idx
            for idx in out["Interest_Rate"].index
        }
        return out

    def _forecast_for(self, country: str) -> str:
        return self._forecast_key.get(
            self._canon(country), "United States"
        )

    def _fc_row(self, var_short: str, sheet: str) -> pd.Series:
        if sheet in getattr(self, "_hist_fallback", {}):
            country = "United States"
        else:
            country = sheet
        return self.forecasts[_VAR_KEY[var_short]].loc[
            self._forecast_for(country)
        ]

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

