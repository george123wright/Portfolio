# Portfolio Optimisation Toolkit

This repository contains a collection of scripts for building equity return forecasts and constructing optimised portfolios. The code pulls data from online sources, generates valuation models and exports the results to Excel workbooks.

## Installation

The project requires Python 3.10+ and a number of scientific packages. Install the dependencies using:

```bash
pip install -r requirements.txt
```

Several scripts expect local Excel files as inputs/outputs (e.g. `Portfolio_Optimisation_Data_YYYY-MM-DD.xlsx`). Update the paths inside the modules if your files live elsewhere.

## Repository Layout

```
fetch_data/       - Utilities that download analyst fundamentals, price history and macro data
data_processing/  - Cleaning functions and ratio loaders used by the forecasting models
forecasts/        - Forecasting models (DCF, DCFE, Prophet, SARIMAX, regression) and the combined forecast
functions/        - Reusable helpers: regression, CAPM/Black–Litterman, covariance utilities, Excel export
Optimiser/        - Portfolio optimisation routines and the main optimisation script
indicators/       - Technical analysis scores and Reddit sentiment scrapers
maps/             - Mapping tables for currencies, sectors and indexes
rel_val/          - Relative valuation models (PE, PS, PBV, EV/Sales, Graham, etc.)
```

## Typical Workflow

1. **Download data**
   - `python fetch_data/financial_data.py` retrieves analyst information, financial statements and historical prices via Yahoo Finance.
   - `python fetch_data/fetch_macro_data.py` downloads macroeconomic time series from FRED and major index prices.

2. **Generate indicators**
   - `python indicators/technical_indicators.py` calculates EMA, MACD, Bollinger Bands, RSI and other signals and writes them back to the workbook.
   - `python indicators/wallstreetbets_scrapping.py` scrapes r/wallstreetbets posts to gauge sentiment.

3. **Create forecasts**
   - Run individual models such as `forecasts/prophet_model.py`, `forecasts/Sarimax.py`, `forecasts/dcf.py`, `forecasts/dcfe.py` and `forecasts/ri.py` to produce price/return predictions.
   - `forecasts/Combination_Forecast.py` aggregates the outputs from all models and produces a blended forecast plus a scoring table.

4. **Optimise the portfolio**
   - Execute `python Optimiser/Portfolio_Optimisation.py` to calculate covariance matrices, apply a range of optimisers (max Sharpe, Sortino, MIR, etc.) and export portfolio weights, performance metrics and breakdown tables.

Intermediate data and results are written to dated Excel files so they can be inspected or further processed outside Python.

## Notes

Some scripts reference absolute paths under the author's home directory. Adjust these paths to match your environment before running the code. A few models rely on historical files that are not included in the repository.

## License

This project is provided as-is under the MIT License.
