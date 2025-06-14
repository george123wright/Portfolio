# Portfolio

This repository contains various utilities for financial data processing,
forecast generation and portfolio optimisation.

Several scripts require access to local Excel files that were previously
hard‑coded. Paths can now be configured using environment variables. The
most common variables are:

* `PORTFOLIO_DIR` – Base directory for input/output Excel files.
* `PORTFOLIO_STOCK_DATA_DIR` – Location of downloaded stock analysis data.
* `PORTFOLIO_HIST_PATH` and `PORTFOLIO_FORECAST_PATH` – Macro data files.
* `PORTFOLIO_DCF_FILE` – Location of the DCF Excel workbook.

For example, run a forecast script with custom locations:

```bash
export PORTFOLIO_DIR=/path/to/files
export PORTFOLIO_STOCK_DATA_DIR=/path/to/stock_data
python forecasts/dcfe.py
```

If the variables are not set, the original paths are used as defaults.
