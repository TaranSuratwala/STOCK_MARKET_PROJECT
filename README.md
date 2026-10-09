<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=0:b45309,100:0f766e&height=120&section=header&text=Stock%20Market%20Project&fontSize=34&fontColor=F8FAFC&fontAlignY=60" alt="Stock Market Project banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/TaranSuratwala/STOCK_MARKET_PROJECT?style=flat-square" alt="Last commit" />
  <img src="https://img.shields.io/github/languages/top/TaranSuratwala/STOCK_MARKET_PROJECT?style=flat-square" alt="Top language" />
  <img src="https://img.shields.io/github/repo-size/TaranSuratwala/STOCK_MARKET_PROJECT?style=flat-square" alt="Repo size" />
</p>

# Stock Market Project

A Python-based repository for experimenting with stock screening and analysis workflows focused on NSE equities.

## Overview

This project brings together reusable scoring utilities, report-generation logic, and research prototypes used to evaluate stocks and present screening outcomes in HTML format.

## Key Capabilities

- Technical and fundamental scoring helpers
- Stock rating and categorization logic
- Automated HTML summary report generation
- Included sample ticker datasets and generated reports

## Repository Structure

| Path | Description |
| --- | --- |
| `stock_ratings.py` | Rating and categorization utilities |
| `report_generator.py` | HTML report creation functions |
| `STOCK_SCREENER_AND_ANALYSER.py` | Prototype screener script (currently commented) |
| `prototype.py` | Experimental analysis routines (currently commented) |
| `Ticker_List_NSE_India.csv`, `ind_nifty500list.csv` | Input ticker universes |
| `stock_screening_report_*.html` | Example generated outputs |
| `docs/` | GitHub Pages-friendly documentation assets |

## Requirements

- Python 3.9 or later

### Base Dependencies

```bash
pip install pandas plotly
```

### Optional Prototype Dependencies

Install these only if you plan to run prototype experimentation scripts:

```bash
pip install yfinance numpy ta scikit-learn tensorflow selenium webdriver-manager textblob
```

## Usage

Example: generate a summary report from tabular screening results.

```python
import pandas as pd
from report_generator import create_summary_report

results = pd.read_csv("your_results.csv")
create_summary_report(results)
```

Expected columns include:
`Symbol`, `Market Cap`, `Category`, `Technical Score`, `Momentum`, `Volume`, `Trend`, and `Overall Rating`.

## Notes

- Some larger prototype scripts are intentionally commented out.
- Uncomment relevant sections and install optional dependencies before running those scripts.

## Disclaimer

This repository is intended for educational and research purposes only and should not be treated as financial advice.
