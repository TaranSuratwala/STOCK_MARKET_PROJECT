<p align="center">
	<img src="https://capsule-render.vercel.app/api?type=rect&color=0:b45309,100:0f766e&height=120&section=header&text=Stock%20Market%20Project&fontSize=34&fontColor=F8FAFC&fontAlignY=60" alt="Stock Market Project banner" />
</p>

<p align="center">
	<img src="https://img.shields.io/github/last-commit/TaranSuratwala/STOCK_MARKET_PROJECT?style=flat-square" alt="Last commit" />
	<img src="https://img.shields.io/github/languages/top/TaranSuratwala/STOCK_MARKET_PROJECT?style=flat-square" alt="Top language" />
	<img src="https://img.shields.io/github/repo-size/TaranSuratwala/STOCK_MARKET_PROJECT?style=flat-square" alt="Repo size" />
</p>

# Stock Market Project

Collection of stock screening and analysis prototypes for NSE equities. Includes research scripts, rating utilities, and sample HTML reports.

## Highlights

- Technical and fundamental scoring helpers
- HTML report generator for screening results
- Sample outputs and ticker lists for NSE and Nifty 500

## Site Theme

A GitHub Pages ready theme lives in `docs/`. Enable Pages to serve from the `/docs` folder.

## Repository Contents

- `STOCK_SCREENER_AND_ANALYSER.py` (prototype script, currently commented)
- `prototype.py` (experimental analysis routines, commented)
- `report_generator.py` (HTML report builder)
- `stock_ratings.py` (rating and categorization logic)
- CSV ticker lists and `stock_screening_report_*.html` outputs

## Setup

Python 3.9+ recommended.

Base dependencies:

```bash
pip install pandas plotly
```

Optional dependencies for the prototype scripts:

```bash
pip install yfinance numpy ta scikit-learn tensorflow selenium webdriver-manager textblob
```

## Usage

Example: generate a summary report from results data.

```python
import pandas as pd
from report_generator import create_summary_report

results = pd.read_csv("your_results.csv")
create_summary_report(results)
```

The report expects columns such as `Symbol`, `Market Cap`, `Category`, `Technical Score`, `Momentum`, `Volume`, `Trend`, and `Overall Rating`.

## Notes

- The larger prototype scripts are commented out. Uncomment sections and install optional dependencies before running them.
