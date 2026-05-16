# Stock Market Project

Collection of stock screening and analysis prototypes for NSE equities. Includes research scripts, rating utilities, and sample HTML reports.

## Highlights

- Technical and fundamental scoring helpers
- HTML report generator for screening results
- Sample outputs and ticker lists for NSE and Nifty 500

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
