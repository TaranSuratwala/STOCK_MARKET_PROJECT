# Artha Drishti — ML-Based Advanced Stock Screener

**Artha Drishti** is a machine-learning powered stock analysis and screening platform built with [Streamlit](https://streamlit.io/). It supports Indian (NSE/BSE) and global exchanges and combines technical analysis, fundamental analysis, ML predictions, backtesting, risk analytics, news sentiment, and portfolio management in one interactive web application.

---

## Features

| Module | Description |
|---|---|
| **Stock Screener** | Screen Nifty 50 / Nifty 500 / NSE stocks with customizable technical & ML filters |
| **Stock Analysis** | Deep-dive into any stock: price prediction, support/resistance, news sentiment |
| **Multi-Strategy** | Build, evaluate, and compare rule-based trading strategies |
| **Backtesting** | Test strategies against historical OHLCV data with full performance metrics |
| **Risk Analytics** | Sharpe ratio, Sortino ratio, Beta, VaR, Max Drawdown, stock comparison, portfolio risk |
| **Market Overview** | Global indices, sector performance, sector rotation detection, correlation finder |
| **Portfolio** | Track your holdings, analyze P&L, and compute portfolio-level risk |
| **Watchlist** | Save and monitor your favourite stocks |Collapse commentComment on line R18Copilot commented on Feb 24, 2026 CopilotAIon Feb 24, 2026AuthorMore actionsInconsistent spelling: "favourite" (line 18) uses British English while the rest of the documentation appears to mix British and American English. Consider using consistent spelling throughout, either "favorite" (American) or "favourite" (British) and ensuring similar words follow the same convention.
  
    
      
        Suggested change
      
    
    
      
          
            
            | **Watchlist** | Save and monitor your favourite stocks |
          
          
            
            | **Watchlist** | Save and monitor your favorite stocks |
          
      
    
    Unable to apply suggestions when viewing a specific commit range.
  
Copilot uses AI. Check for mistakes.Positive FeedbackNegative FeedbackReactWrite a replyResolve comment
| **Price Alerts** | Create price-trigger alerts and view alert history |
| **Fundamental Analysis** | P/E, P/B, PEG, EV/EBITDA, ROE, ROCE, debt ratios, and more via yfinance |
| **News Sentiment** | Live news fetched from NewsAPI and scored with TextBlob |
| **Authentication** | User registration, login, and per-user data persistence (SHA-256 + salt) |

---

## Tech Stack

- **Frontend / UI**: Streamlit, Plotly
- **Data**: yfinance, pandas, NumPy
- **Technical Analysis**: `ta` library, mplfinance
- **Machine Learning**: scikit-learn (Random Forest, Gradient Boosting), TensorFlow / Keras, XGBoost
- **NLP / Sentiment**: TextBlob, NLTK
- **Visualisation**: Plotly, Matplotlib, Seaborn
- **Other**: SciPy, NetworkX, Optuna, Transformers (HuggingFace), Flask (export API)
Collapse commentComment on lines R31 to R35Copilot commented on Feb 24, 2026 CopilotAIon Feb 24, 2026AuthorMore actionsThe tech stack lists several libraries (XGBoost, Flask, Optuna, Transformers, NetworkX) that are installed in requirements.txt but do not appear to be used in the codebase. Consider removing these from the tech stack list to accurately reflect the technologies actively used in the application, or clarify that these are available for future use.
  
    
      
        Suggested change
      
    
    
      
          
            
            - **Machine Learning**: scikit-learn (Random Forest, Gradient Boosting), TensorFlow / Keras, XGBoost
          
          
            
            - **NLP / Sentiment**: TextBlob, NLTK
          
          
            
            - **Visualisation**: Plotly, Matplotlib, Seaborn
          
          
            
            - **Other**: SciPy, NetworkX, Optuna, Transformers (HuggingFace), Flask (export API)
          
          
            
            - **Machine Learning**: scikit-learn (Random Forest, Gradient Boosting), TensorFlow / Keras
          
          
            
            - **NLP / Sentiment**: TextBlob, NLTK
          
          
            
            - **Visualisation**: Plotly, Matplotlib, Seaborn
          
          
            
            - **Other**: SciPy
          
          
            
            - **Optional / Planned (available via requirements)**: XGBoost, NetworkX, Optuna, Transformers (HuggingFace), Flask (for potential API export)
          
      
    
    Unable to apply suggestions when viewing a specific commit range.
  
Copilot uses AI. Check for mistakes.Positive FeedbackNegative FeedbackReactWrite a replyResolve comment
---

## Project Structure

```
STOCK_MARKET_PROJECT/
├── app.py                   # Main Streamlit application (entry point)
├── auth.py                  # User authentication & session management
├── advanced_strategies.py   # Strategy engine & sector rotation detector
├── backtester.py            # Historical strategy backtester
├── enhanced_screener.py     # ML-powered stock screener
├── export_utils.py          # Report / data export utilities
├── fundamental_analysis.py  # Fundamental metrics via yfinance
├── multi_exchange.py        # Multi-exchange support (NSE, BSE, NYSE, NASDAQ, LSE)
├── prototype.py             # Prototype / experimental features
├── report_generator.py      # HTML report generation
├── risk_analytics.py        # Risk metrics (Sharpe, VaR, Beta, etc.)
├── stock_ratings.py         # Stock rating & scoring engine
├── technical_patterns.py    # Chart pattern & support/resistance detection
├── requirements.txt         # Python dependencies
├── Nifty50.csv              # Nifty 50 stock list
├── Ticker_List_NSE_India.csv# Full NSE ticker list
├── ind_nifty500list.csv     # Nifty 500 stock list
├── Strategy1.json           # Sample saved strategy
└── ml_config.json           # ML model configuration
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/TaranSuratwala/STOCK_MARKET_PROJECT.git
cd STOCK_MARKET_PROJECT

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## Configuration

### News API Key

To enable the live news sentiment feature, obtain a free API key from [newsapi.org](https://newsapi.org/) and set it as an environment variable before running the app:

```bash
export NEWS_API_KEY="your_newsapi_key_here"   # Linux / macOS
set NEWS_API_KEY=your_newsapi_key_here         # Windows
```

Alternatively, you can set it directly in `app.py`, but **avoid committing secrets to version control**.Collapse commentComment on line R108Copilot commented on Feb 24, 2026 CopilotAIon Feb 24, 2026AuthorMore actionsThe README advises against committing secrets to version control, but the codebase currently contains a hardcoded NEWS_API_KEY in app.py (line 46). This is a security concern. The README should explicitly mention that the current hardcoded key in app.py should be removed or replaced, and all users should set their own keys via environment variables as described.
  
    
      
        Suggested change
      
    
    
      
          
            
            Alternatively, you can set it directly in `app.py`, but **avoid committing secrets to version control**.
          
          
            
            The `app.py` file currently contains a hardcoded `NEWS_API_KEY` placeholder used for demonstration. **You should remove or replace this value and rely on the `NEWS_API_KEY` environment variable shown above.** Do not commit real API keys or other secrets to version control.
          
      
    
    Unable to apply suggestions when viewing a specific commit range.
  
Copilot uses AI. Check for mistakes.Positive FeedbackNegative FeedbackReactWrite a replyResolve comment

---

## Usage

1. **Register / Login** — Create an account or continue as a guest.
2. **Stock Screener** — Select an index (Nifty 50 / Nifty 500 / All NSE), apply filters, and run the screener to get a ranked list of stocks.
3. **Stock Analysis** — Enter any NSE ticker to view price predictions, technical indicators, fundamental data, and news sentiment.
4. **Multi-Strategy** — Load or create a trading strategy, evaluate it on a stock, and compare multiple strategies side-by-side.
5. **Backtesting** — Select a strategy and date range to backtest it against historical data and view P&L, win rate, and drawdown charts.
6. **Risk Analytics** — Compute risk metrics for a single stock, compare multiple stocks, or analyse your portfolio risk.
7. **Market Overview** — View global indices, sector heatmaps, rotation signals, and correlation matrices.
8. **Portfolio / Watchlist / Alerts** — Manage your holdings, watchlist, and price-trigger alerts.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---
