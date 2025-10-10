# import os
# import warnings
# import numpy as np
# import pandas as pd
# import yfinance as yf
# import ta
# import plotly.graph_objects as go
# import requests
# import multiprocessing as mp
# from tqdm import tqdm
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import mean_squared_error
# import tensorflow as tf
# from tensorflow.keras import Model, Input
# from tensorflow.keras.layers import (Conv1D, Bidirectional, LSTM, GRU, Dropout,
#                                      MultiHeadAttention, Add, LayerNormalization,
#                                      GlobalAveragePooling1D, Dense, Concatenate)
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager

# warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CACHE_DIR = "cache_data"
# os.makedirs(CACHE_DIR, exist_ok=True)

# features = ["Close", "RSI", "MACD", "MACD_signal", "BB_MID", "BB_UP", "BB_LOW",
#             "ADX", "DI_POS", "DI_NEG", "OBV", "SMA_20", "EMA_20", "EMA_50", "EMA_200",
#             "ATR", "stoch_k", "stoch_d", "CCI", "PE_Ratio", "PB_Ratio", "Market_Cap",
#             "Dividend_Yield", "EPS", "ROE", "ROCE", "VWAP", "MFI", "Williams_R", "TSI",
#             "Aroon_Up", "Aroon_Down", "Volume_SMA", "Price_ROC", "Volume_ROC", "BB_WIDTH",
#             "Keltner_Up", "Keltner_Low", "Ichimoku_A", "Ichimoku_B", "High_Low_Ratio",
#             "Close_Open_Ratio", "Intraday_Range", "Trend_Strength", "Historical_Volatility"]

# # Fallback Nifty 500 stock list in case fetching fails
# FALLBACK_NIFTY_500_STOCKS = [
#     "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "SBIN", "BHARTIARTL",
#     "ITC", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE", "HCLTECH", "KOTAKBANK",
#     "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "ONGC", "TATAMOTORS", "NTPC",
#     "BAJAJFINSV", "POWERGRID", "ADANIPORTS", "COALINDIA", "M&M", "JSWSTEEL", "TECHM",
#     "TATASTEEL", "INDUSINDBK", "GRASIM", "DRREDDY", "HINDALCO", "CIPLA", "BRITANNIA",
#     "DIVISLAB", "EICHERMOT", "BPCL", "SHREECEM", "APOLLOHOSP", "TATAPOWER", "ADANIENT",
#     "TATACONSUM", "SIEMENS", "BAJAJ-AUTO", "HAVELLS", "GODREJCP", "UPL", "PIDILITIND",
#     "TORNTPHARM", "BERGEPAINT", "DABUR", "AMBUJACEM", "BANKBARODA", "INDIGO", "HEROMOTOCO",
#     "BOSCHLTD", "MARICO", "HDFCLIFE", "SBILIFE", "GAIL", "IOC", "HINDPETRO", "DLF",
#     "VEDL", "PNB", "SAIL", "CANBK", "LUPIN", "ACC", "NMDC", "MOTHERSON", "BANDHANBNK",
#     "INDUSTOWER", "TRENT", "IDEA", "ZEEL", "JUBLFOOD", "MUTHOOTFIN", "GODREJPROP", "PEL",
#     "ESCORTS", "COLPAL", "CONCOR", "SBICARD", "BALKRISIND", "ATUL", "ASTRAL", "CHAMBLFERT",
#     "AUROPHARMA", "ABFRL", "VOLTAS", "ASHOKLEY", "DIXON", "NATIONALUM", "MPHASIS", "NAUKRI",
#     "BIOCON", "PERSISTENT", "LTTS", "LTIM", "COFORGE", "LALPATHLAB", "DMART", "ZOMATO",
#     "PAYTM", "POLICYBZR", "NYKAA", "ADANIGREEN", "ADANIPOWER", "ADANITRANS", "ATGL",
#     "IRCTC", "CHOLAFIN", "LICHSGFIN", "RECLTD", "PFC", "CUMMINSIND", "ABB", "CROMPTON",
#     "BLUESTARCO", "WHIRLPOOL", "HONAUT", "KANSAINER", "AUBANK", "IDFCFIRSTB", "FEDERALBNK",
#     "RBLBANK", "INDHOTEL", "WESTLIFE", "BSOFT", "POLYCAB", "KEI", "FINOLEX", "APLAPOLLO",
#     "JINDALSTEL", "TATACHEM", "DEEPAKNTR", "GNFC", "COROMANDEL", "PIIND", "APOLLOTYRE",
#     "CEAT", "MRF", "SUNDRMFAST", "EXIDEIND", "AMARAJABAT", "BATAINDIA", "RELAXO",
#     "PAGEIND", "LAURUSLABS", "ALKEM", "GRANULES", "NATCOPHARMA", "IPCALAB", "GLAXO",
#     "ABBOTINDIA", "PFIZER", "SANOFI", "JUBLPHARMA", "ASTRAZEN", "ZYDUSLIFE", "CENTRALBK",
#     "IOB", "UNIONBANK", "MAHABANK", "INDIANB", "UCOBANK", "JKBANK", "SOUTHBANK", "PSB",
#     "BANKINDIA", "MANAPPURAM", "SHRIRAMFIN", "IIFL", "BAJAJHLDNG", "PNBHOUSING",
#     "CANFINHOME", "AIAENG", "SCHAEFFLER", "TIMKEN", "SKFINDIA", "FINCABLES", "HINDCOPPER",
#     "RATNAMANI", "WELCORP", "WELSPUNIND", "RAYMOND", "SRTRANSFIN", "AAVAS", "HOMEFIRST",
#     "MAXHEALTH", "FORTIS", "NARAYANA", "METROPOLIS", "THYROCARE", "KIMS", "RAINBOW",
#     "HATSUN", "HERITGFOOD", "VADILALIND", "DODLA", "PGHH", "GILLETTE", "JYOTHYLAB",
#     "VGUARD", "SYMPHONY", "ORIENTELEC", "FINEORG", "AARTIIND", "NAVINFLUOR",
#     "SUMICHEM", "TATACOFFEE", "JKLAKSHMI", "RAMCOCEM", "DALBHARAT", "HEIDELBERG", "JKCEMENT",
#     "STARCEMENT", "MCDOWELL-N", "CCL", "RADICO", "GLOBUSSPR", "GRINDWELL", "CARBORUNIV",
#     "KAJARIACER", "ORIENTBELL", "PRISMCEM", "SUPREMEIND", "NILKAMAL", "HINDWARE", "CERA",
#     "SOMANYCERA", "VINATIORGA", "SRF", "EIDPARRY", "DHAMPURSUG", "BALRAMCHIN", "JKPAPER", "TNPL",
#     "ANDHRAPAP", "RPOWER", "JPPOWER", "TORNTPOWER", "CESC", "NHPC", "SJVN", "IRCON", "RVNL",
#     "RAILTEL", "BEL", "HAL", "BEML", "GRSE", "COCHINSHIP", "MAZDOCK", "HFCL", "TATACOMM",
#     "GOCOLORS", "TATAELXSI", "ROUTE", "TANLA", "EASEMYTRIP", "HAPPSTMNDS", "ZENSARTECH", "CAMS",
#     "CDSL", "IEX", "IIFLSEC", "ANGELONE", "MOTILALOFS", "5PAISA", "SWSOLAR", "CLEAN",
#     "SUZLON", "INOXWIND", "BHEL", "THERMAX", "TIINDIA", "ELECON", "PRAKASH", "CAMPUS",
#     "TCNSBRANDS", "ROSSARI", "HERANBA", "SUPRAJIT", "SUBROS", "JTEKTINDIA", "GABRIEL",
#     "ENDURANCE", "SANDHAR", "ICICIGI", "ICICIPRULI", "LINDEINDIA", "INOXGREEN", "PVRINOX",
#     "DEVYANI", "SAPPHIRE", "SHYAMMETL", "CAPLIPOINT", "GOKEX", "GLENMARK", "MANKIND",
#     "JBCHEPHARM", "TASTYBITE", "HATHWAY", "ZENTEC", "FSL", "TTKPRESTIG", "CENTURYTEX", "HUDCO",
#     "NBCC", "NLCINDIA", "GMDCLTD", "MOIL", "KPIL", "RCF", "NFL", "FACT", "GSFC"
# ]


# # THIS FUNCTION AUTOMATICALLY FETCHES THE NIFTY 500 LIST
# def get_nifty500_symbols():
#     driver = None
#     try:
#         print("Initializing a headless browser with webdriver-manager Chromedriver...")

#         chrome_options = Options()
#         chrome_options.add_argument("--headless=new")
#         chrome_options.add_argument("--disable-gpu")
#         chrome_options.add_argument("--no-sandbox")
#         chrome_options.add_argument("--disable-dev-shm-usage")
#         chrome_options.add_argument("--window-size=1920,1080")
#         chrome_options.add_argument("--log-level=3")

#         # Disable images and CSS for faster loading
#         prefs = {
#             "profile.managed_default_content_settings.images": 2,
#             "profile.managed_default_content_settings.stylesheets": 2,
#             "profile.managed_default_content_settings.cookies": 2,
#             "profile.managed_default_content_settings.javascript": 1,
#             "profile.managed_default_content_settings.plugins": 2,
#             "profile.managed_default_content_settings.popups": 2,
#             "profile.managed_default_content_settings.geolocation": 2,
#             "profile.managed_default_content_settings.media_stream": 2,
#         }
#         chrome_options.add_experimental_option("prefs", prefs)

#         driver = webdriver.Chrome(
#             service=Service(ChromeDriverManager().install()),
#             options=chrome_options
#         )

#         url = "https://en.wikipedia.org/wiki/NIFTY_500"
#         driver.get(url)

#         WebDriverWait(driver, 15).until(
#             EC.presence_of_element_located((By.ID, "constituents"))
#         )

#         html_source = driver.page_source
#         tables = pd.read_html(html_source)
#         nifty500_df = None
#         symbol_col_name = None
#         possible_colnames = ['Symbol', 'Ticker']

#         for table in tables:
#             table.columns = [str(col).strip() for col in table.columns]
#             for col_name in possible_colnames:
#                 if col_name in table.columns:
#                     nifty500_df = table
#                     symbol_col_name = col_name
#                     break
#             if nifty500_df is not None:
#                 break

#         if nifty500_df is None:
#             raise ValueError("Selenium loaded the page, but still could not find the table.")

#         symbols = nifty500_df[symbol_col_name].str.strip().tolist()
#         print(f"✅ Successfully fetched {len(symbols)} symbols using webdriver-manager.")
#         return symbols

#     except Exception as e:
#         print(f"⚠️ Failed to fetch NIFTY 500 symbols via Selenium: {e}")
#         print("Falling back to the hardcoded list.")
#         return FALLBACK_NIFTY_500_STOCKS

#     finally:
#         if driver:
#             driver.quit()

# def fetch_data(symbol="TATAPOWER", period="5y", interval="1d", force_download=False):
#     """
#     IMPROVED: More robust data fetching using yf.Ticker as the primary method.
#     """
#     yf_symbol = f"{symbol}.NS" if "." not in symbol else symbol
#     cache_path = os.path.join(CACHE_DIR, f"{yf_symbol.replace('.', '_')}_{period}_{interval}.csv")

#     if os.path.exists(cache_path) and not force_download:
#         try:
#             df = pd.read_csv(cache_path, index_col=0)
#             df.index = pd.to_datetime(df.index, errors="coerce")
#             df = df[~df.index.isna()]
#             if not df.empty and all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]):
#                 return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
#         except Exception:
#             # If cache is corrupted, proceed to download
#             pass

#     try:
#         # Primary Method: Use yf.Ticker which is more reliable for individual symbols
#         ticker = yf.Ticker(yf_symbol)
#         df = ticker.history(period=period, interval=interval, auto_adjust=False)

#         if df is None or df.empty:
#             # Fallback Method: Use yf.download
#             df = yf.download(yf_symbol, period=period, interval=interval, progress=False, timeout=10)

#         if df is None or df.empty:
#             return None

#         required_cols = ["Open", "High", "Low", "Close", "Volume"]
#         if not all(col in df.columns for col in required_cols):
#             return None

#         df.to_csv(cache_path)
#         return df.dropna(subset=required_cols)
#     except Exception:
#         return None

# def fetch_fundamental_data(symbol="TATAPOWER"):
#     try:
#         ticker = yf.Ticker(f"{symbol}.NS" if "." not in symbol else symbol)
#         info = ticker.info
#         return pd.DataFrame({
#             "PE_Ratio": [info.get("trailingPE", 0.0) or 0.0],
#             "EPS": [info.get("trailingEps", 0.0) or 0.0],
#             "PB_Ratio": [info.get("priceToBook", 0.0) or 0.0],
#             "Market_Cap": [info.get("marketCap", 0.0) or 0.0],
#             "Dividend_Yield": [info.get("dividendYield", 0.0) or 0.0],
#             "ROE": [info.get("returnOnEquity", 0.0) or 0.0],
#             "ROCE": [info.get("returnOnCapitalEmployed", 0.0) or 0.0]
#         })
#     except Exception:
#         return pd.DataFrame()

# def compute_indicators_fast(df, symbol="TATAPOWER"):
#     """Optimized indicator computation"""
#     if df is None or df.empty:
#         return pd.DataFrame()
#     df = df.copy()
#     for col in ["Close", "High", "Low", "Volume", "Open"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#     df = df.dropna(subset=["Close", "High", "Low", "Volume", "Open"])

#     close, high, low, vol, open_price = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]

#     # Compute all indicators in one pass
#     df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
#     macd = ta.trend.MACD(close)
#     df["MACD"], df["MACD_signal"] = macd.macd(), macd.macd_signal()
#     bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
#     df["BB_MID"], df["BB_UP"], df["BB_LOW"] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
#     adx = ta.trend.ADXIndicator(high, low, close, window=14)
#     df["ADX"], df["DI_POS"], df["DI_NEG"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
#     df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
#     df["SMA_20"] = close.rolling(20).mean()
#     df["EMA_20"], df["EMA_50"], df["EMA_200"] = close.ewm(span=20).mean(), close.ewm(span=50).mean(), close.ewm(span=200).mean()
#     df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
#     stoch = ta.momentum.StochasticOscillator(high, low, close)
#     df["stoch_k"], df["stoch_d"] = stoch.stoch(), stoch.stoch_signal()
#     df["CCI"] = ta.trend.CCIIndicator(high, low, close).cci()
#     df["VWAP"] = (vol * (high + low + close) / 3).cumsum() / vol.cumsum()
#     df["MFI"] = ta.volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()
#     df["Williams_R"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
#     df["TSI"] = ta.momentum.TSIIndicator(close).tsi()
#     aroon = ta.trend.AroonIndicator(high=high, low=low, window=25)
#     df["Aroon_Up"], df["Aroon_Down"] = aroon.aroon_up(), aroon.aroon_down()
#     df["Volume_SMA"] = vol.rolling(20).mean()
#     df["Price_ROC"] = ta.momentum.ROCIndicator(close, window=12).roc()
#     df["Volume_ROC"] = vol.pct_change(periods=12) * 100
#     df["BB_WIDTH"] = (df["BB_UP"] - df["BB_LOW"]) / df["BB_MID"]
#     keltner = ta.volatility.KeltnerChannel(high, low, close, window=20)
#     df["Keltner_Up"], df["Keltner_Low"] = keltner.keltner_channel_hband(), keltner.keltner_channel_lband()
#     ichimoku = ta.trend.IchimokuIndicator(high, low)
#     df["Ichimoku_A"], df["Ichimoku_B"] = ichimoku.ichimoku_a(), ichimoku.ichimoku_b()
#     df["High_Low_Ratio"] = np.where(low > 0, high / low, 1.0)
#     df["Close_Open_Ratio"] = np.where(open_price > 0, close / open_price, 1.0)
#     df["Intraday_Range"] = np.where(open_price > 0, (high - low) / open_price, 0.0)
#     rolling_mean, rolling_std = close.rolling(50).mean(), close.rolling(50).std()
#     df["Trend_Strength"] = np.where(rolling_std > 0, (close - rolling_mean) / rolling_std, 0.0)
#     df["Historical_Volatility"] = close.pct_change().rolling(20).std() * np.sqrt(252)

#     fund_df = fetch_fundamental_data(symbol)
#     if not fund_df.empty:
#         for key, value in fund_df.iloc[0].to_dict().items():
#             df[key] = value

#     df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').dropna()
#     for col in df.columns:
#         if col not in ['Date'] and pd.api.types.is_numeric_dtype(df[col]):
#             col_mean, col_std = df[col].mean(), df[col].std()
#             if col_std > 0:
#                 df[col] = df[col].clip(lower=col_mean - 5*col_std, upper=col_mean + 5*col_std)
#     return df

# def make_sequences_fast(df, window=90, train_ratio=0.8, symbol="TATAPOWER"):
#     """Optimized sequence creation"""
#     df_ind = compute_indicators_fast(df, symbol=symbol)
#     unique_features = [f for f in features if f in df_ind.columns]
#     df_feat = df_ind[unique_features].values
#     n_features = len(unique_features)

#     n_samples = len(df_feat)
#     X_raw = np.zeros((n_samples - 1, window, n_features), dtype=np.float32)

#     for i in range(1, n_samples):
#         seq_len = min(i, window)
#         X_raw[i-1, -seq_len:] = df_feat[i-seq_len:i]

#     y_raw = df_feat[1:, unique_features.index("Close")].reshape(-1, 1)
#     last_close_raw = df_feat[:-1, unique_features.index("Close")]
#     dates = df_ind.index[1:].values

#     split = int(len(X_raw) * train_ratio)
#     X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
#     y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

#     feature_scaler = RobustScaler()
#     X_train = feature_scaler.fit_transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
#     X_test = feature_scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)

#     target_scaler = RobustScaler()
#     y_train = target_scaler.fit_transform(y_train_raw).flatten()
#     y_test = target_scaler.transform(y_test_raw).flatten()

#     return (X_train, y_train, X_test, y_test, y_train_raw.flatten(), y_test_raw.flatten(),
#             last_close_raw[split:], dates[split:], feature_scaler, target_scaler, df_ind, unique_features)

# def build_model_fast(input_shape, lr=1e-3):
#     """
#     IMPROVED: Deeper model architecture with an added GRU layer.
#     """
#     inp = Input(shape=input_shape)

#     # Parallel convolutions for feature extraction
#     conv1 = Conv1D(32, 3, padding="causal", activation="relu")(inp)
#     conv2 = Conv1D(32, 5, padding="causal", activation="relu")(inp)
#     x = Concatenate()([conv1, conv2])
#     x = Dropout(0.2)(x)

#     # First recurrent layer (LSTM)
#     x = Bidirectional(LSTM(64, return_sequences=True))(x)
#     x = LayerNormalization()(x)
#     x = Dropout(0.2)(x)

#     # Second recurrent layer (GRU) to capture more complex patterns
#     x = Bidirectional(GRU(32, return_sequences=True))(x)
#     x = LayerNormalization()(x)
#     x = Dropout(0.2)(x)

#     # Attention mechanism
#     att = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
#     x = Add()([x, att])
#     x = LayerNormalization()(x)

#     # Pooling layer
#     avg_pool = GlobalAveragePooling1D()(x)
#     max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
#     x = Concatenate()([avg_pool, max_pool])

#     # Dense layers for final prediction
#     x = Dense(128, activation="relu")(x)
#     x = Dropout(0.3)(x)
#     x = Dense(64, activation="relu")(x)
#     x = Dropout(0.2)(x)
#     out = Dense(1)(x)

#     model = Model(inp, out)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="huber", metrics=["mae"])
#     return model

# def calculate_confidence(ensemble_preds, current_price):
#     mean_pred = np.mean(ensemble_preds)
#     std_pred = np.std(ensemble_preds)
#     cv = (std_pred / mean_pred * 100) if mean_pred != 0 else 100
#     pct_change = abs((mean_pred - current_price) / current_price * 100)
#     confidence = 100 - min(cv * 5, 50) - min(pct_change * 2, 30)
#     return max(0, min(100, confidence)), mean_pred

# def generate_confident_signals(all_preds, actual_prices, conf_threshold=75, signal_gap=7):
#     signals, confidences = [], []
#     last_signal_idx = -signal_gap

#     for i in range(len(all_preds[0])):
#         ensemble_preds = [pred[i] for pred in all_preds]
#         confidence, mean_pred = calculate_confidence(ensemble_preds, actual_prices[i])
#         confidences.append(confidence)

#         if confidence >= conf_threshold and (i - last_signal_idx >= signal_gap):
#             pct_change = ((mean_pred - actual_prices[i]) / actual_prices[i]) * 100
#             if pct_change > 4.0:
#                 signals.append('BUY')
#                 last_signal_idx = i
#             elif pct_change < -4.0:
#                 signals.append('SELL')
#                 last_signal_idx = i
#             else:
#                 signals.append('HOLD')
#         else:
#             signals.append('HOLD')

#     return signals, confidences

# def backtest_strategy(actual_prices, signals, initial_capital=100000):
#     capital, position, trades, portfolio_values = initial_capital, 0, [], [initial_capital]

#     for i, signal in enumerate(signals):
#         price = actual_prices[i]
#         if signal == 'BUY' and position == 0:
#             shares = int((capital * 0.95) / price)
#             if shares > 0:
#                 cost = shares * price * 1.001
#                 if cost <= capital:
#                     position, capital = shares, capital - cost
#                     trades.append({'type': 'BUY', 'price': price, 'index': i})
#         elif signal == 'SELL' and position > 0:
#             proceeds = position * price * 0.999
#             capital += proceeds
#             trades.append({'type': 'SELL', 'price': price, 'index': i})
#             position = 0
#         portfolio_values.append(capital + (position * price if position > 0 else 0))

#     if position > 0:
#         capital += position * actual_prices[-1] * 0.999

#     buy_hold = initial_capital
#     if len(actual_prices) > 0 and actual_prices[0] > 0:
#         shares_bh = int((initial_capital * 0.999) / actual_prices[0])
#         buy_hold = shares_bh * actual_prices[-1] * 0.999

#     return {
#         'final_value': capital,
#         'total_return': (capital - initial_capital) / initial_capital * 100,
#         'buy_hold_return': (buy_hold - initial_capital) / initial_capital * 100,
#         'excess_return': ((capital - initial_capital) / initial_capital - (buy_hold - initial_capital) / initial_capital) * 100,
#         'num_trades': len(trades),
#         'portfolio_values': portfolio_values
#     }

# def analyze_growth_potential(symbol):
#     """
#     FIXED & IMPROVED: Comprehensive, coherent analysis for weekly growth potential.
#     """
#     try:
#         df = fetch_data(symbol, period="2y", force_download=False)
#         if df is None or len(df) < 200:
#             return None

#         df_ind = compute_indicators_fast(df, symbol=symbol)
#         if df_ind.empty or len(df_ind) < 50:
#             return None

#         latest = df_ind.iloc[-1]
#         prev = df_ind.iloc[-2]
#         recent = df_ind.tail(30)

#         score = 0
#         signals = {}

#         # === 1. MOMENTUM & TREND (40 points) ===
#         # RSI (10 pts)
#         if 45 < latest['RSI'] < 65 and latest['RSI'] > df_ind['RSI'].tail(5).mean():
#             score += 10; signals['rsi'] = 'BUILDING_MOMENTUM'
#         elif latest['RSI'] > 65:
#             score += 5; signals['rsi'] = 'STRONG'
#         elif latest['RSI'] < 30:
#             score -= 5; signals['rsi'] = 'OVERSOLD'

#         # MACD (10 pts)
#         macd_hist = latest['MACD'] - latest['MACD_signal']
#         prev_macd_hist = prev['MACD'] - prev['MACD_signal']
#         is_fresh_cross = macd_hist > 0 and prev_macd_hist <= 0
#         if is_fresh_cross and macd_hist > prev_macd_hist:
#             score += 10; signals['macd_signal'] = 'FRESH_BULLISH_CROSS'
#         elif macd_hist > 0 and macd_hist > prev_macd_hist:
#             score += 7; signals['macd_signal'] = 'ACCELERATING'
#         elif macd_hist > 0:
#             score += 3

#         # ADX (10 pts)
#         if latest['ADX'] > 25 and latest['DI_POS'] > latest['DI_NEG']:
#             score += 10; signals['adx'] = 'STRONG_UPTREND'
#         elif latest['ADX'] > 20 and latest['DI_POS'] > latest['DI_NEG']:
#             score += 5

#         # MA Alignment (10 pts)
#         is_perfect_align = latest['Close'] > latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200']
#         is_overextended = (latest['Close'] - latest['EMA_20']) / latest['EMA_20'] > 0.08
#         if is_perfect_align and not is_overextended:
#             score += 10; signals['trend'] = 'PERFECT_ALIGN'
#         elif latest['Close'] > latest['EMA_50']:
#             score += 5; signals['trend'] = 'ABOVE_MA'

#         # === 2. PRICE ACTION & VOLATILITY (30 points) ===
#         # Bollinger Band Squeeze (10 pts)
#         bb_width = latest['BB_WIDTH']
#         is_squeeze = bb_width < df_ind['BB_WIDTH'].tail(50).quantile(0.1) # Top 10% tightest
#         if is_squeeze and latest['Close'] > latest['BB_MID']:
#             score += 10; signals['volatility'] = 'SQUEEZE_BREAKOUT_POTENTIAL'

#         # Price Structure (10 pts)
#         hh_count = (recent['High'].diff().dropna() > 0).sum()
#         hl_count = (recent['Low'].diff().dropna() > 0).sum()
#         if hh_count > 15 and hl_count > 15: # Consistently making higher highs/lows
#              score += 10; signals['structure'] = 'STRONG_HH_HL'

#         # Breakout Potential (10 pts)
#         if latest['Close'] > df_ind['High'].tail(50).max():
#             score += 10; signals['breakout'] = '50D_HIGH_BREAK'
#         elif latest['Close'] > df_ind['High'].tail(20).max():
#             score += 7; signals['breakout'] = '20D_HIGH_BREAK'

#         # === 3. VOLUME & MONEY FLOW (20 points) ===
#         # Volume Confirmation (10 pts)
#         price_change_5d = (latest['Close'] - df_ind['Close'].iloc[-5]) / df_ind['Close'].iloc[-5]
#         vol_change_5d = df_ind['Volume'].tail(5).mean() / df_ind['Volume'].tail(30).mean()
#         if price_change_5d > 0.03 and vol_change_5d > 1.5:
#             score += 10; signals['volume_trend'] = 'SURGE_CONFIRM'
#         elif price_change_5d > 0 and vol_change_5d > 1.2:
#             score += 5; signals['volume_trend'] = 'INCREASING'

#         # OBV & MFI (10 pts)
#         obv_rising = latest['OBV'] > df_ind['OBV'].tail(10).mean()
#         mfi_healthy = 40 < latest['MFI'] < 80
#         if obv_rising and mfi_healthy:
#             score += 10; signals['flow'] = 'ACCUMULATION'
#         elif obv_rising or mfi_healthy:
#             score += 5

#         # === 4. FUNDAMENTAL QUALITY (10 points) ===
#         pe, roe, pb = latest.get('PE_Ratio', 999), latest.get('ROE', 0), latest.get('PB_Ratio', 999)
#         if 0 < pe < 35: score += 4
#         if roe > 0.15: score += 4
#         if 0 < pb < 5: score += 2

#         # === FINAL CALCULATION ===
#         growth_score = max(0, min(100, score))
#         price_change_7d = ((latest['Close'] - df_ind['Close'].iloc[-7]) / df_ind['Close'].iloc[-7]) * 100
#         atr_pct = (latest['ATR'] / latest['Close']) * 100

#         # Estimate growth based on score and recent momentum
#         estimated_growth = (price_change_7d * 0.5) + (growth_score / 100 * atr_pct * 2.5)

#         return {
#             'symbol': symbol,
#             'growth_score': round(growth_score, 2),
#             'current_price': round(latest['Close'], 2),
#             'estimated_weekly_growth': round(estimated_growth, 2),
#             'rsi': round(latest['RSI'], 2),
#             'macd_signal': signals.get('macd_signal', 'NEUTRAL'),
#             'trend': signals.get('trend', 'NEUTRAL'),
#             'volume_trend': signals.get('volume_trend', 'NEUTRAL'),
#             'key_signals': ', '.join([f"{k}:{v}" for k, v in signals.items()])
#         }
#     except Exception as e:
#         # This will help debug if a specific stock fails
#         # print(f"Error analyzing {symbol}: {e}")
#         return None

# def scan_growth_stocks(stock_list, top_n=20, min_score=60, max_workers=4):
#     """
#     IMPROVED: Uses multiprocessing for true parallel CPU execution, significantly
#     speeding up the scanning process.
#     """
#     print(f"\n{'='*80}")
#     print(f"🚀 SCANNING {len(stock_list)} STOCKS FOR WEEKLY GROWTH POTENTIAL 🚀")
#     print(f"Analyzing: Technical + Momentum + Fundamentals + Patterns")
#     print(f"Using {max_workers} parallel processes for maximum speed.")
#     print(f"{'='*80}\n")

#     all_results = []

#     # Use multiprocessing.Pool for CPU-bound tasks like indicator calculation
#     with mp.Pool(processes=max_workers) as pool:
#         # imap_unordered is efficient for applying a function to a list in parallel
#         results_iterator = pool.imap_unordered(analyze_growth_potential, stock_list)

#         for result in tqdm(results_iterator, total=len(stock_list), desc="Analyzing stocks"):
#             if result:
#                 all_results.append(result)

#     if not all_results:
#         print("\n⚠️ No valid data could be processed for any stock.")
#         return pd.DataFrame()

#     df_results = pd.DataFrame(all_results)
#     df_results = df_results.sort_values('growth_score', ascending=False)

#     filtered_results = df_results[df_results['growth_score'] >= min_score]

#     # Show statistics
#     scores = df_results['growth_score'].dropna()
#     print(f"\n{'='*80}")
#     print(f"📊 SCAN STATISTICS:")
#     print(f"   Total stocks analyzed: {len(all_results)}")
#     print(f"   Stocks meeting threshold (>={min_score}): {len(filtered_results)}")
#     print(f"   Score range: {scores.min():.1f} - {scores.max():.1f}")
#     print(f"   Average score: {scores.mean():.1f}")
#     print(f"{'='*80}")

#     if filtered_results.empty:
#         print(f"\n⚠️ No stocks found with growth score >= {min_score}")
#         print(f"Showing TOP {top_n} stocks by score instead...\n")
#         display_df = df_results.head(top_n)
#     else:
#         display_df = filtered_results.head(top_n)

#     print(f"\n{'='*80}")
#     print(f"🏆 TOP {len(display_df)} HIGH-POTENTIAL GROWTH STOCKS 🏆")
#     print(f"(Minimum Score: {min_score}/100)")
#     print(f"{'='*80}\n")

#     display_cols = ['symbol', 'growth_score', 'estimated_weekly_growth',
#                     'current_price', 'rsi', 'macd_signal', 'trend', 'volume_trend']

#     print(display_df[display_cols].to_string(index=False))

#     print(f"\n{'='*80}")
#     print("SIGNAL LEGEND:")
#     print("- Growth Score: 0-100 (Higher = Better potential)")
#     print("- Est. Weekly Growth: Projected % change in next 7 days based on score & momentum")
#     print(f"{'='*80}\n")

#     return display_df

# def run(symbol="TATAPOWER", window=90, epochs=100, batch_size=32, force_download=False, n_forecast_days=30, ensemble_n=5):
#     """Full detailed analysis with optimized parameters"""
#     print(f"\n{'='*70}")
#     print(f"DEEP DIVE ANALYSIS: {symbol}")
#     print(f"{'='*70}\n")

#     df = fetch_data(symbol, force_download=force_download)
#     if df is None:
#         print(f"Error: Could not fetch data for {symbol}")
#         return None

#     (X_train, y_train, X_test, y_test, y_train_raw, y_test_raw, last_close_test, test_dates,
#      feature_scaler, target_scaler, df_ind, unique_features) = make_sequences_fast(df, window, symbol=symbol)

#     if len(y_test_raw) == 0:
#         print("Not enough test data to perform analysis.")
#         return

#     models, all_test_preds = [], []
#     for i in range(ensemble_n):
#         print(f"Training Model {i+1}/{ensemble_n}...", end=' ', flush=True)
#         model = build_model_fast((X_train.shape[1], X_train.shape[2]), lr=1e-3 * (0.8 + 0.4 * np.random.random()))
#         es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, min_lr=1e-6, verbose=0)
#         history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size,
#                   callbacks=[es, reduce_lr], verbose=0)
#         print(f"Done (epochs: {len(history.history['loss'])})")
#         models.append(model)
#         pred = target_scaler.inverse_transform(model.predict(X_test, verbose=0).reshape(-1, 1)).flatten()
#         all_test_preds.append(pred)

#     pred_price = np.mean(all_test_preds, axis=0)
#     rmse = np.sqrt(mean_squared_error(y_test_raw, pred_price))
#     mae = np.mean(np.abs(y_test_raw - pred_price))
#     mape = np.mean(np.abs((y_test_raw - pred_price) / y_test_raw)) * 100
#     corr = np.corrcoef(y_test_raw, pred_price)[0, 1]

#     print(f"\nRMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}% | Corr: {corr:.4f}")

#     signals, test_conf = generate_confident_signals(all_test_preds, last_close_test, conf_threshold=75, signal_gap=7)
#     bt = backtest_strategy(y_test_raw, signals)
#     print(f"Return: {bt['total_return']:.2f}% | B&H: {bt['buy_hold_return']:.2f}% | Excess: {bt['excess_return']:.2f}%")
#     print(f"Trades: {bt['num_trades']} | BUY: {signals.count('BUY')} | SELL: {signals.count('SELL')}\n")

#     # Forecast
#     last_seq = df_ind[unique_features].iloc[-window:].values
#     future_preds_all, forecast_sigs, forecast_conf = [], [], []

#     print("Forecasting 30 days...")
#     for day in tqdm(range(n_forecast_days), desc="Forecasting"):
#         last_seq_scaled = feature_scaler.transform(last_seq).reshape(1, window, -1)
#         step_preds = [target_scaler.inverse_transform(m.predict(last_seq_scaled, verbose=0)).flatten()[0] for m in models]
#         future_preds_all.append(step_preds)

#         curr_price = df_ind["Close"].iloc[-1] if day == 0 else np.mean(future_preds_all[day-1])
#         conf, mean_pred = calculate_confidence(step_preds, curr_price)
#         forecast_conf.append(conf)

#         if conf >= 75:
#             pct_chg = ((mean_pred - curr_price) / curr_price) * 100
#             forecast_sigs.append('BUY' if pct_chg > 4.0 else 'SELL' if pct_chg < -4.0 else 'HOLD')
#         else:
#             forecast_sigs.append('HOLD')

#         new_row = last_seq[-1].copy()
#         try:
#             new_row[unique_features.index("Close")] = mean_pred
#         except ValueError:
#             pass
#         last_seq = np.vstack([last_seq[1:], new_row])

#     future_preds_mean = [np.mean(fp) for fp in future_preds_all]
#     future_dates = pd.date_range(df.index[-1], periods=n_forecast_days+1, freq="B")[1:]
#     print(f"Forecast: BUY={forecast_sigs.count('BUY')} | SELL={forecast_sigs.count('SELL')} | Avg Conf: {np.mean(forecast_conf):.1f}%\n")

#     # Plot
#     df_plot = df[df.index >= (df.index[-1] - pd.Timedelta(days=730))].copy()
#     fig = go.Figure()
#     fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot["Open"], high=df_plot["High"],
#         low=df_plot["Low"], close=df_plot["Close"], name="Price", increasing_line_color="green",
#         decreasing_line_color="red"))
#     fig.add_trace(go.Scatter(x=test_dates, y=pred_price, mode="lines", name="Predicted",
#         line=dict(color="cyan", width=2)))

#     buy_idx = [i for i, s in enumerate(signals) if s == 'BUY']
#     sell_idx = [i for i, s in enumerate(signals) if s == 'SELL']
#     if buy_idx:
#         fig.add_trace(go.Scatter(x=test_dates[buy_idx], y=pred_price[buy_idx], mode='markers',
#             name='BUY (≥75%)', marker=dict(symbol='triangle-up', size=16, color='lime', line=dict(width=2, color='darkgreen'))))
#     if sell_idx:
#         fig.add_trace(go.Scatter(x=test_dates[sell_idx], y=pred_price[sell_idx], mode='markers',
#             name='SELL (≥75%)', marker=dict(symbol='triangle-down', size=16, color='red', line=dict(width=2, color='darkred'))))

#     fig.add_trace(go.Scatter(x=future_dates, y=future_preds_mean, mode="lines", name="Forecast",
#         line=dict(color="gold", dash="dash", width=3)))

#     fore_buy = [i for i, s in enumerate(forecast_sigs) if s == 'BUY']
#     fore_sell = [i for i, s in enumerate(forecast_sigs) if s == 'SELL']
#     if fore_buy:
#         fig.add_trace(go.Scatter(x=future_dates[fore_buy], y=np.array(future_preds_mean)[fore_buy],
#             mode='markers', name='Forecast BUY (≥75%)', marker=dict(symbol='star', size=18, color='lightgreen',
#             line=dict(width=2, color='green'))))
#     if fore_sell:
#         fig.add_trace(go.Scatter(x=future_dates[fore_sell], y=np.array(future_preds_mean)[fore_sell],
#             mode='markers', name='Forecast SELL (≥75%)', marker=dict(symbol='star', size=18, color='orange',
#             line=dict(width=2, color='darkred'))))

#     fig.update_layout(title=f"{symbol} | Predictions + Forecast (Confidence ≥ 75%)", xaxis_title="Date",
#         yaxis_title="Price", template="plotly_dark", height=700, width=1400,
#         xaxis=dict(rangeslider=dict(visible=True), showgrid=True), yaxis=dict(showgrid=True))
#     fig.show()

#     return {'rmse': rmse, 'mae': mae, 'mape': mape, 'corr': corr, 'backtest': bt,
#             'test_signals': {'BUY': signals.count('BUY'), 'SELL': signals.count('SELL')},
#             'forecast_signals': {'BUY': forecast_sigs.count('BUY'), 'SELL': forecast_sigs.count('SELL')}}

# if __name__ == "__main__":
#     mp.freeze_support() # Recommended for multiprocessing in packaged apps

#     print("\n" + "="*70)
#     print("🚀 ADVANCED STOCK ANALYSIS & SCANNING SYSTEM 🚀")
#     print("="*70)

#     choice = input("\nChoose mode:\n1. [RECOMMENDED] GROWTH POTENTIAL SCANNER (Nifty 500)\n2. Analyze a specific stock\nEnter choice (1/2): ").strip()

#     if choice == '1':
#         print("\n" + "="*70)
#         print("GROWTH POTENTIAL SCANNER")
#         print("Multi-factor analysis: Technical + Momentum + Fundamentals + Patterns")
#         print("="*70)

#         num_stocks = input("\nEnter number of top stocks to display (default 20): ").strip()
#         top_n = int(num_stocks) if num_stocks.isdigit() else 20

#         min_score = input("Enter minimum growth score threshold (0-100, default 65): ").strip()
#         min_score = int(min_score) if min_score.isdigit() else 65

#         # Automatically suggest workers based on CPU count, but allow override
#         default_workers = min(max(mp.cpu_count() - 1, 1), 8)
#         workers = input(f"Enter number of parallel workers (recommended {default_workers}, max 12): ").strip()
#         max_workers = min(int(workers) if workers.isdigit() else default_workers, 12)

#         # Get the latest Nifty 500 stock list automatically
#         nifty500_list = get_nifty500_symbols()
#         top_stocks = scan_growth_stocks(stock_list=nifty500_list, top_n=top_n, min_score=min_score, max_workers=max_workers)

#         if not top_stocks.empty:
#             print("\n" + "="*70)
#             print("DETAILED SIGNAL BREAKDOWN FOR TOP 5 STOCKS:")
#             print("="*70)
#             for _, row in top_stocks.head(5).iterrows():
#                 print(f"\n{row['symbol']} (Score: {row['growth_score']})")
#                 print(f"  Current Price: ₹{row['current_price']:.2f} | Est. Weekly Growth: {row['estimated_weekly_growth']:.2f}%")
#                 print(f"  Key Signals: {row['key_signals']}")

#             analyze_choice = input("\n\nDo you want to run a detailed ML analysis on any of these stocks? (y/n): ").strip().lower()
#             if analyze_choice == 'y':
#                 symbol = input("\nEnter stock symbol from the list above: ").strip().upper()
#                 if symbol:
#                     res = run(symbol=symbol, force_download=True) # Force fresh download for detailed analysis
#                     if res:
#                          print(f"\n{'='*70}\nFINAL SUMMARY - {symbol}\n{'='*70}")
#                          print(f"RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f} | Corr: {res['corr']:.4f}")
#                          print(f"Strategy Return: {res['backtest']['total_return']:.2f}% | Buy&Hold: {res['backtest']['buy_hold_return']:.2f}% | Excess: {res['backtest']['excess_return']:.2f}%")
#                          print(f"{'='*70}")
#     else:
#         symbol = input("Enter stock symbol (e.g., TATAPOWER, RELIANCE, INFY): ").strip().upper() or "TATAPOWER"
#         res = run(symbol=symbol)
#         if res:
#             print(f"\n{'='*70}\nFINAL SUMMARY - {symbol}\n{'='*70}")
#             print(f"RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f} | Corr: {res['corr']:.4f}")
#             print(f"Strategy Return: {res['backtest']['total_return']:.2f}% | Buy&Hold: {res['backtest']['buy_hold_return']:.2f}% | Excess: {res['backtest']['excess_return']:.2f}%")
#             print(f"{'='*70}")

















# import os
# import warnings
# import numpy as np
# import pandas as pd
# import yfinance as yf
# import ta
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')  # Non-GUI backend
# from io import BytesIO
# import requests
# import multiprocessing as mp
# from tqdm import tqdm
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.utils import class_weight
# import tensorflow as tf
# from tensorflow.keras import Model, Input
# from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
#                                      BatchNormalization, Add, Activation, GlobalAveragePooling2D, Concatenate,
#                                      Conv1D, Bidirectional, LSTM, GRU, LayerNormalization, MultiHeadAttention,
#                                      GlobalAveragePooling1D)
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# import pickle

# warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CACHE_DIR = "cache_data"
# CHART_CACHE_DIR = "chart_cache"
# PATTERN_MODEL_DIR = "pattern_models"
# os.makedirs(CACHE_DIR, exist_ok=True)
# os.makedirs(CHART_CACHE_DIR, exist_ok=True)
# os.makedirs(PATTERN_MODEL_DIR, exist_ok=True)

# # ============================================================================
# # OPTIMIZED SEQUENTIAL PATTERN LABELS (BASED ON OUTCOME)
# # ============================================================================
# # Labels now represent the future price outcome, which is a clearer signal for the model.
# OUTCOME_PATTERN_NAMES = [
#     'STRONG_UP',          # Significant upward movement, low volatility
#     'STRONG_DOWN',        # Significant downward movement, low volatility
#     'SIDEWAYS',           # Low volatility, minimal net movement
#     'VOLATILE_UP',        # High volatility with upward bias
#     'VOLATILE_DOWN',      # High volatility with downward bias
#     'NEUTRAL'             # No clear pattern or insignificant movement
# ]

# # Create mappings for the new outcome-based labels
# OUTCOME_SEQUENTIAL_LABELS = {i: name for i, name in enumerate(OUTCOME_PATTERN_NAMES)}
# OUTCOME_NAME_TO_SEQ_LABEL = {name: i for i, name in enumerate(OUTCOME_PATTERN_NAMES)}

# # PATTERN_WEIGHTS now reflect the new outcome-based labels.
# # Positive values for upward movement, negative for downward.
# PATTERN_WEIGHTS = {
#     'STRONG_UP': 0.9,
#     'VOLATILE_UP': 0.65,
#     'STRONG_DOWN': -0.9,
#     'VOLATILE_DOWN': -0.65,
#     'SIDEWAYS': 0.0,
#     'NEUTRAL': 0.0
# }
# # ============================================================================

# # Global variable and initializer for multiprocessing
# _GLOBAL_PATTERN_MODEL = None

# def init_worker(model_path):
#     """Initializer for multiprocessing pool to load the model once per worker."""
#     global _GLOBAL_PATTERN_MODEL
#     if os.path.exists(model_path):
#         _GLOBAL_PATTERN_MODEL = tf.keras.models.load_model(model_path, compile=False)
#         # Re-compile if needed, especially if custom metrics/losses are used
#         _GLOBAL_PATTERN_MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     else:
#         _GLOBAL_PATTERN_MODEL = None

# def get_global_pattern_model():
#     """Get the global pattern model from the worker's context."""
#     return _GLOBAL_PATTERN_MODEL

# features = ["Close", "RSI", "MACD", "MACD_signal", "BB_MID", "BB_UP", "BB_LOW",
#             "ADX", "DI_POS", "DI_NEG", "OBV", "SMA_20", "EMA_20", "EMA_50", "EMA_200",
#             "ATR", "stoch_k", "stoch_d", "CCI", "PE_Ratio", "PB_Ratio", "Market_Cap",
#             "Dividend_Yield", "EPS", "ROE", "ROCE", "VWAP", "MFI", "Williams_R", "TSI",
#             "Aroon_Up", "Aroon_Down", "Volume_SMA", "Price_ROC", "Volume_ROC", "BB_WIDTH",
#             "Keltner_Up", "Keltner_Low", "Ichimoku_A", "Ichimoku_B", "High_Low_Ratio",
#             "Close_Open_Ratio", "Intraday_Range", "Trend_Strength", "Historical_Volatility"]

# df = pd.read_csv('C:\\Users\\Lenovo\\Documents\\VS CODE codes(files)\\helloworld\\STOCK_MARKET_PROJECT\\Ticker_List_NSE_India.csv')
# FALLBACK_NIFTY_500_STOCKS = df['SYMBOL'].tolist()

# # ============================================================================
# # OPTIMIZED CHART PATTERN CNN MODEL (ResNet-inspired)
# # ============================================================================

# def resnet_block(x, filters, kernel_size=3, stride=1):
#     """A ResNet block with batch normalization and skip connection."""
#     shortcut = x
#     # First conv layer
#     y = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#     # Second conv layer
#     y = Conv2D(filters, kernel_size, padding='same')(y)
#     y = BatchNormalization()(y)

#     # Adjust shortcut if dimensions change (due to strides or filter changes)
#     if stride != 1 or x.shape[-1] != filters:
#         shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
#         shortcut = BatchNormalization()(shortcut)

#     # Add shortcut to the main path
#     y = Add()([y, shortcut])
#     y = Activation('relu')(y)
#     return y

# def build_optimized_cnn(input_shape=(128, 128, 3), num_classes=len(OUTCOME_SEQUENTIAL_LABELS)):
#     """
#     Build an optimized CNN model for chart pattern recognition using ResNet-inspired blocks.
#     """
#     inp = Input(shape=input_shape)

#     # Initial Convolutional Layer
#     x = Conv2D(64, 7, strides=2, padding='same')(inp)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D(3, strides=2, padding='same')(x)

#     # ResNet blocks
#     x = resnet_block(x, 64)
#     x = resnet_block(x, 64)
#     x = resnet_block(x, 128, stride=2) # Downsample
#     x = resnet_block(x, 128)
#     x = resnet_block(x, 256, stride=2) # Downsample
#     x = resnet_block(x, 256)

#     # Final Layers
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     out = Dense(num_classes, activation='softmax')(x)

#     model = Model(inp, out)
#     return model


# def generate_chart_image_with_features(df, lookback=60, img_size=(128, 128)):
#     """
#     OPTIMIZED: Generate a multi-channel chart image encoding indicators.
#     - Channel R: Candlesticks
#     - Channel G: EMA(20) and EMA(50)
#     - Channel B: Bollinger Bands
#     """
#     if df is None or len(df) < lookback:
#         return None

#     df_chart = df.tail(lookback).copy()
#     if df_chart.empty:
#         return None

#     # Normalize all data for plotting
#     min_val = df_chart['Low'].min()
#     max_val = df_chart['High'].max()
#     range_val = max_val - min_val
#     if range_val == 0:
#         return None

#     def normalize(series):
#         return (series - min_val) / range_val

#     df_norm = df_chart[['Open', 'High', 'Low', 'Close']].apply(normalize)
#     df_norm['EMA20'] = normalize(df_chart['EMA_20'])
#     df_norm['EMA50'] = normalize(df_chart['EMA_50'])
#     df_norm['BB_UP'] = normalize(df_chart['BB_UP'])
#     df_norm['BB_LOW'] = normalize(df_chart['BB_LOW'])

#     # Create an empty 3-channel image array
#     img_array = np.zeros((img_size[1], img_size[0], 3), dtype=np.float32)

#     # Create a Matplotlib figure in memory
#     fig, ax = plt.subplots(figsize=(img_size[0]/32, img_size[1]/32), dpi=32)
#     ax.axis('off')
#     fig.patch.set_visible(False)
#     ax.set_ylim(0, 1)
#     ax.set_xlim(-1, len(df_chart))

#     # --- Render Channel R: Candlesticks ---
#     for i in range(len(df_norm)):
#         row = df_norm.iloc[i]
#         color_val = 1.0 # White for candlesticks
#         ax.plot([i, i], [row['Low'], row['High']], color=(color_val, 0, 0), linewidth=0.5)
#         body_bottom = min(row['Open'], row['Close'])
#         body_height = abs(row['Close'] - row['Open'])
#         ax.add_patch(plt.Rectangle((i - 0.3, body_bottom), 0.6, body_height, facecolor=(color_val, 0, 0)))

#     # --- Render Channel G: EMAs ---
#     ax.plot(range(len(df_norm)), df_norm['EMA20'], color=(0, 1.0, 0), linewidth=1)
#     ax.plot(range(len(df_norm)), df_norm['EMA50'], color=(0, 0.7, 0), linewidth=1)

#     # --- Render Channel B: Bollinger Bands ---
#     ax.plot(range(len(df_norm)), df_norm['BB_UP'], color=(0, 0, 1.0), linewidth=0.8, linestyle='--')
#     ax.plot(range(len(df_norm)), df_norm['BB_LOW'], color=(0, 0, 1.0), linewidth=0.8, linestyle='--')
#     ax.fill_between(range(len(df_norm)), df_norm['BB_LOW'], df_norm['BB_UP'], color=(0, 0, 1.0), alpha=0.1)

#     plt.tight_layout(pad=0)

#     # Convert plot to numpy array
#     buf = BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
#     buf.seek(0)
#     plt.close(fig)

#     from PIL import Image
#     img = Image.open(buf).resize(img_size)
#     final_img_array = np.array(img)[:, :, :3] / 255.0  # Keep only RGB and normalize

#     return final_img_array


# def label_pattern_from_outcome(df_window, lookback=60, outcome_period=20):
#     """
#     OPTIMIZED: Analyzes the price action *after* the lookback period to assign a clear,
#     outcome-based label. This creates a much higher quality ground truth.
#     """
#     if len(df_window) < lookback + outcome_period:
#         return OUTCOME_NAME_TO_SEQ_LABEL['NEUTRAL']

#     chart_period = df_window.iloc[:lookback]
#     outcome_data = df_window.iloc[lookback:lookback + outcome_period]

#     start_price = outcome_data['Close'].iloc[0]
#     max_price = outcome_data['High'].max()
#     min_price = outcome_data['Low'].min()
#     end_price = outcome_data['Close'].iloc[-1]
#     volatility = outcome_data['Close'].pct_change().std()

#     # Calculate max gain and loss percentages
#     max_gain = (max_price - start_price) / start_price * 100
#     max_loss = (min_price - start_price) / start_price * 100
#     net_change = (end_price - start_price) / start_price * 100

#     vol_threshold_high = 0.025
#     gain_threshold = 7.0
#     loss_threshold = -7.0

#     # Decision logic based on outcome
#     is_volatile = volatility > vol_threshold_high

#     if not is_volatile:
#         if net_change > gain_threshold:
#             return OUTCOME_NAME_TO_SEQ_LABEL['STRONG_UP']
#         elif net_change < loss_threshold:
#             return OUTCOME_NAME_TO_SEQ_LABEL['STRONG_DOWN']
#         else:
#             return OUTCOME_NAME_TO_SEQ_LABEL['SIDEWAYS']
#     else: # Volatile conditions
#         if net_change > gain_threshold / 2: # Positive bias
#             return OUTCOME_NAME_TO_SEQ_LABEL['VOLATILE_UP']
#         elif net_change < loss_threshold / 2: # Negative bias
#             return OUTCOME_NAME_TO_SEQ_LABEL['VOLATILE_DOWN']
#         else:
#             return OUTCOME_NAME_TO_SEQ_LABEL['NEUTRAL']

# def create_pattern_dataset_from_stocks(stock_list, samples_per_stock=10, lookback=60):
#     """
#     Create a labeled dataset by analyzing historical data from multiple stocks.
#     Uses the optimized image generation and labeling functions.
#     """
#     print(f"\n{'='*70}")
#     print("CREATING OPTIMIZED CHART PATTERN DATASET")
#     print(f"{'='*70}\n")

#     X_images = []
#     y_labels = []

#     for symbol in tqdm(stock_list[:150], desc="Processing stocks"): # Limit to first 150 for speed
#         try:
#             df_full = fetch_data(symbol, period="5y", force_download=False)
#             if df_full is None or len(df_full) < lookback + 50:
#                 continue

#             # Pre-calculate indicators once for the entire dataframe
#             df_full = compute_indicators_fast(df_full, symbol)
#             if df_full.empty:
#                 continue

#             max_start = len(df_full) - lookback - 20 # 20 is outcome period
#             if max_start < samples_per_stock:
#                 continue

#             sample_indices = np.random.choice(max_start,
#                                               min(samples_per_stock, max_start),
#                                               replace=False)

#             for idx in sample_indices:
#                 window_df = df_full.iloc[idx : idx + lookback + 20]

#                 # Generate feature-rich chart image from the lookback period
#                 img = generate_chart_image_with_features(window_df.iloc[:lookback], lookback=lookback)
#                 if img is None:
#                     continue

#                 # Auto-label based on price action after the lookback period
#                 label = label_pattern_from_outcome(window_df, lookback=lookback)

#                 X_images.append(img)
#                 y_labels.append(label)

#         except Exception as e:
#             continue

#     if len(X_images) == 0:
#         print("⚠️ No valid samples generated!")
#         return None, None

#     X = np.array(X_images)
#     y = np.array(y_labels)

#     print(f"\n✅ Dataset created: {len(X)} samples")
#     print(f"Outcome distribution:")
#     unique, counts = np.unique(y, return_counts=True)
#     for pattern_id, count in zip(unique, counts):
#         if pattern_id in OUTCOME_SEQUENTIAL_LABELS:
#             print(f"  {OUTCOME_SEQUENTIAL_LABELS[pattern_id]}: {count}")

#     return X, y

# def train_pattern_recognition_model(X, y, model_path=None, epochs=75, batch_size=32):
#     """
#     Train the OPTIMIZED CNN model for pattern recognition.
#     """
#     if X is None or y is None or len(X) == 0:
#         print("⚠️ No data available for training!")
#         return None

#     print(f"\n{'='*70}")
#     print("TRAINING OPTIMIZED PATTERN RECOGNITION MODEL (ResNet-style)")
#     print(f"{'='*70}\n")

#     # Split data
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
#                                                       random_state=42, stratify=y)

#     print(f"Training samples: {len(X_train)}")
#     print(f"Validation samples: {len(X_val)}")

#     # Calculate class weights to handle data imbalance
#     class_weights = class_weight.compute_class_weight(
#         'balanced',
#         classes=np.unique(y_train),
#         y=y_train
#     )
#     class_weights_dict = dict(enumerate(class_weights))
#     print("\nApplying class weights to handle data imbalance...")

#     # Data augmentation
#     datagen = ImageDataGenerator(
#         rotation_range=8,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         zoom_range=0.1,
#         brightness_range=[0.9, 1.1], # Add brightness jitter
#         horizontal_flip=False # Don't flip financial charts
#     )
#     datagen.fit(X_train)

#     # Build optimized model
#     model = build_optimized_cnn(input_shape=X_train.shape[1:],
#                                   num_classes=len(OUTCOME_SEQUENTIAL_LABELS))

#     # OPTIMIZED: Learning Rate Schedule
#     initial_learning_rate = 0.001
#     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate,
#         decay_steps=len(X_train) // batch_size * 10, # Decay every 10 epochs
#         decay_rate=0.7,
#         staircase=True)

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     # Callbacks
#     callbacks = [
#         EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
#         # The ExponentialDecay schedule is already handling the learning rate, so ReduceLROnPlateau is not needed.
#         # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7),
#     ]
    
#     if model_path:
#         callbacks.append(ModelCheckpoint(model_path, monitor='val_accuracy',
#                                          save_best_only=True, verbose=1))

#     # Train the model
#     history = model.fit(
#         datagen.flow(X_train, y_train, batch_size=batch_size),
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         callbacks=callbacks,
#         verbose=1,
#         class_weight=class_weights_dict
#     )

#     # Evaluate
#     y_pred = np.argmax(model.predict(X_val), axis=1)
#     print("\n" + "="*70)
#     print("VALIDATION RESULTS")
#     print("="*70)
#     print("\nClassification Report:")

#     report_labels = np.union1d(np.unique(y_val), np.unique(y_pred))
#     target_names = [OUTCOME_SEQUENTIAL_LABELS[i] for i in report_labels if i in OUTCOME_SEQUENTIAL_LABELS]

#     print(classification_report(y_val, y_pred, labels=report_labels,
#                                 target_names=target_names, zero_division=0))

#     return model


# def predict_chart_pattern(model, df, lookback=60):
#     """
#     Predict the chart pattern from recent price data using the optimized model and image generator.
#     """
#     if model is None or df is None or len(df) < lookback:
#         return 'NEUTRAL', 0.0

#     # We need indicators for the feature-rich image, so we compute them
#     df_with_ind = compute_indicators_fast(df, symbol="TEMP")
#     if df_with_ind.empty:
#         return 'NEUTRAL', 0.0

#     img = generate_chart_image_with_features(df_with_ind, lookback=lookback)
#     if img is None:
#         return 'NEUTRAL', 0.0

#     img_batch = np.expand_dims(img, axis=0)
#     predictions = model.predict(img_batch, verbose=0)[0]

#     pattern_id = np.argmax(predictions)
#     confidence = predictions[pattern_id] * 100

#     return OUTCOME_SEQUENTIAL_LABELS.get(pattern_id, 'UNKNOWN'), confidence


# # ============================================================================
# # ORIGINAL FUNCTIONS (Unchanged unless necessary for integration)
# # ============================================================================

# def get_nifty500_symbols():
#     driver = None
#     try:
#         print("Initializing a headless browser with webdriver-manager Chromedriver...")
#         chrome_options = Options()
#         chrome_options.add_argument("--headless=new")
#         chrome_options.add_argument("--disable-gpu")
#         chrome_options.add_argument("--no-sandbox")
#         chrome_options.add_argument("--disable-dev-shm-usage")
#         chrome_options.add_argument("--window-size=1920,1080")
#         chrome_options.add_argument("--log-level=3")

#         prefs = {
#             "profile.managed_default_content_settings.images": 2,
#             "profile.managed_default_content_settings.stylesheets": 2,
#         }
#         chrome_options.add_experimental_option("prefs", prefs)

#         driver = webdriver.Chrome(
#             service=Service(ChromeDriverManager().install()),
#             options=chrome_options
#         )

#         url = "https://en.wikipedia.org/wiki/NIFTY_500"
#         driver.get(url)

#         WebDriverWait(driver, 15).until(
#             EC.presence_of_element_located((By.ID, "constituents"))
#         )

#         html_source = driver.page_source
#         tables = pd.read_html(html_source)
#         nifty500_df = None
#         symbol_col_name = None
#         possible_colnames = ['Symbol', 'Ticker']

#         for table in tables:
#             table.columns = [str(col).strip() for col in table.columns]
#             for col_name in possible_colnames:
#                 if col_name in table.columns:
#                     nifty500_df = table
#                     symbol_col_name = col_name
#                     break
#             if nifty500_df is not None:
#                 break

#         if nifty500_df is None:
#             raise ValueError("Selenium loaded the page, but still could not find the table.")

#         symbols = nifty500_df[symbol_col_name].str.strip().tolist()
#         print(f"✅ Successfully fetched {len(symbols)} symbols using webdriver-manager.")
#         return symbols

#     except Exception as e:
#         print(f"⚠️ Failed to fetch NIFTY 500 symbols via Selenium: {e}")
#         print("Falling back to the hardcoded list.")
#         return FALLBACK_NIFTY_500_STOCKS

#     finally:
#         if driver:
#             driver.quit()


# def fetch_data(symbol="TATAPOWER", period="5y", interval="1d", force_download=False):
#     yf_symbol = f"{symbol}.NS" if "." not in symbol else symbol
#     cache_path = os.path.join(CACHE_DIR, f"{yf_symbol.replace('.', '_')}_{period}_{interval}.csv")

#     if os.path.exists(cache_path) and not force_download:
#         try:
#             df = pd.read_csv(cache_path, index_col=0)
#             df.index = pd.to_datetime(df.index, errors="coerce")
#             df = df[~df.index.isna()]
#             if not df.empty and all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]):
#                 return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
#         except Exception:
#             pass

#     try:
#         ticker = yf.Ticker(yf_symbol)
#         df = ticker.history(period=period, interval=interval, auto_adjust=False)

#         if df is None or df.empty:
#             df = yf.download(yf_symbol, period=period, interval=interval, progress=False, timeout=10)

#         if df is None or df.empty:
#             return None

#         required_cols = ["Open", "High", "Low", "Close", "Volume"]
#         if not all(col in df.columns for col in required_cols):
#             return None

#         df.to_csv(cache_path)
#         return df.dropna(subset=required_cols)
#     except Exception:
#         return None


# def fetch_fundamental_data(symbol="TATAPOWER"):
#     try:
#         ticker = yf.Ticker(f"{symbol}.NS" if "." not in symbol else symbol)
#         info = ticker.info
#         return pd.DataFrame({
#             "PE_Ratio": [info.get("trailingPE", 0.0) or 0.0],
#             "EPS": [info.get("trailingEps", 0.0) or 0.0],
#             "PB_Ratio": [info.get("priceToBook", 0.0) or 0.0],
#             "Market_Cap": [info.get("marketCap", 0.0) or 0.0],
#             "Dividend_Yield": [info.get("dividendYield", 0.0) or 0.0],
#             "ROE": [info.get("returnOnEquity", 0.0) or 0.0],
#             "ROCE": [info.get("returnOnCapitalEmployed", 0.0) or 0.0]
#         })
#     except Exception:
#         return pd.DataFrame()


# def compute_indicators_fast(df, symbol="TATAPOWER"):
#     if df is None or df.empty:
#         return pd.DataFrame()
#     df = df.copy()
#     for col in ["Close", "High", "Low", "Volume", "Open"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#     df = df.dropna(subset=["Close", "High", "Low", "Volume", "Open"])
#     if df.empty:
#         return df

#     close, high, low, vol, open_price = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]

#     df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
#     macd = ta.trend.MACD(close)
#     df["MACD"], df["MACD_signal"] = macd.macd(), macd.macd_signal()
#     bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
#     df["BB_MID"], df["BB_UP"], df["BB_LOW"] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
#     adx = ta.trend.ADXIndicator(high, low, close, window=14)
#     df["ADX"], df["DI_POS"], df["DI_NEG"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
#     df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
#     df["SMA_20"] = close.rolling(20).mean()
#     df["EMA_20"], df["EMA_50"], df["EMA_200"] = close.ewm(span=20).mean(), close.ewm(span=50).mean(), close.ewm(span=200).mean()
#     df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
#     stoch = ta.momentum.StochasticOscillator(high, low, close)
#     df["stoch_k"], df["stoch_d"] = stoch.stoch(), stoch.stoch_signal()
#     df["CCI"] = ta.trend.CCIIndicator(high, low, close).cci()
#     df["VWAP"] = (vol * (high + low + close) / 3).cumsum() / vol.cumsum()
#     df["MFI"] = ta.volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()
#     df["Williams_R"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
#     df["TSI"] = ta.momentum.TSIIndicator(close).tsi()
#     aroon = ta.trend.AroonIndicator(high=high, low=low, window=25)
#     df["Aroon_Up"], df["Aroon_Down"] = aroon.aroon_up(), aroon.aroon_down()
#     df["Volume_SMA"] = vol.rolling(20).mean()
#     df["Price_ROC"] = ta.momentum.ROCIndicator(close, window=12).roc()
#     df["Volume_ROC"] = vol.pct_change(periods=12) * 100
#     df["BB_WIDTH"] = (df["BB_UP"] - df["BB_LOW"]) / df["BB_MID"]
#     keltner = ta.volatility.KeltnerChannel(high, low, close, window=20)
#     df["Keltner_Up"], df["Keltner_Low"] = keltner.keltner_channel_hband(), keltner.keltner_channel_lband()
#     ichimoku = ta.trend.IchimokuIndicator(high, low)
#     df["Ichimoku_A"], df["Ichimoku_B"] = ichimoku.ichimoku_a(), ichimoku.ichimoku_b()
#     df["High_Low_Ratio"] = np.where(low > 0, high / low, 1.0)
#     df["Close_Open_Ratio"] = np.where(open_price > 0, close / open_price, 1.0)
#     df["Intraday_Range"] = np.where(open_price > 0, (high - low) / open_price, 0.0)
#     rolling_mean, rolling_std = close.rolling(50).mean(), close.rolling(50).std()
#     df["Trend_Strength"] = np.where(rolling_std > 0, (close - rolling_mean) / rolling_std, 0.0)
#     df["Historical_Volatility"] = close.pct_change().rolling(20).std() * np.sqrt(252)

#     fund_df = fetch_fundamental_data(symbol)
#     if not fund_df.empty:
#         for key, value in fund_df.iloc[0].to_dict().items():
#             df[key] = value

#     df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').dropna()
#     for col in df.columns:
#         if col not in ['Date'] and pd.api.types.is_numeric_dtype(df[col]):
#             col_mean, col_std = df[col].mean(), df[col].std()
#             if col_std > 0:
#                 df[col] = df[col].clip(lower=col_mean - 5*col_std, upper=col_mean + 5*col_std)
#     return df


# def make_sequences_fast(df, window=90, train_ratio=0.8, symbol="TATAPOWER"):
#     df_ind = compute_indicators_fast(df, symbol=symbol)
#     unique_features = [f for f in features if f in df_ind.columns]
#     df_feat = df_ind[unique_features].values
#     n_features = len(unique_features)

#     n_samples = len(df_feat)
#     X_raw = np.zeros((n_samples - 1, window, n_features), dtype=np.float32)

#     for i in range(1, n_samples):
#         seq_len = min(i, window)
#         X_raw[i-1, -seq_len:] = df_feat[i-seq_len:i]

#     y_raw = df_feat[1:, unique_features.index("Close")].reshape(-1, 1)
#     last_close_raw = df_feat[:-1, unique_features.index("Close")]
#     dates = df_ind.index[1:].values

#     split = int(len(X_raw) * train_ratio)
#     X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
#     y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

#     feature_scaler = RobustScaler()
#     X_train = feature_scaler.fit_transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
#     X_test = feature_scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)

#     target_scaler = RobustScaler()
#     y_train = target_scaler.fit_transform(y_train_raw).flatten()
#     y_test = target_scaler.transform(y_test_raw).flatten()

#     return (X_train, y_train, X_test, y_test, y_train_raw.flatten(), y_test_raw.flatten(),
#             last_close_raw[split:], dates[split:], feature_scaler, target_scaler, df_ind, unique_features)


# def build_model_fast(input_shape, lr=1e-3):
#     inp = Input(shape=input_shape)

#     conv1 = Conv1D(32, 3, padding="causal", activation="relu")(inp)
#     conv2 = Conv1D(32, 5, padding="causal", activation="relu")(inp)
#     x = Concatenate()([conv1, conv2])
#     x = Dropout(0.2)(x)

#     x = Bidirectional(LSTM(64, return_sequences=True))(x)
#     x = LayerNormalization()(x)
#     x = Dropout(0.2)(x)

#     x = Bidirectional(GRU(32, return_sequences=True))(x)
#     x = LayerNormalization()(x)
#     x = Dropout(0.2)(x)

#     att = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
#     x = Add()([x, att])
#     x = LayerNormalization()(x)

#     avg_pool = GlobalAveragePooling1D()(x)
#     max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
#     x = Concatenate()([avg_pool, max_pool])

#     x = Dense(128, activation="relu")(x)
#     x = Dropout(0.3)(x)
#     x = Dense(64, activation="relu")(x)
#     x = Dropout(0.2)(x)
#     out = Dense(1)(x)

#     model = Model(inp, out)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="huber", metrics=["mae"])
#     return model


# def calculate_confidence(ensemble_preds, current_price):
#     mean_pred = np.mean(ensemble_preds)
#     std_pred = np.std(ensemble_preds)
#     cv = (std_pred / mean_pred * 100) if mean_pred != 0 else 100
#     pct_change = abs((mean_pred - current_price) / current_price * 100) if current_price != 0 else 100
#     confidence = 100 - min(cv * 5, 50) - min(pct_change * 2, 30)
#     return max(0, min(100, confidence)), mean_pred


# def generate_confident_signals(all_preds, actual_prices, conf_threshold=75, signal_gap=7):
#     signals, confidences = [], []
#     last_signal_idx = -signal_gap

#     for i in range(len(all_preds[0])):
#         ensemble_preds = [pred[i] for pred in all_preds]
#         confidence, mean_pred = calculate_confidence(ensemble_preds, actual_prices[i])
#         confidences.append(confidence)

#         if confidence >= conf_threshold and (i - last_signal_idx >= signal_gap):
#             pct_change = ((mean_pred - actual_prices[i]) / actual_prices[i]) * 100 if actual_prices[i] != 0 else 0
#             if pct_change > 4.0:
#                 signals.append('BUY')
#                 last_signal_idx = i
#             elif pct_change < -4.0:
#                 signals.append('SELL')
#                 last_signal_idx = i
#             else:
#                 signals.append('HOLD')
#         else:
#             signals.append('HOLD')

#     return signals, confidences


# def backtest_strategy(actual_prices, signals, initial_capital=100000):
#     capital, position, trades, portfolio_values = initial_capital, 0, [], [initial_capital]

#     for i, signal in enumerate(signals):
#         price = actual_prices[i]
#         if signal == 'BUY' and position == 0:
#             shares = int((capital * 0.95) / price)
#             if shares > 0:
#                 cost = shares * price * 1.001
#                 if cost <= capital:
#                     position, capital = shares, capital - cost
#                     trades.append({'type': 'BUY', 'price': price, 'index': i})
#         elif signal == 'SELL' and position > 0:
#             proceeds = position * price * 0.999
#             capital += proceeds
#             trades.append({'type': 'SELL', 'price': price, 'index': i})
#             position = 0
#         portfolio_values.append(capital + (position * price if position > 0 else 0))

#     if position > 0:
#         capital += position * actual_prices[-1] * 0.999

#     buy_hold = initial_capital
#     if len(actual_prices) > 0 and actual_prices[0] > 0:
#         shares_bh = int((initial_capital * 0.999) / actual_prices[0])
#         buy_hold = shares_bh * actual_prices[-1] * 0.999

#     return {
#         'final_value': capital,
#         'total_return': (capital - initial_capital) / initial_capital * 100,
#         'buy_hold_return': (buy_hold - initial_capital) / initial_capital * 100,
#         'excess_return': ((capital - initial_capital) / initial_capital - (buy_hold - initial_capital) / initial_capital) * 100,
#         'num_trades': len(trades),
#         'portfolio_values': portfolio_values
#     }


# def analyze_growth_potential(symbol):
#     """
#     ENHANCED: Uses CNN pattern recognition from the global worker model.
#     """
#     pattern_model = get_global_pattern_model()

#     try:
#         df = fetch_data(symbol, period="2y", force_download=False)
#         if df is None or len(df) < 200:
#             return None

#         # Pass symbol to compute_indicators for fundamental data fetching
#         df_ind = compute_indicators_fast(df, symbol=symbol)
#         if df_ind.empty or len(df_ind) < 50:
#             return None

#         latest = df_ind.iloc[-1]
#         prev = df_ind.iloc[-2]
#         recent = df_ind.tail(30)

#         score = 0
#         signals = {}

#         # === 1. MOMENTUM & TREND (35 points) ===
#         if 45 < latest['RSI'] < 65 and latest['RSI'] > df_ind['RSI'].tail(5).mean():
#             score += 9; signals['rsi'] = 'BUILDING_MOMENTUM'
#         elif latest['RSI'] > 65:
#             score += 5; signals['rsi'] = 'STRONG'
#         elif latest['RSI'] < 30:
#             score -= 5; signals['rsi'] = 'OVERSOLD'

#         macd_hist = latest['MACD'] - latest['MACD_signal']
#         prev_macd_hist = prev['MACD'] - prev['MACD_signal']
#         is_fresh_cross = macd_hist > 0 and prev_macd_hist <= 0
#         if is_fresh_cross and macd_hist > prev_macd_hist:
#             score += 9; signals['macd_signal'] = 'FRESH_BULLISH_CROSS'
#         elif macd_hist > 0 and macd_hist > prev_macd_hist:
#             score += 6; signals['macd_signal'] = 'ACCELERATING'
#         elif macd_hist > 0:
#             score += 3

#         if latest['ADX'] > 25 and latest['DI_POS'] > latest['DI_NEG']:
#             score += 9; signals['adx'] = 'STRONG_UPTREND'
#         elif latest['ADX'] > 20 and latest['DI_POS'] > latest['DI_NEG']:
#             score += 5

#         is_perfect_align = latest['Close'] > latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200']
#         is_overextended = (latest['Close'] - latest['EMA_20']) / latest['EMA_20'] > 0.08
#         if is_perfect_align and not is_overextended:
#             score += 8; signals['trend'] = 'PERFECT_ALIGN'
#         elif latest['Close'] > latest['EMA_50']:
#             score += 4; signals['trend'] = 'ABOVE_MA'

#         # === 2. PRICE ACTION & VOLATILITY (25 points) ===
#         bb_width = latest['BB_WIDTH']
#         is_squeeze = bb_width < df_ind['BB_WIDTH'].tail(50).quantile(0.1)
#         if is_squeeze and latest['Close'] > latest['BB_MID']:
#             score += 9; signals['volatility'] = 'SQUEEZE_BREAKOUT_POTENTIAL'

#         hh_count = (recent['High'].diff().dropna() > 0).sum()
#         hl_count = (recent['Low'].diff().dropna() > 0).sum()
#         if hh_count > 15 and hl_count > 15:
#             score += 8; signals['structure'] = 'STRONG_HH_HL'

#         if latest['Close'] > df_ind['High'].tail(50).max():
#             score += 8; signals['breakout'] = '50D_HIGH_BREAK'
#         elif latest['Close'] > df_ind['High'].tail(20).max():
#             score += 6; signals['breakout'] = '20D_HIGH_BREAK'

#         # === 3. VOLUME & MONEY FLOW (20 points) ===
#         price_change_5d = (latest['Close'] - df_ind['Close'].iloc[-5]) / df_ind['Close'].iloc[-5]
#         vol_change_5d = df_ind['Volume'].tail(5).mean() / df_ind['Volume'].tail(30).mean() if df_ind['Volume'].tail(30).mean() > 0 else 1
#         if price_change_5d > 0.03 and vol_change_5d > 1.5:
#             score += 10; signals['volume_trend'] = 'SURGE_CONFIRM'
#         elif price_change_5d > 0 and vol_change_5d > 1.2:
#             score += 5; signals['volume_trend'] = 'INCREASING'

#         obv_rising = latest['OBV'] > df_ind['OBV'].tail(10).mean()
#         mfi_healthy = 40 < latest['MFI'] < 80
#         if obv_rising and mfi_healthy:
#             score += 10; signals['flow'] = 'ACCUMULATION'
#         elif obv_rising or mfi_healthy:
#             score += 5

#         # === 4. COMPREHENSIVE PATTERN RECOGNITION (30 points) ===
#         if pattern_model:
#             pattern_name, pattern_conf = predict_chart_pattern(pattern_model, df, lookback=60)
#             pattern_weight = PATTERN_WEIGHTS.get(pattern_name, 0.0)
#             signals['chart_pattern'] = f"CNN: {pattern_name} ({pattern_conf:.1f}%)"

#             if pattern_conf > 50: # Confidence threshold
#                 pattern_score = pattern_weight * 30  # Max 30 points
#                 score += pattern_score
#                 if pattern_weight > 0:
#                     signals['pattern_signal'] = 'BULLISH'
#                 elif pattern_weight < 0:
#                     signals['pattern_signal'] = 'BEARISH'
#         else:
#              signals['chart_pattern'] = "N/A (No Model)"

#         # === 5. FUNDAMENTAL QUALITY (10 points) ===
#         pe, roe, pb = latest.get('PE_Ratio', 999), latest.get('ROE', 0), latest.get('PB_Ratio', 999)
#         if 0 < pe < 35: score += 4
#         if roe > 0.15: score += 4
#         if 0 < pb < 5: score += 2

#         # === FINAL CALCULATION ===
#         growth_score = max(0, min(100, score))
#         price_change_7d = ((latest['Close'] - df_ind['Close'].iloc[-7]) / df_ind['Close'].iloc[-7]) * 100
#         atr_pct = (latest['ATR'] / latest['Close']) * 100 if latest['Close'] > 0 else 0

#         estimated_growth = (price_change_7d * 0.5) + (growth_score / 100 * atr_pct * 2.5)

#         return {
#             'symbol': symbol,
#             'growth_score': round(growth_score, 2),
#             'current_price': round(latest['Close'], 2),
#             'estimated_weekly_growth': round(estimated_growth, 2),
#             'rsi': round(latest['RSI'], 2),
#             'macd_signal': signals.get('macd_signal', 'NEUTRAL'),
#             'trend': signals.get('trend', 'NEUTRAL'),
#             'volume_trend': signals.get('volume_trend', 'NEUTRAL'),
#             'chart_pattern': signals.get('chart_pattern', 'NEUTRAL'),
#             'key_signals': ', '.join([f"{k}:{v}" for k, v in signals.items()])
#         }
#     except Exception as e:
#         return None


# def scan_growth_stocks_with_patterns(stock_list, pattern_model_path, top_n=20, min_score=60, max_workers=4):
#     """
#     ENHANCED: Uses CNN pattern recognition in screening with a robust multiprocessing initializer.
#     """
#     print(f"\n{'='*80}")
#     print(f"🚀 OPTIMIZED SCANNING WITH CNN PATTERN RECOGNITION 🚀")
#     print(f"{'='*80}")
#     print(f"Scanning {len(stock_list)} stocks...")
#     print(f"Analysis: Technical + Momentum + Fundamentals + CNN Chart Outcomes")
#     print(f"Using {max_workers} parallel processes.")
#     print(f"{'='*80}\n")

#     all_results = []
    
#     # Use initializer to load the model in each worker process
#     with mp.Pool(processes=max_workers, initializer=init_worker, initargs=(pattern_model_path,)) as pool:
#         results_iterator = pool.imap_unordered(analyze_growth_potential, stock_list)

#         for result in tqdm(results_iterator, total=len(stock_list), desc="Analyzing stocks"):
#             if result:
#                 all_results.append(result)

#     if not all_results:
#         print("\n⚠️ No valid data could be processed for any stock.")
#         return pd.DataFrame()

#     df_results = pd.DataFrame(all_results)
#     df_results = df_results.sort_values('growth_score', ascending=False)

#     filtered_results = df_results[df_results['growth_score'] >= min_score]

#     scores = df_results['growth_score'].dropna()
#     print(f"\n{'='*80}")
#     print(f"📊 SCAN STATISTICS:")
#     print(f"   Total stocks analyzed: {len(all_results)}")
#     print(f"   Stocks meeting threshold (>={min_score}): {len(filtered_results)}")
#     if not scores.empty:
#         print(f"   Score range: {scores.min():.1f} - {scores.max():.1f}")
#         print(f"   Average score: {scores.mean():.1f}")
#     print(f"{'='*80}")

#     if filtered_results.empty:
#         print(f"\n⚠️ No stocks found with growth score >= {min_score}")
#         print(f"Showing TOP {top_n} stocks by score instead...\n")
#         display_df = df_results.head(top_n)
#     else:
#         display_df = filtered_results.head(top_n)

#     if display_df.empty:
#         print("\nNo stocks to display.")
#         return display_df

#     print(f"\n{'='*80}")
#     print(f"🏆 TOP {len(display_df)} HIGH-POTENTIAL GROWTH STOCKS 🏆")
#     print(f"{'='*80}\n")

#     display_cols = ['symbol', 'growth_score', 'estimated_weekly_growth',
#                     'current_price', 'rsi', 'chart_pattern', 'macd_signal',
#                     'trend', 'volume_trend']

#     print(display_df[display_cols].to_string(index=False))

#     print(f"\n{'='*80}")
#     print("SIGNAL LEGEND:")
#     print("- Growth Score: 0-100 (Higher = Better potential)")
#     print("- Chart Pattern: CNN-detected future price outcome with confidence %")
#     print("- Est. Weekly Growth: Projected % change in next 7 days")
#     print(f"{'='*80}\n")

#     return display_df


# def run(symbol="TATAPOWER", window=90, epochs=100, batch_size=32, force_download=False,
#       n_forecast_days=30, ensemble_n=5, pattern_model=None):
#     """
#     ENHANCED: Full analysis with CNN pattern recognition integrated.
#     """
#     print(f"\n{'='*70}")
#     print(f"DEEP DIVE ANALYSIS: {symbol}")
#     print(f"{'='*70}\n")

#     df = fetch_data(symbol, force_download=force_download)
#     if df is None:
#         print(f"Error: Could not fetch data for {symbol}")
#         return None

#     # CNN Pattern Detection
#     pattern_name, pattern_conf = 'N/A', 0.0
#     if pattern_model is not None:
#         pattern_name, pattern_conf = predict_chart_pattern(pattern_model, df, lookback=60)
#         print(f"📊 CNN Predicted Outcome: {pattern_name} (Confidence: {pattern_conf:.1f}%)")
#         print(f"{'='*70}\n")

#     (X_train, y_train, X_test, y_test, y_train_raw, y_test_raw, last_close_test, test_dates,
#      feature_scaler, target_scaler, df_ind, unique_features) = make_sequences_fast(df, window, symbol=symbol)

#     if len(y_test_raw) == 0:
#         print("Not enough test data to perform analysis.")
#         return

#     models, all_test_preds = [], []
#     for i in range(ensemble_n):
#         print(f"Training Model {i+1}/{ensemble_n}...", end=' ', flush=True)
#         model = build_model_fast((X_train.shape[1], X_train.shape[2]), lr=1e-3 * (0.8 + 0.4 * np.random.random()))
#         es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, min_lr=1e-6, verbose=0)
#         history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size,
#                   callbacks=[es, reduce_lr], verbose=0)
#         print(f"Done (epochs: {len(history.history['loss'])})")
#         models.append(model)
#         pred = target_scaler.inverse_transform(model.predict(X_test, verbose=0).reshape(-1, 1)).flatten()
#         all_test_preds.append(pred)

#     pred_price = np.mean(all_test_preds, axis=0)
#     rmse = np.sqrt(mean_squared_error(y_test_raw, pred_price))
#     mae = np.mean(np.abs(y_test_raw - pred_price))
#     mape = np.mean(np.abs((y_test_raw - pred_price) / y_test_raw)) * 100 if np.mean(y_test_raw) > 0 else 0
#     corr = np.corrcoef(y_test_raw, pred_price)[0, 1]

#     print(f"\nRMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}% | Corr: {corr:.4f}")

#     signals, test_conf = generate_confident_signals(all_test_preds, last_close_test, conf_threshold=75, signal_gap=7)
#     bt = backtest_strategy(y_test_raw, signals)
#     print(f"Return: {bt['total_return']:.2f}% | B&H: {bt['buy_hold_return']:.2f}% | Excess: {bt['excess_return']:.2f}%")
#     print(f"Trades: {bt['num_trades']} | BUY: {signals.count('BUY')} | SELL: {signals.count('SELL')}\n")

#     # Forecast
#     last_seq = df_ind[unique_features].iloc[-window:].values
#     future_preds_all, forecast_sigs, forecast_conf = [], [], []

#     print("Forecasting 30 days...")
#     for day in tqdm(range(n_forecast_days), desc="Forecasting"):
#         last_seq_scaled = feature_scaler.transform(last_seq).reshape(1, window, -1)
#         step_preds = [target_scaler.inverse_transform(m.predict(last_seq_scaled, verbose=0)).flatten()[0] for m in models]
#         future_preds_all.append(step_preds)

#         curr_price = df_ind["Close"].iloc[-1] if day == 0 else np.mean(future_preds_all[day-1])
#         conf, mean_pred = calculate_confidence(step_preds, curr_price)
#         forecast_conf.append(conf)

#         if conf >= 75:
#             pct_chg = ((mean_pred - curr_price) / curr_price) * 100 if curr_price != 0 else 0
#             forecast_sigs.append('BUY' if pct_chg > 4.0 else 'SELL' if pct_chg < -4.0 else 'HOLD')
#         else:
#             forecast_sigs.append('HOLD')

#         new_row = last_seq[-1].copy()
#         try:
#             new_row[unique_features.index("Close")] = mean_pred
#         except ValueError:
#             pass
#         last_seq = np.vstack([last_seq[1:], new_row])

#     future_preds_mean = [np.mean(fp) for fp in future_preds_all]
#     future_dates = pd.date_range(df.index[-1], periods=n_forecast_days+1, freq="B")[1:]
#     print(f"Forecast: BUY={forecast_sigs.count('BUY')} | SELL={forecast_sigs.count('SELL')} | Avg Conf: {np.mean(forecast_conf):.1f}%\n")

#     # Plot
#     df_plot = df[df.index >= (df.index[-1] - pd.Timedelta(days=730))].copy()
#     fig = go.Figure()
#     fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot["Open"], high=df_plot["High"],
#         low=df_plot["Low"], close=df_plot["Close"], name="Price", increasing_line_color="green",
#         decreasing_line_color="red"))
#     fig.add_trace(go.Scatter(x=test_dates, y=pred_price, mode="lines", name="Predicted",
#         line=dict(color="cyan", width=2)))

#     buy_idx = [i for i, s in enumerate(signals) if s == 'BUY']
#     sell_idx = [i for i, s in enumerate(signals) if s == 'SELL']
#     if buy_idx:
#         fig.add_trace(go.Scatter(x=test_dates[buy_idx], y=pred_price[buy_idx], mode='markers',
#             name='BUY (≥75%)', marker=dict(symbol='triangle-up', size=16, color='lime', line=dict(width=2, color='darkgreen'))))
#     if sell_idx:
#         fig.add_trace(go.Scatter(x=test_dates[sell_idx], y=pred_price[sell_idx], mode='markers',
#             name='SELL (≥75%)', marker=dict(symbol='triangle-down', size=16, color='red', line=dict(width=2, color='darkred'))))

#     fig.add_trace(go.Scatter(x=future_dates, y=future_preds_mean, mode="lines", name="Forecast",
#         line=dict(color="gold", dash="dash", width=3)))

#     fore_buy = [i for i, s in enumerate(forecast_sigs) if s == 'BUY']
#     fore_sell = [i for i, s in enumerate(forecast_sigs) if s == 'SELL']
#     if fore_buy:
#         fig.add_trace(go.Scatter(x=future_dates[fore_buy], y=np.array(future_preds_mean)[fore_buy],
#             mode='markers', name='Forecast BUY (≥75%)', marker=dict(symbol='star', size=18, color='lightgreen',
#             line=dict(width=2, color='green'))))
#     if fore_sell:
#         fig.add_trace(go.Scatter(x=future_dates[fore_sell], y=np.array(future_preds_mean)[fore_sell],
#             mode='markers', name='Forecast SELL (≥75%)', marker=dict(symbol='star', size=18, color='orange',
#             line=dict(width=2, color='darkred'))))

#     title_text = f"{symbol} | Predictions + Forecast"
#     if pattern_model is not None:
#         title_text += f" | Predicted Outcome: {pattern_name} ({pattern_conf:.1f}%)"

#     fig.update_layout(title=title_text, xaxis_title="Date",
#         yaxis_title="Price", template="plotly_dark", height=700, width=1400,
#         xaxis=dict(rangeslider=dict(visible=True), showgrid=True), yaxis=dict(showgrid=True))
#     fig.show()

#     return {'rmse': rmse, 'mae': mae, 'mape': mape, 'corr': corr, 'backtest': bt,
#             'test_signals': {'BUY': signals.count('BUY'), 'SELL': signals.count('SELL')},
#             'forecast_signals': {'BUY': forecast_sigs.count('BUY'), 'SELL': forecast_sigs.count('SELL')},
#             'pattern': pattern_name if pattern_model else 'N/A',
#             'pattern_confidence': pattern_conf if pattern_model else 0.0}

# if __name__ == "__main__":
#     mp.freeze_support()

#     print("\n" + "="*70)
#     print("🚀 ADVANCED STOCK ANALYSIS WITH OPTIMIZED CNN 🚀")
#     print("="*70)

#     pattern_model_path = os.path.join(PATTERN_MODEL_DIR, "optimized_chart_pattern_cnn.keras")
#     pattern_model = None

#     if os.path.exists(pattern_model_path):
#         try:
#             pattern_model = tf.keras.models.load_model(pattern_model_path)
#             print("\n✅ Loaded pre-trained OPTIMIZED CNN pattern model.")
#         except Exception as e:
#             print(f"\n⚠️ Could not load existing model due to error: {e}. Will train new one if needed.")

#     print("\nChoose mode:")
#     print("1. [IMPORTANT] Train new OPTIMIZED CNN model (recommended for first run)")
#     print("2. Enhanced growth scanner (with OPTIMIZED CNN patterns)")
#     print("3. Analyze specific stock (with OPTIMIZED CNN patterns)")
#     choice = input("\nEnter choice (1/2/3): ").strip()

#     if choice == '1':
#         print("\n" + "="*70)
#         print("TRAINING OPTIMIZED CNN PATTERN RECOGNITION MODEL")
#         print("="*70)

#         nifty500_list = get_nifty500_symbols()
#         X_train, y_train = create_pattern_dataset_from_stocks(
#             nifty500_list,
#             samples_per_stock=15,
#             lookback=60
#         )

#         if X_train is not None and len(X_train) > 100:
#             pattern_model = train_pattern_recognition_model(
#                 X_train,
#                 y_train,
#                 model_path=pattern_model_path,
#                 epochs=75 # ResNet style models benefit from slightly more epochs
#             )
#             if pattern_model:
#                 print(f"\n✅ Model trained and saved to: {pattern_model_path}")
#                 print("You can now use option 2 or 3 to analyze stocks with this powerful model.")
#         else:
#             print("\n⚠️ Insufficient data for training. Please try again later.")

#     elif choice == '2':
#         print("\n" + "="*70)
#         print("ENHANCED GROWTH POTENTIAL SCANNER WITH OPTIMIZED CNN")
#         print("="*70)

#         if pattern_model is None:
#             print("\n⚠️ No pre-trained pattern model found. The scanner will run without pattern analysis.")
#             print("It's highly recommended to train a model first using Option 1 for best results.")
#             if input("Continue without pattern model? (y/n): ").strip().lower() != 'y':
#                 exit()

#         num_stocks = input("\nEnter number of top stocks to display (default 20): ").strip()
#         top_n = int(num_stocks) if num_stocks.isdigit() else 20
#         min_score = input("Enter minimum growth score threshold (0-100, default 65): ").strip()
#         min_score = int(min_score) if min_score.isdigit() else 65
#         default_workers = min(max(mp.cpu_count() - 1, 1), 8)
#         workers = input(f"Enter number of parallel workers (recommended {default_workers}): ").strip()
#         max_workers = min(int(workers) if workers.isdigit() else default_workers, 12)

#         nifty500_list = get_nifty500_symbols()
#         top_stocks = scan_growth_stocks_with_patterns(
#             stock_list=nifty500_list,
#             pattern_model_path=pattern_model_path,
#             top_n=top_n,
#             min_score=min_score,
#             max_workers=max_workers
#         )

#         if not top_stocks.empty:
#             print("\n" + "="*70)
#             print("DETAILED SIGNAL BREAKDOWN FOR TOP 5 STOCKS:")
#             print("="*70)
#             for _, row in top_stocks.head(5).iterrows():
#                 print(f"\n{row['symbol']} (Score: {row['growth_score']})")
#                 print(f"  Current Price: ₹{row['current_price']:.2f} | Est. Weekly Growth: {row['estimated_weekly_growth']:.2f}%")
#                 print(f"  Chart Outcome: {row.get('chart_pattern', 'N/A')}")
#                 print(f"  Key Signals: {row['key_signals']}")

#             analyze_choice = input("\n\nRun detailed ML analysis on any stock? (y/n): ").strip().lower()
#             if analyze_choice == 'y':
#                 symbol = input("\nEnter stock symbol from the list above: ").strip().upper()
#                 if symbol:
#                     res = run(symbol=symbol, force_download=True, pattern_model=pattern_model)

#     else:  # Option 3
#         if pattern_model is None:
#              print("\n⚠️ No pre-trained pattern model found. Analysis will run without it.")
#              print("It's highly recommended to train a model first using Option 1 for best results.")

#         symbol = input("\nEnter stock symbol (e.g., TATAPOWER, RELIANCE, INFY): ").strip().upper() or "TATAPOWER"
#         res = run(symbol=symbol, pattern_model=pattern_model)
#         if res:
#             print(f"\n{'='*70}\nFINAL SUMMARY - {symbol}\n{'='*70}")
#             print(f"RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f} | Corr: {res['corr']:.4f}")
#             print(f"Predicted Outcome: {res['pattern']} (Confidence: {res['pattern_confidence']:.1f}%)")
#             print(f"Strategy Return: {res['backtest']['total_return']:.2f}% | Buy&Hold: {res['backtest']['buy_hold_return']:.2f}%")
#             print(f"Excess Return: {res['backtest']['excess_return']:.2f}%")
#             print(f"{'='*70}")







import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import requests
import multiprocessing as mp
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, Add, Activation, GlobalAveragePooling2D, Concatenate,
                                     Conv1D, Bidirectional, LSTM, GRU, LayerNormalization, MultiHeadAttention,
                                     GlobalAveragePooling1D, LeakyReLU, Lambda)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pickle
from collections import deque
import json
from datetime import datetime
from PIL import Image
import random

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CACHE_DIR = "cache_data"
CHART_CACHE_DIR = "chart_cache"
PATTERN_MODEL_DIR = "pattern_models"
RL_MODEL_DIR = "rl_models"
RL_MEMORY_DIR = "rl_memory"

for dir_path in [CACHE_DIR, CHART_CACHE_DIR, PATTERN_MODEL_DIR, RL_MODEL_DIR, RL_MEMORY_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# ENHANCED PATTERN LABELS WITH MORE GRANULARITY
# ============================================================================
OUTCOME_PATTERN_NAMES = [
    'STRONG_BULLISH',      # Strong upward with high confidence
    'MODERATE_BULLISH',    # Moderate upward movement
    'WEAK_BULLISH',        # Weak upward bias
    'STRONG_BEARISH',      # Strong downward movement
    'MODERATE_BEARISH',    # Moderate downward movement
    'WEAK_BEARISH',        # Weak downward bias
    'CONSOLIDATION',       # Tight range, low volatility
    'HIGH_VOLATILITY',     # High volatility, no clear direction
    'BREAKOUT_UP',         # Breakout pattern upward
    'BREAKOUT_DOWN',       # Breakout pattern downward
    'REVERSAL_UP',         # Reversal from downtrend
    'REVERSAL_DOWN',       # Reversal from uptrend
]

OUTCOME_SEQUENTIAL_LABELS = {i: name for i, name in enumerate(OUTCOME_PATTERN_NAMES)}
OUTCOME_NAME_TO_SEQ_LABEL = {name: i for i, name in enumerate(OUTCOME_PATTERN_NAMES)}

# Enhanced weights for better scoring
PATTERN_WEIGHTS = {
    'STRONG_BULLISH': 1.0,
    'MODERATE_BULLISH': 0.7,
    'WEAK_BULLISH': 0.4,
    'BREAKOUT_UP': 0.9,
    'REVERSAL_UP': 0.85,
    'STRONG_BEARISH': -1.0,
    'MODERATE_BEARISH': -0.7,
    'WEAK_BEARISH': -0.4,
    'BREAKOUT_DOWN': -0.9,
    'REVERSAL_DOWN': -0.85,
    'CONSOLIDATION': 0.0,
    'HIGH_VOLATILITY': 0.0
}

# ============================================================================
# REINFORCEMENT LEARNING AGENT
# ============================================================================

class StockTradingAgent:
    """
    Deep Q-Network (DQN) Agent for stock trading that learns from predictions.
    Uses experience replay and learns from both correct and incorrect predictions.
    """
    def __init__(self, state_size=50, action_size=3, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size  # BUY, SELL, HOLD
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Performance tracking
        self.performance_log = []
        self.trade_history = []
        
    def _build_model(self):
        """Enhanced DQN architecture with Dueling structure"""
        inp = Input(shape=(self.state_size,))
        
        # Feature extraction
        x = Dense(256)(inp)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        
        # Dueling DQN: separate value and advantage streams
        value_stream = Dense(32, activation='relu')(x)
        value = Dense(1)(value_stream)
        
        advantage_stream = Dense(32, activation='relu')(x)
        advantage = Dense(self.action_size)(advantage_stream)
        
        # Combine value and advantage using a Lambda layer for the mean operation
        advantage_mean = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
        q_values = value + (advantage - advantage_mean)
        
        model = Model(inp, q_values)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='huber')
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=32):
        """Train on batch of experiences using Double DQN"""
        if len(self.memory) < batch_size:
            return 0
        
        minibatch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        minibatch = [self.memory[i] for i in minibatch_indices]
        
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])
        
        # Double DQN: use online network to select actions, target network to evaluate
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        target_q_values = self.target_model.predict(next_states, verbose=0)
        
        targets = rewards + (1 - dones) * self.gamma * target_q_values[np.arange(batch_size), next_actions]
        
        target_f = self.model.predict(states, verbose=0)
        target_f[np.arange(batch_size), actions] = targets
        
        history = self.model.fit(states, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def calculate_reward(self, action, actual_return, predicted_return, confidence):
        """
        Enhanced reward function that considers:
        - Correctness of prediction
        - Magnitude of returns
        - Confidence level
        """
        action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
        
        # Base reward from actual return
        if action == 0:  # BUY
            reward = actual_return * 100  # Profit/loss from holding
        elif action == 1:  # SELL
            reward = -actual_return * 100  # Inverse (profit from shorting)
        else:  # HOLD
            reward = -abs(actual_return) * 10  # Penalty for missing opportunities
        
        # Bonus for correct directional prediction
        if (action == 0 and predicted_return > 0 and actual_return > 0) or \
           (action == 1 and predicted_return < 0 and actual_return < 0):
            reward += confidence * 2  # Confidence-weighted bonus
        
        # Penalty for wrong direction
        elif (action == 0 and actual_return < -0.02) or \
             (action == 1 and actual_return > 0.02):
            reward -= confidence * 3  # Higher penalty for confident mistakes
        
        # Bonus for high-confidence correct predictions on large moves
        if abs(actual_return) > 0.05 and np.sign(predicted_return) == np.sign(actual_return):
            reward += 10
        
        return reward
    
    def save(self, path):
        """Save model and memory"""
        self.model.save(os.path.join(path, 'rl_model.keras'))
        self.target_model.save(os.path.join(path, 'rl_target_model.keras'))
        
        # Save memory and metadata
        memory_data = {
            'memory': list(self.memory)[-1000:],  # Save last 1000 experiences
            'epsilon': self.epsilon,
            'performance_log': self.performance_log,
            'trade_history': self.trade_history
        }
        
        with open(os.path.join(path, 'rl_memory.pkl'), 'wb') as f:
            pickle.dump(memory_data, f)
    
    def load(self, path):
        """Load model and memory"""
        try:
            self.model = tf.keras.models.load_model(os.path.join(path, 'rl_model.keras'))
            self.target_model = tf.keras.models.load_model(os.path.join(path, 'rl_target_model.keras'))
            
            with open(os.path.join(path, 'rl_memory.pkl'), 'rb') as f:
                memory_data = pickle.load(f)
                self.memory = deque(memory_data['memory'], maxlen=10000)
                self.epsilon = memory_data['epsilon']
                self.performance_log = memory_data.get('performance_log', [])
                self.trade_history = memory_data.get('trade_history', [])
            
            return True
        except Exception as e:
            print(f"Could not load RL model: {e}")
            return False

# ============================================================================
# GLOBAL VARIABLES FOR MULTIPROCESSING
# ============================================================================
_GLOBAL_PATTERN_MODEL = None
_GLOBAL_RL_AGENT = None

def init_worker(pattern_model_path, rl_model_path=None):
    """Initializer for multiprocessing pool"""
    global _GLOBAL_PATTERN_MODEL, _GLOBAL_RL_AGENT
    
    if os.path.exists(pattern_model_path):
        _GLOBAL_PATTERN_MODEL = tf.keras.models.load_model(pattern_model_path, compile=False)
        _GLOBAL_PATTERN_MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        _GLOBAL_PATTERN_MODEL = None
    
    if rl_model_path and os.path.exists(os.path.join(rl_model_path, 'rl_model.keras')):
        _GLOBAL_RL_AGENT = StockTradingAgent()
        _GLOBAL_RL_AGENT.load(rl_model_path)
    else:
        _GLOBAL_RL_AGENT = None

def get_global_pattern_model():
    return _GLOBAL_PATTERN_MODEL

def get_global_rl_agent():
    return _GLOBAL_RL_AGENT

# ============================================================================
# DATA FETCHING AND FEATURE ENGINEERING
# ============================================================================

df_fallback = pd.read_csv('C:\\Users\\Lenovo\\Documents\\VS CODE codes(files)\\helloworld\\STOCK_MARKET_PROJECT\\Ticker_List_NSE_India.csv')
FALLBACK_NIFTY_500_STOCKS = df_fallback['SYMBOL'].tolist()

def get_nifty500_symbols():
    driver = None
    try:
        print("Initializing browser to fetch NIFTY 500 list...")
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--log-level=3")
        
        prefs = {"profile.managed_default_content_settings.images": 2,
                 "profile.managed_default_content_settings.stylesheets": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                                  options=chrome_options)
        url = "https://en.wikipedia.org/wiki/NIFTY_500"
        driver.get(url)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "constituents")))
        
        html_source = driver.page_source
        tables = pd.read_html(html_source)
        
        for table in tables:
            table.columns = [str(col).strip() for col in table.columns]
            for col_name in ['Symbol', 'Ticker']:
                if col_name in table.columns:
                    symbols = table[col_name].str.strip().tolist()
                    print(f"✅ Fetched {len(symbols)} symbols from Wikipedia.")
                    return symbols
        
        raise ValueError("Could not find symbol column")
    
    except Exception as e:
        print(f"⚠️ Could not fetch from Wikipedia, falling back to local CSV list: {e}")
        return FALLBACK_NIFTY_500_STOCKS
    finally:
        if driver:
            driver.quit()

def fetch_data(symbol="TATAPOWER", period="5y", interval="1d", force_download=False):
    yf_symbol = f"{symbol}.NS" if "." not in symbol else symbol
    cache_path = os.path.join(CACHE_DIR, f"{yf_symbol.replace('.', '_')}_{period}_{interval}.csv")
    
    if os.path.exists(cache_path) and not force_download:
        try:
            df = pd.read_csv(cache_path, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
            if not df.empty and all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]):
                return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        except:
            pass
    
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False, timeout=10)
        if df is None or df.empty:
            return None
        
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            return None
        
        df.to_csv(cache_path)
        return df.dropna(subset=required_cols)
    except:
        return None

def fetch_fundamental_data(symbol="TATAPOWER"):
    try:
        ticker = yf.Ticker(f"{symbol}.NS" if "." not in symbol else symbol)
        info = ticker.info
        return pd.DataFrame({
            "PE_Ratio": [info.get("trailingPE", 0.0) or 0.0],
            "EPS": [info.get("trailingEps", 0.0) or 0.0],
            "PB_Ratio": [info.get("priceToBook", 0.0) or 0.0],
            "Market_Cap": [info.get("marketCap", 0.0) or 0.0],
            "Dividend_Yield": [info.get("dividendYield", 0.0) or 0.0],
            "ROE": [info.get("returnOnEquity", 0.0) or 0.0],
            "ROCE": [info.get("returnOnCapitalEmployed", 0.0) or 0.0]
        })
    except:
        return pd.DataFrame()

def compute_indicators_fast(df, symbol="TATAPOWER"):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    
    for col in ["Close", "High", "Low", "Volume", "Open"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close", "High", "Low", "Volume", "Open"])
    
    if df.empty:
        return df
    
    close, high, low, vol, open_price = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    
    # Technical indicators
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"], df["MACD_signal"] = macd.macd(), macd.macd_signal()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_MID"], df["BB_UP"], df["BB_LOW"] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["ADX"], df["DI_POS"], df["DI_NEG"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["SMA_20"] = close.rolling(20).mean()
    df["EMA_20"], df["EMA_50"], df["EMA_200"] = close.ewm(span=20).mean(), close.ewm(span=50).mean(), close.ewm(span=200).mean()
    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"], df["stoch_d"] = stoch.stoch(), stoch.stoch_signal()
    df["CCI"] = ta.trend.CCIIndicator(high, low, close).cci()
    df["VWAP"] = (vol * (high + low + close) / 3).cumsum() / vol.cumsum()
    df["MFI"] = ta.volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()
    df["Williams_R"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    df["TSI"] = ta.momentum.TSIIndicator(close).tsi()
    aroon = ta.trend.AroonIndicator(high=high, low=low, window=25)
    df["Aroon_Up"], df["Aroon_Down"] = aroon.aroon_up(), aroon.aroon_down()
    df["Volume_SMA"] = vol.rolling(20).mean()
    df["Price_ROC"] = ta.momentum.ROCIndicator(close, window=12).roc()
    df["Volume_ROC"] = vol.pct_change(periods=12) * 100
    df["BB_WIDTH"] = (df["BB_UP"] - df["BB_LOW"]) / df["BB_MID"]
    keltner = ta.volatility.KeltnerChannel(high, low, close, window=20)
    df["Keltner_Up"], df["Keltner_Low"] = keltner.keltner_channel_hband(), keltner.keltner_channel_lband()
    ichimoku = ta.trend.IchimokuIndicator(high, low)
    df["Ichimoku_A"], df["Ichimoku_B"] = ichimoku.ichimoku_a(), ichimoku.ichimoku_b()
    df["High_Low_Ratio"] = np.where(low > 0, high / low, 1.0)
    df["Close_Open_Ratio"] = np.where(open_price > 0, close / open_price, 1.0)
    df["Intraday_Range"] = np.where(open_price > 0, (high - low) / open_price, 0.0)
    rolling_mean, rolling_std = close.rolling(50).mean(), close.rolling(50).std()
    df["Trend_Strength"] = np.where(rolling_std > 0, (close - rolling_mean) / rolling_std, 0.0)
    df["Historical_Volatility"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Fundamental data
    fund_df = fetch_fundamental_data(symbol)
    if not fund_df.empty:
        for key, value in fund_df.iloc[0].to_dict().items():
            df[key] = value
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').dropna()
    
    # Outlier handling
    for col in df.columns:
        if col not in ['Date'] and pd.api.types.is_numeric_dtype(df[col]):
            col_mean, col_std = df[col].mean(), df[col].std()
            if col_std > 0:
                df[col] = df[col].clip(lower=col_mean - 5*col_std, upper=col_mean + 5*col_std)
    
    return df

# ============================================================================
# CNN MODEL AND DATASET CREATION
# ============================================================================

def squeeze_excitation_block(x, ratio=16):
    """Squeeze-and-Excitation block for channel attention"""
    channels = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return tf.keras.layers.Multiply()([x, se])

def resnet_block_enhanced(x, filters, kernel_size=3, stride=1):
    """Enhanced ResNet block with SE attention"""
    shortcut = x
    
    y = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization()(y)
    
    # Add Squeeze-Excitation
    y = squeeze_excitation_block(y)
    
    if stride != 1 or x.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    y = Add()([y, shortcut])
    y = Activation('relu')(y)
    return y

def build_optimized_cnn(input_shape=(128, 128, 3), num_classes=len(OUTCOME_SEQUENTIAL_LABELS)):
    """Enhanced CNN with SE blocks and better regularization"""
    inp = Input(shape=input_shape)
    
    # Initial conv with larger kernel for better feature extraction
    x = Conv2D(64, 7, strides=2, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Enhanced ResNet blocks with SE
    x = resnet_block_enhanced(x, 64)
    x = resnet_block_enhanced(x, 64)
    x = resnet_block_enhanced(x, 128, stride=2)
    x = resnet_block_enhanced(x, 128)
    x = resnet_block_enhanced(x, 256, stride=2)
    x = resnet_block_enhanced(x, 256)
    x = resnet_block_enhanced(x, 512, stride=2)
    
    # Global pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with better regularization
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    
    out = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inp, out)
    return model

def label_pattern_from_outcome_enhanced(df_window, lookback=60, outcome_period=20):
    """Enhanced pattern labeling with more granular patterns"""
    if len(df_window) < lookback + outcome_period:
        return OUTCOME_NAME_TO_SEQ_LABEL['CONSOLIDATION']
    
    chart_period = df_window.iloc[:lookback]
    outcome_data = df_window.iloc[lookback:lookback + outcome_period]
    
    start_price = outcome_data['Close'].iloc[0]
    max_price = outcome_data['High'].max()
    min_price = outcome_data['Low'].min()
    end_price = outcome_data['Close'].iloc[-1]
    
    # Calculate metrics
    volatility = outcome_data['Close'].pct_change().std()
    net_change = (end_price - start_price) / start_price * 100
    
    # Check for breakout (volume spike with directional move)
    vol_spike = outcome_data['Volume'].iloc[:5].mean() / chart_period['Volume'].mean() > 1.5
    
    # Check for reversal (trend change)
    pre_trend = (chart_period['Close'].iloc[-1] - chart_period['Close'].iloc[-20]) / chart_period['Close'].iloc[-20] * 100
    is_reversal = (pre_trend > 5 and net_change < -3) or (pre_trend < -5 and net_change > 3)
    
    # Thresholds
    strong_threshold = 8.0
    moderate_threshold = 4.0
    weak_threshold = 2.0
    vol_threshold = 0.03
    
    # Pattern classification
    if is_reversal:
        if net_change > moderate_threshold:
            return OUTCOME_NAME_TO_SEQ_LABEL['REVERSAL_UP']
        elif net_change < -moderate_threshold:
            return OUTCOME_NAME_TO_SEQ_LABEL['REVERSAL_DOWN']
    
    if vol_spike:
        if net_change > moderate_threshold:
            return OUTCOME_NAME_TO_SEQ_LABEL['BREAKOUT_UP']
        elif net_change < -moderate_threshold:
            return OUTCOME_NAME_TO_SEQ_LABEL['BREAKOUT_DOWN']
    
    if volatility > vol_threshold:
        return OUTCOME_NAME_TO_SEQ_LABEL['HIGH_VOLATILITY']
    
    if net_change > strong_threshold:
        return OUTCOME_NAME_TO_SEQ_LABEL['STRONG_BULLISH']
    elif net_change > moderate_threshold:
        return OUTCOME_NAME_TO_SEQ_LABEL['MODERATE_BULLISH']
    elif net_change > weak_threshold:
        return OUTCOME_NAME_TO_SEQ_LABEL['WEAK_BULLISH']
    elif net_change < -strong_threshold:
        return OUTCOME_NAME_TO_SEQ_LABEL['STRONG_BEARISH']
    elif net_change < -moderate_threshold:
        return OUTCOME_NAME_TO_SEQ_LABEL['MODERATE_BEARISH']
    elif net_change < -weak_threshold:
        return OUTCOME_NAME_TO_SEQ_LABEL['WEAK_BEARISH']
    else:
        return OUTCOME_NAME_TO_SEQ_LABEL['CONSOLIDATION']

def generate_chart_image_with_features(df, lookback=60, img_size=(128, 128)):
    """Generate multi-channel chart image"""
    if df is None or len(df) < lookback:
        return None
    
    df_chart = df.tail(lookback).copy()
    if df_chart.empty:
        return None
    
    min_val = df_chart['Low'].min()
    max_val = df_chart['High'].max()
    range_val = max_val - min_val
    if range_val == 0:
        return None
    
    def normalize(series):
        return (series - min_val) / range_val
    
    df_norm = df_chart[['Open', 'High', 'Low', 'Close']].apply(normalize)
    df_norm['EMA20'] = normalize(df_chart['EMA_20'])
    df_norm['EMA50'] = normalize(df_chart['EMA_50'])
    df_norm['BB_UP'] = normalize(df_chart['BB_UP'])
    df_norm['BB_LOW'] = normalize(df_chart['BB_LOW'])
    
    fig, ax = plt.subplots(figsize=(img_size[0]/32, img_size[1]/32), dpi=32)
    ax.axis('off')
    fig.patch.set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, len(df_chart))
    
    # Candlesticks (Red channel)
    for i in range(len(df_norm)):
        row = df_norm.iloc[i]
        color_val = 1.0
        ax.plot([i, i], [row['Low'], row['High']], color=(color_val, 0, 0), linewidth=0.5)
        body_bottom = min(row['Open'], row['Close'])
        body_height = abs(row['Close'] - row['Open'])
        ax.add_patch(plt.Rectangle((i - 0.3, body_bottom), 0.6, body_height, facecolor=(color_val, 0, 0)))
    
    # EMAs (Green channel)
    ax.plot(range(len(df_norm)), df_norm['EMA20'], color=(0, 1.0, 0), linewidth=1)
    ax.plot(range(len(df_norm)), df_norm['EMA50'], color=(0, 0.7, 0), linewidth=1)
    
    # Bollinger Bands (Blue channel)
    ax.plot(range(len(df_norm)), df_norm['BB_UP'], color=(0, 0, 1.0), linewidth=0.8, linestyle='--')
    ax.plot(range(len(df_norm)), df_norm['BB_LOW'], color=(0, 0, 1.0), linewidth=0.8, linestyle='--')
    ax.fill_between(range(len(df_norm)), df_norm['BB_LOW'], df_norm['BB_UP'], color=(0, 0, 1.0), alpha=0.1)
    
    plt.tight_layout(pad=0)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    img = Image.open(buf).resize(img_size)
    final_img_array = np.array(img)[:, :, :3] / 255.0
    
    return final_img_array

def create_pattern_dataset_from_stocks(stock_list, samples_per_stock=10, lookback=60):
    """Create labeled dataset with enhanced patterns"""
    print(f"\n{'='*70}")
    print("CREATING ENHANCED CHART PATTERN DATASET")
    print(f"{'='*70}\n")
    
    X_images = []
    y_labels = []
    
    for symbol in tqdm(stock_list[:150], desc="Processing stocks"):
        try:
            df_full = fetch_data(symbol, period="5y", force_download=False)
            if df_full is None or len(df_full) < lookback + 50:
                continue
            
            df_full = compute_indicators_fast(df_full, symbol)
            if df_full.empty:
                continue
            
            max_start = len(df_full) - lookback - 20
            if max_start < samples_per_stock:
                continue
            
            sample_indices = np.random.choice(max_start, min(samples_per_stock, max_start), replace=False)
            
            for idx in sample_indices:
                window_df = df_full.iloc[idx : idx + lookback + 20]
                img = generate_chart_image_with_features(window_df.iloc[:lookback], lookback=lookback)
                if img is None:
                    continue
                
                label = label_pattern_from_outcome_enhanced(window_df, lookback=lookback)
                X_images.append(img)
                y_labels.append(label)
        
        except Exception as e:
            continue
    
    if len(X_images) == 0:
        print("⚠️ No valid samples generated!")
        return None, None
    
    X = np.array(X_images)
    y = np.array(y_labels)
    
    print(f"\n✅ Dataset created: {len(X)} samples")
    print(f"Pattern distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for pattern_id, count in zip(unique, counts):
        if pattern_id in OUTCOME_SEQUENTIAL_LABELS:
            print(f"  {OUTCOME_SEQUENTIAL_LABELS[pattern_id]}: {count}")
    
    return X, y

def train_pattern_recognition_model(X, y, model_path=None, epochs=100, batch_size=32):
    """Train enhanced CNN model"""
    if X is None or y is None or len(X) == 0:
        print("⚠️ No data available for training!")
        return None
    
    print(f"\n{'='*70}")
    print("TRAINING ENHANCED PATTERN RECOGNITION MODEL")
    print(f"{'='*70}\n")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    print("\nApplying class weights...")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        horizontal_flip=False
    )
    datagen.fit(X_train)
    
    # Build model
    model = build_optimized_cnn(input_shape=X_train.shape[1:], num_classes=len(OUTCOME_SEQUENTIAL_LABELS))
    
    # Learning rate schedule with warmup
    initial_lr = 0.0001
    warmup_epochs = 5
    
    def lr_schedule(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            return initial_lr * (0.95 ** (epoch - warmup_epochs))
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        lr_scheduler
    ]
    
    if model_path:
        callbacks.append(ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1))
    
    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights_dict
    )
    
    # Evaluate
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    report_labels = np.union1d(np.unique(y_val), np.unique(y_pred))
    target_names = [OUTCOME_SEQUENTIAL_LABELS[i] for i in report_labels if i in OUTCOME_SEQUENTIAL_LABELS]
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, labels=report_labels, target_names=target_names, zero_division=0))
    
    return model

# ============================================================================
# OPTIMIZED REINFORCEMENT LEARNING AND ANALYSIS
# ============================================================================

def predict_chart_pattern(model, df_with_indicators, lookback=60):
    """
    Predict chart pattern from a dataframe that already contains indicators.
    This version is optimized to avoid re-calculating indicators.
    """
    if model is None or df_with_indicators is None or len(df_with_indicators) < lookback:
        return 'CONSOLIDATION', 0.0
    
    # Generate image directly from the provided dataframe with indicators
    img = generate_chart_image_with_features(df_with_indicators, lookback=lookback)
    if img is None:
        return 'CONSOLIDATION', 0.0
    
    img_batch = np.expand_dims(img, axis=0)
    predictions = model.predict(img_batch, verbose=0)[0]
    
    pattern_id = np.argmax(predictions)
    confidence = predictions[pattern_id] * 100
    
    return OUTCOME_SEQUENTIAL_LABELS.get(pattern_id, 'UNKNOWN'), confidence

def create_state_vector(df_window):
    """Create state vector from price window for RL agent"""
    try:
        if len(df_window) < 50:
            return None
        
        latest = df_window.iloc[-1]
        
        # Ensure indicators are present, fill with 0 if not
        required_indicators = ['RSI', 'MACD', 'MACD_signal', 'ADX', 'Close', 'BB_LOW', 'BB_UP', 
                               'stoch_k', 'CCI', 'MFI', 'Williams_R', 'Aroon_Up', 'Aroon_Down',
                               'Volume', 'EMA_20', 'EMA_50', 'EMA_200', 'ATR', 'Historical_Volatility',
                               'High_Low_Ratio', 'Close_Open_Ratio']
        for ind in required_indicators:
            if ind not in latest:
                latest[ind] = 0

        # Technical indicators
        state = [
            latest['RSI'] / 100.0,
            (latest['MACD'] - latest['MACD_signal']) / latest['Close'] if latest['Close'] > 0 else 0,
            latest['ADX'] / 100.0,
            (latest['Close'] - latest['BB_LOW']) / (latest['BB_UP'] - latest['BB_LOW']) if latest['BB_UP'] != latest['BB_LOW'] else 0.5,
            latest['stoch_k'] / 100.0,
            np.clip(latest['CCI'] / 200.0, -1, 1),
            latest['MFI'] / 100.0,
            latest['Williams_R'] / -100.0,
            latest['Aroon_Up'] / 100.0,
            latest['Aroon_Down'] / 100.0,
        ]
        
        # Price momentum features
        for period in [5, 10, 20]:
            if len(df_window) >= period and df_window.iloc[-period]['Close'] > 0:
                price_change = (latest['Close'] - df_window.iloc[-period]['Close']) / df_window.iloc[-period]['Close']
                state.append(np.clip(price_change, -0.5, 0.5))
            else:
                state.append(0)
        
        # Volume features
        mean_vol = df_window['Volume'].mean()
        vol_ratio = latest['Volume'] / mean_vol if mean_vol > 0 else 1
        state.append(min(vol_ratio, 5.0) / 5.0)
        
        # Trend features
        if latest['EMA_20'] > 0:
            state.append(np.clip((latest['Close'] - latest['EMA_20']) / latest['EMA_20'], -0.5, 0.5))
            state.append(np.clip((latest['EMA_20'] - latest['EMA_50']) / latest['EMA_50'], -0.5, 0.5) if latest['EMA_50'] > 0 else 0)
            state.append(np.clip((latest['EMA_50'] - latest['EMA_200']) / latest['EMA_200'], -0.5, 0.5) if latest['EMA_200'] > 0 else 0)
        else:
            state.extend([0, 0, 0])
        
        # Volatility
        state.append(min(latest['ATR'] / latest['Close'], 0.2) * 5 if latest['Close'] > 0 else 0)
        state.append(min(latest.get('Historical_Volatility', 0), 1.0))
        
        # Price action
        state.append(np.clip(latest['High_Low_Ratio'] - 1.0, 0, 0.5))
        state.append(np.clip(latest['Close_Open_Ratio'] - 1.0, -0.2, 0.2))
        
        # Recent performance
        returns_20d = df_window['Close'].tail(20).pct_change().dropna()
        if len(returns_20d) > 0:
            state.append(np.clip(returns_20d.mean(), -0.1, 0.1))
            state.append(min(returns_20d.std(), 0.1))
        else:
            state.extend([0, 0])
        
        # Pad or trim to exactly 50 features
        while len(state) < 50:
            state.append(0)
        state = state[:50]
        
        return np.array(state, dtype=np.float32)
    
    except Exception:
        return None

def process_stock_for_training(args):
    """
    Pre-processes a single stock's historical data to generate training samples.
    This function is designed to be run in parallel.
    """
    symbol, pattern_model = args
    try:
        df = fetch_data(symbol, period="3y", force_download=False)
        if df is None or len(df) < 200:
            return []

        df = compute_indicators_fast(df, symbol)
        if df.empty:
            return []

        experiences = []
        lookback_window = 90
        simulation_start = len(df) - 378 # Last 1.5 years for more data
        
        for i in range(max(simulation_start, lookback_window), len(df) - 10, 5):
            # Current state
            current_window = df.iloc[i-lookback_window:i]
            state = create_state_vector(current_window)
            if state is None:
                continue

            # Next state
            next_window = df.iloc[i+5-lookback_window:i+5]
            next_state = create_state_vector(next_window)
            if next_state is None:
                continue
            
            # CNN Pattern Prediction (on data up to the point of decision)
            df_for_pattern = df.iloc[:i]
            pattern, pattern_conf = predict_chart_pattern(pattern_model, df_for_pattern, lookback=60)

            # Actual outcome for reward calculation
            current_price = df.iloc[i-1]['Close']
            future_price = df.iloc[i+4]['Close']
            actual_return = (future_price - current_price) / current_price if current_price > 0 else 0

            experiences.append((state, next_state, pattern, pattern_conf, actual_return))
        
        return experiences
    except Exception:
        return []

def train_rl_agent_on_historical(stock_list, pattern_model, rl_agent=None, episodes=5):
    """
    Optimized training for the RL agent using pre-computed, parallelized data.
    """
    print(f"\n{'='*70}")
    print("OPTIMIZED REINFORCEMENT LEARNING AGENT TRAINING")
    print(f"{'='*70}\n")
    
    if rl_agent is None:
        rl_agent = StockTradingAgent(state_size=50, action_size=3)
    
    # --- Step 1: Parallel Data Pre-computation ---
    print("Phase 1: Pre-computing training data for all stocks...")
    all_experiences = []
    
    tasks = [(symbol, pattern_model) for symbol in stock_list[:150]] # Process 150 stocks
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_stock_for_training, tasks), total=len(tasks), desc="Processing Stocks"))

    for stock_experiences in results:
        all_experiences.extend(stock_experiences)

    if not all_experiences:
        print("⚠️ No training data could be generated. Aborting training.")
        return rl_agent

    print(f"\n✅ Pre-computation complete. Generated {len(all_experiences)} total training samples.")

    # --- Step 2: Agent Training Loop ---
    print("\nPhase 2: Training the RL agent...")
    for episode in range(episodes):
        random.shuffle(all_experiences) # Shuffle for each episode
        episode_reward = 0
        trades_made = 0
        
        pbar = tqdm(all_experiences, desc=f"Episode {episode + 1}/{episodes}")
        for state, next_state, pattern, pattern_conf, actual_return in pbar:
            # Agent takes action
            action = rl_agent.act(state, training=True)
            
            # Calculate predicted return and reward
            predicted_return = PATTERN_WEIGHTS.get(pattern, 0) * pattern_conf / 100.0
            reward = rl_agent.calculate_reward(action, actual_return, predicted_return, pattern_conf / 100.0)
            
            # Store experience in memory
            rl_agent.remember(state, action, reward, next_state, False)
            
            episode_reward += reward
            trades_made += 1
            
            # Train (replay) from memory periodically
            if trades_made % 10 == 0 and len(rl_agent.memory) > 64:
                loss = rl_agent.replay(batch_size=64)
                if loss is not None:
                    pbar.set_postfix(loss=f"{loss:.4f}", epsilon=f"{rl_agent.epsilon:.3f}")
        
        # Update target network at the end of each episode
        rl_agent.update_target_model()
        
        avg_reward = episode_reward / trades_made if trades_made > 0 else 0
        print(f"Episode {episode + 1} - Avg Reward: {avg_reward:.4f}, Epsilon: {rl_agent.epsilon:.4f}, Total Steps: {trades_made}")
        
        rl_agent.performance_log.append({
            'episode': episode + 1, 'avg_reward': avg_reward,
            'epsilon': rl_agent.epsilon, 'trades': trades_made
        })
    
    print(f"\n✅ RL Agent training complete!")
    print(f"Final epsilon: {rl_agent.epsilon:.4f}")
    
    return rl_agent


def analyze_growth_potential_with_rl(symbol, pattern_model, rl_agent=None):
    """Enhanced analysis with RL agent prediction"""
    try:
        df = fetch_data(symbol, period="2y", force_download=False)
        if df is None or len(df) < 200:
            return None
        
        df_ind = compute_indicators_fast(df, symbol=symbol)
        if df_ind.empty or len(df_ind) < 50:
            return None
        
        latest = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        recent = df_ind.tail(30)
        
        score = 0
        signals = {}
        
        # === TECHNICAL ANALYSIS (50 points) ===
        # RSI
        if 45 < latest['RSI'] < 65 and latest['RSI'] > df_ind['RSI'].tail(5).mean():
            score += 8
            signals['rsi'] = 'BUILDING_MOMENTUM'
        elif latest['RSI'] > 70:
            score -= 4 # Penalize overbought slightly
            signals['rsi'] = 'OVERBOUGHT'
        elif latest['RSI'] < 30:
            score += 3 # Reward potential reversal from oversold
            signals['rsi'] = 'OVERSOLD'
        
        # MACD
        macd_hist = latest['MACD'] - latest['MACD_signal']
        prev_macd_hist = prev['MACD'] - prev['MACD_signal']
        if macd_hist > 0 and prev_macd_hist <= 0:
            score += 10
            signals['macd'] = 'BULLISH_CROSS'
        elif macd_hist > 0 and macd_hist > prev_macd_hist:
            score += 6
            signals['macd'] = 'ACCELERATING'
        
        # Trend
        if latest['ADX'] > 25 and latest['DI_POS'] > latest['DI_NEG']:
            score += 8
            signals['trend'] = 'STRONG_UPTREND'
        
        if latest['Close'] > latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200']:
            is_overextended = (latest['Close'] - latest['EMA_20']) / latest['EMA_20'] > 0.08
            if not is_overextended:
                score += 10
                signals['ema_align'] = 'PERFECT'
        
        # Volume
        vol_ratio = recent['Volume'].tail(5).mean() / recent['Volume'].mean()
        price_change_5d = (latest['Close'] - df_ind['Close'].iloc[-5]) / df_ind['Close'].iloc[-5]
        if price_change_5d > 0.03 and vol_ratio > 1.5:
            score += 8
            signals['volume'] = 'SURGE_CONFIRM'
        
        # === CNN PATTERN (30 points) ===
        if pattern_model:
            pattern_name, pattern_conf = predict_chart_pattern(pattern_model, df_ind, lookback=60)
            pattern_weight = PATTERN_WEIGHTS.get(pattern_name, 0.0)
            signals['pattern'] = f"{pattern_name} ({pattern_conf:.1f}%)"
            
            if pattern_conf > 60:
                pattern_score = pattern_weight * 30
                score += pattern_score
        
        # === RL AGENT RECOMMENDATION (20 points) ===
        rl_action_str = 'N/A'
        if rl_agent:
            state = create_state_vector(df_ind.tail(90))
            if state is not None:
                rl_action = rl_agent.act(state, training=False)
                q_values = rl_agent.model.predict(state.reshape(1, -1), verbose=0)[0]
                q_values_softmax = tf.nn.softmax(q_values).numpy()
                rl_confidence = q_values_softmax[rl_action] * 100

                action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
                rl_action_str = f"{action_map[rl_action]} ({rl_confidence:.1f}%)"
                signals['rl_action'] = rl_action_str
                
                if rl_action == 0:  # BUY
                    score += 20 * (rl_confidence / 100)
                elif rl_action == 1:  # SELL
                    score -= 15 * (rl_confidence / 100)
        
        # === FUNDAMENTALS (bonus points) ===
        pe, roe = latest.get('PE_Ratio', 999), latest.get('ROE', 0)
        if 0 < pe < 30:
            score += 3
        if roe > 0.15:
            score += 3
        
        growth_score = max(0, min(100, score))
        
        # Estimated growth
        atr_pct = (latest['ATR'] / latest['Close']) * 100 if latest['Close'] > 0 else 0
        price_change_7d = ((latest['Close'] - df_ind['Close'].iloc[-7]) / df_ind['Close'].iloc[-7]) * 100
        estimated_growth = (price_change_7d * 0.5) + (growth_score / 100 * atr_pct * 2.5)
        
        return {
            'symbol': symbol,
            'growth_score': round(growth_score, 2),
            'current_price': round(latest['Close'], 2),
            'estimated_weekly_growth': round(estimated_growth, 2),
            'rsi': round(latest['RSI'], 2),
            'pattern': signals.get('pattern', 'N/A'),
            'rl_action': rl_action_str,
            'key_signals': ', '.join([f"{k}:{v}" for k, v in signals.items()])
        }
    
    except Exception as e:
        return None

def scan_growth_stocks_enhanced(stock_list, pattern_model, rl_agent, top_n=20, min_score=65):
    """Enhanced screening with RL"""
    print(f"\n{'='*80}")
    print(f"🚀 ENHANCED SCREENING WITH CNN + RL 🚀")
    print(f"{'='*80}\n")
    
    all_results = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Create a list of arguments for the executor map
        args_list = [(symbol, pattern_model, rl_agent) for symbol in stock_list]
        
        # We need a wrapper to unpack arguments for the executor
        def analyze_wrapper(p):
            return analyze_growth_potential_with_rl(*p)

        futures = executor.map(analyze_wrapper, args_list)
        
        for result in tqdm(futures, total=len(stock_list), desc="Analyzing"):
            if result:
                all_results.append(result)

    if not all_results:
        print("\n⚠️ No valid data processed.")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('growth_score', ascending=False)
    
    filtered = df_results[df_results['growth_score'] >= min_score]
    
    print(f"\n{'='*80}")
    print(f"📊 SCAN STATISTICS:")
    print(f"    Total analyzed: {len(all_results)}")
    print(f"    Meeting threshold (>={min_score}): {len(filtered)}")
    print(f"{'='*80}")
    
    display_df = filtered.head(top_n) if not filtered.empty else pd.DataFrame()
    
    if not display_df.empty:
        print(f"\n🏆 TOP {len(display_df)} HIGH-POTENTIAL STOCKS 🏆\n")
        print(display_df[['symbol', 'growth_score', 'estimated_weekly_growth', 
                          'current_price', 'rsi', 'pattern', 'rl_action']].to_string(index=False))
    else:
        print("\nNo stocks met the minimum growth score threshold.")

    return display_df

def perform_detailed_analysis(symbol, pattern_model, rl_agent):
    """Performs and prints a detailed analysis for a single stock."""
    print(f"\n{'='*80}")
    print(f"🔬 DETAILED ANALYSIS FOR: {symbol.upper()} 🔬")
    print(f"{'='*80}")

    analysis_result = analyze_growth_potential_with_rl(symbol, pattern_model, rl_agent)

    if not analysis_result:
        print(f"Could not perform analysis for {symbol}. Data might be unavailable or insufficient.")
        return

    # --- Overview ---
    print(f"\n--- 📈 OVERVIEW ---")
    print(f"  Symbol:                {analysis_result['symbol']}")
    print(f"  Current Price:         ₹{analysis_result['current_price']:.2f}")
    print(f"  Overall Growth Score:  {analysis_result['growth_score']}/100")
    print(f"  Est. Weekly Growth %:  {analysis_result['estimated_weekly_growth']:.2f}%")

    # --- Recommendation ---
    score = analysis_result['growth_score']
    if score > 80:
        recommendation = "STRONG BUY"
    elif score > 65:
        recommendation = "POTENTIAL BUY"
    elif score > 50:
        recommendation = "NEUTRAL / HOLD"
    else:
        recommendation = "AVOID / SELL"
    print(f"  Recommendation:        {recommendation}")

    # --- AI/ML Predictions ---
    print(f"\n--- 🤖 AI/ML PREDICTIONS ---")
    print(f"  CNN Pattern:           {analysis_result['pattern']}")
    print(f"  RL Agent Action:       {analysis_result['rl_action']}")

    # --- Key Signals ---
    print(f"\n--- 🔑 KEY SIGNALS DETECTED ---")
    signals = analysis_result['key_signals'].split(', ')
    if signals and signals[0]:
        for signal in signals:
            print(f"  - {signal.replace(':', ': ')}")
    else:
        print("  - No strong signals detected.")
        
    # --- Detailed Technicals ---
    df = fetch_data(symbol, period="1y")
    if df is not None:
        df = compute_indicators_fast(df, symbol)
        if not df.empty:
            latest = df.iloc[-1]
            print(f"\n--- 📊 TECHNICAL INDICATORS (Current Values) ---")
            print(f"  RSI (14):              {latest.get('RSI', 0):.2f}")
            print(f"  MACD Histogram:        {(latest.get('MACD', 0) - latest.get('MACD_signal', 0)):.2f}")
            print(f"  ADX (14):              {latest.get('ADX', 0):.2f} (Trend Strength)")
            print(f"  Price vs EMA20:        {'Above' if latest.get('Close', 0) > latest.get('EMA_20', 0) else 'Below'}")
            print(f"  EMA20 vs EMA50:        {'Bullish' if latest.get('EMA_20', 0) > latest.get('EMA_50', 0) else 'Bearish'}")
            vol_sma = latest.get('Volume_SMA', 0)
            vol_ratio = latest.get('Volume', 0) / vol_sma if vol_sma > 0 else 0
            print(f"  Volume (vs 20-day avg): {vol_ratio:.2f}x")
            print(f"  Bollinger Band Width:  {latest.get('BB_WIDTH', 0):.4f}")

    print(f"\n{'='*80}\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    mp.freeze_support()
    
    print("\n" + "="*70)
    print("🚀 ADVANCED STOCK ANALYSIS WITH CNN + RL 🚀")
    print("="*70)
    
    pattern_model_path = os.path.join(PATTERN_MODEL_DIR, "enhanced_cnn_pattern.keras")
    rl_model_path = RL_MODEL_DIR
    
    print("\nChoose mode:")
    print("1. Train CNN pattern model (Requires significant time and data)")
    print("2. Train RL agent (Requires a pre-trained CNN model)")
    print("3. Enhanced growth scanner (CNN + RL)")
    print("4. Analyze specific stock (CNN + RL)")
    
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    
    if choice == '1':
        nifty500 = get_nifty500_symbols()
        X, y = create_pattern_dataset_from_stocks(nifty500, samples_per_stock=15, lookback=60)
        
        if X is not None and len(X) > 100:
            model = train_pattern_recognition_model(X, y, model_path=pattern_model_path, epochs=100)
            if model:
                print(f"\n✅ Model saved to: {pattern_model_path}")
    
    elif choice == '2':
        if not os.path.exists(pattern_model_path):
            print("\n⚠️ CNN model not found. Please train it first (option 1).")
        else:
            pattern_model = tf.keras.models.load_model(pattern_model_path)
            nifty500 = get_nifty500_symbols()
            
            rl_agent = StockTradingAgent()
            if rl_agent.load(rl_model_path):
                 print("\nLoaded existing RL agent. Continuing training...")
            else:
                 print("\nCreating new RL agent.")

            rl_agent = train_rl_agent_on_historical(nifty500, pattern_model, rl_agent, episodes=10)
            rl_agent.save(rl_model_path)
            print(f"\n✅ RL agent saved to: {rl_model_path}")
    
    elif choice in ['3', '4']:
        # Load models once for scanning or single analysis
        pattern_model = None
        if os.path.exists(pattern_model_path):
            print(f"\nLoading CNN model from {pattern_model_path}...")
            pattern_model = tf.keras.models.load_model(pattern_model_path)
        else:
            print(f"\n⚠️ CNN model not found at {pattern_model_path}. Pattern analysis will be skipped.")

        rl_agent = StockTradingAgent()
        if os.path.exists(os.path.join(rl_model_path, 'rl_model.keras')):
            print(f"Loading RL agent from {rl_model_path}...")
            rl_agent.load(rl_model_path)
        else:
            print(f"\n⚠️ RL agent not found at {rl_model_path}. RL-based analysis will be skipped.")
            rl_agent = None

        if choice == '3':
            nifty500 = get_nifty500_symbols()
            results = scan_growth_stocks_enhanced(nifty500, pattern_model, rl_agent, top_n=20, min_score=65)
            
            if not results.empty:
                while True:
                    analyze = input("\nAnalyze a stock from the list for details? (Enter symbol or 'n' to exit): ").strip().upper()
                    if analyze == 'N':
                        break
                    if analyze in results['symbol'].values:
                        perform_detailed_analysis(analyze, pattern_model, rl_agent)
                    elif analyze:
                        print("Symbol not in the top list. Please enter a valid symbol or 'n'.")

        elif choice == '4':
            while True:
                symbol = input("\nEnter stock symbol (or 'q' to quit): ").strip().upper()
                if symbol == 'Q':
                    break
                if symbol:
                    perform_detailed_analysis(symbol, pattern_model, rl_agent)

    else:
        print("Invalid choice. Please run the script again.")