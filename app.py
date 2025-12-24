"""
Artha Drishti - ML Based Advanced Stock Screener
Features: All NSE stocks, Custom strategies, Portfolio analysis, News sentiment, Price predictions
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ta
from datetime import datetime, timedelta
import json
import time
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import requests
from io import StringIO
import concurrent.futures
from functools import lru_cache
from textblob import TextBlob
import re
from scipy import stats

warnings.filterwarnings("ignore")

# NewsAPI Configuration
NEWS_API_KEY = "fe303478-aba4-479e-8bdb-b2b8feb2db64"  # Replace with your NewsAPI key from newsapi.org
NEWS_API_URL = "https://newsapi.org/v2/everything"

# ============================================================================
# NEWS SENTIMENT ANALYZER
# ============================================================================

class NewsSentimentAnalyzer:
    """Analyze news sentiment for stocks"""
    
    def __init__(self, api_key):
        self.api_key = api_key
    
    def get_news(self, symbol, company_name, days=7):
        """Fetch news for a stock"""
        try:
            # Search query
            query = f"{company_name} OR {symbol}"
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 10,
                'apiKey': self.api_key
            }
            
            response = requests.get(NEWS_API_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                return articles
            else:
                return []
        except:
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = "Positive"
                score = polarity
            elif polarity < -0.1:
                sentiment = "Negative"
                score = polarity
            else:
                sentiment = "Neutral"
                score = polarity
            
            return sentiment, score
        except:
            return "Neutral", 0
    
    def analyze_stock_news(self, symbol, company_name):
        """Complete news sentiment analysis"""
        articles = self.get_news(symbol, company_name)
        
        if not articles:
            return {
                'overall_sentiment': 'Neutral',
                'sentiment_score': 0,
                'news_count': 0,
                'articles': [],
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiments = []
        article_data = []
        
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}"
            
            sentiment, score = self.analyze_sentiment(text)
            sentiments.append(score)
            
            article_data.append({
                'title': title,
                'description': description,
                'sentiment': sentiment,
                'score': score,
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', '')
            })
        
        # Calculate overall sentiment
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        
        if avg_sentiment > 0.1:
            overall = "Positive"
        elif avg_sentiment < -0.1:
            overall = "Negative"
        else:
            overall = "Neutral"
        
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        return {
            'overall_sentiment': overall,
            'sentiment_score': avg_sentiment,
            'news_count': len(articles),
            'articles': article_data,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }

# ============================================================================
# ADVANCED ML-BASED PRICE PREDICTOR
# ============================================================================

class PricePredictor:
    """Advanced ML-based price prediction with target, buy, sell, and stop loss using ensemble methods"""
    
    def __init__(self):
        self.model_price = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, 
                                                     max_depth=5, random_state=42)
        self.model_buy = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        self.model_stop = GradientBoostingRegressor(n_estimators=80, learning_rate=0.08, 
                                                    max_depth=5, random_state=42)
        self.model_direction = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.scaler = StandardScaler()
        self.models_trained = False
    
    def _extract_comprehensive_features(self, df, sentiment_score=0):
        """Extract comprehensive features from price data"""
        if len(df) < 60:
            return None
        
        features = pd.DataFrame(index=df.index)
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        try:
            # Price-based features
            features['price_normalized'] = close / close.rolling(50).mean()
            features['returns_1d'] = close.pct_change(1)
            features['returns_5d'] = close.pct_change(5)
            features['returns_10d'] = close.pct_change(10)
            features['returns_20d'] = close.pct_change(20)
            
            # Volatility features
            features['volatility_10'] = close.pct_change().rolling(10).std()
            features['volatility_20'] = close.pct_change().rolling(20).std()
            features['volatility_30'] = close.pct_change().rolling(30).std()
            
            # Momentum indicators
            features['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
            features['rsi_21'] = ta.momentum.RSIIndicator(close, window=21).rsi()
            features['rsi_slope'] = features['rsi_14'].diff(5)
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(high, low, close)
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()
            
            # MACD
            macd = ta.trend.MACD(close)
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()
            features['macd_slope'] = features['macd'].diff(3)
            
            # ADX
            adx = ta.trend.ADXIndicator(high, low, close, window=14)
            features['adx'] = adx.adx()
            features['adx_pos'] = adx.adx_pos()
            features['adx_neg'] = adx.adx_neg()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            features['bb_upper'] = bb.bollinger_hband()
            features['bb_lower'] = bb.bollinger_lband()
            features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # ATR
            atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
            features['atr'] = atr.average_true_range()
            features['atr_pct'] = features['atr'] / close
            
            # Volume features
            features['volume_ratio'] = volume / volume.rolling(20).mean()
            features['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()
            features['volume_change'] = volume.pct_change()
            
            # Trend features
            sma_12 = close.rolling(12).mean()
            sma_26 = close.rolling(26).mean()
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()
            
            features['price_sma12'] = close / sma_12
            features['price_sma26'] = close / sma_26
            features['price_sma50'] = close / sma_50
            features['price_sma200'] = close / sma_200
            features['sma12_sma26'] = sma_12 / sma_26
            features['sma50_sma200'] = sma_50 / sma_200
            
            # Support and Resistance
            features['support'] = low.rolling(20).min()
            features['resistance'] = high.rolling(20).max()
            features['distance_support'] = (close - features['support']) / features['support']
            features['distance_resistance'] = (features['resistance'] - close) / features['resistance']
            
            # Pattern detection - Higher High/Low patterns
            features['hh_ll'] = ((high > high.shift(1)) & (low > low.shift(1))).astype(int)
            features['lh_ll'] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
            
            # Range and volatility patterns
            high_low_range = high - low
            features['candle_range'] = high_low_range / close
            features['candle_body'] = abs(close - df['Open']) / close
            
            # CCI (Commodity Channel Index)
            cci = ta.trend.CCIIndicator(high, low, close, window=20)
            features['cci'] = cci.cci()
            
            # Sentiment integration
            features['sentiment_factor'] = sentiment_score
            
            # Rate of Change
            features['roc'] = ta.momentum.ROCIndicator(close, window=12).roc()
            
            # Williams %R
            williams = ta.momentum.WilliamsRIndicator(high, low, close, lperiod=14)
            features['williams_r'] = williams.williams_r()
            
            # Fill NaN values
            features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            return features
        
        except Exception as e:
            return None
    
    def _create_labels(self, df, lookahead=5):
        """Create labels for price targets"""
        labels_price = np.zeros(len(df))
        labels_buy = np.zeros(len(df))
        labels_stop = np.zeros(len(df))
        labels_direction = np.zeros(len(df))
        
        for i in range(len(df) - lookahead):
            current_price = df['Close'].iloc[i]
            future_high = df['High'].iloc[i:i+lookahead].max()
            future_low = df['Low'].iloc[i:i+lookahead].min()
            future_close = df['Close'].iloc[i+lookahead]
            
            labels_price[i] = future_high
            labels_buy[i] = future_low * 0.98
            labels_stop[i] = future_low * 0.95
            labels_direction[i] = 1 if future_close > current_price else 0
        
        return labels_price, labels_buy, labels_stop, labels_direction
    
    def _train_models(self, df, sentiment_score=0):
        """Train ensemble models on historical data"""
        try:
            if len(df) < 100:
                return False
            
            features = self._extract_comprehensive_features(df, sentiment_score)
            if features is None or features.empty:
                return False
            
            # Create labels
            labels_price, labels_buy, labels_stop, labels_direction = self._create_labels(df, lookahead=5)
            
            # Prepare training data
            valid_idx = ~(features.isna().any(axis=1) | pd.isna(labels_price))
            features_train = features[valid_idx].iloc[:-5]
            labels_price_train = labels_price[valid_idx][:-5]
            labels_buy_train = labels_buy[valid_idx][:-5]
            labels_stop_train = labels_stop[valid_idx][:-5]
            labels_direction_train = labels_direction[valid_idx][:-5]
            
            if len(features_train) < 50:
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(features_train)
            
            # Train models with Time Series Split
            tscv = TimeSeriesSplit(n_splits=3)
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_price_train, y_price_val = labels_price_train.iloc[train_idx], labels_price_train.iloc[val_idx]
                y_buy_train, y_buy_val = labels_buy_train.iloc[train_idx], labels_buy_train.iloc[val_idx]
                y_stop_train, y_stop_val = labels_stop_train.iloc[train_idx], labels_stop_train.iloc[val_idx]
                y_dir_train, y_dir_val = labels_direction_train.iloc[train_idx], labels_direction_train.iloc[val_idx]
                
                self.model_price.fit(X_train, y_price_train)
                self.model_buy.fit(X_train, y_buy_train)
                self.model_stop.fit(X_train, y_stop_train)
                self.model_direction.fit(X_train, y_dir_train)
            
            self.models_trained = True
            return True
        
        except Exception as e:
            return False
    
    def calculate_support_resistance(self, df, window=20):
        """Calculate support and resistance levels using multiple windows"""
        highs = df['High'].rolling(window).max()
        lows = df['Low'].rolling(window).min()
        
        resistance = highs.iloc[-1]
        support = lows.iloc[-1]
        
        return support, resistance
    
    def calculate_pivot_points(self, df):
        """Calculate pivot points"""
        high = df['High'].iloc[-1]
        low = df['Low'].iloc[-1]
        close = df['Close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            's1': s1,
            's2': s2
        }
    
    def calculate_atr_stops(self, df, multiplier=2):
        """Calculate ATR-based stop loss"""
        atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        current_price = df['Close'].iloc[-1]
        atr_value = atr.iloc[-1]
        
        stop_loss = current_price - (multiplier * atr_value)
        take_profit = current_price + (multiplier * atr_value * 1.5)
        
        return stop_loss, take_profit
    
    def predict_target_price(self, df, sentiment_score=0, fundamental_score=50):
        """Predict target price with ML models + technical analysis"""
        try:
            current_price = df['Close'].iloc[-1]
            
            if len(df) < 60:
                return None
            
            # Train models on historical data
            if not self.models_trained:
                self._train_models(df, sentiment_score)
            
            # Extract features for current state
            features = self._extract_comprehensive_features(df, sentiment_score)
            if features is None or features.empty:
                return None
            
            latest_features = features.iloc[-1:].values
            X_scaled = self.scaler.transform(latest_features)
            
            # Make predictions
            if self.models_trained:
                ml_target_price = self.model_price.predict(X_scaled)[0]
                ml_buy_price = self.model_buy.predict(X_scaled)[0]
                ml_stop_loss = self.model_stop.predict(X_scaled)[0]
                direction_pred = self.model_direction.predict(X_scaled)[0]
                direction_prob = self.model_direction.predict_proba(X_scaled)[0]
            else:
                ml_target_price = current_price * 1.05
                ml_buy_price = current_price * 0.98
                ml_stop_loss = current_price * 0.95
                direction_pred = 1
                direction_prob = [0.5, 0.5]
            
            # Technical Analysis Component
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            trend_strength = (current_price - sma_50) / sma_50 * 100
            
            rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
            rsi_score = (rsi - 50) / 50
            
            macd = ta.trend.MACD(df['Close'])
            macd_value = macd.macd().iloc[-1]
            macd_signal = macd.macd_signal().iloc[-1]
            macd_bullish = 1 if macd_value > macd_signal else -1
            
            volatility = df['Close'].pct_change().std() * np.sqrt(252)
            
            support, resistance = self.calculate_support_resistance(df)
            distance_to_resistance = (resistance - current_price) / current_price * 100
            
            pivots = self.calculate_pivot_points(df)
            atr_stop, atr_target = self.calculate_atr_stops(df)
            
            # Technical score
            technical_score = (
                (trend_strength / 10) * 0.3 +
                rsi_score * 0.2 +
                macd_bullish * 0.3 +
                (distance_to_resistance / 10) * 0.2
            ) * 100
            
            # Sentiment Component
            sentiment_component = sentiment_score * 100
            
            # Fundamental Component
            fundamental_component = (fundamental_score - 50) / 50 * 100
            
            # Combined Score
            combined_score = (
                technical_score * 0.35 +
                sentiment_component * 0.25 +
                fundamental_component * 0.25 +
                direction_prob[1] * 40
            )
            
            # Blend ML and technical predictions
            target_price = (ml_target_price * 0.6 + (current_price * (1 + combined_score / 100 * volatility)) * 0.4)
            conservative_target = target_price * 0.85
            aggressive_target = target_price * 1.15
            
            buy_price = (ml_buy_price * 0.5 + min(current_price * 0.98, pivots['s1']) * 0.5)
            sell_price = max(target_price * 1.05, pivots['r1'])
            
            stop_loss = (ml_stop_loss * 0.5 + max(atr_stop, buy_price * 0.97) * 0.5)
            
            # Calculate confidence
            ml_confidence = direction_prob[1] * 70 + 15
            
            technical_confidence = 0
            if 40 < rsi < 60:
                technical_confidence += 20
            elif 30 < rsi < 70:
                technical_confidence += 15
            else:
                technical_confidence += 5
            
            if trend_strength > 5:
                technical_confidence += 20
            elif trend_strength > 0:
                technical_confidence += 10
            
            if macd_bullish > 0:
                technical_confidence += 15
            else:
                technical_confidence += 5
            
            volume_ratio = df['Volume'].iloc[-5:].mean() / df['Volume'].mean()
            if volume_ratio > 1.2:
                technical_confidence += 10
            elif volume_ratio > 0.8:
                technical_confidence += 5
            
            confidence = (ml_confidence * 0.6 + technical_confidence * 0.4)
            
            if volatility > 0.5:
                confidence -= 10
            
            confidence = min(max(confidence, 20), 95)
            
            time_horizon = int(30 / (volatility + 0.1))
            time_horizon = min(max(time_horizon, 7), 90)
            
            expected_return = ((target_price - current_price) / current_price) * 100
            
            return {
                'current_price': current_price,
                'target_price': target_price,
                'conservative_target': conservative_target,
                'aggressive_target': aggressive_target,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'time_horizon': time_horizon,
                'expected_return': expected_return,
                'technical_score': technical_score,
                'sentiment_score': sentiment_component,
                'fundamental_score': fundamental_component,
                'support': support,
                'resistance': resistance,
                'pivot_points': pivots,
                'atr_stop': atr_stop,
                'atr_target': atr_target,
                'risk_reward': abs((target_price - buy_price) / (buy_price - stop_loss)) if (buy_price - stop_loss) > 0 else 0
            }
        
        except Exception as e:
            return None

# ============================================================================
# DATA LOADER
# ============================================================================

@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_nse_stocks():
    """Load all NSE stock symbols"""
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(StringIO(response.text))
        symbols = df['SYMBOL'].str.strip().tolist()
        return sorted(symbols)
    except Exception as e:
        st.error(f"Failed to load NSE stocks: {e}")
        # Fallback to sample stocks
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", 
                "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
                "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI"]

@st.cache_data(ttl=86400)
def load_nifty50():
    """Load NIFTY50 stocks"""
    nifty50 = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR", 
               "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", 
               "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "WIPRO", "POWERINDIA",
               "JSWSTEEL", "BAJAJFINSV", "HDFC", "LUPIN", "TECHM", "ULTRACEMCO",
               "DRREDDY", "ADANIPORTS", "ADANIGREEN", "ADANITRANS", "TITAN", "HCLTECH",
               "NESTLEIND", "BRITANNIA", "SIEMENS", "CUMMINSIND", "INDIGO", "DIVISLAB",
               "SHREECEM", "EICHERMOT", "HEROMOTOCO", "SBICARD", "MUTHOOTFIN", "LTIM",
               "PAGEIND", "TATASTEEL", "M&M", "BOSCHLTD", "GAIL", "IOC"]
    return sorted(nifty50)

@st.cache_data(ttl=86400)
def load_nifty500():
    """Load NIFTY500 stocks (top 500 by market cap)"""
    try:
        # Try to fetch from NSE, filter to NIFTY500
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(StringIO(response.text))
        symbols = df['SYMBOL'].str.strip().tolist()
        return sorted(symbols[:500])  # Return top 500
    except:
        # Fallback to NIFTY50 + additional stocks
        nifty50 = load_nifty50()
        additional = ["AARTI", "ABB", "ACRYSIL", "ADORWELD", "ADVANIHOTELS", 
                     "AEGISCHEM", "AGRITECH", "AMBUJACEM", "ANIKINDUSTRY", "APARINDS",
                     "APOLLOHOSP", "APOLLOTYRE", "ARVINDFARM", "ARVIND", "ASAHISONG"]
        return sorted(nifty50 + additional)

@lru_cache(maxsize=500)
def get_stock_data_cached(symbol, period="2y"):
    """Cached stock data retrieval"""
    try:
        ticker = yf.Ticker(symbol + ".NS")
        df = ticker.history(period=period)
        return df if not df.empty and len(df) > 100 else None
    except:
        return None

# ============================================================================
# STRATEGY ENGINE
# ============================================================================

class StrategyEngine:
    """Custom screening strategy builder"""
    
    AVAILABLE_INDICATORS = {
        'MACD': {'type': 'trend', 'params': ['fast', 'slow', 'signal']},
        'RSI': {'type': 'momentum', 'params': ['period', 'overbought', 'oversold']},
        'ADX': {'type': 'trend', 'params': ['period', 'threshold']},
        'Bollinger': {'type': 'volatility', 'params': ['period', 'std']},
        'Volume': {'type': 'volume', 'params': ['period', 'multiplier']},
        'SMA_Cross': {'type': 'trend', 'params': ['fast', 'slow']},
        'EMA_Cross': {'type': 'trend', 'params': ['fast', 'slow']},
        'Stochastic': {'type': 'momentum', 'params': ['k', 'd', 'smooth']},
        'CCI': {'type': 'momentum', 'params': ['period', 'threshold']},
        'ATR': {'type': 'volatility', 'params': ['period']},
    }
    
    def __init__(self):
        self.strategies = self.load_default_strategies()
    
    def load_default_strategies(self):
        """Load default strategies"""
        return {
            'Swing Trader': {
                'description': 'Medium-term momentum with trend confirmation',
                'rules': [
                    {'indicator': 'MACD', 'condition': 'bullish_cross', 'weight': 2.0},
                    {'indicator': 'RSI', 'condition': 'between', 'min': 50, 'max': 70, 'weight': 1.5},
                    {'indicator': 'ADX', 'condition': 'above', 'threshold': 25, 'weight': 1.5},
                    {'indicator': 'Volume', 'condition': 'above_avg', 'multiplier': 1.2, 'weight': 1.0},
                ]
            },
            'Momentum Hunter': {
                'description': 'High momentum stocks breaking out',
                'rules': [
                    {'indicator': 'RSI', 'condition': 'above', 'threshold': 60, 'weight': 2.0},
                    {'indicator': 'MACD', 'condition': 'bullish', 'weight': 1.5},
                    {'indicator': 'Volume', 'condition': 'above_avg', 'multiplier': 1.5, 'weight': 2.0},
                    {'indicator': 'SMA_Cross', 'condition': 'golden', 'fast': 20, 'slow': 50, 'weight': 1.5},
                ]
            },
            'Value Bounce': {
                'description': 'Oversold stocks with reversal potential',
                'rules': [
                    {'indicator': 'RSI', 'condition': 'below', 'threshold': 40, 'weight': 2.0},
                    {'indicator': 'Bollinger', 'condition': 'near_lower', 'weight': 1.5},
                    {'indicator': 'MACD', 'condition': 'turning_up', 'weight': 1.5},
                    {'indicator': 'Volume', 'condition': 'above_avg', 'multiplier': 1.0, 'weight': 1.0},
                ]
            },
            'Trend Rider': {
                'description': 'Strong trending stocks',
                'rules': [
                    {'indicator': 'ADX', 'condition': 'above', 'threshold': 30, 'weight': 2.5},
                    {'indicator': 'EMA_Cross', 'condition': 'bullish', 'fast': 12, 'slow': 26, 'weight': 2.0},
                    {'indicator': 'RSI', 'condition': 'above', 'threshold': 55, 'weight': 1.5},
                    {'indicator': 'Volume', 'condition': 'trending_up', 'weight': 1.0},
                ]
            }
        }
    
    def evaluate_strategy(self, df, strategy_name):
        """Evaluate a strategy on stock data"""
        if strategy_name not in self.strategies:
            return None
        
        strategy = self.strategies[strategy_name]
        df = compute_indicators(df)
        
        if df is None or len(df) < 50:
            return None
        
        total_score = 0
        total_weight = 0
        conditions_met = []
        
        for rule in strategy['rules']:
            indicator = rule['indicator']
            condition = rule['condition']
            weight = rule.get('weight', 1.0)
            
            met = self._evaluate_rule(df, rule)
            
            if met:
                total_score += weight
                conditions_met.append(f"{indicator}:{condition}")
            
            total_weight += weight
        
        confidence = (total_score / total_weight * 100) if total_weight > 0 else 0
        
        return {
            'confidence': confidence,
            'conditions_met': len(conditions_met),
            'total_conditions': len(strategy['rules']),
            'details': conditions_met
        }
    
    def _evaluate_rule(self, df, rule):
        """Evaluate a single rule"""
        try:
            indicator = rule['indicator']
            condition = rule['condition']
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            if indicator == 'MACD':
                if condition == 'bullish_cross':
                    return latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']
                elif condition == 'bullish':
                    return latest['MACD'] > 0
                elif condition == 'turning_up':
                    return latest['MACD'] > prev['MACD']
            
            elif indicator == 'RSI':
                if condition == 'above':
                    return latest['RSI'] > rule.get('threshold', 50)
                elif condition == 'below':
                    return latest['RSI'] < rule.get('threshold', 50)
                elif condition == 'between':
                    return rule.get('min', 0) < latest['RSI'] < rule.get('max', 100)
            
            elif indicator == 'ADX':
                if condition == 'above':
                    return latest['ADX'] > rule.get('threshold', 25)
            
            elif indicator == 'Bollinger':
                if condition == 'near_lower':
                    distance = (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
                    return distance < 0.2
            
            elif indicator == 'Volume':
                if condition == 'above_avg':
                    return latest['Volume'] > latest['Vol_MA'] * rule.get('multiplier', 1.0)
                elif condition == 'trending_up':
                    vol_5 = df['Volume'].iloc[-5:].mean()
                    vol_20 = df['Volume'].iloc[-20:].mean()
                    return vol_5 > vol_20 * 1.1
            
            elif indicator == 'SMA_Cross':
                if condition == 'golden':
                    fast = rule.get('fast', 20)
                    slow = rule.get('slow', 50)
                    sma_fast = df['Close'].rolling(fast).mean()
                    sma_slow = df['Close'].rolling(slow).mean()
                    return sma_fast.iloc[-1] > sma_slow.iloc[-1] and sma_fast.iloc[-2] <= sma_slow.iloc[-2]
            
            elif indicator == 'EMA_Cross':
                if condition == 'bullish':
                    fast = rule.get('fast', 12)
                    slow = rule.get('slow', 26)
                    ema_fast = df['Close'].ewm(span=fast).mean()
                    ema_slow = df['Close'].ewm(span=slow).mean()
                    return ema_fast.iloc[-1] > ema_slow.iloc[-1]
            
            return False
        except:
            return False

# ============================================================================
# ML PREDICTOR (Enhanced)
# ============================================================================

class EnhancedMLPredictor:
    """Enhanced ML predictor with ensemble methods"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
        }
        self.scaler = StandardScaler()
    
    def create_advanced_features(self, df):
        """Create advanced features"""
        features = pd.DataFrame(index=df.index)
        
        try:
            features['returns_1d'] = df['Close'].pct_change(1)
            features['returns_5d'] = df['Close'].pct_change(5)
            features['returns_20d'] = df['Close'].pct_change(20)
            
            features['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
            features['volatility_30d'] = df['Close'].pct_change().rolling(30).std()
            
            features['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            features['rsi_slope'] = features['rsi'].diff(5)
            
            macd = ta.trend.MACD(df['Close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()
            
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
            features['adx'] = adx.adx()
            
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            return features.fillna(method='ffill').fillna(0)
        except:
            return pd.DataFrame()
    
    def predict(self, symbol):
        """Make prediction for a symbol"""
        try:
            df = get_stock_data_cached(symbol)
            if df is None or len(df) < 150:
                return None
            
            features = self.create_advanced_features(df)
            if features.empty:
                return None
            
            # Simple prediction based on recent trend
            recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1)
            
            if recent_return > 0.05:
                prediction = 2
                confidence = min(abs(recent_return) * 1000, 85)
            elif recent_return > 0.02:
                prediction = 1
                confidence = min(abs(recent_return) * 800, 70)
            else:
                prediction = 0
                confidence = 50
            
            return {
                'prediction': prediction,
                'confidence': confidence
            }
        except:
            return None

# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================

class PortfolioManager:
    """Portfolio analysis and management"""
    
    def __init__(self):
        self.portfolio = []
    
    def add_stock(self, symbol, quantity, buy_price):
        """Add stock to portfolio"""
        self.portfolio.append({
            'symbol': symbol,
            'quantity': quantity,
            'buy_price': buy_price,
            'buy_date': datetime.now().strftime('%Y-%m-%d')
        })
    
    def remove_stock(self, symbol):
        """Remove stock from portfolio"""
        self.portfolio = [s for s in self.portfolio if s['symbol'] != symbol]
    
    def analyze_portfolio(self):
        """Analyze entire portfolio"""
        if not self.portfolio:
            return None
        
        results = []
        total_invested = 0
        total_current = 0
        
        for stock in self.portfolio:
            symbol = stock['symbol']
            quantity = stock['quantity']
            buy_price = stock['buy_price']
            
            df = get_stock_data_cached(symbol)
            if df is None:
                continue
            
            current_price = df['Close'].iloc[-1]
            invested = quantity * buy_price
            current_value = quantity * current_price
            pnl = current_value - invested
            pnl_pct = (pnl / invested) * 100
            
            total_invested += invested
            total_current += current_value
            
            results.append({
                'Symbol': symbol,
                'Quantity': quantity,
                'Buy Price': buy_price,
                'Current Price': current_price,
                'Invested': invested,
                'Current Value': current_value,
                'P&L': pnl,
                'P&L %': pnl_pct,
                'Buy Date': stock['buy_date']
            })
        
        summary = {
            'total_invested': total_invested,
            'total_current': total_current,
            'total_pnl': total_current - total_invested,
            'total_pnl_pct': ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0,
            'stocks': results
        }
        
        return summary

# ============================================================================
# STOCK ANALYZER
# ============================================================================

class StockAnalyzer:
    """Comprehensive stock analysis"""
    
    def analyze_stock(self, symbol):
        """Comprehensive stock analysis"""
        try:
            ticker = yf.Ticker(symbol + ".NS")
            df = ticker.history(period="2y")
            
            if df.empty:
                return None
            
            df = compute_indicators(df)
            info = ticker.info
            
            # Calculate metrics
            latest = df.iloc[-1]
            
            # Price metrics
            week_52_high = df['High'].rolling(252).max().iloc[-1]
            week_52_low = df['Low'].rolling(252).min().iloc[-1]
            current_price = latest['Close']
            
            # Returns
            returns_1m = (current_price / df['Close'].iloc[-21] - 1) * 100 if len(df) > 21 else 0
            returns_3m = (current_price / df['Close'].iloc[-63] - 1) * 100 if len(df) > 63 else 0
            returns_6m = (current_price / df['Close'].iloc[-126] - 1) * 100 if len(df) > 126 else 0
            returns_1y = (current_price / df['Close'].iloc[-252] - 1) * 100 if len(df) > 252 else 0
            
            # Volatility
            volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
            
            # Volume
            avg_volume = df['Volume'].mean()
            current_volume = latest['Volume']
            volume_ratio = current_volume / avg_volume
            
            # Technical signals
            rsi = latest['RSI']
            macd = latest['MACD']
            macd_signal = latest['MACD_Signal']
            adx = latest['ADX']
            
            # AI Recommendation
            recommendation = self.generate_recommendation(df, info)
            
            analysis = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'current_price': current_price,
                'week_52_high': week_52_high,
                'week_52_low': week_52_low,
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'returns_1m': returns_1m,
                'returns_3m': returns_3m,
                'returns_6m': returns_6m,
                'returns_1y': returns_1y,
                'volatility': volatility,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'adx': adx,
                'recommendation': recommendation,
                'df': df
            }
            
            return analysis
            
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def generate_recommendation(self, df, info):
        """Generate AI recommendation"""
        try:
            latest = df.iloc[-1]
            
            score = 0
            reasons = []
            
            # RSI analysis
            if latest['RSI'] > 70:
                score -= 2
                reasons.append("Overbought (RSI > 70)")
            elif latest['RSI'] < 30:
                score += 2
                reasons.append("Oversold (RSI < 30)")
            elif 50 < latest['RSI'] < 60:
                score += 1
                reasons.append("Healthy RSI")
            
            # MACD analysis
            if latest['MACD'] > latest['MACD_Signal']:
                score += 2
                reasons.append("Bullish MACD")
            else:
                score -= 1
                reasons.append("Bearish MACD")
            
            # ADX analysis
            if latest['ADX'] > 25:
                score += 1
                reasons.append("Strong trend")
            
            # Volume analysis
            volume_ratio = latest['Volume'] / df['Volume'].mean()
            if volume_ratio > 1.5:
                score += 1
                reasons.append("High volume")
            
            # Price vs SMA
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            if latest['Close'] > sma_50:
                score += 1
                reasons.append("Above 50 SMA")
            else:
                score -= 1
                reasons.append("Below 50 SMA")
            
            # Generate recommendation
            if score >= 5:
                rec = "STRONG BUY"
                color = "success"
            elif score >= 3:
                rec = "BUY"
                color = "success"
            elif score >= 1:
                rec = "HOLD"
                color = "warning"
            elif score >= -2:
                rec = "SELL"
                color = "warning"
            else:
                rec = "STRONG SELL"
                color = "error"
            
            return {
                'action': rec,
                'score': score,
                'reasons': reasons,
                'color': color
            }
            
        except:
            return {
                'action': 'HOLD',
                'score': 0,
                'reasons': ['Insufficient data'],
                'color': 'warning'
            }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_indicators(df):
    """Compute technical indicators"""
    if df is None or len(df) < 30:
        return df
    
    try:
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['+DMI'] = adx.adx_pos()
        df['-DMI'] = adx.adx_neg()
        df['ADX'] = adx.adx()
        
        df['Vol_MA'] = df['Volume'].rolling(20).mean()
    except:
        pass
    
    return df

def format_number(num):
    """Format large numbers"""
    if num >= 1e12:
        return f"₹{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"₹{num/1e9:.2f}B"
    elif num >= 1e7:
        return f"₹{num/1e7:.2f}Cr"
    elif num >= 1e5:
        return f"₹{num/1e5:.2f}L"
    else:
        return f"₹{num:.2f}"

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Artha Drishti - ML Stock Screener",
    page_icon="C:\\Users\\Lenovo\\Documents\\VS CODE codes(files)\\helloworld\\BTP SEM-5\\arthadrishti_logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with logo and background
st.markdown("""
    <style>
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        }
        50% {
            box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
        }
    }
    
    * {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 25%, #16213e 50%, #0f3460 75%, #0a0e27 100%);
        min-height: 100vh;
        background-attachment: fixed;
        position: relative;
    }
    
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(118, 75, 162, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: transparent;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .stMarkdown {
        transition: all 0.3s ease;
    }
    
    p, span, label {
        color: #ffffff !important;
    }
    
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #e0e0ff !important;
        font-weight: 500;
    }
    
    input, textarea, select {
        background: rgba(102, 126, 234, 0.08) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    input:hover, textarea:hover, select:hover {
        border-color: rgba(102, 126, 234, 0.4) !important;
        background: rgba(102, 126, 234, 0.12) !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.1) !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.15) !important;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.2) !important;
        outline: none !important;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 2rem 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .header-container:hover {
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.3),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.1);
        transform: translateY(-4px);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.18) 0%, rgba(118, 75, 162, 0.18) 100%);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .logo-image {
        max-width: 280px;
        height: auto;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        filter: drop-shadow(0 12px 30px rgba(102, 126, 234, 0.35)) 
                brightness(1.05);
        animation: float 3s ease-in-out infinite;
    }
    
    .logo-image:hover {
        transform: scale(1.12) rotate(2deg);
        filter: drop-shadow(0 20px 40px rgba(102, 126, 234, 0.5)) 
                brightness(1.15);
        animation: glow 2s ease-in-out infinite;
    }
    
    img {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .header-text {
        flex: 1;
        margin-left: 2rem;
        color: white;
        transition: all 0.3s ease;
    }
    .header-text h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -1px;
        transition: all 0.3s ease;
    }
    .header-text p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
        transition: opacity 0.3s ease;
        font-weight: 500;
    }
    
    .nav-menu {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .nav-button {
        flex: 1;
        min-width: 180px;
        padding: 14px 24px;
        border: 1.5px solid rgba(102, 126, 234, 0.25);
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        color: #e0e0ff;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(5px);
        position: relative;
        overflow: hidden;
    }
    
    .nav-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .nav-button:hover::before {
        left: 100%;
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        transform: translateX(6px) translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.25),
                    inset 0 1px 1px rgba(255, 255, 255, 0.05);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border-color: #667eea;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5),
                    inset 0 1px 1px rgba(255, 255, 255, 0.1);
        transform: scale(1.02);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .logo-container {
        text-align: center;
        padding: 2rem 0;
    }
    .tagline {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-top: -1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.08) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.25);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.05);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.3),
                    inset 0 1px 1px rgba(255, 255, 255, 0.1);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.18) 0%, rgba(118, 75, 162, 0.14) 100%);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .recommendation-strong-buy { 
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; 
        padding: 0.75rem 1.25rem; 
        border-radius: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .recommendation-strong-buy:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
    }
    
    .recommendation-buy { 
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        color: white; 
        padding: 0.75rem 1.25rem; 
        border-radius: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(52, 211, 153, 0.3);
    }
    
    .recommendation-buy:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(52, 211, 153, 0.4);
    }
    
    .recommendation-hold { 
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white; 
        padding: 0.75rem 1.25rem; 
        border-radius: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.3);
    }
    
    .recommendation-hold:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(251, 191, 36, 0.4);
    }
    
    .recommendation-sell { 
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white; 
        padding: 0.75rem 1.25rem; 
        border-radius: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
    }
    
    .recommendation-sell:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(249, 115, 22, 0.4);
    }
    
    .recommendation-strong-sell { 
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white; 
        padding: 0.75rem 1.25rem; 
        border-radius: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    .recommendation-strong-sell:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(239, 68, 68, 0.4);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(15px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.1);
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        transition: all 0.5s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-12px);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4),
                    inset 0 1px 1px rgba(255, 255, 255, 0.15);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .prediction-card:hover::before {
        top: -20%;
        right: -20%;
    }
    .sentiment-positive {
        color: #10b981;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .sentiment-positive:hover {
        transform: scale(1.05);
    }
    
    .sentiment-negative {
        color: #ef4444;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .sentiment-negative:hover {
        transform: scale(1.05);
    }
    
    .sentiment-neutral {
        color: #f59e0b;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .sentiment-neutral:hover {
        transform: scale(1.05);
    }
    
    button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    button:hover {
        transform: translateY(-2px) !important;
    }
    
    [role="tablist"] button {
        transition: all 0.3s ease;
    }
    
    [role="tab"]:hover {
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'nse_stocks' not in st.session_state:
    st.session_state.nse_stocks = None
if 'screened_stocks' not in st.session_state:
    st.session_state.screened_stocks = None
if 'strategy_engine' not in st.session_state:
    st.session_state.strategy_engine = StrategyEngine()
if 'ml_predictor' not in st.session_state:
    st.session_state.ml_predictor = EnhancedMLPredictor()
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()
if 'stock_analyzer' not in st.session_state:
    st.session_state.stock_analyzer = StockAnalyzer()
if 'news_analyzer' not in st.session_state:
    st.session_state.news_analyzer = NewsSentimentAnalyzer(NEWS_API_KEY)
if 'price_predictor' not in st.session_state:
    st.session_state.price_predictor = PricePredictor()

# Load NSE stocks
if st.session_state.nse_stocks is None:
    with st.spinner("Loading NSE stocks..."):
        st.session_state.nse_stocks = load_nse_stocks()

# Header with centered logo
st.markdown("""
    <style>
    .header-logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem 0;
    }
    </style>
    <div class="header-logo-container">
""", unsafe_allow_html=True)

header_col1, header_col2, header_col3 = st.columns([1, 1, 1])

with header_col2:
    st.image("C:\\Users\\Lenovo\\Documents\\VS CODE codes(files)\\helloworld\\BTP SEM-5\\arthadrishti_logo.png", 
             width=280)

st.markdown("""
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .divider {
        border-top: 2px solid rgba(102, 126, 234, 0.3);
        margin: 20px 0;
    }
    </style>
    <div class="divider"></div>
""", unsafe_allow_html=True)

# Initialize page session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Stock Screener"

# Enhanced Navigation Menu with better styling
st.markdown("""
    <style>
    .nav-container {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .nav-item {
        flex: 1;
        min-width: 180px;
    }
    </style>
""", unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4, gap="small")

nav_buttons = [
    ("Stock Screener", "btn_screener", nav_col1),
    ("Custom Strategy", "btn_strategy", nav_col2),
    ("Portfolio", "btn_portfolio", nav_col3),
    ("Stock Analysis", "btn_analysis", nav_col4)
]

for btn_text, btn_key, col in nav_buttons:
    with col:
        is_active = st.session_state.current_page == btn_text
        btn_type = "primary" if is_active else "secondary"
        
        if st.button(btn_text, use_container_width=True, key=btn_key, type=btn_type):
            st.session_state.current_page = btn_text
            st.rerun()

st.markdown("""
    <style>
    .info-bar {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.4) 0%, rgba(118, 75, 162, 0.4) 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2),
                    inset 0 1px 1px rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out;
        letter-spacing: 0.5px;
    }
    
    .info-bar:hover {
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4),
                    inset 0 1px 1px rgba(255, 255, 255, 0.15);
        transform: translateY(-4px);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.5) 0%, rgba(118, 75, 162, 0.5) 100%);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7b8ff5 0%, #8659b5 100%);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    </style>
    <div class="info-bar">
        Stocks Loaded: """ + str(len(st.session_state.nse_stocks)) + """ | Ready to Analyze
    </div>
""", unsafe_allow_html=True)

page = st.session_state.current_page

# ============================================================================
# HELPER FUNCTION FOR STOCK ANALYSIS DISPLAY
# ============================================================================

def _display_stock_analysis(analysis, news_data, prediction):
    """Helper function to display stock analysis"""
    
    # Header info
    st.markdown(f"### {analysis['company_name']} ({analysis['symbol']})")
    st.markdown(f"**Sector:** {analysis['sector']} | **Industry:** {analysis['industry']}")
    
    # AI Recommendation
    rec = analysis['recommendation']
    st.markdown(f"<div class='recommendation-{rec['action'].lower().replace(' ', '-')}'>"
               f"AI RECOMMENDATION: {rec['action']} (Score: {rec['score']})</div>",
               unsafe_allow_html=True)
    
    st.markdown("**Reasons:**")
    for reason in rec['reasons']:
        st.markdown(f"- {reason}")
    
    st.markdown("---")
    
    # ============ PRICE PREDICTION SECTION ============
    if prediction:
        st.markdown("## Price Prediction & Trading Levels")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
                <div class='prediction-card'>
                    <h3 style='margin-top: 0;'>Target Price Prediction</h3>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
                        <div>
                            <p style='margin: 0; opacity: 0.8;'>Conservative Target</p>
                            <h2 style='margin: 0.5rem 0;'>₹{prediction['conservative_target']:.2f}</h2>
                        </div>
                        <div>
                            <p style='margin: 0; opacity: 0.8;'>Primary Target</p>
                            <h2 style='margin: 0.5rem 0;'>₹{prediction['target_price']:.2f}</h2>
                        </div>
                        <div>
                            <p style='margin: 0; opacity: 0.8;'>Aggressive Target</p>
                            <h2 style='margin: 0.5rem 0;'>₹{prediction['aggressive_target']:.2f}</h2>
                        </div>
                        <div>
                            <p style='margin: 0; opacity: 0.8;'>Expected Return</p>
                            <h2 style='margin: 0.5rem 0;'>{prediction['expected_return']:.2f}%</h2>
                        </div>
                    </div>
                    <p style='margin-top: 1rem; opacity: 0.9;'>
                        <strong>Confidence:</strong> {prediction['confidence']:.1f}% | 
                        <strong>Time Horizon:</strong> {prediction['time_horizon']} days
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Trading levels
            st.markdown("### Key Levels")
            st.metric("Current Price", f"₹{prediction['current_price']:.2f}")
            st.metric("Buy Price", f"₹{prediction['buy_price']:.2f}", 
                     delta=f"{((prediction['buy_price']/prediction['current_price']-1)*100):.2f}%")
            st.metric("Sell Price", f"₹{prediction['sell_price']:.2f}",
                     delta=f"{((prediction['sell_price']/prediction['current_price']-1)*100):.2f}%")
            st.metric("Stop Loss", f"₹{prediction['stop_loss']:.2f}",
                     delta=f"{((prediction['stop_loss']/prediction['current_price']-1)*100):.2f}%")
        
        # Detailed prediction metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Risk/Reward", f"{prediction['risk_reward']:.2f}")
        with col2:
            st.metric("Support", f"₹{prediction['support']:.2f}")
        with col3:
            st.metric("Resistance", f"₹{prediction['resistance']:.2f}")
        with col4:
            st.metric("Pivot Point", f"₹{prediction['pivot_points']['pivot']:.2f}")
        
        # Score breakdown
        st.markdown("#### Prediction Score Breakdown")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tech_score = prediction['technical_score']
            st.metric("Technical Score", f"{tech_score:.1f}/100")
            st.progress(min(abs(tech_score)/100, 1.0))
        
        with col2:
            sent_score = prediction['sentiment_score']
            st.metric("Sentiment Score", f"{sent_score:.1f}/100")
            st.progress(min(abs(sent_score)/100, 1.0))
        
        with col3:
            fund_score = prediction['fundamental_score']
            st.metric("Fundamental Score", f"{fund_score:.1f}/100")
            st.progress(min(abs(fund_score)/100, 1.0))
        
        st.markdown("---")
    
    # ============ NEWS SENTIMENT SECTION ============
    if news_data and news_data.get('news_count', 0) > 0:
        st.markdown("## News Sentiment Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sentiment = news_data['overall_sentiment']
            sentiment_class = f"sentiment-{sentiment.lower()}"
            st.markdown(f"**Overall Sentiment**")
            st.markdown(f"<p class='{sentiment_class}'>{sentiment}</p>", unsafe_allow_html=True)
        
        with col2:
            st.metric("Total News", news_data['news_count'])
        
        with col3:
            st.metric("Positive", news_data['positive_count'], 
                     delta=f"{(news_data['positive_count']/news_data['news_count']*100):.0f}%")
        
        with col4:
            st.metric("Negative", news_data['negative_count'],
                     delta=f"{(news_data['negative_count']/news_data['news_count']*100):.0f}%",
                     delta_color="inverse")
        
        # Sentiment score gauge
        score = news_data['sentiment_score']
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Score"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "lightcoral"},
                    {'range': [-0.3, 0.3], 'color': "lightyellow"},
                    {'range': [0.3, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Recent news articles
        st.markdown("### 📑 Recent News")
        
        for i, article in enumerate(news_data['articles'][:5], 1):
            sentiment_class = f"sentiment-{article['sentiment'].lower()}"
            
            with st.expander(f"{i}. {article['title'][:100]}..."):
                st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{article['sentiment']}</span> "
                          f"(Score: {article['score']:.2f})", unsafe_allow_html=True)
                st.markdown(f"**Published:** {article['publishedAt'][:10]}")
                st.markdown(f"**Description:** {article['description']}")
                st.markdown(f"[Read more]({article['url']})")
        
        st.markdown("---")
    elif news_data and news_data.get('news_count', 0) == 0:
        st.info("No recent news found for this stock")
        st.markdown("---")
    
    # Key Metrics
    st.markdown("### Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Current Price", f"₹{analysis['current_price']:.2f}")
    with col2:
        st.metric("Market Cap", format_number(analysis['market_cap']))
    with col3:
        st.metric("P/E Ratio", f"{analysis['pe_ratio']:.2f}")
    with col4:
        st.metric("P/B Ratio", f"{analysis['pb_ratio']:.2f}")
    with col5:
        st.metric("Div Yield", f"{analysis['dividend_yield']:.2f}%")
    
    st.markdown("---")
    
    # Performance Metrics
    st.markdown("### Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("1 Month", f"{analysis['returns_1m']:.2f}%")
    with col2:
        st.metric("3 Months", f"{analysis['returns_3m']:.2f}%")
    with col3:
        st.metric("6 Months", f"{analysis['returns_6m']:.2f}%")
    with col4:
        st.metric("1 Year", f"{analysis['returns_1y']:.2f}%")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("52W High", f"₹{analysis['week_52_high']:.2f}")
    with col2:
        st.metric("52W Low", f"₹{analysis['week_52_low']:.2f}")
    with col3:
        st.metric("Volatility", f"{analysis['volatility']:.2f}%")
    
    st.markdown("---")
    
    # Technical Indicators
    st.markdown("### Technical Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rsi = analysis['rsi']
        rsi_color = "🟢" if 30 < rsi < 70 else "🔴"
        st.metric("RSI", f"{rsi:.2f} {rsi_color}")
    
    with col2:
        macd = analysis['macd']
        macd_color = "🟢" if macd > analysis['macd_signal'] else "🔴"
        st.metric("MACD", f"{macd:.2f} {macd_color}")
    
    with col3:
        adx = analysis['adx']
        adx_color = "🟢" if adx > 25 else "🟡"
        st.metric("ADX", f"{adx:.2f} {adx_color}")
    
    with col4:
        vol_ratio = analysis['volume_ratio']
        vol_color = "🟢" if vol_ratio > 1 else "🔴"
        st.metric("Volume Ratio", f"{vol_ratio:.2f} {vol_color}")
    
    st.markdown("---")
    
    # Charts
    st.markdown("### Charts")
    
    df = analysis['df']
    
    # Price chart with indicators
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Price & Bollinger Bands', 'RSI', 'MACD')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], 
                            line=dict(dash='dash', color='gray', width=1),
                            name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], 
                            line=dict(dash='dash', color='gray', width=1),
                            name='BB Lower'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], 
                            line=dict(color='purple'),
                            name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], 
                            line=dict(color='blue'),
                            name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], 
                            line=dict(color='red'),
                            name='Signal'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False, xaxis_rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Volume chart
    st.markdown("### Volume Analysis")
    
    fig_vol = go.Figure()
    
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
             for i in range(len(df))]
    
    fig_vol.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        marker_color=colors,
        name='Volume'
    ))
    
    fig_vol.add_trace(go.Scatter(
        x=df.index,
        y=df['Vol_MA'],
        line=dict(color='orange', width=2),
        name='Avg Volume'
    ))
    
    fig_vol.update_layout(height=300)
    st.plotly_chart(fig_vol, use_container_width=True)


# ============================================================================
# PAGE 1: STOCK SCREENER
# ============================================================================

if page == "Stock Screener":
    st.markdown("## Stock Screener")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy_name = st.selectbox(
            "Select Strategy",
            list(st.session_state.strategy_engine.strategies.keys())
        )
    
    with col2:
        min_confidence = st.slider("Min Confidence %", 50, 90, 60)
    
    with col3:
        stock_list_option = st.selectbox(
            "Select Stock Universe",
            ["NIFTY50", "NIFTY500", "All NSE Stocks"]
        )
    
    # Display strategy description
    strategy_desc = st.session_state.strategy_engine.strategies[strategy_name]['description']
    st.info(f"**Strategy:** {strategy_desc}")
    
    if st.button("Start Screening", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if stock_list_option == "NIFTY50":
            stocks_to_scan = load_nifty50()
        elif stock_list_option == "NIFTY500":
            stocks_to_scan = load_nifty500()
        else:  # All NSE Stocks
            stocks_to_scan = st.session_state.nse_stocks
        
        results = []
        
        for i, symbol in enumerate(stocks_to_scan):
            status_text.text(f"Screening {symbol}... ({i+1}/{len(stocks_to_scan)})")
            
            try:
                df = get_stock_data_cached(symbol)
                if df is None:
                    continue
                
                result = st.session_state.strategy_engine.evaluate_strategy(df, strategy_name)
                
                if result and result['confidence'] >= min_confidence:
                    ml_pred = st.session_state.ml_predictor.predict(symbol)
                    current_price = df['Close'].iloc[-1]
                    
                    # Get quick price predictions
                    try:
                        prediction = st.session_state.price_predictor.predict_target_price(
                            df, sentiment_score=0, fundamental_score=50
                        )
                        buy_price = prediction['buy_price'] if prediction else current_price * 0.98
                        target_price = prediction['target_price'] if prediction else current_price * 1.05
                        stop_loss = prediction['stop_loss'] if prediction else current_price * 0.97
                        expected_return = prediction['expected_return'] if prediction else 5.0
                        pred_confidence = prediction['confidence'] if prediction else 50
                    except:
                        buy_price = current_price * 0.98
                        target_price = current_price * 1.05
                        stop_loss = current_price * 0.97
                        expected_return = 5.0
                        pred_confidence = 50
                    
                    results.append({
                        'Symbol': symbol,
                        'Current Price': current_price,
                        'Buy Price': buy_price,
                        'Target Price': target_price,
                        'Stop Loss': stop_loss,
                        'Expected Return %': expected_return,
                        'Strategy Confidence': result['confidence'],
                        'Prediction Confidence': pred_confidence,
                        'Conditions Met': result['conditions_met'],
                        'Total Conditions': result['total_conditions'],
                        'ML Prediction': ml_pred['prediction'] if ml_pred else 0,
                        'ML Confidence': ml_pred['confidence'] if ml_pred else 0
                    })
            except:
                pass
            
            progress_bar.progress((i + 1) / len(stocks_to_scan))
        
        status_text.text("Screening Complete!")
        
        if results:
            st.session_state.screened_stocks = pd.DataFrame(results).sort_values(
                'Strategy Confidence', ascending=False
            )
            st.success(f"Found {len(results)} stocks matching criteria!")
        else:
            st.warning("No stocks found. Try adjusting filters.")
    
    # Display results
    if st.session_state.screened_stocks is not None:
        st.markdown("---")
        st.markdown("## Results")
        
        df = st.session_state.screened_stocks
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stocks", len(df))
        with col2:
            st.metric("Avg Strategy Confidence", f"{df['Strategy Confidence'].mean():.1f}%")
        with col3:
            high = len(df[df['Strategy Confidence'] >= 75])
            st.metric("High Confidence", high)
        with col4:
            bullish = len(df[df['ML Prediction'] >= 1])
            st.metric("Bullish (ML)", bullish)
        
        # Format display columns
        display_df = df.copy()
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"₹{x:.2f}")
        display_df['Buy Price'] = display_df['Buy Price'].apply(lambda x: f"₹{x:.2f}")
        display_df['Target Price'] = display_df['Target Price'].apply(lambda x: f"₹{x:.2f}")
        display_df['Stop Loss'] = display_df['Stop Loss'].apply(lambda x: f"₹{x:.2f}")
        display_df['Expected Return %'] = display_df['Expected Return %'].apply(lambda x: f"{x:.2f}%")
        display_df['Strategy Confidence'] = display_df['Strategy Confidence'].apply(lambda x: f"{x:.1f}%")
        display_df['Prediction Confidence'] = display_df['Prediction Confidence'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                f"screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Quick analyze option
            selected_symbol = st.selectbox(
                "Select stock to analyze in detail",
                df['Symbol'].tolist(),
                key="quick_analyze_select"
            )
        
        if st.button("🔍 Analyze Selected Stock", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {selected_symbol}..."):
                # Switch to analysis page and analyze
                analysis = st.session_state.stock_analyzer.analyze_stock(selected_symbol)
                
                if analysis:
                    st.session_state.current_analysis = analysis
                    
                    # Get news sentiment
                    with st.spinner("Analyzing news sentiment..."):
                        news_data = st.session_state.news_analyzer.analyze_stock_news(
                            selected_symbol, analysis.get('company_name', selected_symbol)
                        )
                        st.session_state.current_news = news_data
                    
                    # Get price prediction
                    with st.spinner("Predicting target prices..."):
                        prediction = st.session_state.price_predictor.predict_target_price(
                            analysis['df'],
                            sentiment_score=news_data['sentiment_score'],
                            fundamental_score=50
                        )
                        st.session_state.current_prediction = prediction
                    
                    st.success(f"Analysis complete! Scroll down to view detailed analysis of {selected_symbol}")
                    
                    # Display analysis immediately below
                    st.markdown("---")
                    st.markdown(f"## Detailed Analysis: {selected_symbol}")
                    
                    # Show the full analysis
                    _display_stock_analysis(analysis, news_data, prediction)
                else:
                    st.error("Failed to analyze stock")

# ============================================================================
# PAGE 2: CUSTOM STRATEGY
# ============================================================================

elif page == "Custom Strategy":
    st.markdown("## Custom Strategy Builder")
    
    st.info("Build your own screening strategy by selecting indicators and conditions")
    
    tab1, tab2 = st.tabs(["Create Strategy", "Manage Strategies"])
    
    with tab1:
        st.markdown("### Create New Strategy")
        
        strategy_name = st.text_input("Strategy Name", "My Custom Strategy")
        strategy_desc = st.text_area("Description", "Enter strategy description...")
        
        st.markdown("#### Add Rules")
        
        num_rules = st.number_input("Number of Rules", 1, 10, 3)
        
        rules = []
        for i in range(num_rules):
            st.markdown(f"**Rule {i+1}**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                indicator = st.selectbox(f"Indicator", 
                    list(st.session_state.strategy_engine.AVAILABLE_INDICATORS.keys()),
                    key=f"ind_{i}")
            
            with col2:
                if indicator in ['MACD', 'EMA_Cross', 'SMA_Cross']:
                    condition = st.selectbox(f"Condition",
                        ['bullish', 'bullish_cross', 'turning_up'],
                        key=f"cond_{i}")
                elif indicator == 'RSI':
                    condition = st.selectbox(f"Condition",
                        ['above', 'below', 'between'],
                        key=f"cond_{i}")
                else:
                    condition = st.selectbox(f"Condition",
                        ['above', 'below', 'above_avg'],
                        key=f"cond_{i}")
            
            with col3:
                threshold = st.number_input(f"Threshold", 0, 100, 50, key=f"thresh_{i}")
            
            with col4:
                weight = st.slider(f"Weight", 0.5, 3.0, 1.0, 0.5, key=f"weight_{i}")
            
            rules.append({
                'indicator': indicator,
                'condition': condition,
                'threshold': threshold,
                'weight': weight
            })
        
        if st.button("Save Strategy", type="primary"):
            st.session_state.strategy_engine.strategies[strategy_name] = {
                'description': strategy_desc,
                'rules': rules
            }
            st.success(f"Strategy '{strategy_name}' saved!")
    
    with tab2:
        st.markdown("### Saved Strategies")
        
        for name, strategy in st.session_state.strategy_engine.strategies.items():
            with st.expander(f"📋 {name}"):
                st.write(f"**Description:** {strategy['description']}")
                st.write(f"**Rules:** {len(strategy['rules'])}")
                
                for i, rule in enumerate(strategy['rules'], 1):
                    st.write(f"{i}. {rule['indicator']} - {rule['condition']} (weight: {rule.get('weight', 1.0)})")

# ============================================================================
# PAGE 3: PORTFOLIO
# ============================================================================

elif page == "Portfolio":
    st.markdown("## Portfolio Management")
    
    tab1, tab2, tab3 = st.tabs(["My Portfolio", "Add Stock", "Analysis"])
    
    with tab1:
        st.markdown("### My Portfolio")
        
        if st.session_state.portfolio_manager.portfolio:
            analysis = st.session_state.portfolio_manager.analyze_portfolio()
            
            if analysis:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Invested", format_number(analysis['total_invested']))
                with col2:
                    st.metric("Current Value", format_number(analysis['total_current']))
                with col3:
                    pnl = analysis['total_pnl']
                    st.metric("P&L", format_number(pnl), delta=f"{analysis['total_pnl_pct']:.2f}%")
                with col4:
                    returns = analysis['total_pnl_pct']
                    st.metric("Returns", f"{returns:.2f}%")
                
                # Portfolio table
                st.markdown("---")
                df_portfolio = pd.DataFrame(analysis['stocks'])
                st.dataframe(df_portfolio, use_container_width=True)
                
                # Remove stock
                st.markdown("---")
                stock_to_remove = st.selectbox(
                    "Remove Stock",
                    [s['symbol'] for s in st.session_state.portfolio_manager.portfolio]
                )
                if st.button("Remove"):
                    st.session_state.portfolio_manager.remove_stock(stock_to_remove)
                    st.rerun()
        else:
            st.info("📊 Your portfolio is empty. Add stocks to get started!")
    
    with tab2:
        st.markdown("### Add Stock to Portfolio")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox("Stock Symbol", st.session_state.nse_stocks)
        
        with col2:
            quantity = st.number_input("Quantity", min_value=1, value=100)
        
        with col3:
            buy_price = st.number_input("Buy Price (₹)", min_value=0.01, value=100.0)
        
        if st.button("Add to Portfolio", type="primary"):
            st.session_state.portfolio_manager.add_stock(symbol, quantity, buy_price)
            st.success(f"Added {quantity} shares of {symbol} to portfolio!")
            st.rerun()
    
    with tab3:
        st.markdown("### Portfolio Analysis")
        
        if st.session_state.portfolio_manager.portfolio:
            st.markdown("#### Individual Stock Performance")
            
            for stock in st.session_state.portfolio_manager.portfolio:
                symbol = stock['symbol']
                
                with st.expander(f"📊 {symbol}"):
                    analysis = st.session_state.stock_analyzer.analyze_stock(symbol)
                    
                    if analysis:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Price", f"₹{analysis['current_price']:.2f}")
                        with col2:
                            st.metric("1M Return", f"{analysis['returns_1m']:.2f}%")
                        with col3:
                            rec = analysis['recommendation']['action']
                            st.markdown(f"<div class='recommendation-{rec.lower().replace(' ', '-')}'>{rec}</div>", 
                                      unsafe_allow_html=True)
                        
                        # Mini chart
                        if analysis['df'] is not None:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=analysis['df'].index[-60:],
                                y=analysis['df']['Close'][-60:],
                                mode='lines',
                                name='Price',
                                line=dict(color='#667eea', width=2)
                            ))
                            fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add stocks to your portfolio first!")

# ============================================================================
# PAGE 4: STOCK ANALYSIS
# ============================================================================

elif page == "Stock Analysis":
    st.markdown("## Detailed Stock Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.selectbox("Select Stock", st.session_state.nse_stocks)
    
    with col2:
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {symbol}..."):
                analysis = st.session_state.stock_analyzer.analyze_stock(symbol)
                
                if analysis:
                    st.session_state.current_analysis = analysis
                    
                    # Get news sentiment
                    with st.spinner("Analyzing news sentiment..."):
                        news_data = st.session_state.news_analyzer.analyze_stock_news(
                            symbol, analysis.get('company_name', symbol)
                        )
                        st.session_state.current_news = news_data
                    
                    # Get price prediction
                    with st.spinner("Predicting target prices..."):
                        prediction = st.session_state.price_predictor.predict_target_price(
                            analysis['df'],
                            sentiment_score=news_data['sentiment_score'],
                            fundamental_score=50  # Can be enhanced with actual fundamental data
                        )
                        st.session_state.current_prediction = prediction
                else:
                    st.error("Failed to analyze stock")
    
    if 'current_analysis' in st.session_state:
        analysis = st.session_state.current_analysis
        news_data = st.session_state.get('current_news', {})
        prediction = st.session_state.get('current_prediction', None)
        
        # Header info
        st.markdown(f"### {analysis['company_name']} ({analysis['symbol']})")
        st.markdown(f"**Sector:** {analysis['sector']} | **Industry:** {analysis['industry']}")
        
        # AI Recommendation
        rec = analysis['recommendation']
        st.markdown(f"<div class='recommendation-{rec['action'].lower().replace(' ', '-')}'>"
                   f"AI RECOMMENDATION: {rec['action']} (Score: {rec['score']})</div>",
                   unsafe_allow_html=True)
        
        st.markdown("**Reasons:**")
        for reason in rec['reasons']:
            st.markdown(f"- {reason}")
        
        st.markdown("---")
        
        # ============ PRICE PREDICTION SECTION ============
        if prediction:
            st.markdown("## Price Prediction & Trading Levels")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                    <div class='prediction-card'>
                        <h3 style='margin-top: 0;'>Target Price Prediction</h3>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
                            <div>
                                <p style='margin: 0; opacity: 0.8;'>Conservative Target</p>
                                <h2 style='margin: 0.5rem 0;'>₹{prediction['conservative_target']:.2f}</h2>
                            </div>
                            <div>
                                <p style='margin: 0; opacity: 0.8;'>Primary Target</p>
                                <h2 style='margin: 0.5rem 0;'>₹{prediction['target_price']:.2f}</h2>
                            </div>
                            <div>
                                <p style='margin: 0; opacity: 0.8;'>Aggressive Target</p>
                                <h2 style='margin: 0.5rem 0;'>₹{prediction['aggressive_target']:.2f}</h2>
                            </div>
                            <div>
                                <p style='margin: 0; opacity: 0.8;'>Expected Return</p>
                                <h2 style='margin: 0.5rem 0;'>{prediction['expected_return']:.2f}%</h2>
                            </div>
                        </div>
                        <p style='margin-top: 1rem; opacity: 0.9;'>
                            <strong>Confidence:</strong> {prediction['confidence']:.1f}% | 
                            <strong>Time Horizon:</strong> {prediction['time_horizon']} days
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Trading levels
                st.markdown("### Key Levels")
                st.metric("Current Price", f"₹{prediction['current_price']:.2f}")
                st.metric("Buy Price", f"₹{prediction['buy_price']:.2f}", 
                         delta=f"{((prediction['buy_price']/prediction['current_price']-1)*100):.2f}%")
                st.metric("Sell Price", f"₹{prediction['sell_price']:.2f}",
                         delta=f"{((prediction['sell_price']/prediction['current_price']-1)*100):.2f}%")
                st.metric("Stop Loss", f"₹{prediction['stop_loss']:.2f}",
                         delta=f"{((prediction['stop_loss']/prediction['current_price']-1)*100):.2f}%")
            
            # Detailed prediction metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Risk/Reward", f"{prediction['risk_reward']:.2f}")
            with col2:
                st.metric("Support", f"₹{prediction['support']:.2f}")
            with col3:
                st.metric("Resistance", f"₹{prediction['resistance']:.2f}")
            with col4:
                st.metric("Pivot Point", f"₹{prediction['pivot_points']['pivot']:.2f}")
            
            # Score breakdown
            st.markdown("#### Prediction Score Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tech_score = prediction['technical_score']
                st.metric("Technical Score", f"{tech_score:.1f}/100")
                st.progress(min(abs(tech_score)/100, 1.0))
            
            with col2:
                sent_score = prediction['sentiment_score']
                st.metric("Sentiment Score", f"{sent_score:.1f}/100")
                st.progress(min(abs(sent_score)/100, 1.0))
            
            with col3:
                fund_score = prediction['fundamental_score']
                st.metric("Fundamental Score", f"{fund_score:.1f}/100")
                st.progress(min(abs(fund_score)/100, 1.0))
            
            st.markdown("---")
        
        # ============ NEWS SENTIMENT SECTION ============
        if news_data and news_data.get('news_count', 0) > 0:
            st.markdown("## News Sentiment Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sentiment = news_data['overall_sentiment']
                sentiment_class = f"sentiment-{sentiment.lower()}"
                st.markdown(f"**Overall Sentiment**")
                st.markdown(f"<p class='{sentiment_class}'>{sentiment}</p>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Total News", news_data['news_count'])
            
            with col3:
                st.metric("Positive", news_data['positive_count'], 
                         delta=f"{(news_data['positive_count']/news_data['news_count']*100):.0f}%")
            
            with col4:
                st.metric("Negative", news_data['negative_count'],
                         delta=f"{(news_data['negative_count']/news_data['news_count']*100):.0f}%",
                         delta_color="inverse")
            
            # Sentiment score gauge
            score = news_data['sentiment_score']
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sentiment Score"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "lightcoral"},
                        {'range': [-0.3, 0.3], 'color': "lightyellow"},
                        {'range': [0.3, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Recent news articles
            st.markdown("### 📑 Recent News")
            
            for i, article in enumerate(news_data['articles'][:5], 1):
                sentiment_class = f"sentiment-{article['sentiment'].lower()}"
                
                with st.expander(f"{i}. {article['title'][:100]}..."):
                    st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{article['sentiment']}</span> "
                              f"(Score: {article['score']:.2f})", unsafe_allow_html=True)
                    st.markdown(f"**Published:** {article['publishedAt'][:10]}")
                    st.markdown(f"**Description:** {article['description']}")
                    st.markdown(f"[Read more]({article['url']})")
            
            st.markdown("---")
        elif news_data and news_data.get('news_count', 0) == 0:
            st.info("No recent news found for this stock")
            st.markdown("---")
        
        # Key Metrics
        st.markdown("### Key Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"₹{analysis['current_price']:.2f}")
        with col2:
            st.metric("Market Cap", format_number(analysis['market_cap']))
        with col3:
            st.metric("P/E Ratio", f"{analysis['pe_ratio']:.2f}")
        with col4:
            st.metric("P/B Ratio", f"{analysis['pb_ratio']:.2f}")
        with col5:
            st.metric("Div Yield", f"{analysis['dividend_yield']:.2f}%")
        
        st.markdown("---")
        
        # Performance Metrics
        st.markdown("### Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("1 Month", f"{analysis['returns_1m']:.2f}%")
        with col2:
            st.metric("3 Months", f"{analysis['returns_3m']:.2f}%")
        with col3:
            st.metric("6 Months", f"{analysis['returns_6m']:.2f}%")
        with col4:
            st.metric("1 Year", f"{analysis['returns_1y']:.2f}%")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("52W High", f"₹{analysis['week_52_high']:.2f}")
        with col2:
            st.metric("52W Low", f"₹{analysis['week_52_low']:.2f}")
        with col3:
            st.metric("Volatility", f"{analysis['volatility']:.2f}%")
        
        st.markdown("---")
        
        # Technical Indicators
        st.markdown("### Technical Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = analysis['rsi']
            rsi_color = "🟢" if 30 < rsi < 70 else "🔴"
            st.metric("RSI", f"{rsi:.2f} {rsi_color}")
        
        with col2:
            macd = analysis['macd']
            macd_color = "🟢" if macd > analysis['macd_signal'] else "🔴"
            st.metric("MACD", f"{macd:.2f} {macd_color}")
        
        with col3:
            adx = analysis['adx']
            adx_color = "🟢" if adx > 25 else "🟡"
            st.metric("ADX", f"{adx:.2f} {adx_color}")
        
        with col4:
            vol_ratio = analysis['volume_ratio']
            vol_color = "🟢" if vol_ratio > 1 else "🔴"
            st.metric("Volume Ratio", f"{vol_ratio:.2f} {vol_color}")
        
        st.markdown("---")
        
        # Charts
        st.markdown("### Charts")
        
        df = analysis['df']
        
        # Price chart with indicators
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price & Bollinger Bands', 'RSI', 'MACD')
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], 
                                line=dict(dash='dash', color='gray', width=1),
                                name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], 
                                line=dict(dash='dash', color='gray', width=1),
                                name='BB Lower'), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], 
                                line=dict(color='purple'),
                                name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], 
                                line=dict(color='blue'),
                                name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], 
                                line=dict(color='red'),
                                name='Signal'), row=3, col=1)
        
        fig.update_layout(height=800, showlegend=False, xaxis_rangeslider_visible=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Volume chart
        st.markdown("### Volume Analysis")
        
        fig_vol = go.Figure()
        
        colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                 for i in range(len(df))]
        
        fig_vol.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color=colors,
            name='Volume'
        ))
        
        fig_vol.add_trace(go.Scatter(
            x=df.index,
            y=df['Vol_MA'],
            line=dict(color='orange', width=2),
            name='Avg Volume'
        ))
        
        fig_vol.update_layout(height=300)
        st.plotly_chart(fig_vol, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3 style='color: #667eea;'>👁️ ARTHA DRISHTI</h3>
        <p><strong>ML Based Advanced Stock Screener v3.0</strong></p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
        <p>Data: Yahoo Finance | News: NewsAPI | Built with Streamlit & ML</p>
    </div>
""", unsafe_allow_html=True)