# import yfinance as yf
# import pandas as pd
# import numpy as np
# import ta
# import warnings
# import time
# from tqdm import tqdm
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.ensemble import RandomForestClassifier
# from textblob import TextBlob
# from datetime import datetime
# import json
# import os
# import multiprocessing as mp
# import math
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import webbrowser

# # Import rating system and report generator
# from stock_ratings import get_stock_rating
# from report_generator import create_summary_report

# warnings.filterwarnings("ignore")

# # === Parallel Processing Functions ===
# def process_stock_chunk(chunk_data):
#     """Process a chunk of stocks in parallel"""
#     try:
#         symbols = chunk_data['symbols']
#         custom_conditions = chunk_data.get('custom_conditions')
#         use_ml = chunk_data.get('use_ml', True)
#         use_sentiment = chunk_data.get('use_sentiment', True)
#         unavailable_stocks = chunk_data.get('unavailable_stocks', set())
        
#         if not symbols:  # Check for empty chunk
#             return [], {}
            
#         results = []
#         analysis_data = {}
        
#         for symbol in symbols:
#             try:
#                 # Get stock data and perform basic screening
#                 stock_data = get_stock_data(symbol)
#                 if stock_data is None:
#                     continue
                    
#                 # Store data for visualization
#                 analysis_data[symbol] = compute_indicators(stock_data.copy())
                    
#                 # Basic technical screening
#                 result = check_conditions(symbol, custom_conditions)
#                 if not result:
#                     continue
                    
#                 # Add ML predictions if enabled
#                 if use_ml:
#                     ml_results = train_ml_model(symbol)
#                     if ml_results:
#                         result['ML_Score'] = f"{ml_results['avg_score']:.2f} (±{ml_results['std_score']:.2f})"
#                         result['ML_Prediction'] = "Bullish" if ml_results['avg_score'] > 0.6 else "Neutral"
                
#                 # Add sentiment if enabled
#                 if use_sentiment:
#                     sentiment = get_sentiment_score(symbol)
#                     result['Sentiment'] = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
#                     result['Sentiment_Score'] = f"{sentiment:.2f}"
                    
#                 results.append(result)
#             except Exception as e:
#                 print(f"Error processing {symbol}: {str(e)}")
#                 continue
                
#         return results, analysis_data
#     except Exception as e:
#         print(f"Error processing chunk: {str(e)}")
#         return [], {}


# # === Interactive Visualization Functions ===
# def create_stock_analysis_chart(symbol, data):
#     """Create an interactive chart for stock analysis"""
#     fig = make_subplots(rows=3, cols=1,
#                        vertical_spacing=0.05,
#                        row_heights=[0.6, 0.2, 0.2],
#                        shared_xaxes=True)
    
#     # Candlestick chart
#     fig.add_trace(
#         go.Candlestick(x=data.index,
#                        open=data['Open'],
#                        high=data['High'],
#                        low=data['Low'],
#                        close=data['Close'],
#                        name='Price'),
#         row=1, col=1
#     )
    
#     # Add Bollinger Bands if available
#     if 'BB_upper' in data.columns:
#         fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper',
#                                 line=dict(color='gray', dash='dash')), row=1, col=1)
#         fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower',
#                                 line=dict(color='gray', dash='dash')), row=1, col=1)
    
#     # Volume
#     fig.add_trace(
#         go.Bar(x=data.index, y=data['Volume'], name='Volume'),
#         row=2, col=1
#     )
    
#     # RSI
#     if 'RSI' in data.columns:
#         fig.add_trace(
#             go.Scatter(x=data.index, y=data['RSI'], name='RSI'),
#             row=3, col=1
#         )
#         fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
#         fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
#     fig.update_layout(
#         title=f'{symbol} Technical Analysis',
#         yaxis_title='Price',
#         yaxis2_title='Volume',
#         yaxis3_title='RSI',
#         xaxis3_title='Date',
#         showlegend=False,
#         height=800
#     )
    
#     return fig

# def create_stock_summary_charts(df):
#     """Create interactive charts summarizing the stock screening results"""
#     fig = make_subplots(rows=2, cols=2,
#                        subplot_titles=('Stocks by Rating', 'Stocks by Category',
#                                      'Distribution of Technical Scores', 'Distribution of Market Cap'))
    
#     # Rating distribution
#     rating_counts = df['Overall Rating'].value_counts()
#     fig.add_trace(
#         go.Bar(x=rating_counts.index, y=rating_counts.values, name='Ratings'),
#         row=1, col=1
#     )
    
#     # Category distribution
#     category_counts = df['Category'].value_counts()
#     fig.add_trace(
#         go.Bar(x=category_counts.index, y=category_counts.values, name='Categories'),
#         row=1, col=2
#     )
    
#     # Technical Score distribution
#     fig.add_trace(
#         go.Histogram(x=df['Technical Score'], name='Technical Scores',
#                      nbinsx=20, histnorm='percent'),
#         row=2, col=1
#     )
    
#     # Market Cap distribution (log scale)
#     market_caps = pd.to_numeric(df['Market Cap (Raw)'], errors='coerce')
#     fig.add_trace(
#         go.Histogram(x=np.log10(market_caps[market_caps > 0]), 
#                      name='Market Cap (log10)', nbinsx=20),
#         row=2, col=2
#     )
    
#     fig.update_layout(height=800, showlegend=False)
#     return fig

# def generate_analysis_explanation(analysis_results):
#     """Generate a detailed explanation of the stock's rating and technical analysis"""
#     explanations = []
#     signals = []
#     met_conditions = []
#     failed_conditions = []
    
#     # Technical Score Components
#     tech_score = analysis_results.get('Technical Score', 0)
#     if tech_score >= 80:
#         explanations.append(f"Very strong technical indicators across multiple timeframes (Score: {tech_score}/100)")
#     elif tech_score >= 60:
#         explanations.append(f"Generally positive technical indicators (Score: {tech_score}/100)")
#     else:
#         explanations.append(f"Mixed technical signals (Score: {tech_score}/100)")
    
#     # Momentum Analysis
#     momentum_score = 0
#     if analysis_results.get('Daily MACD Up', False):
#         met_conditions.append("✅ Daily MACD is trending upward")
#         signals.append(("✅ Daily MACD is trending upward", "Showing positive short-term price momentum"))
#         momentum_score += 1
#     else:
#         failed_conditions.append("❌ Daily MACD needs improvement")
#         signals.append(("❌ Daily MACD is not trending upward", "Short-term momentum needs strengthening"))
        
#     if analysis_results.get('Weekly MACD Up', False):
#         met_conditions.append("✅ Weekly MACD is bullish")
#         signals.append(("✅ Weekly MACD is bullish", "Demonstrating strong medium-term trend"))
#         momentum_score += 1
#     else:
#         failed_conditions.append("❌ Weekly MACD needs improvement")
#         signals.append(("❌ Weekly MACD is not bullish", "Medium-term momentum needs development"))
        
#     if analysis_results.get('Monthly MACD Up', False):
#         met_conditions.append("✅ Monthly MACD is positive")
#         signals.append(("✅ Monthly MACD is positive", "Strong long-term upward trend"))
#         momentum_score += 1
#     else:
#         failed_conditions.append("❌ Monthly MACD needs improvement")
#         signals.append(("❌ Monthly MACD is not positive", "Long-term trend needs strengthening"))
    
#     # Trend Strength
#     trend_score = 0
#     if analysis_results.get('RSI > 55', False):
#         met_conditions.append("✅ RSI above 55")
#         signals.append(("✅ RSI above 55", "Strong momentum without being overbought"))
#         trend_score += 1
#     else:
#         failed_conditions.append("❌ RSI below 55")
#         signals.append(("❌ RSI below 55", "Momentum needs to improve"))
        
#     if analysis_results.get('+DMI > -DMI', False):
#         met_conditions.append("✅ Positive Directional Movement")
#         signals.append(("✅ +DMI > -DMI", "Price showing upward directional strength"))
#         trend_score += 1
#     else:
#         failed_conditions.append("❌ Negative Directional Movement")
#         signals.append(("❌ +DMI < -DMI", "Price showing downward pressure"))
        
#     if analysis_results.get('ADX Rising', False):
#         met_conditions.append("✅ Rising Trend Strength")
#         signals.append(("✅ ADX Rising", "Overall trend strength is increasing"))
#         trend_score += 1
#     else:
#         failed_conditions.append("❌ Weakening Trend")
#         signals.append(("❌ ADX Not Rising", "Trend strength is declining"))
    
#     # Volume Analysis
#     if analysis_results.get('Vol > MA (90%)', False):
#         met_conditions.append("✅ Above-Average Volume")
#         signals.append(("✅ Volume above average", "Strong trading interest and liquidity"))
#     else:
#         failed_conditions.append("❌ Below-Average Volume")
#         signals.append(("❌ Volume below average", "Low trading interest"))
    
#     # Market Cap Analysis
#     market_cap = analysis_results.get('Market Cap', 'N/A')
#     if market_cap and 'Cr' in market_cap:
#         signals.append(("ℹ️ Market Cap: " + market_cap, "Company size and liquidity indication"))
    
#     # Add overall summary
#     total_conditions = len(met_conditions) + len(failed_conditions)
#     met_count = len(met_conditions)
#     explanations.append(f"\nMet {met_count} out of {total_conditions} technical conditions")
    
#     if met_count >= 6:
#         explanations.append("Exceptional technical setup with multiple confirming signals")
#     elif met_count >= 4:
#         explanations.append("Strong technical setup with required MACD confirmation")
#     else:
#         explanations.append("Basic criteria met but needs more confirming signals")
    
#     # Add detailed breakdowns
#     if met_conditions:
#         explanations.append("\nStrong Points:")
#         explanations.extend(met_conditions)
    
#     if failed_conditions:
#         explanations.append("\nAreas for Improvement:")
#         explanations.extend(failed_conditions)
    
#     return explanations, signals
    
#     # Trend Strength
#     if analysis_results.get('RSI > 55', False):
#         signals.append(("✅ RSI above 55", "Showing good momentum while avoiding overbought levels"))
#     else:
#         signals.append(("❌ RSI below 55", "Momentum is currently weak"))
        
#     if analysis_results.get('+DMI > -DMI', False):
#         signals.append(("✅ +DMI > -DMI", "Positive directional movement"))
#     else:
#         signals.append(("❌ +DMI < -DMI", "Negative directional movement"))
        
#     if analysis_results.get('ADX Rising', False):
#         signals.append(("✅ ADX Rising", "Trend strength is increasing"))
#     else:
#         signals.append(("❌ ADX Not Rising", "Trend strength is decreasing"))
    
#     # Volume Analysis
#     if analysis_results.get('Vol > MA (90%)', False):
#         signals.append(("✅ Volume above average", "Strong trading interest"))
#     else:
#         signals.append(("❌ Volume below average", "Low trading interest"))
    
#     # Market Cap Analysis
#     market_cap = analysis_results.get('Market Cap', 'N/A')
#     if market_cap and 'Cr' in market_cap:
#         signals.append(("ℹ️ Market Cap: " + market_cap, "Company size and liquidity indication"))
    
#     return explanations, signals

# def save_analysis_report(symbol, data, analysis_results):
#     """Generate and save an HTML analysis report"""
#     report_dir = "stock_reports"
#     if not os.path.exists(report_dir):
#         os.makedirs(report_dir)
        
#     filename = f"{report_dir}/{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
#     # Create the technical analysis chart
#     fig = create_stock_analysis_chart(symbol, data)
    
#     # Format key metrics
#     tech_score = analysis_results.get('Technical Score', 'N/A')
#     momentum = analysis_results.get('Momentum', 'N/A')
#     trend = analysis_results.get('Trend', 'N/A')
#     rating = analysis_results.get('Overall Rating', 'N/A')
#     category = analysis_results.get('Category', 'N/A')
    
#     # Generate detailed explanation
#     explanations, signals = generate_analysis_explanation(analysis_results)
    
#     # Generate HTML report
#     html_content = f"""
#     <html>
#     <head>
#         <title>{symbol} Stock Analysis Report</title>
#         <style>
#             body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
#             .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
#             .section {{ margin: 20px 0; padding: 20px; border-radius: 5px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
#             .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
#             .metric-card {{ background-color: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s; }}
#             .metric-card:hover {{ transform: translateY(-2px); }}
#             .Strong.Buy {{ color: #2E7D32; font-weight: bold; }}
#             .Buy {{ color: #388E3C; font-weight: bold; }}
#             .Hold {{ color: #FFA000; font-weight: bold; }}
#             .Watch {{ color: #757575; font-weight: bold; }}
#             .chart-container {{ margin-top: 20px; }}
#             .signal-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 15px; }}
#             .signal-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; transition: transform 0.2s; }}
#             .signal-card:hover {{ transform: translateY(-2px); }}
#             .signal-title {{ font-weight: bold; margin-bottom: 5px; }}
#             .signal-explanation {{ color: #666; font-size: 0.9em; }}
#             .explanation-text {{ font-size: 1.1em; color: #333; margin: 10px 0; padding: 10px; background-color: #e3f2fd; border-radius: 5px; }}
#             .rating-explanation {{ 
#                 padding: 20px;
#                 background-color: {
#                     '#e8f5e9' if rating == 'Strong Buy'
#                     else '#c8e6c9' if rating == 'Buy'
#                     else '#fff3e0' if rating == 'Hold'
#                     else '#f5f5f5'
#                 };
#                 border-radius: 8px;
#                 margin: 20px 0;
#             }}
#         </style>
#     </head>
#     <body>
#         <div class="header">
#             <h1>{symbol} Stock Analysis Report</h1>
#             <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
#         </div>
        
#         <div class="section">
#             <h2>Technical Analysis Summary</h2>
#             <div class="metrics">
#                 <div class="metric-card">
#                     <h3>Overall Rating</h3>
#                     <p class="{rating.replace(' ', '.')}">{rating}</p>
#                 </div>
#                 <div class="metric-card">
#                     <h3>Category</h3>
#                     <p>{category}</p>
#                 </div>
#                 <div class="metric-card">
#                     <h3>Technical Score</h3>
#                     <p>{tech_score}/100</p>
#                 </div>
#                 <div class="metric-card">
#                     <h3>Momentum</h3>
#                     <p>{momentum}</p>
#                 </div>
#                 <div class="metric-card">
#                     <h3>Trend</h3>
#                     <p>{trend}</p>
#                 </div>
#             </div>
            
#             <div class="rating-explanation">
#                 <h3>Why This Rating?</h3>
#                 {chr(10).join(f'<p class="explanation-text">• {exp}</p>' for exp in explanations)}
#             </div>
            
#             <h3>Technical Signals Breakdown</h3>
#             <div class="signal-grid">
#                 {chr(10).join(f'''
#                     <div class="signal-card">
#                         <div class="signal-title">{signal[0]}</div>
#                         <div class="signal-explanation">{signal[1]}</div>
#                     </div>
#                 ''' for signal in signals)}
#             </div>
#         </div>
        
#         <div class="section">
#             <h2>Technical Analysis Chart</h2>
#             <div class="chart-container">
#                 {fig.to_html(full_html=False, include_plotlyjs='cdn')}
#             </div>
#         </div>
#     </body>
#     </html>
#     """
    
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write(html_content)
    
#     return filename

# # === Strategy Builder ===

# class StrategyBuilder:
#     def __init__(self):
#         self.conditions = []
        
#         # Technical Indicators with descriptions
#         self.tech_indicators = {
#             "MA": {
#                 "name": "Moving Average",
#                 "conditions": ["price above", "price below", "golden cross", "death cross"],
#                 "description": "Shows average price over specific time period"
#             },
#             "MACD": {
#                 "name": "MACD",
#                 "conditions": ["bullish crossover", "bearish crossover", "above zero", "below zero"],
#                 "description": "Momentum indicator showing relationship between moving averages"
#             },
#             "RSI": {
#                 "name": "Relative Strength Index",
#                 "conditions": ["overbought (>70)", "oversold (<30)", "above 50", "below 50"],
#                 "description": "Momentum oscillator measuring price change velocity"
#             },
#             "BB": {
#                 "name": "Bollinger Bands",
#                 "conditions": ["price above upper", "price below lower", "squeeze", "expansion"],
#                 "description": "Shows volatility and potential price levels"
#             },
#             "VOL": {
#                 "name": "Volume",
#                 "conditions": ["above average", "below average", "increasing", "decreasing"],
#                 "description": "Trading volume analysis"
#             },
#             "ADX": {
#                 "name": "Average Directional Index",
#                 "conditions": ["strong trend (>25)", "weak trend (<20)", "strengthening", "weakening"],
#                 "description": "Measures trend strength"
#             }
#         }
        
#         # Fundamental Indicators with descriptions
#         self.fund_indicators = {
#             "MCAP": {
#                 "name": "Market Cap",
#                 "conditions": ["micro (<$300M)", "small ($300M-$2B)", "mid ($2B-$10B)", "large (>$10B)"],
#                 "description": "Total market value of company's shares"
#             },
#             "PE": {
#                 "name": "P/E Ratio", 
#                 "conditions": ["low (<15)", "moderate (15-25)", "high (>25)"],
#                 "description": "Price to earnings ratio"
#             },
#             "PB": {
#                 "name": "P/B Ratio",
#                 "conditions": ["below industry", "above industry"],
#                 "description": "Price to book value ratio"
#             },
#             "ROE": {
#                 "name": "Return on Equity",
#                 "conditions": ["strong (>15%)", "moderate (10-15%)", "weak (<10%)"],
#                 "description": "Net income relative to shareholder equity"
#             },
#             "DE": {
#                 "name": "Debt/Equity",
#                 "conditions": ["low (<0.5)", "moderate (0.5-1.5)", "high (>1.5)"],
#                 "description": "Total debt relative to equity"
#             }
#         }
    
#     def get_predefined_strategies(self):
#         """Return a dictionary of predefined screening strategies"""
#         return {
#             "PIOTROSKI": {
#                 "name": "Piotroski F-Score",
#                 "description": "Value investing strategy based on 9 fundamental criteria",
#                 "conditions": [
#                     "Fundamental: ROA positive",
#                     "Fundamental: Operating Cash Flow positive",
#                     "Fundamental: ROA increasing",
#                     "Fundamental: Operating Cash Flow > ROA",
#                     "Fundamental: Debt ratio decreasing",
#                     "Fundamental: Current ratio increasing",
#                     "Fundamental: No new shares issued",
#                     "Fundamental: Gross margin increasing",
#                     "Fundamental: Asset turnover increasing"
#                 ]
#             },
#             "TREND_VALUE": {
#                 "name": "Trending Value Strategy",
#                 "description": "Combines momentum with value metrics",
#                 "conditions": [
#                     "Technical: RSI above 50",
#                     "Technical: MA price above",
#                     "Technical: VOL above average",
#                     "Fundamental: PE low (<15)",
#                     "Fundamental: PB below industry"
#                 ]
#             },
#             "QUALITY": {
#                 "name": "Quality Growth Strategy",
#                 "description": "Focuses on high-quality growth stocks",
#                 "conditions": [
#                     "Fundamental: Revenue growth > 15%",
#                     "Fundamental: ROE strong (>15%)",
#                     "Fundamental: DE low (<0.5)",
#                     "Technical: MACD bullish crossover",
#                     "Technical: RSI above 50"
#                 ]
#             },
#             "DIVIDEND": {
#                 "name": "Dividend Growth Strategy",
#                 "description": "Targets stable dividend-paying companies",
#                 "conditions": [
#                     "Fundamental: Dividend yield > 2%",
#                     "Fundamental: Payout ratio < 75%",
#                     "Technical: BB squeeze",
#                     "Technical: ADX weak trend (<20)",
#                     "Technical: VOL above average"
#                 ]
#             }
#         }

#     def create_strategy(self):
#         """Create a new screening strategy interactively through command line"""
#         print("\n=== Stock Screening Strategy Builder ===")
#         print("Build your screening strategy by adding conditions.")
#         print("\nAvailable options:")
#         print("1. Use Predefined Strategy")
#         print("2. Build Custom Strategy")
        
#         choice = input("\nEnter your choice (1-2): ")
        
#         if choice == "1":
#             strategies = self.get_predefined_strategies()
#             print("\nAvailable Predefined Strategies:")
#             for key, strategy in strategies.items():
#                 print(f"\n{key}:")
#                 print(f"Name: {strategy['name']}")
#                 print(f"Description: {strategy['description']}")
            
#             strategy_choice = input("\nEnter strategy name (or press Enter to go back): ").strip()
#             if strategy_choice in strategies:
#                 self.conditions = strategies[strategy_choice]['conditions'].copy()
#                 print(f"\nLoaded {strategies[strategy_choice]['name']} strategy")
#                 return
        
#         print("\nBuild your custom screening strategy.")
#         print("Available indicator types:")
#         print("1. Technical Analysis")
#         print("2. Fundamental Analysis")
        
#         while True:
#             print("\nOptions:")
#             print("1. Add Technical Indicator")
#             print("2. Add Fundamental Indicator")
#             print("3. View Current Strategy")
#             print("4. Save Strategy")
#             print("5. Load Strategy")
#             print("6. Finish")
            
#             choice = input("\nEnter your choice (1-6): ")
            
#             if choice == "1":
#                 self.add_tech_condition()
#             elif choice == "2":
#                 self.add_fund_condition()
#             elif choice == "3":
#                 self.view_strategy()
#             elif choice == "4":
#                 self.save_strategy()
#             elif choice == "5":
#                 self.load_strategy()
#             elif choice == "6":
#                 if not self.conditions:
#                     confirm = input("No conditions added. Are you sure you want to finish? (y/n): ")
#                     if confirm.lower() != 'y':
#                         continue
#                 break
#             else:
#                 print("Invalid choice. Please try again.")
    
#     def add_tech_condition(self):
#         """Add a technical indicator condition through command line"""
#         print("\nAvailable Technical Indicators:")
#         for code, indicator in self.tech_indicators.items():
#             print(f"{code} - {indicator['name']}")
#             print(f"Description: {indicator['description']}")
            
#         code = input("\nEnter indicator code: ").upper()
#         if code not in self.tech_indicators:
#             print("Invalid indicator code")
#             return
            
#         print("\nAvailable conditions:")
#         for i, condition in enumerate(self.tech_indicators[code]['conditions'], 1):
#             print(f"{i}. {condition}")
            
#         try:
#             choice = int(input("\nSelect condition number: "))
#             if 1 <= choice <= len(self.tech_indicators[code]['conditions']):
#                 condition = self.tech_indicators[code]['conditions'][choice-1]
#                 value = ""
                
#                 if any(word in condition for word in ['above', 'below', 'between']):
#                     value = input("Enter value (if applicable): ")
                    
#                 cond_str = f"Technical: {self.tech_indicators[code]['name']} {condition}"
#                 if value:
#                     cond_str += f" {value}"
                    
#                 self.conditions.append(cond_str)
#                 print("Condition added successfully!")
#             else:
#                 print("Invalid condition number")
#         except ValueError:
#             print("Invalid input")
    
#     def add_fund_condition(self):
#         """Add a fundamental indicator condition through command line"""
#         print("\nAvailable Fundamental Indicators:")
#         for code, indicator in self.fund_indicators.items():
#             print(f"{code} - {indicator['name']}")
#             print(f"Description: {indicator['description']}")
            
#         code = input("\nEnter indicator code: ").upper()
#         if code not in self.fund_indicators:
#             print("Invalid indicator code")
#             return
            
#         print("\nAvailable conditions:")
#         for i, condition in enumerate(self.fund_indicators[code]['conditions'], 1):
#             print(f"{i}. {condition}")
            
#         try:
#             choice = int(input("\nSelect condition number: "))
#             if 1 <= choice <= len(self.fund_indicators[code]['conditions']):
#                 condition = self.fund_indicators[code]['conditions'][choice-1]
#                 cond_str = f"Fundamental: {self.fund_indicators[code]['name']} {condition}"
#                 self.conditions.append(cond_str)
#                 print("Condition added successfully!")
#             else:
#                 print("Invalid condition number")
#         except ValueError:
#             print("Invalid input")
    
#     def view_strategy(self):
#         """Display current strategy conditions"""
#         if not self.conditions:
#             print("\nNo conditions set in the strategy.")
#             return
            
#         print("\nCurrent Strategy Conditions:")
#         for i, condition in enumerate(self.conditions, 1):
#             print(f"{i}. {condition}")
    
#     def clear_conditions(self):
#         """Clear all conditions from the strategy"""
#         self.conditions = []
#         print("\nAll conditions cleared.")
    
#     def save_strategy(self):
#         """Save strategy to a file"""
#         if not self.conditions:
#             print("No strategy to save.")
#             return
            
#         filename = input("\nEnter filename to save strategy (press Enter for default 'strategy.json'): ")
#         if not filename:
#             filename = "strategy.json"
#         if not filename.endswith('.json'):
#             filename += '.json'
            
#         try:
#             with open(filename, 'w') as f:
#                 json.dump(self.conditions, f, indent=4)
#             print(f"Strategy saved to {filename}")
#         except Exception as e:
#             print(f"Error saving strategy: {e}")
    
#     def load_strategy(self):
#         """Load strategy from a file"""
#         filename = input("\nEnter filename to load strategy (press Enter for default 'strategy.json'): ")
#         if not filename:
#             filename = "strategy.json"
#         if not filename.endswith('.json'):
#             filename += '.json'
            
#         try:
#             with open(filename, 'r') as f:
#                 self.conditions = json.load(f)
#             print("Strategy loaded successfully!")
#             self.view_strategy()
#         except FileNotFoundError:
#             print(f"File {filename} not found.")
#         except Exception as e:
#             print(f"Error loading strategy: {e}")
    
#     def run(self):
#         """Main method to run the strategy builder"""
#         self.create_strategy()
#         return self.conditions
    
#     def apply_strategy(self):
#         """Apply the current strategy"""
#         if not self.conditions:
#             print("No conditions in strategy to apply.")
#             return False
#         return True

# # === Step 1: NSE Stock List and Unavailable Stocks Cache ===
# def load_unavailable_stocks():
#     """Load the list of previously identified unavailable stocks"""
#     try:
#         if os.path.exists('unavailable_stocks.json'):
#             with open('unavailable_stocks.json', 'r') as f:
#                 return set(json.load(f))
#     except Exception as e:
#         print(f"Error loading unavailable stocks cache: {e}")
#     return set()

# def save_unavailable_stocks(unavailable_stocks):
#     """Save the list of unavailable stocks"""
#     try:
#         with open('unavailable_stocks.json', 'w') as f:
#             json.dump(list(unavailable_stocks), f)
#     except Exception as e:
#         print(f"Error saving unavailable stocks cache: {e}")

# def get_nse_stocks():
#     """Get NSE stocks list excluding known unavailable stocks"""
#     try:
#         url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
#         stocks_df = pd.read_csv(url)
#         all_stocks = stocks_df["SYMBOL"].tolist()
        
#         # Load and filter out known unavailable stocks
#         unavailable_stocks = load_unavailable_stocks()
#         available_stocks = [s for s in all_stocks if s not in unavailable_stocks]
        
#         print(f"Loaded {len(all_stocks)} total stocks")
#         print(f"Filtered out {len(unavailable_stocks)} known unavailable stocks")
#         print(f"Proceeding with {len(available_stocks)} available stocks")
        
#         return available_stocks, unavailable_stocks
#     except Exception as e:
#         print(f"Error fetching NSE stocks: {e}")
#         return [], set()

# # === Step 2: Fetch Stock Data ===
# def get_stock_data(symbol, period="5y", interval="1d", retries=5, unavailable_stocks=None):
#     """Get stock data with retries and fallback periods"""
#     if unavailable_stocks is None:
#         unavailable_stocks = set()
        
#     if symbol in unavailable_stocks:
#         print(f"\rSkipping known unavailable stock: {symbol}")
#         return None
        
#     periods = ["5y", "2y", "1y", "6mo", "3mo", "1mo"]  # More fallback periods
    
#     for attempt in range(retries):
#         for test_period in periods:
#             try:
#                 if attempt > 0:  # Add increasing delays between retries
#                     delay = attempt * 2  # 2, 4, 6, 8, 10 seconds
#                     time.sleep(delay)
                
#                 print(f"\rFetching data for {symbol} ({test_period})...", end="")
#                 ticker = yf.Ticker(symbol + ".NS")
#                 df = ticker.history(period=test_period, interval=interval)
                
#                 if not df.empty:
#                     if len(df) >= 30:  # Reduced minimum required data points
#                         print(f"\rSuccessfully fetched {len(df)} days of data for {symbol}")
#                         return df
#                     else:
#                         print(f"\rInsufficient data points for {symbol} with {test_period} period ({len(df)} points)")
#                         continue
#             except Exception as e:
#                 error_msg = str(e).lower()
#                 if "not found" in error_msg or "delisted" in error_msg:
#                     print(f"\rSkipping {symbol}: Stock not found or delisted")
#                     unavailable_stocks.add(symbol)
#                     save_unavailable_stocks(unavailable_stocks)
#                     return None
#                 elif "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
#                     print(f"\rNetwork error for {symbol}, will retry...")
#                     time.sleep(3)  # Additional delay for network errors
#                     continue
#                 else:
#                     print(f"\rError fetching {symbol}: {str(e)}")
#                     continue
    
#     print(f"\rFailed to fetch data for {symbol} after {retries} attempts with all periods")
#     # Add to unavailable stocks if all attempts fail
#     unavailable_stocks.add(symbol)
#     save_unavailable_stocks(unavailable_stocks)
#     return None

# # === Step 3: Extended Technical Indicators ===
# def compute_indicators(df, window=14):
#     if df is None or len(df) < window + 1:
#         return df
        
#     # Bollinger Bands
#     bollinger = ta.volatility.BollingerBands(df["Close"], window=20)
#     df["BB_upper"] = bollinger.bollinger_hband()
#     df["BB_lower"] = bollinger.bollinger_lband()
#     df["BB_mid"] = bollinger.bollinger_mavg()

#     # On-Balance Volume (OBV)
#     df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()

#     # RSI
#     df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

#     # MACD
#     macd = ta.trend.MACD(df["Close"])
#     df["MACD"] = macd.macd()
#     df["MACD_Signal"] = macd.macd_signal()

#     # DMI & ADX
#     if len(df) >= window + 1:
#         try:
#             adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=window)
#             df["+DMI"] = adx.adx_pos()
#             df["-DMI"] = adx.adx_neg()
#             df["ADX"] = adx.adx()
#         except Exception:
#             df["+DMI"], df["-DMI"], df["ADX"] = None, None, None

#     # Volume MA
#     if "Volume" in df:
#         df["Vol_MA"] = df["Volume"].rolling(20).mean()

#     return df

# # === Step 4: Format Market Cap ===
# def format_market_cap(market_cap):
#     """Convert Market Cap into human readable units."""
#     if market_cap is None:
#         return None
#     try:
#         market_cap = float(market_cap)
#         if market_cap >= 1e12:   # Lakh Crore
#             return f"{market_cap/1e12:.2f} Lakh Cr"
#         elif market_cap >= 1e7:  # Crore
#             return f"{market_cap/1e7:.2f} Cr"
#         elif market_cap >= 1e9:  # Billion (fallback for foreign listings)
#             return f"{market_cap/1e9:.2f} Bn"
#         else:
#             return f"{market_cap:.0f}"
#     except:
#         return None

# # === Step 5: Check Conditions ===
# def calculate_piotroski_score(symbol):
#     """Calculate Piotroski F-Score (0-9) for a stock"""
#     try:
#         ticker = yf.Ticker(symbol + ".NS")
#         info = ticker.info
#         financials = ticker.financials
#         balance_sheet = ticker.balance_sheet
#         cash_flow = ticker.cashflow
        
#         if any(x is None for x in [financials, balance_sheet, cash_flow]):
#             print(f"Missing fundamental data for {symbol}")
#             return None
            
#         score = 0
        
#         # 1. Return on Assets (ROA) positive
#         try:
#             net_income = financials.loc['Net Income'].iloc[0]
#             total_assets = balance_sheet.loc['Total Assets'].iloc[0]
#             roa = net_income / total_assets
#             if roa > 0:
#                 score += 1
#         except:
#             pass
            
#         # 2. Operating Cash Flow positive
#         try:
#             operating_cash_flow = cash_flow.loc['Operating Cash Flow'].iloc[0]
#             if operating_cash_flow > 0:
#                 score += 1
#         except:
#             pass
            
#         # 3. ROA increasing
#         try:
#             roa_prev = financials.loc['Net Income'].iloc[1] / balance_sheet.loc['Total Assets'].iloc[1]
#             if roa > roa_prev:
#                 score += 1
#         except:
#             pass
            
#         # 4. OCF > ROA (Operating cash flow exceeds ROA)
#         try:
#             if operating_cash_flow / total_assets > roa:
#                 score += 1
#         except:
#             pass
            
#         # 5. Decreasing Debt Ratio
#         try:
#             long_term_debt = balance_sheet.loc['Long Term Debt'].iloc[0]
#             long_term_debt_prev = balance_sheet.loc['Long Term Debt'].iloc[1]
#             if long_term_debt < long_term_debt_prev:
#                 score += 1
#         except:
#             pass
            
#         # 6. Increasing Current Ratio
#         try:
#             current_assets = balance_sheet.loc['Current Assets'].iloc[0]
#             current_liab = balance_sheet.loc['Current Liabilities'].iloc[0]
#             current_ratio = current_assets / current_liab
            
#             current_assets_prev = balance_sheet.loc['Current Assets'].iloc[1]
#             current_liab_prev = balance_sheet.loc['Current Liabilities'].iloc[1]
#             current_ratio_prev = current_assets_prev / current_liab_prev
            
#             if current_ratio > current_ratio_prev:
#                 score += 1
#         except:
#             pass
            
#         # 7. No New Shares Issued
#         try:
#             shares = balance_sheet.loc['Share Issued'].iloc[0]
#             shares_prev = balance_sheet.loc['Share Issued'].iloc[1]
#             if shares <= shares_prev:
#                 score += 1
#         except:
#             pass
            
#         # 8. Increasing Gross Margin
#         try:
#             revenue = financials.loc['Total Revenue'].iloc[0]
#             cogs = financials.loc['Cost Of Revenue'].iloc[0]
#             gross_margin = (revenue - cogs) / revenue
            
#             revenue_prev = financials.loc['Total Revenue'].iloc[1]
#             cogs_prev = financials.loc['Cost Of Revenue'].iloc[1]
#             gross_margin_prev = (revenue_prev - cogs_prev) / revenue_prev
            
#             if gross_margin > gross_margin_prev:
#                 score += 1
#         except:
#             pass
            
#         # 9. Increasing Asset Turnover
#         try:
#             asset_turnover = revenue / total_assets
#             asset_turnover_prev = revenue_prev / balance_sheet.loc['Total Assets'].iloc[1]
#             if asset_turnover > asset_turnover_prev:
#                 score += 1
#         except:
#             pass
            
#         return score
            
#     except Exception as e:
#         print(f"Error calculating Piotroski score for {symbol}: {str(e)}")
#         return None

# def calculate_quality_metrics(symbol):
#     """Calculate quality metrics for a stock"""
#     try:
#         ticker = yf.Ticker(symbol + ".NS")
#         info = ticker.info
#         financials = ticker.financials
        
#         metrics = {}
        
#         # ROE (Return on Equity)
#         try:
#             net_income = financials.loc['Net Income'].iloc[0]
#             total_equity = ticker.balance_sheet.loc['Total Stockholder Equity'].iloc[0]
#             metrics['ROE'] = (net_income / total_equity) * 100
#         except:
#             metrics['ROE'] = None
            
#         # Profit Margin
#         try:
#             revenue = financials.loc['Total Revenue'].iloc[0]
#             metrics['Profit_Margin'] = (net_income / revenue) * 100
#         except:
#             metrics['Profit_Margin'] = None
            
#         # Debt to Equity
#         try:
#             total_debt = ticker.balance_sheet.loc['Total Debt'].iloc[0]
#             metrics['Debt_to_Equity'] = total_debt / total_equity
#         except:
#             metrics['Debt_to_Equity'] = None
            
#         # Revenue Growth
#         try:
#             current_revenue = financials.loc['Total Revenue'].iloc[0]
#             prev_revenue = financials.loc['Total Revenue'].iloc[1]
#             metrics['Revenue_Growth'] = ((current_revenue - prev_revenue) / prev_revenue) * 100
#         except:
#             metrics['Revenue_Growth'] = None
            
#         return metrics
            
#     except Exception as e:
#         print(f"Error calculating quality metrics for {symbol}: {str(e)}")
#         return None

# def check_fundamental_conditions(symbol, conditions):
#     """Check if a stock meets the fundamental conditions"""
#     try:
#         # Get fundamental metrics
#         f_score = calculate_piotroski_score(symbol)
#         quality = calculate_quality_metrics(symbol)
        
#         if f_score is None or quality is None:
#             return False
            
#         # Check Piotroski score
#         if any('piotroski' in cond.lower() for cond in conditions):
#             if f_score < 6:  # Usually consider stocks with F-score >= 6
#                 return False
                
#         # Check quality metrics
#         for metric, value in quality.items():
#             if value is None:
#                 continue
                
#             if metric == 'ROE' and value < 15:  # ROE > 15%
#                 return False
#             elif metric == 'Profit_Margin' and value < 10:  # Profit margin > 10%
#                 return False
#             elif metric == 'Debt_to_Equity' and value > 1:  # D/E < 1
#                 return False
#             elif metric == 'Revenue_Growth' and value < 15:  # Growth > 15%
#                 return False
                
#         return True
            
#     except Exception as e:
#         print(f"Error checking fundamental conditions for {symbol}: {str(e)}")
#         return False

# def check_conditions(symbol, custom_conditions=None):
#     """Check if a stock meets the screening conditions with more flexible validation.
    
#     Args:
#         symbol (str): Stock symbol to check
#         custom_conditions (list, optional): List of custom conditions to check. Defaults to None.
        
#     Returns:
#         dict: Results of condition checks, or None if conditions not met
#     """
#     def safe_get_value(df, col, default=None):
#         """Safely get numeric value from dataframe with default"""
#         try:
#             val = df[col]
#             return float(val) if not pd.isna(val) else default
#         except:
#             return default

#     try:
#         # Check fundamental conditions if specified
#         if custom_conditions and any('fundamental:' in cond.lower() for cond in custom_conditions):
#             if not check_fundamental_conditions(symbol, custom_conditions):
#                 print(f"Failed fundamental check for {symbol}")
#                 return None
                
#         # Get stock data
#         df = get_stock_data(symbol)
#         if df is None or df.empty:
#             print(f"No data available for {symbol}")
#             return None

#         # Ensure minimum data length with flexible requirement
#         min_required_length = 30  # Reduced from 50
#         if len(df) < min_required_length:
#             print(f"Insufficient data for {symbol} (need {min_required_length} points, got {len(df)})")
#             return None

#         # Compute technical indicators with fallback
#         df = compute_indicators(df)
#         if df is None:
#             print(f"Failed to compute indicators for {symbol}")
#             return None
            
#         # Initialize conditions and attempt to calculate
#         cond1 = cond2 = cond3 = cond4 = cond5 = cond6 = cond7 = False
        
#         if len(df) >= 2:
#             latest = df.iloc[-1].copy()
#             prev = df.iloc[-2].copy()
            
#             # Daily MACD
#             macd_latest = safe_get_value(latest, 'MACD')
#             macd_prev = safe_get_value(prev, 'MACD')
#             if macd_latest is not None and macd_prev is not None:
#                 cond1 = macd_latest > 0 and macd_latest > macd_prev

#             # RSI with more flexible threshold
#             rsi_latest = safe_get_value(latest, 'RSI')
#             if rsi_latest is not None:
#                 cond4 = rsi_latest > 55  # Slightly more lenient

#             # DMI
#             dmi_plus = safe_get_value(latest, '+DMI')
#             dmi_minus = safe_get_value(latest, '-DMI')
#             if dmi_plus is not None and dmi_minus is not None:
#                 cond5 = dmi_plus > dmi_minus

#             # ADX
#             adx_latest = safe_get_value(latest, 'ADX')
#             adx_prev = safe_get_value(prev, 'ADX')
#             if adx_latest is not None and adx_prev is not None:
#                 cond6 = adx_latest > adx_prev

#             # Volume
#             vol_latest = safe_get_value(latest, 'Volume')
#             vol_ma = safe_get_value(latest, 'Vol_MA')
#             if vol_latest is not None and vol_ma is not None:
#                 cond7 = vol_latest > vol_ma * 0.9  # 90% of MA is acceptable

#         # Weekly MACD with fallback
#         try:
#             df_weekly = df.resample("W").last()
#             if len(df_weekly) >= 3:
#                 df_weekly = compute_indicators(df_weekly)
#                 if "MACD" in df_weekly.columns:
#                     w_latest = df_weekly.iloc[-1]
#                     w_prev = df_weekly.iloc[-2]
#                     w_macd_latest = safe_get_value(w_latest, 'MACD')
#                     w_macd_prev = safe_get_value(w_prev, 'MACD')
#                     if w_macd_latest is not None and w_macd_prev is not None:
#                         cond2 = w_macd_latest > 0 and w_macd_latest > w_macd_prev
#         except Exception as e:
#             print(f"Weekly MACD calculation failed for {symbol}: {str(e)}")

#         # Monthly MACD with fallback
#         try:
#             df_monthly = df.resample("M").last()
#             if len(df_monthly) >= 3:
#                 df_monthly = compute_indicators(df_monthly)
#                 if "MACD" in df_monthly.columns:
#                     m_latest = df_monthly.iloc[-1]
#                     m_prev = df_monthly.iloc[-2]
#                     m_macd_latest = safe_get_value(m_latest, 'MACD')
#                     m_macd_prev = safe_get_value(m_prev, 'MACD')
#                     if m_macd_latest is not None and m_macd_prev is not None:
#                         cond3 = m_macd_latest > 0 and m_macd_latest > m_macd_prev
#         except Exception as e:
#             print(f"Monthly MACD calculation failed for {symbol}: {str(e)}")

#         # Get market cap with fallback
#         market_cap = None
#         try:
#             ticker = yf.Ticker(symbol + ".NS")
#             market_cap = ticker.fast_info.get("market_cap", None)
#             if market_cap is None:
#                 market_cap = ticker.info.get("marketCap", None)
#         except Exception as e:
#             print(f"Market cap retrieval failed for {symbol}: {str(e)}")

#         market_cap_fmt = format_market_cap(market_cap)

#         # Create results dictionary
#         conditions = {
#             "Daily MACD Up": cond1,
#             "Weekly MACD Up": cond2,
#             "Monthly MACD Up": cond3,
#             "RSI > 55": cond4,  # Updated label
#             "+DMI > -DMI": cond5,
#             "ADX Rising": cond6,
#             "Vol > MA (90%)": cond7  # Updated label
#         }

#         satisfied = sum(conditions.values())

#         # Check if all MACD conditions are met (mandatory requirement)
#         macd_conditions = [
#             conditions["Daily MACD Up"],
#             conditions["Weekly MACD Up"],
#             conditions["Monthly MACD Up"]
#         ]
        
#         # Stock must have positive MACD across all timeframes
#         if not all(macd_conditions):
#             return None
            
#         # Must meet at least 4 conditions total (including MACD conditions)
#         min_conditions = 4
#         if satisfied >= min_conditions:
#             return {
#                 "Symbol": symbol,
#                 "Market Cap": market_cap_fmt,
#                 "Market Cap (Raw)": market_cap,
#                 "Conditions Met": satisfied,
#                 **conditions
#             }
#         return None

#     except Exception as e:
#         print(f"Error in check_conditions for {symbol}: {str(e)}")
#         return None

#     except Exception as e:
#         print(f"Error in check_conditions for {symbol}: {str(e)}")
#         return None

#         # Initialize all conditions
#         cond1 = cond2 = cond3 = cond4 = cond5 = cond6 = cond7 = False
        
#         # Helper function to safely check numeric values
#         def is_valid_and_greater(val1, val2=None):
#             if pd.isna(val1) or (val2 is not None and pd.isna(val2)):
#                 return False
#             if val2 is None:
#                 return True
#             return float(val1) > float(val2)

#         # Calculate daily conditions
#         if not pd.isna(latest["MACD"]) and not pd.isna(prev["MACD"]):
#             cond1 = float(latest["MACD"]) > 0 and float(latest["MACD"]) > float(prev["MACD"])
        
#         if not pd.isna(latest["RSI"]):
#             cond4 = float(latest["RSI"]) > 60

#         if "+DMI" in df.columns and "-DMI" in df.columns:
#             if not pd.isna(latest["+DMI"]) and not pd.isna(latest["-DMI"]):
#                 cond5 = float(latest["+DMI"]) > float(latest["-DMI"])

#         if "ADX" in df.columns:
#             if not pd.isna(latest["ADX"]) and not pd.isna(prev["ADX"]):
#                 cond6 = float(latest["ADX"]) > float(prev["ADX"])

#         if not pd.isna(latest["Volume"]) and not pd.isna(latest["Vol_MA"]):
#             cond7 = float(latest["Volume"]) > float(latest["Vol_MA"])

#         # Calculate weekly conditions
#         df_weekly = df.resample("W").last()
#         df_weekly = compute_indicators(df_weekly)
#         if len(df_weekly) >= 3 and "MACD" in df_weekly.columns:
#             latest_w = df_weekly.iloc[-1]
#             prev_w = df_weekly.iloc[-2]
#             if not pd.isna(latest_w["MACD"]) and not pd.isna(prev_w["MACD"]):
#                 cond2 = float(latest_w["MACD"]) > 0 and float(latest_w["MACD"]) > float(prev_w["MACD"])

#         # Calculate monthly conditions
#         df_monthly = df.resample("M").last()
#         df_monthly = compute_indicators(df_monthly)
#         if len(df_monthly) >= 3 and "MACD" in df_monthly.columns:
#             latest_m = df_monthly.iloc[-1]
#             prev_m = df_monthly.iloc[-2]
#             if not pd.isna(latest_m["MACD"]) and not pd.isna(prev_m["MACD"]):
#                 cond3 = float(latest_m["MACD"]) > 0 and float(latest_m["MACD"]) > float(prev_m["MACD"])

#         # Get market cap
#         market_cap = None
#         try:
#             ticker = yf.Ticker(symbol + ".NS")
#             market_cap = ticker.fast_info.get("market_cap", None)
#             if market_cap is None:
#                 market_cap = ticker.info.get("marketCap", None)
#         except:
#             pass

#         market_cap_fmt = format_market_cap(market_cap)

#         # Create results dictionary
#         conditions = {
#             "Daily MACD Up": cond1,
#             "Weekly MACD Up": cond2,
#             "Monthly MACD Up": cond3,
#             "RSI > 60": cond4,
#             "+DMI > -DMI": cond5,
#             "ADX Rising": cond6,
#             "Vol > MA": cond7
#         }

#         satisfied = sum(conditions.values())

#         return {
#             "Symbol": symbol,
#             "Market Cap": market_cap_fmt,    # formatted for display
#             "Market Cap (Raw)": market_cap,  # numeric for sorting
#             "Conditions Met": satisfied,
#             **conditions
#         }
#     except Exception as e:
#         print(f"Error in check_conditions for {symbol}: {str(e)}")
#         return None

# # === ML and Backtesting Functions ===
# def prepare_features(df):
#     """Prepare features for ML model."""
#     features = df[['RSI', 'MACD', 'ADX', 'BB_upper', 'BB_lower', 'OBV']].copy()
#     features = features.fillna(method='ffill')
#     return features

# def prepare_labels(df, forward_days=5):
#     """Create labels for ML based on future returns."""
#     df['Future_Returns'] = df['Close'].shift(-forward_days) / df['Close'] - 1
#     labels = (df['Future_Returns'] > 0).astype(int)
#     return labels[:-forward_days]  # Remove last rows where we don't have future data

# def train_ml_model(symbol, cv_splits=5):
#     """Train ML model with cross-validation."""
#     df = get_stock_data(symbol)
#     if df is None:
#         return None
    
#     df = compute_indicators(df)
#     features = prepare_features(df)
#     labels = prepare_labels(df)
    
#     if features.empty or labels.empty:
#         return None
    
#     # Time series cross-validation
#     tscv = TimeSeriesSplit(n_splits=cv_splits)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
    
#     scores = []
#     for train_idx, test_idx in tscv.split(features):
#         X_train = features.iloc[train_idx]
#         X_test = features.iloc[test_idx]
#         y_train = labels.iloc[train_idx]
#         y_test = labels.iloc[test_idx]
        
#         model.fit(X_train, y_train)
#         score = model.score(X_test, y_test)
#         scores.append(score)
    
#     return {
#         'model': model,
#         'avg_score': np.mean(scores),
#         'std_score': np.std(scores)
#     }

# def get_sentiment_score(symbol, days=30):
#     """Get news sentiment score for a stock."""
#     try:
#         ticker = yf.Ticker(symbol + ".NS")
#         news = ticker.news
        
#         if not news:
#             return 0
        
#         sentiments = []
#         for article in news:
#             analysis = TextBlob(article.get('title', ''))
#             sentiments.append(analysis.sentiment.polarity)
        
#         return np.mean(sentiments) if sentiments else 0
#     except:
#         return 0

# # === Step 6: Main Screener ===
# def screen_stocks(custom_conditions=None, min_conditions=4, use_ml=True, use_sentiment=True, progress_callback=None):
#     stock_list, unavailable_stocks = get_nse_stocks()
#     total_stocks = len(stock_list)
    
#     # Print initial statistics
#     print("\n=== Stock Screening Statistics ===")
#     print(f"Total stocks in NSE: {total_stocks + len(unavailable_stocks)}")
#     print(f"Previously identified unavailable stocks: {len(unavailable_stocks)}")
#     print(f"Stocks to be screened: {total_stocks}")
#     print("================================\n")
    
#     # Determine optimal chunk size and number of workers
#     num_cores = min(mp.cpu_count(), max(1, total_stocks // 20))  # Limit cores based on dataset size
#     chunk_size = max(1, math.ceil(total_stocks / (num_cores * 4)))  # Ensure minimum chunk size of 1
#     stock_chunks = [stock_list[i:i + chunk_size] for i in range(0, len(stock_list), chunk_size)]
    
#     if total_stocks < 100:  # For very small datasets, use fewer cores
#         num_cores = max(1, num_cores // 2)
    
#     print(f"\n🔍 Screening NSE Stocks using {num_cores} CPU cores...")
#     all_results = []
#     all_analysis_data = {}
    
#     # Initialize progress bar with more info
#     pbar = tqdm(total=total_stocks, 
#                 desc="Screening Stocks", 
#                 unit="stock",
#                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]')
    
#     successful_stocks = 0
#     failed_stocks = 0
    
#     # Prepare chunk data
#     chunk_data_list = [
#         {
#             'symbols': chunk,
#             'custom_conditions': custom_conditions,
#             'use_ml': use_ml,
#             'use_sentiment': use_sentiment
#         }
#         for chunk in stock_chunks
#     ]
    
#     # Use ProcessPoolExecutor for parallel processing
#     with ProcessPoolExecutor(max_workers=num_cores) as executor:
#         # Submit all chunks for processing
#         future_to_chunk = {executor.submit(process_stock_chunk, chunk_data): chunk_data['symbols'] 
#                           for chunk_data in chunk_data_list}
        
#         # Process completed chunks
#         for future in as_completed(future_to_chunk):
#             chunk = future_to_chunk[future]
#             try:
#                 chunk_results, chunk_analysis = future.result()
#                 all_results.extend(chunk_results)
#                 all_analysis_data.update(chunk_analysis)
                
#                 # Update progress with more detailed stats
#                 successful_chunk = len([r for r in chunk_results if r is not None])
#                 failed_chunk = len(chunk) - successful_chunk
#                 successful_stocks += successful_chunk
#                 failed_stocks += failed_chunk
                
#                 # Update progress bar
#                 pbar.update(len(chunk))
#                 # Add detailed statistics to progress bar
#                 pbar.set_postfix_str(
#                     f"Success: {successful_stocks}, Failed: {failed_stocks}, "
#                     f"Found: {len(all_results)} matches"
#                 )
#             except Exception as e:
#                 print(f"Error processing chunk: {e}")
    
#     # Close progress bar
#     pbar.close()
    
#     # Add analysis links to results
#     for result in all_results:
#         symbol = result['Symbol']
#         result['Analysis'] = f"Click to view detailed analysis of {symbol}"
    
#     return all_results, all_analysis_data



# # === Main Function ===

# def main():
#     print("\n=== Advanced Stock Screening Tool ===")
#     print("This tool helps you find promising stocks based on technical and fundamental analysis.")
#     print("\nWhat it analyzes:")
#     print("✓ Price trends and momentum")
#     print("✓ Volume patterns")
#     print("✓ Market capitalization")
#     print("✓ Technical indicators (MACD, RSI, ADX)")
#     print("✓ Multiple timeframes (Daily, Weekly, Monthly)")
    
#     # Initialize strategy builder
#     strategy_builder = StrategyBuilder()
    
#     # Get user preferences
#     print("\nChoose screening mode:")
#     print("1. Quick Scan (Default settings)")
#     print("2. Custom Strategy (Build your own)")
#     print("3. Deep Analysis (ML + Sentiment)")
    
#     mode = input("\nEnter mode (1-3): ").strip()
    
#     custom_conditions = None
#     use_ml = False
#     use_sentiment = False
    
#     if mode == "2":
#         custom_conditions = strategy_builder.run()
#     elif mode == "3":
#         use_ml = True
#         use_sentiment = True
#         print("\nInitiating deep analysis mode...")
    
#     print("\nStarting stock screening...")
#     print("This may take a few minutes. The system will analyze multiple timeframes for each stock.")
    
#     results, analysis_data = screen_stocks(
#         custom_conditions=custom_conditions,
#         use_ml=use_ml,
#         use_sentiment=use_sentiment
#     )
    
#     # Process results with rating system
#     from stock_ratings import get_stock_rating
#     from report_generator import create_summary_report
    
#     if results:
#         # Convert each result to a rating
#         rated_results = [get_stock_rating(result) for result in results if result]
#         rated_results = [r for r in rated_results if r]  # Remove None values
        
#         if rated_results:
#             # Create a user-friendly report
#             report_file = create_summary_report(rated_results)
            
#             print("\n=== Screening Complete! ===")
#             print(f"Found {len(rated_results)} stocks matching criteria.")
#             print("\nResults Summary:")
            
#             # Convert to DataFrame for analysis
#             results_df = pd.DataFrame(rated_results)
            
#     if not results_df.empty:
#         # Sort results by Technical Score and Market Cap
#         results_df = results_df.sort_values(
#             by=["Technical Score", "Market Cap"], 
#             ascending=[False, False]
#         ).reset_index(drop=True)

#         # Create timestamp for consistent filenames
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
#         # Pre-generate reports for all stocks
#         print("\nGenerating individual stock reports...")
#         stock_reports = {}
        
#         # Create reports directory if it doesn't exist
#         if not os.path.exists('stock_reports'):
#             os.makedirs('stock_reports')
            
#         for symbol in tqdm(results_df['Symbol'], desc="Generating stock reports"):
#             if symbol in analysis_data:
#                 report_path = f"stock_reports/{symbol}_analysis_{timestamp}.html"
#                 save_analysis_report(
#                     symbol,
#                     analysis_data[symbol],
#                     results_df[results_df["Symbol"] == symbol].to_dict('records')[0]
#                 )
#                 stock_reports[symbol] = report_path
        
#         # Group by rating and show counts
#         rating_counts = results_df['Overall Rating'].value_counts()
        
#         print("\nStocks by Rating:")
#         for rating, count in rating_counts.items():
#             print(f"  {rating}: {count} stocks")
        
#         # Show top picks in each category
#         print("\nTop Picks by Category:")
#         for rating in ['Strong Buy', 'Buy', 'Hold', 'Watch']:
#             if rating in rating_counts:
#                 category_stocks = results_df[results_df['Overall Rating'] == rating]
#                 print(f"\n{rating} Recommendations:")
#                 for _, stock in category_stocks.head(3).iterrows():
#                     print(f"  • {stock['Symbol']}")
#                     print(f"    - Category: {stock['Category']}")
#                     print(f"    - Technical Score: {stock['Technical Score']}/100")
#                     print(f"    - Momentum: {stock['Momentum']}")
#                     print(f"    - Trend: {stock['Trend']}")
        
#         # Add chart links to results DataFrame
#         results_df['Chart_Link'] = results_df['Symbol'].map(lambda x: stock_reports.get(x, ''))
        
#         # Generate main HTML report
#         report_file = f"stock_screening_report_{timestamp}.html"
#         create_summary_report(results_df, report_file)
        
#         print(f"\nDetailed HTML report has been generated: {report_file}")
        
#         # Ask to open report
#         if input("\nOpen detailed report in browser? (y/n): ").lower() == 'y':
#             webbrowser.open(f'file://{os.path.abspath(report_file)}')
        
#         # View individual stock analysis with pre-generated reports
#         while True:
#             symbol = input("\nEnter stock symbol for detailed technical analysis (or 'q' to quit): ").upper()
#             if symbol == 'Q':
#                 break
            
#             if symbol in stock_reports:
#                 webbrowser.open(f'file://{os.path.abspath(stock_reports[symbol])}')
#                 print(f"Technical analysis report opened in your browser")
#             else:
#                 print("Symbol not found in analysis data")
#     else:
#         print("No stocks matched the screening criteria.")
    
#     print("\nScreening completed!")
    
#     return results_df

# if __name__ == "__main__":
#     df = main()









import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings
import time
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from datetime import datetime
import json
import os
import multiprocessing as mp
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

# Import rating system and report generator
# Note: These files 'stock_ratings.py' and 'report_generator.py' must exist in the same directory.
# Since their code is not provided, their functions are assumed to work as named.
# from stock_ratings import get_stock_rating
# from report_generator import create_summary_report

warnings.filterwarnings("ignore")

# === Placeholder functions for missing imports ===
# These are placeholders since the original files were not provided.
def get_stock_rating(result):
    """Placeholder for the actual get_stock_rating function."""
    score = result.get('Conditions Met', 0) * 10
    if result.get('ML_Score'):
        try:
            score += float(result['ML_Score'].split(' ')[0]) * 10
        except:
            pass
    
    rating_map = {
        "Technical Score": score,
        "Momentum": "Strong" if result.get("Daily MACD Up") else "Weak",
        "Trend": "Positive" if result.get("+DMI > -DMI") else "Negative"
    }
    
    if score > 80:
        rating_map["Overall Rating"] = "Strong Buy"
        rating_map["Category"] = "High Momentum"
    elif score > 60:
        rating_map["Overall Rating"] = "Buy"
        rating_map["Category"] = "Trending"
    elif score > 40:
        rating_map["Overall Rating"] = "Hold"
        rating_map["Category"] = "Neutral"
    else:
        rating_map["Overall Rating"] = "Watch"
        rating_map["Category"] = "Needs Confirmation"
        
    return {**result, **rating_map}

def create_summary_report(results_df, report_file):
    """Placeholder for the actual create_summary_report function."""
    if isinstance(results_df, list):
        results_df = pd.DataFrame(results_df)
    
    if not results_df.empty:
        results_df.to_html(report_file, escape=False, index=False)
        print(f"Generated placeholder report at {report_file}")
        return report_file
    return None

# === Parallel Processing Functions ===
def process_stock_chunk(chunk_data):
    """Process a chunk of stocks in parallel"""
    try:
        symbols = chunk_data['symbols']
        custom_conditions = chunk_data.get('custom_conditions')
        use_ml = chunk_data.get('use_ml', True)
        use_sentiment = chunk_data.get('use_sentiment', True)
        unavailable_stocks = chunk_data.get('unavailable_stocks', set())
        
        if not symbols:  # Check for empty chunk
            return [], {}
            
        results = []
        analysis_data = {}
        
        for symbol in symbols:
            try:
                # Get stock data and perform basic screening
                stock_data = get_stock_data(symbol)
                if stock_data is None:
                    continue
                    
                # Store data for visualization
                analysis_data[symbol] = compute_indicators(stock_data.copy())
                    
                # Basic technical screening
                result = check_conditions(symbol, custom_conditions)
                if not result:
                    continue
                    
                # Add ML predictions if enabled
                if use_ml:
                    ml_results = train_ml_model(symbol)
                    if ml_results:
                        result['ML_Score'] = f"{ml_results['avg_score']:.2f} (±{ml_results['std_score']:.2f})"
                        result['ML_Prediction'] = "Bullish" if ml_results['avg_score'] > 0.6 else "Neutral"
                
                # Add sentiment if enabled
                if use_sentiment:
                    sentiment = get_sentiment_score(symbol)
                    result['Sentiment'] = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
                    result['Sentiment_Score'] = f"{sentiment:.2f}"
                    
                results.append(result)
            except Exception as e:
                # This will catch errors from process_stock_chunk, but check_conditions has its own
                print(f"Error processing {symbol}: {str(e)}")
                continue
                
        return results, analysis_data
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return [], {}


# === Interactive Visualization Functions ===
def create_stock_analysis_chart(symbol, data):
    """Create an interactive chart for stock analysis"""
    fig = make_subplots(rows=3, cols=1,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2],
                        shared_xaxes=True)
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Price'),
        row=1, col=1
    )
    
    # Add Bollinger Bands if available
    if 'BB_upper' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper',
                                 line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower',
                                 line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume'),
        row=2, col=1
    )
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI'),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        yaxis_title='Price',
        yaxis2_title='Volume',
        yaxis3_title='RSI',
        xaxis3_title='Date',
        showlegend=False,
        height=800
    )
    
    return fig

def create_stock_summary_charts(df):
    """Create interactive charts summarizing the stock screening results"""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Stocks by Rating', 'Stocks by Category',
                                        'Distribution of Technical Scores', 'Distribution of Market Cap'))
    
    # Rating distribution
    rating_counts = df['Overall Rating'].value_counts()
    fig.add_trace(
        go.Bar(x=rating_counts.index, y=rating_counts.values, name='Ratings'),
        row=1, col=1
    )
    
    # Category distribution
    category_counts = df['Category'].value_counts()
    fig.add_trace(
        go.Bar(x=category_counts.index, y=category_counts.values, name='Categories'),
        row=1, col=2
    )
    
    # Technical Score distribution
    fig.add_trace(
        go.Histogram(x=df['Technical Score'], name='Technical Scores',
                     nbinsx=20, histnorm='percent'),
        row=2, col=1
    )
    
    # Market Cap distribution (log scale)
    market_caps = pd.to_numeric(df['Market Cap (Raw)'], errors='coerce')
    fig.add_trace(
        go.Histogram(x=np.log10(market_caps[market_caps > 0]), 
                     name='Market Cap (log10)', nbinsx=20),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def generate_analysis_explanation(analysis_results):
    """Generate a detailed explanation of the stock's rating and technical analysis"""
    explanations = []
    signals = []
    met_conditions = []
    failed_conditions = []
    
    # Technical Score Components
    tech_score = analysis_results.get('Technical Score', 0)
    if tech_score >= 80:
        explanations.append(f"Very strong technical indicators across multiple timeframes (Score: {tech_score}/100)")
    elif tech_score >= 60:
        explanations.append(f"Generally positive technical indicators (Score: {tech_score}/100)")
    else:
        explanations.append(f"Mixed technical signals (Score: {tech_score}/100)")
    
    # Momentum Analysis
    momentum_score = 0
    if analysis_results.get('Daily MACD Up', False):
        met_conditions.append("✅ Daily MACD is trending upward")
        signals.append(("✅ Daily MACD is trending upward", "Showing positive short-term price momentum"))
        momentum_score += 1
    else:
        failed_conditions.append("❌ Daily MACD needs improvement")
        signals.append(("❌ Daily MACD is not trending upward", "Short-term momentum needs strengthening"))
        
    if analysis_results.get('Weekly MACD Up', False):
        met_conditions.append("✅ Weekly MACD is bullish")
        signals.append(("✅ Weekly MACD is bullish", "Demonstrating strong medium-term trend"))
        momentum_score += 1
    else:
        failed_conditions.append("❌ Weekly MACD needs improvement")
        signals.append(("❌ Weekly MACD is not bullish", "Medium-term momentum needs development"))
        
    if analysis_results.get('Monthly MACD Up', False):
        met_conditions.append("✅ Monthly MACD is positive")
        signals.append(("✅ Monthly MACD is positive", "Strong long-term upward trend"))
        momentum_score += 1
    else:
        failed_conditions.append("❌ Monthly MACD needs improvement")
        signals.append(("❌ Monthly MACD is not positive", "Long-term trend needs strengthening"))
    
    # Trend Strength
    trend_score = 0
    if analysis_results.get('RSI > 55', False):
        met_conditions.append("✅ RSI above 55")
        signals.append(("✅ RSI above 55", "Strong momentum without being overbought"))
        trend_score += 1
    else:
        failed_conditions.append("❌ RSI below 55")
        signals.append(("❌ RSI below 55", "Momentum needs to improve"))
        
    if analysis_results.get('+DMI > -DMI', False):
        met_conditions.append("✅ Positive Directional Movement")
        signals.append(("✅ +DMI > -DMI", "Price showing upward directional strength"))
        trend_score += 1
    else:
        failed_conditions.append("❌ Negative Directional Movement")
        signals.append(("❌ +DMI < -DMI", "Price showing downward pressure"))
        
    if analysis_results.get('ADX Rising', False):
        met_conditions.append("✅ Rising Trend Strength")
        signals.append(("✅ ADX Rising", "Overall trend strength is increasing"))
        trend_score += 1
    else:
        failed_conditions.append("❌ Weakening Trend")
        signals.append(("❌ ADX Not Rising", "Trend strength is declining"))
    
    # Volume Analysis
    if analysis_results.get('Vol > MA (90%)', False):
        met_conditions.append("✅ Above-Average Volume")
        signals.append(("✅ Volume above average", "Strong trading interest and liquidity"))
    else:
        failed_conditions.append("❌ Below-Average Volume")
        signals.append(("❌ Volume below average", "Low trading interest"))
    
    # Market Cap Analysis
    market_cap = analysis_results.get('Market Cap', 'N/A')
    if market_cap and 'Cr' in market_cap:
        signals.append(("ℹ️ Market Cap: " + market_cap, "Company size and liquidity indication"))
    
    # Add overall summary
    total_conditions = len(met_conditions) + len(failed_conditions)
    met_count = len(met_conditions)
    explanations.append(f"\nMet {met_count} out of {total_conditions} technical conditions")
    
    if met_count >= 6:
        explanations.append("Exceptional technical setup with multiple confirming signals")
    elif met_count >= 4:
        explanations.append("Strong technical setup with required MACD confirmation")
    else:
        explanations.append("Basic criteria met but needs more confirming signals")
    
    # Add detailed breakdowns
    if met_conditions:
        explanations.append("\nStrong Points:")
        explanations.extend(met_conditions)
    
    if failed_conditions:
        explanations.append("\nAreas for Improvement:")
        explanations.extend(failed_conditions)
    
    return explanations, signals

def save_analysis_report(symbol, data, analysis_results):
    """Generate and save an HTML analysis report"""
    report_dir = "stock_reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    filename = f"{report_dir}/{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    # Create the technical analysis chart
    fig = create_stock_analysis_chart(symbol, data)
    
    # Format key metrics
    tech_score = analysis_results.get('Technical Score', 'N/A')
    momentum = analysis_results.get('Momentum', 'N/A')
    trend = analysis_results.get('Trend', 'N/A')
    rating = analysis_results.get('Overall Rating', 'N/A')
    category = analysis_results.get('Category', 'N/A')
    
    # Generate detailed explanation
    explanations, signals = generate_analysis_explanation(analysis_results)
    
    # Generate HTML report
    html_content = f"""
    <html>
    <head>
        <title>{symbol} Stock Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .section {{ margin: 20px 0; padding: 20px; border-radius: 5px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
            .metric-card {{ background-color: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s; }}
            .metric-card:hover {{ transform: translateY(-2px); }}
            .Strong.Buy {{ color: #2E7D32; font-weight: bold; }}
            .Buy {{ color: #388E3C; font-weight: bold; }}
            .Hold {{ color: #FFA000; font-weight: bold; }}
            .Watch {{ color: #757575; font-weight: bold; }}
            .chart-container {{ margin-top: 20px; }}
            .signal-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 15px; }}
            .signal-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; transition: transform 0.2s; }}
            .signal-card:hover {{ transform: translateY(-2px); }}
            .signal-title {{ font-weight: bold; margin-bottom: 5px; }}
            .signal-explanation {{ color: #666; font-size: 0.9em; }}
            .explanation-text {{ font-size: 1.1em; color: #333; margin: 10px 0; padding: 10px; background-color: #e3f2fd; border-radius: 5px; }}
            .rating-explanation {{ 
                padding: 20px;
                background-color: {
                    '#e8f5e9' if rating == 'Strong Buy'
                    else '#c8e6c9' if rating == 'Buy'
                    else '#fff3e0' if rating == 'Hold'
                    else '#f5f5f5'
                };
                border-radius: 8px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{symbol} Stock Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Technical Analysis Summary</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>Overall Rating</h3>
                    <p class="{rating.replace(' ', '.')}">{rating}</p>
                </div>
                <div class="metric-card">
                    <h3>Category</h3>
                    <p>{category}</p>
                </div>
                <div class="metric-card">
                    <h3>Technical Score</h3>
                    <p>{tech_score}/100</p>
                </div>
                <div class="metric-card">
                    <h3>Momentum</h3>
                    <p>{momentum}</p>
                </div>
                <div class="metric-card">
                    <h3>Trend</h3>
                    <p>{trend}</p>
                </div>
            </div>
            
            <div class="rating-explanation">
                <h3>Why This Rating?</h3>
                {chr(10).join(f'<p class="explanation-text">• {exp}</p>' for exp in explanations)}
            </div>
            
            <h3>Technical Signals Breakdown</h3>
            <div class="signal-grid">
                {chr(10).join(f'''
                    <div class="signal-card">
                        <div class="signal-title">{signal[0]}</div>
                        <div class="signal-explanation">{signal[1]}</div>
                    </div>
                ''' for signal in signals)}
            </div>
        </div>
        
        <div class="section">
            <h2>Technical Analysis Chart</h2>
            <div class="chart-container">
                {fig.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename

# === Strategy Builder ===

class StrategyBuilder:
    def __init__(self):
        self.conditions = []
        
        # Technical Indicators with descriptions
        self.tech_indicators = {
            "MA": {
                "name": "Moving Average",
                "conditions": ["price above", "price below", "golden cross", "death cross"],
                "description": "Shows average price over specific time period"
            },
            "MACD": {
                "name": "MACD",
                "conditions": ["bullish crossover", "bearish crossover", "above zero", "below zero"],
                "description": "Momentum indicator showing relationship between moving averages"
            },
            "RSI": {
                "name": "Relative Strength Index",
                "conditions": ["overbought (>70)", "oversold (<30)", "above 50", "below 50"],
                "description": "Momentum oscillator measuring price change velocity"
            },
            "BB": {
                "name": "Bollinger Bands",
                "conditions": ["price above upper", "price below lower", "squeeze", "expansion"],
                "description": "Shows volatility and potential price levels"
            },
            "VOL": {
                "name": "Volume",
                "conditions": ["above average", "below average", "increasing", "decreasing"],
                "description": "Trading volume analysis"
            },
            "ADX": {
                "name": "Average Directional Index",
                "conditions": ["strong trend (>25)", "weak trend (<20)", "strengthening", "weakening"],
                "description": "Measures trend strength"
            }
        }
        
        # Fundamental Indicators with descriptions
        self.fund_indicators = {
            "MCAP": {
                "name": "Market Cap",
                "conditions": ["micro (<$300M)", "small ($300M-$2B)", "mid ($2B-$10B)", "large (>$10B)"],
                "description": "Total market value of company's shares"
            },
            "PE": {
                "name": "P/E Ratio", 
                "conditions": ["low (<15)", "moderate (15-25)", "high (>25)"],
                "description": "Price to earnings ratio"
            },
            "PB": {
                "name": "P/B Ratio",
                "conditions": ["below industry", "above industry"],
                "description": "Price to book value ratio"
            },
            "ROE": {
                "name": "Return on Equity",
                "conditions": ["strong (>15%)", "moderate (10-15%)", "weak (<10%)"],
                "description": "Net income relative to shareholder equity"
            },
            "DE": {
                "name": "Debt/Equity",
                "conditions": ["low (<0.5)", "moderate (0.5-1.5)", "high (>1.5)"],
                "description": "Total debt relative to equity"
            }
        }
    
    def get_predefined_strategies(self):
        """Return a dictionary of predefined screening strategies"""
        return {
            "PIOTROSKI": {
                "name": "Piotroski F-Score",
                "description": "Value investing strategy based on 9 fundamental criteria",
                "conditions": [
                    "Fundamental: ROA positive",
                    "Fundamental: Operating Cash Flow positive",
                    "Fundamental: ROA increasing",
                    "Fundamental: Operating Cash Flow > ROA",
                    "Fundamental: Debt ratio decreasing",
                    "Fundamental: Current ratio increasing",
                    "Fundamental: No new shares issued",
                    "Fundamental: Gross margin increasing",
                    "Fundamental: Asset turnover increasing"
                ]
            },
            "TREND_VALUE": {
                "name": "Trending Value Strategy",
                "description": "Combines momentum with value metrics",
                "conditions": [
                    "Technical: RSI above 50",
                    "Technical: MA price above",
                    "Technical: VOL above average",
                    "Fundamental: PE low (<15)",
                    "Fundamental: PB below industry"
                ]
            },
            "QUALITY": {
                "name": "Quality Growth Strategy",
                "description": "Focuses on high-quality growth stocks",
                "conditions": [
                    "Fundamental: Revenue growth > 15%",
                    "Fundamental: ROE strong (>15%)",
                    "Fundamental: DE low (<0.5)",
                    "Technical: MACD bullish crossover",
                    "Technical: RSI above 50"
                ]
            },
            "DIVIDEND": {
                "name": "Dividend Growth Strategy",
                "description": "Targets stable dividend-paying companies",
                "conditions": [
                    "Fundamental: Dividend yield > 2%",
                    "Fundamental: Payout ratio < 75%",
                    "Technical: BB squeeze",
                    "Technical: ADX weak trend (<20)",
                    "Technical: VOL above average"
                ]
            }
        }

    def create_strategy(self):
        """Create a new screening strategy interactively through command line"""
        print("\n=== Stock Screening Strategy Builder ===")
        print("Build your screening strategy by adding conditions.")
        print("\nAvailable options:")
        print("1. Use Predefined Strategy")
        print("2. Build Custom Strategy")
        
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == "1":
            strategies = self.get_predefined_strategies()
            print("\nAvailable Predefined Strategies:")
            for key, strategy in strategies.items():
                print(f"\n{key}:")
                print(f"Name: {strategy['name']}")
                print(f"Description: {strategy['description']}")
            
            strategy_choice = input("\nEnter strategy name (or press Enter to go back): ").strip()
            if strategy_choice in strategies:
                self.conditions = strategies[strategy_choice]['conditions'].copy()
                print(f"\nLoaded {strategies[strategy_choice]['name']} strategy")
                return
        
        print("\nBuild your custom screening strategy.")
        print("Available indicator types:")
        print("1. Technical Analysis")
        print("2. Fundamental Analysis")
        
        while True:
            print("\nOptions:")
            print("1. Add Technical Indicator")
            print("2. Add Fundamental Indicator")
            print("3. View Current Strategy")
            print("4. Save Strategy")
            print("5. Load Strategy")
            print("6. Finish")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == "1":
                self.add_tech_condition()
            elif choice == "2":
                self.add_fund_condition()
            elif choice == "3":
                self.view_strategy()
            elif choice == "4":
                self.save_strategy()
            elif choice == "5":
                self.load_strategy()
            elif choice == "6":
                if not self.conditions:
                    confirm = input("No conditions added. Are you sure you want to finish? (y/n): ")
                    if confirm.lower() != 'y':
                        continue
                break
            else:
                print("Invalid choice. Please try again.")
    
    def add_tech_condition(self):
        """Add a technical indicator condition through command line"""
        print("\nAvailable Technical Indicators:")
        for code, indicator in self.tech_indicators.items():
            print(f"{code} - {indicator['name']}")
            print(f"Description: {indicator['description']}")
            
        code = input("\nEnter indicator code: ").upper()
        if code not in self.tech_indicators:
            print("Invalid indicator code")
            return
            
        print("\nAvailable conditions:")
        for i, condition in enumerate(self.tech_indicators[code]['conditions'], 1):
            print(f"{i}. {condition}")
            
        try:
            choice = int(input("\nSelect condition number: "))
            if 1 <= choice <= len(self.tech_indicators[code]['conditions']):
                condition = self.tech_indicators[code]['conditions'][choice-1]
                value = ""
                
                if any(word in condition for word in ['above', 'below', 'between']):
                    value = input("Enter value (if applicable): ")
                    
                cond_str = f"Technical: {self.tech_indicators[code]['name']} {condition}"
                if value:
                    cond_str += f" {value}"
                    
                self.conditions.append(cond_str)
                print("Condition added successfully!")
            else:
                print("Invalid condition number")
        except ValueError:
            print("Invalid input")
    
    def add_fund_condition(self):
        """Add a fundamental indicator condition through command line"""
        print("\nAvailable Fundamental Indicators:")
        for code, indicator in self.fund_indicators.items():
            print(f"{code} - {indicator['name']}")
            print(f"Description: {indicator['description']}")
            
        code = input("\nEnter indicator code: ").upper()
        if code not in self.fund_indicators:
            print("Invalid indicator code")
            return
            
        print("\nAvailable conditions:")
        for i, condition in enumerate(self.fund_indicators[code]['conditions'], 1):
            print(f"{i}. {condition}")
            
        try:
            choice = int(input("\nSelect condition number: "))
            if 1 <= choice <= len(self.fund_indicators[code]['conditions']):
                condition = self.fund_indicators[code]['conditions'][choice-1]
                cond_str = f"Fundamental: {self.fund_indicators[code]['name']} {condition}"
                self.conditions.append(cond_str)
                print("Condition added successfully!")
            else:
                print("Invalid condition number")
        except ValueError:
            print("Invalid input")
    
    def view_strategy(self):
        """Display current strategy conditions"""
        if not self.conditions:
            print("\nNo conditions set in the strategy.")
            return
            
        print("\nCurrent Strategy Conditions:")
        for i, condition in enumerate(self.conditions, 1):
            print(f"{i}. {condition}")
    
    def clear_conditions(self):
        """Clear all conditions from the strategy"""
        self.conditions = []
        print("\nAll conditions cleared.")
    
    def save_strategy(self):
        """Save strategy to a file"""
        if not self.conditions:
            print("No strategy to save.")
            return
            
        filename = input("\nEnter filename to save strategy (press Enter for default 'strategy.json'): ")
        if not filename:
            filename = "strategy.json"
        if not filename.endswith('.json'):
            filename += '.json'
            
        try:
            with open(filename, 'w') as f:
                json.dump(self.conditions, f, indent=4)
            print(f"Strategy saved to {filename}")
        except Exception as e:
            print(f"Error saving strategy: {e}")
    
    def load_strategy(self):
        """Load strategy from a file"""
        filename = input("\nEnter filename to load strategy (press Enter for default 'strategy.json'): ")
        if not filename:
            filename = "strategy.json"
        if not filename.endswith('.json'):
            filename += '.json'
            
        try:
            with open(filename, 'r') as f:
                self.conditions = json.load(f)
            print("Strategy loaded successfully!")
            self.view_strategy()
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except Exception as e:
            print(f"Error loading strategy: {e}")
    
    def run(self):
        """Main method to run the strategy builder"""
        self.create_strategy()
        return self.conditions
    
    def apply_strategy(self):
        """Apply the current strategy"""
        if not self.conditions:
            print("No conditions in strategy to apply.")
            return False
        return True

# === Step 1: NSE Stock List and Unavailable Stocks Cache ===
def load_unavailable_stocks():
    """Load the list of previously identified unavailable stocks"""
    try:
        if os.path.exists('unavailable_stocks.json'):
            with open('unavailable_stocks.json', 'r') as f:
                return set(json.load(f))
    except Exception as e:
        print(f"Error loading unavailable stocks cache: {e}")
    return set()

def save_unavailable_stocks(unavailable_stocks):
    """Save the list of unavailable stocks"""
    try:
        with open('unavailable_stocks.json', 'w') as f:
            json.dump(list(unavailable_stocks), f)
    except Exception as e:
        print(f"Error saving unavailable stocks cache: {e}")

def get_nse_stocks():
    """Get NSE stocks list excluding known unavailable stocks"""
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        stocks_df = pd.read_csv(url)
        all_stocks = stocks_df["SYMBOL"].tolist()
        
        # Load and filter out known unavailable stocks
        unavailable_stocks = load_unavailable_stocks()
        available_stocks = [s for s in all_stocks if s not in unavailable_stocks]
        
        print(f"Loaded {len(all_stocks)} total stocks")
        print(f"Filtered out {len(unavailable_stocks)} known unavailable stocks")
        print(f"Proceeding with {len(available_stocks)} available stocks")
        
        return available_stocks, unavailable_stocks
    except Exception as e:
        print(f"Error fetching NSE stocks: {e}")
        return [], set()

# === Step 2: Fetch Stock Data ===
def get_stock_data(symbol, period="5y", interval="1d", retries=5, unavailable_stocks=None):
    """Get stock data with retries and fallback periods"""
    if unavailable_stocks is None:
        unavailable_stocks = set()
        
    if symbol in unavailable_stocks:
        #print(f"\rSkipping known unavailable stock: {symbol}")
        return None
        
    periods = ["5y", "2y", "1y", "6mo", "3mo", "1mo"]  # More fallback periods
    
    for attempt in range(retries):
        for test_period in periods:
            try:
                if attempt > 0:  # Add increasing delays between retries
                    delay = attempt * 2  # 2, 4, 6, 8, 10 seconds
                    time.sleep(delay)
                
                #print(f"\rFetching data for {symbol} ({test_period})...", end="")
                ticker = yf.Ticker(symbol + ".NS")
                df = ticker.history(period=test_period, interval=interval)
                
                if not df.empty:
                    if len(df) >= 30:  # Reduced minimum required data points
                        #print(f"\rSuccessfully fetched {len(df)} days of data for {symbol}")
                        return df
                    else:
                        #print(f"\rInsufficient data points for {symbol} with {test_period} period ({len(df)} points)")
                        continue
            except Exception as e:
                error_msg = str(e).lower()
                if "not found" in error_msg or "delisted" in error_msg:
                    #print(f"\rSkipping {symbol}: Stock not found or delisted")
                    unavailable_stocks.add(symbol)
                    save_unavailable_stocks(unavailable_stocks)
                    return None
                elif "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
                    #print(f"\rNetwork error for {symbol}, will retry...")
                    time.sleep(3)  # Additional delay for network errors
                    continue
                else:
                    #print(f"\rError fetching {symbol}: {str(e)}")
                    continue
    
    #print(f"\rFailed to fetch data for {symbol} after {retries} attempts with all periods")
    # Add to unavailable stocks if all attempts fail
    unavailable_stocks.add(symbol)
    save_unavailable_stocks(unavailable_stocks)
    return None

# === Step 3: Extended Technical Indicators ===
def compute_indicators(df, window=14):
    # FIX 1: Min length check increased to 30 to be safe for MACD(26) and BB(20)
    if df is None or len(df) < 30: 
        return df
        
    try:
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df["Close"], window=20)
        df["BB_upper"] = bollinger.bollinger_hband()
        df["BB_lower"] = bollinger.bollinger_lband()
        df["BB_mid"] = bollinger.bollinger_mavg()
    except Exception:
        # FIX 2: Added try/except block for resilience
        df["BB_upper"], df["BB_lower"], df["BB_mid"] = np.nan, np.nan, np.nan

    try:
        # On-Balance Volume (OBV)
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    except Exception:
        # FIX 2: Added try/except block for resilience
        df["OBV"] = np.nan

    try:
        # RSI
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    except Exception:
        # FIX 2: Added try/except block for resilience
        df["RSI"] = np.nan

    try:
        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
    except Exception:
        # FIX 2: Added try/except block for resilience
        df["MACD"], df["MACD_Signal"] = np.nan, np.nan

    # DMI & ADX
    # FIX 1: Check increased to 30 (to match main check)
    if len(df) >= 30:
        try:
            adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=window)
            df["+DMI"] = adx.adx_pos()
            df["-DMI"] = adx.adx_neg()
            df["ADX"] = adx.adx()
        except Exception:
            # Use np.nan for consistency
            df["+DMI"], df["-DMI"], df["ADX"] = np.nan, np.nan, np.nan
    else:
        df["+DMI"], df["-DMI"], df["ADX"] = np.nan, np.nan, np.nan

    try:
        # Volume MA
        if "Volume" in df:
            df["Vol_MA"] = df["Volume"].rolling(20).mean()
    except Exception:
         # FIX 2: Added try/except block for resilience
        df["Vol_MA"] = np.nan

    return df

# === Step 4: Format Market Cap ===
def format_market_cap(market_cap):
    """Convert Market Cap into human readable units."""
    if market_cap is None:
        return None
    try:
        market_cap = float(market_cap)
        if market_cap >= 1e12:   # Lakh Crore
            return f"{market_cap/1e12:.2f} Lakh Cr"
        elif market_cap >= 1e7:  # Crore
            return f"{market_cap/1e7:.2f} Cr"
        elif market_cap >= 1e9:  # Billion (fallback for foreign listings)
            return f"{market_cap/1e9:.2f} Bn"
        else:
            return f"{market_cap:.0f}"
    except:
        return None

# === Step 5: Check Conditions ===
def calculate_piotroski_score(symbol):
    """Calculate Piotroski F-Score (0-9) for a stock"""
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        
        # Check if any financial data is missing
        if financials is None or financials.empty or \
           balance_sheet is None or balance_sheet.empty or \
           cash_flow is None or cash_flow.empty:
            #print(f"Missing fundamental data for {symbol}")
            return None
        
        # Check if there are at least 2 years of data for comparison
        has_prev_year = len(financials.columns) > 1 and len(balance_sheet.columns) > 1
        
        score = 0
        
        # 1. Return on Assets (ROA) positive
        try:
            net_income = financials.loc['Net Income'].iloc[0]
            total_assets = balance_sheet.loc['Total Assets'].iloc[0]
            roa = net_income / total_assets
            if roa > 0:
                score += 1
        except:
            pass
            
        # 2. Operating Cash Flow positive
        try:
            operating_cash_flow = cash_flow.loc['Operating Cash Flow'].iloc[0]
            if operating_cash_flow > 0:
                score += 1
        except:
            pass
            
        # 3. ROA increasing (requires previous year)
        if has_prev_year:
            try:
                roa_prev = financials.loc['Net Income'].iloc[1] / balance_sheet.loc['Total Assets'].iloc[1]
                if roa > roa_prev:
                    score += 1
            except:
                pass
            
        # 4. OCF > ROA (Operating cash flow exceeds ROA)
        try:
            if operating_cash_flow / total_assets > roa:
                score += 1
        except:
            pass
            
        # 5. Decreasing Debt Ratio (requires previous year)
        if has_prev_year:
            try:
                long_term_debt = balance_sheet.loc['Long Term Debt'].iloc[0]
                long_term_debt_prev = balance_sheet.loc['Long Term Debt'].iloc[1]
                if long_term_debt < long_term_debt_prev:
                    score += 1
            except:
                pass
            
        # 6. Increasing Current Ratio (requires previous year)
        if has_prev_year:
            try:
                current_assets = balance_sheet.loc['Current Assets'].iloc[0]
                current_liab = balance_sheet.loc['Current Liabilities'].iloc[0]
                current_ratio = current_assets / current_liab
                
                current_assets_prev = balance_sheet.loc['Current Assets'].iloc[1]
                current_liab_prev = balance_sheet.loc['Current Liabilities'].iloc[1]
                current_ratio_prev = current_assets_prev / current_liab_prev
                
                if current_ratio > current_ratio_prev:
                    score += 1
            except:
                pass
            
        # 7. No New Shares Issued (requires previous year)
        if has_prev_year:
            try:
                shares = balance_sheet.loc['Share Issued'].iloc[0]
                shares_prev = balance_sheet.loc['Share Issued'].iloc[1]
                if shares <= shares_prev:
                    score += 1
            except:
                pass
            
        # 8. Increasing Gross Margin (requires previous year)
        if has_prev_year:
            try:
                revenue = financials.loc['Total Revenue'].iloc[0]
                cogs = financials.loc['Cost Of Revenue'].iloc[0]
                gross_margin = (revenue - cogs) / revenue
                
                revenue_prev = financials.loc['Total Revenue'].iloc[1]
                cogs_prev = financials.loc['Cost Of Revenue'].iloc[1]
                gross_margin_prev = (revenue_prev - cogs_prev) / revenue_prev
                
                if gross_margin > gross_margin_prev:
                    score += 1
            except:
                pass
            
        # 9. Increasing Asset Turnover (requires previous year)
        if has_prev_year:
            try:
                asset_turnover = revenue / total_assets
                asset_turnover_prev = revenue_prev / balance_sheet.loc['Total Assets'].iloc[1]
                if asset_turnover > asset_turnover_prev:
                    score += 1
            except:
                pass
            
        return score
            
    except Exception as e:
        #print(f"Error calculating Piotroski score for {symbol}: {str(e)}")
        return None

def calculate_quality_metrics(symbol):
    """Calculate quality metrics for a stock"""
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        financials = ticker.financials
        
        # Check if any financial data is missing
        if financials is None or financials.empty or \
           ticker.balance_sheet is None or ticker.balance_sheet.empty:
            #print(f"Missing fundamental data for {symbol}")
            return None

        has_prev_year = len(financials.columns) > 1
        
        metrics = {}
        
        # ROE (Return on Equity)
        try:
            net_income = financials.loc['Net Income'].iloc[0]
            total_equity = ticker.balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            metrics['ROE'] = (net_income / total_equity) * 100
        except:
            metrics['ROE'] = None
            
        # Profit Margin
        try:
            revenue = financials.loc['Total Revenue'].iloc[0]
            metrics['Profit_Margin'] = (net_income / revenue) * 100
        except:
            metrics['Profit_Margin'] = None
            
        # Debt to Equity
        try:
            total_debt = ticker.balance_sheet.loc['Total Debt'].iloc[0]
            metrics['Debt_to_Equity'] = total_debt / total_equity
        except:
            metrics['Debt_to_Equity'] = None
            
        # Revenue Growth
        if has_prev_year:
            try:
                current_revenue = financials.loc['Total Revenue'].iloc[0]
                prev_revenue = financials.loc['Total Revenue'].iloc[1]
                metrics['Revenue_Growth'] = ((current_revenue - prev_revenue) / prev_revenue) * 100
            except:
                metrics['Revenue_Growth'] = None
        else:
            metrics['Revenue_Growth'] = None
            
        return metrics
            
    except Exception as e:
        #print(f"Error calculating quality metrics for {symbol}: {str(e)}")
        return None

def check_fundamental_conditions(symbol, conditions):
    """Check if a stock meets the fundamental conditions"""
    try:
        # Get fundamental metrics
        f_score = calculate_piotroski_score(symbol)
        quality = calculate_quality_metrics(symbol)
        
        if f_score is None or quality is None:
            return False
            
        # Check Piotroski score
        if any('piotroski' in cond.lower() for cond in conditions):
            if f_score < 6:  # Usually consider stocks with F-score >= 6
                return False
                
        # Check quality metrics
        for metric, value in quality.items():
            if value is None:
                continue
                
            if metric == 'ROE' and value < 15:  # ROE > 15%
                return False
            elif metric == 'Profit_Margin' and value < 10:  # Profit margin > 10%
                return False
            elif metric == 'Debt_to_Equity' and value > 1:  # D/E < 1
                return False
            elif metric == 'Revenue_Growth' and value < 15:  # Growth > 15%
                return False
                
        return True
            
    except Exception as e:
        #print(f"Error checking fundamental conditions for {symbol}: {str(e)}")
        return False

def check_conditions(symbol, custom_conditions=None):
    """Check if a stock meets the screening conditions with more flexible validation.
    
    Args:
        symbol (str): Stock symbol to check
        custom_conditions (list, optional): List of custom conditions to check. Defaults to None.
        
    Returns:
        dict: Results of condition checks, or None if conditions not met
    """
    def safe_get_value(df, col, default=None):
        """Safely get numeric value from dataframe with default"""
        try:
            val = df[col]
            return float(val) if not pd.isna(val) else default
        except:
            return default

    try:
        # Check fundamental conditions if specified
        if custom_conditions and any('fundamental:' in cond.lower() for cond in custom_conditions):
            if not check_fundamental_conditions(symbol, custom_conditions):
                #print(f"Failed fundamental check for {symbol}")
                return None
                
        # Get stock data
        df = get_stock_data(symbol)
        if df is None or df.empty:
            #print(f"No data available for {symbol}")
            return None

        # Ensure minimum data length with flexible requirement
        min_required_length = 50
        if len(df) < min_required_length:
            #print(f"Insufficient data for {symbol} (need {min_required_length} points, got {len(df)})")
            return None

        # Compute technical indicators with fallback
        df = compute_indicators(df)
        if df is None:
            #print(f"Failed to compute indicators for {symbol}")
            return None
            
        # Initialize conditions and attempt to calculate
        cond1 = cond2 = cond3 = cond4 = cond5 = cond6 = cond7 = False
        
        if len(df) >= 2:
            latest = df.iloc[-1].copy()
            prev = df.iloc[-2].copy()
            
            # Daily MACD
            macd_latest = safe_get_value(latest, 'MACD')
            macd_prev = safe_get_value(prev, 'MACD')
            if macd_latest is not None and macd_prev is not None:
                cond1 = macd_latest > 0 and macd_latest > macd_prev

            # RSI with more flexible threshold
            rsi_latest = safe_get_value(latest, 'RSI')
            if rsi_latest is not None:
                cond4 = rsi_latest > 55  # Slightly more lenient

            # DMI
            dmi_plus = safe_get_value(latest, '+DMI')
            dmi_minus = safe_get_value(latest, '-DMI')
            if dmi_plus is not None and dmi_minus is not None:
                cond5 = dmi_plus > dmi_minus

            # ADX
            adx_latest = safe_get_value(latest, 'ADX')
            adx_prev = safe_get_value(prev, 'ADX')
            if adx_latest is not None and adx_prev is not None:
                cond6 = adx_latest > adx_prev

            # Volume
            vol_latest = safe_get_value(latest, 'Volume')
            vol_ma = safe_get_value(latest, 'Vol_MA')
            if vol_latest is not None and vol_ma is not None and vol_ma > 0: # Add vol_ma > 0 check
                cond7 = vol_latest > vol_ma * 0.9  # 90% of MA is acceptable

        # Weekly MACD with fallback
        try:
            df_weekly = df.resample("W").last()
            # FIX: Check for 30 weeks of data to ensure indicators can be calculated
            if len(df_weekly) >= 30:
                df_weekly = compute_indicators(df_weekly)
                if "MACD" in df_weekly.columns and not df_weekly["MACD"].isnull().all():
                    w_latest = df_weekly.iloc[-1]
                    w_prev = df_weekly.iloc[-2]
                    w_macd_latest = safe_get_value(w_latest, 'MACD')
                    w_macd_prev = safe_get_value(w_prev, 'MACD')
                    if w_macd_latest is not None and w_macd_prev is not None:
                        cond2 = w_macd_latest > 0 and w_macd_latest > w_macd_prev
        except Exception as e:
            #print(f"Weekly MACD calculation failed for {symbol}: {str(e)}")
            pass

        # Monthly MACD with fallback
        try:
            df_monthly = df.resample("M").last()
            # FIX: Check for 30 months of data to ensure indicators can be calculated
            if len(df_monthly) >= 30:
                df_monthly = compute_indicators(df_monthly)
                if "MACD" in df_monthly.columns and not df_monthly["MACD"].isnull().all():
                    m_latest = df_monthly.iloc[-1]
                    m_prev = df_monthly.iloc[-2]
                    m_macd_latest = safe_get_value(m_latest, 'MACD')
                    m_macd_prev = safe_get_value(m_prev, 'MACD')
                    if m_macd_latest is not None and m_macd_prev is not None:
                        cond3 = m_macd_latest > 0 and m_macd_latest > m_macd_prev
        except Exception as e:
            #print(f"Monthly MACD calculation failed for {symbol}: {str(e)}")
            pass

        # =====================================================================
        # OPTIMIZATION: Relaxed filter.
        # Original was: if not all([cond1, cond2, cond3]):
        # This is too strict. We now require at least 2 of 3 timeframes
        # to have a positive MACD signal.
        # =====================================================================
        if sum([cond1, cond2, cond3]) < 2:
            return None

        # Get market cap with fallback
        market_cap = None
        try:
            ticker = yf.Ticker(symbol + ".NS")
            market_cap = ticker.fast_info.get("market_cap", None)
            if market_cap is None:
                market_cap = ticker.info.get("marketCap", None)
        except Exception as e:
            #print(f"Market cap retrieval failed for {symbol}: {str(e)}")
            pass

        market_cap_fmt = format_market_cap(market_cap)

        # Create results dictionary
        conditions = {
            "Daily MACD Up": cond1,
            "Weekly MACD Up": cond2,
            "Monthly MACD Up": cond3,
            "RSI > 55": cond4,
            "+DMI > -DMI": cond5,
            "ADX Rising": cond6,
            "Vol > MA (90%)": cond7
        }

        satisfied = sum(conditions.values())
        
        # After passing the mandatory MACD check (2 of 3), ensure at least
        # one other condition is met for a total of 4.
        min_conditions = 4
        if satisfied >= min_conditions:
            return {
                "Symbol": symbol,
                "Market Cap": market_cap_fmt,
                "Market Cap (Raw)": market_cap,
                "Conditions Met": satisfied,
                **conditions
            }
        return None

    except Exception as e:
        #print(f"Error in check_conditions for {symbol}: {str(e)}")
        return None

# === ML and Backtesting Functions ===
def prepare_features(df):
    """Prepare features for ML model."""
    features = df[['RSI', 'MACD', 'ADX', 'BB_upper', 'BB_lower', 'OBV']].copy()
    features = features.fillna(method='ffill')
    return features

def prepare_labels(df, forward_days=5):
    """Create labels for ML based on future returns."""
    df['Future_Returns'] = df['Close'].shift(-forward_days) / df['Close'] - 1
    labels = (df['Future_Returns'] > 0).astype(int)
    return labels[:-forward_days]  # Remove last rows where we don't have future data

def train_ml_model(symbol, cv_splits=5):
    """Train ML model with cross-validation."""
    df = get_stock_data(symbol)
    if df is None:
        return None
    
    df = compute_indicators(df)
    features = prepare_features(df)
    labels = prepare_labels(df)
    
    if features.empty or labels.empty or features.isnull().values.any() or labels.isnull().values.any():
        return None
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    scores = []
    try:
        for train_idx, test_idx in tscv.split(features):
            X_train = features.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_train = labels.iloc[train_idx]
            y_test = labels.iloc[test_idx]
            
            if X_train.empty or y_train.empty:
                continue

            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
    except Exception as e:
        #print(f"Error during ML training for {symbol}: {e}")
        return None
    
    if not scores:
        return None

    return {
        'model': model,
        'avg_score': np.mean(scores),
        'std_score': np.std(scores)
    }

def get_sentiment_score(symbol, days=30):
    """Get news sentiment score for a stock."""
    try:
        ticker = yf.Ticker(symbol + ".NS")
        news = ticker.news
        
        if not news:
            return 0
        
        sentiments = []
        for article in news:
            analysis = TextBlob(article.get('title', ''))
            sentiments.append(analysis.sentiment.polarity)
        
        return np.mean(sentiments) if sentiments else 0
    except:
        return 0

# === Step 6: Main Screener ===
def screen_stocks(custom_conditions=None, min_conditions=4, use_ml=True, use_sentiment=True, progress_callback=None):
    stock_list, unavailable_stocks = get_nse_stocks()
    total_stocks = len(stock_list)
    
    # Print initial statistics
    print("\n=== Stock Screening Statistics ===")
    print(f"Total stocks in NSE: {total_stocks + len(unavailable_stocks)}")
    print(f"Previously identified unavailable stocks: {len(unavailable_stocks)}")
    print(f"Stocks to be screened: {total_stocks}")
    print("================================\n")
    
    # Determine optimal chunk size and number of workers
    num_cores = min(mp.cpu_count(), max(1, total_stocks // 20))  # Limit cores based on dataset size
    chunk_size = max(1, math.ceil(total_stocks / (num_cores * 4)))  # Ensure minimum chunk size of 1
    stock_chunks = [stock_list[i:i + chunk_size] for i in range(0, len(stock_list), chunk_size)]
    
    if total_stocks < 100:  # For very small datasets, use fewer cores
        num_cores = max(1, num_cores // 2)
    
    print(f"\nScreening NSE Stocks using {num_cores} CPU cores...")
    all_results = []
    all_analysis_data = {}
    
    # Initialize progress bar with more info
    pbar = tqdm(total=total_stocks, 
                desc="Screening Stocks", 
                unit="stock",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]')
    
    successful_stocks = 0
    failed_stocks = 0
    
    # Prepare chunk data
    chunk_data_list = [
        {
            'symbols': chunk,
            'custom_conditions': custom_conditions,
            'use_ml': use_ml,
            'use_sentiment': use_sentiment
        }
        for chunk in stock_chunks
    ]
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all chunks for processing
        future_to_chunk = {executor.submit(process_stock_chunk, chunk_data): chunk_data['symbols'] 
                           for chunk_data in chunk_data_list}
        
        # Process completed chunks
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                chunk_results, chunk_analysis = future.result()
                all_results.extend(chunk_results)
                all_analysis_data.update(chunk_analysis)
                
                # Update progress with more detailed stats
                # A stock is "successful" if it was processed without error
                # not necessarily if it matched the filter.
                # A "Found" stock is one that matched the filter.
                processed_count = len(chunk_analysis)
                found_count = len(chunk_results)
                failed_chunk = len(chunk) - processed_count

                successful_stocks += processed_count
                failed_stocks += failed_chunk
                
                # Update progress bar
                pbar.update(len(chunk))
                # Add detailed statistics to progress bar
                pbar.set_postfix_str(
                    f"Processed: {successful_stocks}, Failed: {failed_stocks}, "
                    f"Found: {len(all_results)} matches"
                )
            except Exception as e:
                print(f"Error processing chunk: {e}")
                pbar.update(len(chunk)) # Ensure pbar updates even on chunk error
    
    # Close progress bar
    pbar.close()
    
    # Add analysis links to results
    for result in all_results:
        symbol = result['Symbol']
        result['Analysis'] = f"Click to view detailed analysis of {symbol}"
    
    return all_results, all_analysis_data



# === Main Function ===

def main():
    print("\n=== Advanced Stock Screening Tool ===")
    print("This tool helps you find promising stocks based on technical and fundamental analysis.")
    print("\nWhat it analyzes:")
    print("* Price trends and momentum")
    print("* Volume patterns")
    print("* Market capitalization")
    print("* Technical indicators (MACD, RSI, ADX)")
    print("* Multiple timeframes (Daily, Weekly, Monthly)")
    
    # Initialize strategy builder
    strategy_builder = StrategyBuilder()
    
    # Get user preferences
    print("\nChoose screening mode:")
    print("1. Quick Scan (Default settings)")
    print("2. Custom Strategy (Build your own)")
    print("3. Deep Analysis (ML + Sentiment)")
    
    mode = input("\nEnter mode (1-3): ").strip()
    
    custom_conditions = None
    use_ml = False
    use_sentiment = False
    results_df = pd.DataFrame() # Initialize empty DataFrame
    
    if mode == "2":
        custom_conditions = strategy_builder.run()
    elif mode == "3":
        use_ml = True
        use_sentiment = True
        print("\nInitiating deep analysis mode...")
    
    print("\nStarting stock screening...")
    print("This may take a few minutes. The system will analyze multiple timeframes for each stock.")
    
    results, analysis_data = screen_stocks(
        custom_conditions=custom_conditions,
        use_ml=use_ml,
        use_sentiment=use_sentiment
    )
    
    if results:
        # Convert each result to a rating
        rated_results = [get_stock_rating(result) for result in results if result]
        rated_results = [r for r in rated_results if r]  # Remove None values
        
        if rated_results:
            print("\n=== Screening Complete! ===")
            print(f"Found {len(rated_results)} stocks matching criteria.")
            print("\nResults Summary:")
            
            # Convert to DataFrame for analysis
            results_df = pd.DataFrame(rated_results)
            
    if not results_df.empty:
        # Sort results by Technical Score and Market Cap
        results_df = results_df.sort_values(
            by=["Technical Score", "Market Cap (Raw)"], 
            ascending=[False, False]
        ).reset_index(drop=True)

        # Create timestamp for consistent filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Pre-generate reports for all stocks
        print("\nGenerating individual stock reports...")
        stock_reports = {}
        
        # Create reports directory if it doesn't exist
        if not os.path.exists('stock_reports'):
            os.makedirs('stock_reports')
            
        for symbol in tqdm(results_df['Symbol'], desc="Generating stock reports"):
            if symbol in analysis_data and analysis_data[symbol] is not None:
                report_path = f"stock_reports/{symbol}_analysis_{timestamp}.html"
                try:
                    save_analysis_report(
                        symbol,
                        analysis_data[symbol],
                        results_df[results_df["Symbol"] == symbol].to_dict('records')[0]
                    )
                    stock_reports[symbol] = report_path
                except Exception as e:
                    print(f"Failed to generate report for {symbol}: {e}")
            else:
                print(f"No analysis data found for {symbol}, skipping report.")
        
        # Group by rating and show counts
        rating_counts = results_df['Overall Rating'].value_counts()
        
        print("\nStocks by Rating:")
        for rating, count in rating_counts.items():
            print(f"  {rating}: {count} stocks")
        
        # Show top picks in each category
        print("\nTop Picks by Category:")
        for rating in ['Strong Buy', 'Buy', 'Hold', 'Watch']:
            if rating in rating_counts:
                category_stocks = results_df[results_df['Overall Rating'] == rating]
                print(f"\n{rating} Recommendations:")
                for _, stock in category_stocks.head(3).iterrows():
                    print(f"  • {stock['Symbol']}")
                    print(f"    - Category: {stock['Category']}")
                    print(f"    - Technical Score: {stock['Technical Score']}/100")
                    print(f"    - Momentum: {stock['Momentum']}")
                    print(f"    - Trend: {stock['Trend']}")
        
        # Add chart links to results DataFrame
        results_df['Chart_Link'] = results_df['Symbol'].map(lambda x: stock_reports.get(x, ''))
        
        # Generate main HTML report
        report_file = f"stock_screening_report_{timestamp}.html"
        create_summary_report(results_df, report_file)
        
        print(f"\nDetailed HTML report has been generated: {report_file}")
        
        # Ask to open report
        try:
            if input("\nOpen detailed report in browser? (y/n): ").lower() == 'y':
                webbrowser.open(f'file://{os.path.abspath(report_file)}')
        except Exception as e:
            print(f"Could not open browser: {e}")
        
        # View individual stock analysis with pre-generated reports
        while True:
            try:
                symbol = input("\nEnter stock symbol for detailed technical analysis (or 'q' to quit): ").upper()
                if symbol == 'Q':
                    break
                
                if symbol in stock_reports:
                    webbrowser.open(f'file://{os.path.abspath(stock_reports[symbol])}')
                    print(f"Technical analysis report opened in your browser")
                else:
                    print("Symbol not found in analysis data or report generation failed.")
            except Exception as e:
                 print(f"An error occurred: {e}")
    else:
        print("No stocks matched the screening criteria.")
    
    print("\nScreening completed!")
    
    return results_df

if __name__ == "__main__":
    # To handle multiprocessing correctly on some platforms (like Windows)
    mp.freeze_support() 
    df = main()

#IF ANY OTHER STOCK NAME ENTERED DETAILED ANALYSIS IS NOT WORKING