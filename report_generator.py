"""Generate user-friendly stock screening reports"""

import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def create_summary_report(results, filename=None):
    """Create a user-friendly HTML summary report"""
    if not filename:
        filename = f"stock_screening_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
    # Convert results to DataFrame if needed
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)
    
    # Group stocks by rating
    rating_groups = results.groupby('Overall Rating')
    
    # Create HTML content
    html_content = f"""
    <html>
    <head>
        <title>Stock Screening Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .stock-card {{ 
                background-color: white; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .strong-buy {{ border-left: 5px solid #28a745; }}
            .buy {{ border-left: 5px solid #17a2b8; }}
            .hold {{ border-left: 5px solid #ffc107; }}
            .watch {{ border-left: 5px solid #6c757d; }}
            .rating {{ font-weight: bold; }}
            .metrics {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .metric {{ 
                background: #f8f9fa; 
                padding: 10px; 
                border-radius: 5px;
                min-width: 150px;
            }}
            .category-label {{
                font-size: 0.9em;
                color: #666;
                margin-bottom: 5px;
            }}
            .value {{
                font-size: 1.1em;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Stock Screening Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total stocks screened: {len(results)}</p>
        </div>
    """
    
    # Add summary statistics
    html_content += """
        <div class="section">
            <h2>Summary</h2>
            <div class="metrics">
    """
    
    for rating in ["Strong Buy", "Buy", "Hold", "Watch"]:
        count = len(rating_groups.get_group(rating)) if rating in rating_groups.groups else 0
        html_content += f"""
            <div class="metric">
                <div class="category-label">{rating}</div>
                <div class="value">{count} stocks</div>
            </div>
        """
    
    html_content += """
            </div>
        </div>
    """
    
    # Add detailed stock cards grouped by rating
    for rating in ["Strong Buy", "Buy", "Hold", "Watch"]:
        if rating in rating_groups.groups:
            html_content += f"""
                <div class="section">
                    <h2>{rating} Stocks</h2>
            """
            
            group = rating_groups.get_group(rating)
            for _, stock in group.iterrows():
                rating_class = rating.lower().replace(" ", "-")
                html_content += f"""
                    <div class="stock-card {rating_class}">
                        <h3>{stock['Symbol']}</h3>
                        <div class="metrics">
                            <div class="metric">
                                <div class="category-label">Market Cap</div>
                                <div class="value">{stock['Market Cap']}</div>
                            </div>
                            <div class="metric">
                                <div class="category-label">Category</div>
                                <div class="value">{stock['Category']}</div>
                            </div>
                            <div class="metric">
                                <div class="category-label">Technical Score</div>
                                <div class="value">{stock['Technical Score']}/100</div>
                            </div>
                            <div class="metric">
                                <div class="category-label">Momentum</div>
                                <div class="value">{stock['Momentum']}</div>
                            </div>
                            <div class="metric">
                                <div class="category-label">Volume</div>
                                <div class="value">{stock['Volume']}</div>
                            </div>
                            <div class="metric">
                                <div class="category-label">Trend</div>
                                <div class="value">{stock['Trend']}</div>
                            </div>
                        </div>
                    </div>
                """
            
            html_content += """
                </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save the report
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename