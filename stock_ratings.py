"""Stock rating and categorization system"""

def calculate_technical_score(conditions):
    """Calculate technical analysis score (0-100)"""
    weights = {
        "Daily MACD Up": 15,
        "Weekly MACD Up": 20,
        "Monthly MACD Up": 25,
        "RSI > 55": 10,
        "+DMI > -DMI": 10,
        "ADX Rising": 10,
        "Vol > MA (90%)": 10
    }
    
    score = 0
    for condition, value in conditions.items():
        if condition in weights and value:
            score += weights[condition]
    return score

def get_market_cap_category(market_cap):
    """Categorize stock by market cap"""
    if market_cap is None:
        return "Unknown"
    try:
        market_cap = float(market_cap)
        if market_cap >= 1e12:  # > 1 lakh crore
            return "Large Cap"
        elif market_cap >= 2e10:  # > 20k crore
            return "Mid Cap"
        elif market_cap >= 5e9:  # > 5k crore
            return "Small Cap"
        else:
            return "Micro Cap"
    except:
        return "Unknown"

def get_momentum_rating(conditions):
    """Get momentum rating (Strong/Moderate/Weak)"""
    score = 0
    if conditions.get("Daily MACD Up"): score += 1
    if conditions.get("Weekly MACD Up"): score += 2
    if conditions.get("Monthly MACD Up"): score += 3
    
    if score >= 5:
        return "Strong Momentum"
    elif score >= 3:
        return "Moderate Momentum"
    else:
        return "Weak Momentum"

def get_volume_rating(conditions):
    """Get volume rating"""
    if conditions.get("Vol > MA (90%)"):
        return "High Volume"
    return "Low Volume"

def get_trend_strength(conditions):
    """Get trend strength"""
    if conditions.get("ADX Rising") and conditions.get("+DMI > -DMI"):
        return "Strong Trend"
    elif conditions.get("+DMI > -DMI"):
        return "Moderate Trend"
    return "Weak Trend"

def get_stock_rating(stock_data):
    """Generate comprehensive stock rating"""
    if not stock_data:
        return None
        
    tech_score = calculate_technical_score(stock_data)
    category = get_market_cap_category(stock_data.get("Market Cap (Raw)"))
    momentum = get_momentum_rating(stock_data)
    volume = get_volume_rating(stock_data)
    trend = get_trend_strength(stock_data)
    
    # Overall rating
    if tech_score >= 80:
        rating = "Strong Buy"
    elif tech_score >= 60:
        rating = "Buy"
    elif tech_score >= 40:
        rating = "Hold"
    else:
        rating = "Watch"
        
    return {
        "Symbol": stock_data["Symbol"],
        "Market Cap": stock_data["Market Cap"],
        "Category": category,
        "Technical Score": tech_score,
        "Overall Rating": rating,
        "Momentum": momentum,
        "Volume": volume,
        "Trend": trend
    }