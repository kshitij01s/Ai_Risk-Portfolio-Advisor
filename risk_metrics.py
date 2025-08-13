import numpy as np
import pandas as pd
import streamlit as st

# ===============================
# 1. Portfolio Volatility
# ===============================
def compute_portfolio_volatility(returns, weights):
    """Annualized portfolio volatility."""
    cov_matrix = returns.cov()
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return vol

# ===============================
# 2. Correlation Matrix
# ===============================
def calculate_correlation_matrix(returns):
    """Calculate correlation matrix of asset returns."""
    return returns.corr()

# ===============================
# 3. Sector Concentration (HHI)
# ===============================
def sector_concentration(portfolio_df, sector_map):
    """Calculate HHI and sector weights."""
    portfolio_df['Sector'] = portfolio_df['Ticker'].map(sector_map)
    sector_weights = portfolio_df.groupby('Sector')['Weight'].sum()
    hhi = (sector_weights ** 2).sum()
    return hhi, sector_weights

# ===============================
# 4. Diversification Suggestions
# ===============================
def generate_suggestions(sector_weights, correlation_matrix):
    """Generate portfolio improvement tips."""
    suggestions = []
    tech_allocation = sector_weights.get("Technology", 0)
    if tech_allocation > 0.3:
        suggestions.append(f"Reduce Tech sector allocation below 30% (currently {tech_allocation:.0%})")
    healthcare_allocation = sector_weights.get("Healthcare", 0)
    if healthcare_allocation < 0.2:
        suggestions.append("Add Healthcare stocks such as JNJ to balance sector exposure")
    bond_allocation = sector_weights.get("Bonds", 0)
    if bond_allocation < 0.1:
        suggestions.append("Consider including bonds or commodities to lower portfolio volatility")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > 0.8:
                suggestions.append(f"High correlation between {correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}")
    return suggestions

# ===============================
# 5. Risk Metrics Functions
# ===============================
def CAGR(returns):
    cumulative_return = (1 + returns).prod()
    n_years = len(returns) / 252
    return cumulative_return ** (1 / n_years) - 1

def volatility(returns):
    return returns.std() * np.sqrt(252)

def sharpe_ratio(returns, risk_free_rate=0.02):
    excess_return = CAGR(returns) - risk_free_rate
    return excess_return / volatility(returns)

def sortino_ratio(returns, risk_free_rate=0.02):
    downside_returns = returns[returns < 0]
    excess_return = CAGR(returns) - risk_free_rate
    downside_vol = downside_returns.std() * np.sqrt(252)
    return excess_return / downside_vol if downside_vol != 0 else np.nan

def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def win_rate(returns):
    return len(returns[returns > 0]) / len(returns)

# ===============================
# 6. Wrap metrics calculation
# ===============================
def calculate_risk_metrics(returns):
    return {
        "CAGR": CAGR(returns),
        "Volatility": volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns),
        "Sortino Ratio": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(returns),
        "Win Rate": win_rate(returns)
    }




