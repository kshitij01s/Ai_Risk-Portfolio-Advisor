import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import joblib

model_path = r"C:\Users\admin\Desktop\ai_portfolio-risk-advisor\models\risk_model.pkl"
model = joblib.load('models/risk_model.pkl')
joblib.dump(model, "risk_model_new.pkl")


IMG_HEIGHT = 128
IMG_WIDTH = 128
class_labels = ["Low Risk", "Medium Risk", "High Risk"]
from src.model import load_model, predict_risk
from src.data_loader import load_portfolio 
from src.data_loader import get_price_data 
from src.risk_metrics import (
    compute_portfolio_volatility, calculate_correlation_matrix,
    sector_concentration, CAGR, volatility, sharpe_ratio,
    sortino_ratio, max_drawdown, win_rate
)

from src.model import predict_risk
from src.diversification import suggest_diversification



# ===============================
# Wrap risk metrics calculation
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

# Sector mapping example
sector_map = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech',
    'TSLA': 'Consumer', 'JPM': 'Finance', 'XOM': 'Energy'
}

st.title("📊 AI Portfolio Risk Advisor")


uploaded_file = st.file_uploader(
    "Upload Portfolio CSV", 
    type=["csv"], 
    help="CSV must contain columns: Ticker, Weight, Return"
)

if uploaded_file is not None:
    try:
        portfolio_df = pd.read_csv(uploaded_file)
        if portfolio_df.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()
        
      
        required_cols = {"Ticker", "Weight", "Return"}
        if not required_cols.issubset(portfolio_df.columns):
            st.error(f"CSV must contain columns: {', '.join(required_cols)}")
            st.stop()
        
        st.subheader("Portfolio Preview")
        st.dataframe(portfolio_df)
        
       
        tickers = portfolio_df["Ticker"].tolist()
        weights = portfolio_df["Weight"].values

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()


    prices_dict = get_price_data(tickers, period="1y", interval="1d")
    price_frames = []
    for ticker, df_price in prices_dict.items():
        if df_price.index.name == 'Date':
            df_price = df_price.reset_index()
        df_price = df_price[['Date', 'Close']].copy()
        df_price.set_index('Date', inplace=True)
        df_price.rename(columns={'Close': ticker}, inplace=True)
        price_frames.append(df_price)

    prices = pd.concat(price_frames, axis=1)
    returns = prices.pct_change().dropna()

    st.subheader("Historical Prices & Returns")
    st.write(prices.head())
    
    # ===============================
    # Portfolio Metrics
    # ===============================
    port_vol = compute_portfolio_volatility(returns, weights)
    corr_matrix = calculate_correlation_matrix(returns)
    avg_corr = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean()
    
    # Sector mapping example
    sector_map = {
        'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech',
        'TSLA': 'Consumer', 'JPM': 'Finance', 'XOM': 'Energy'
    }
    hhi, sector_weights = sector_concentration(portfolio_df, sector_map)

    # ===============================
    # Predict Risk
    # ===============================
    features = [avg_corr, hhi, port_vol]
    risk_score = predict_risk(model, features)

    st.markdown("### 📈 Risk Score")
    st.write(risk_score)

    st.markdown("### 🚨 Top Risk Drivers")
    st.write(f"- Average correlation: {avg_corr:.2f}")
    st.write(f"- Sector HHI: {hhi:.2f}")
    st.write(f"- Portfolio volatility: {port_vol:.2f}")

    st.markdown("### 💡 Diversification Suggestions")
    suggestions = suggest_diversification(sector_weights)
    for s in suggestions:
        st.write("•", s)

    # ===============================
    # Risk Metrics Table
    # ===============================
    def calculate_risk_metrics(returns_series):
        return {
            "CAGR": CAGR(returns_series),
            "Volatility": volatility(returns_series),
            "Sharpe Ratio": sharpe_ratio(returns_series),
            "Sortino Ratio": sortino_ratio(returns_series),
            "Max Drawdown": max_drawdown(returns_series),
            "Win Rate": win_rate(returns_series)
        }

    metrics = calculate_risk_metrics(returns.mean(axis=1))
    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    metrics_df["Value"] = metrics_df["Value"].apply(lambda x: f"{x:.2%}" if abs(x) < 10 else f"{x:.2f}")

    st.subheader("📈 Risk Metrics")
    st.table(metrics_df)

    st.subheader("📌 Portfolio Volatility")
    st.write(f"Annualized Volatility: **{port_vol:.2%}**")

    st.subheader("📌 Sector Concentration (HHI)")
    st.write(f"HHI: **{hhi:.3f}**")
    st.bar_chart(sector_weights)

    st.subheader("📌 Correlation Matrix")
    st.dataframe(corr_matrix)

    # Friendly Sharpe message
    if metrics["Sharpe Ratio"] > 1:
        st.success("✅ Strong risk-adjusted return (Sharpe > 1)")
    else:
        st.warning("⚠️ Risk-adjusted return could be improved (Sharpe ≤ 1)")

else:
    st.info("Upload a CSV file to get started.")

# ===============================
# Example main for testing
# ===============================
def main():
    arr = np.array([1, 2, 3, 4, 5])
    print("Array:", arr)
    mean_value = np.mean(arr)
    print("Mean:", mean_value)
    random_nums = np.random.rand(5)
    print("Random Numbers:", random_nums)

if __name__ == "__main__":
    main()