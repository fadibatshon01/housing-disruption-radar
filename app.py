# app.py

import os
import ssl
import certifi
import urllib.request
from io import StringIO
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ─── Setup ────────────────────────────────────────────────────────────────────────
BASE = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Create an SSL context for FRED downloads
CTX = ssl.create_default_context(cafile=certifi.where())

st.set_page_config(page_title="Housing Disruption Radar", layout="wide")

# ─── Data‐fetching Helpers ─────────────────────────────────────────────────────────
def fetch_fred(series_id: str) -> pd.DataFrame:
    """
    Download a FRED series as CSV and return a DataFrame indexed by Date.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, context=CTX) as resp:
        txt = resp.read().decode("utf-8")
    df = pd.read_csv(StringIO(txt))
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Fetch Housing Starts, Mortgage Rate, Home Price Index, merge them,
    and rename columns for clarity.
    """
    hs = fetch_fred("HOUST")
    mr = fetch_fred("MORTGAGE30US")
    hp = fetch_fred("CSUSHPISA")

    # Save raw CSVs for inspection
    hs.to_csv(os.path.join(DATA_DIR, "housing_starts.csv"))
    mr.to_csv(os.path.join(DATA_DIR, "mortgage_rate.csv"))
    hp.to_csv(os.path.join(DATA_DIR, "home_price_index.csv"))

    # Merge into one DataFrame
    df = hs.join([mr, hp], how="inner").dropna()
    df.columns = ["Housing Starts", "Mortgage Rate", "Home Price Index"]
    return df

# ─── Load the data ────────────────────────────────────────────────────────────────
df = load_data()

# ─── Sidebar Controls ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

start_date, end_date = st.sidebar.date_input(
    "Date Range",
    [df.index.min().date(), df.index.max().date()]
)
# Filter data by date
filtered = df.loc[start_date:end_date]

threshold_mult = st.sidebar.slider(
    "Disruption Threshold = mean + σ ×",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1
)

all_series = ["Housing Starts", "Mortgage Rate", "Home Price Index", "Pressure Index"]
selected = st.sidebar.multiselect(
    "Series to Plot",
    all_series,
    default=all_series[:-1]
)

# ─── Compute Composite Pressure Index ──────────────────────────────────────────────
z_hs = (filtered["Housing Starts"] - filtered["Housing Starts"].mean()) / filtered["Housing Starts"].std()
z_mr = (filtered["Mortgage Rate"]  - filtered["Mortgage Rate"].mean())   / filtered["Mortgage Rate"].std()
z_hp = (filtered["Home Price Index"] - filtered["Home Price Index"].mean()) / filtered["Home Price Index"].std()

filtered["Pressure Index"] = z_mr + z_hp - z_hs
threshold = filtered["Pressure Index"].mean() + threshold_mult * filtered["Pressure Index"].std()
filtered["Disruption Flag"] = filtered["Pressure Index"] > threshold

# ─── Main Dashboard ───────────────────────────────────────────────────────────────
st.title("🏠 Housing Disruption Radar")
st.markdown(f"**Threshold** = mean + {threshold_mult:.1f} × σ → **{threshold:.2f}**")

# Plot selected series
st.line_chart(filtered[selected])

# Show flagged disruption events
st.subheader("⚠️ Flagged Disruption Events")
st.dataframe(
    filtered[filtered["Disruption Flag"]][selected + ["Disruption Flag"]]
)

# ─── Regression Analysis ─────────────────────────────────────────────────────────
st.subheader("📈 Regression: Housing Starts ~ Mortgage Rate + Home Price Index")

# Fit OLS model
X = sm.add_constant(filtered[["Mortgage Rate", "Home Price Index"]])
model = sm.OLS(filtered["Housing Starts"], X).fit()

# Build summary DataFrame
summary_df = pd.DataFrame({
    "Coefficient": model.params,
    "Std. Error": model.bse,
    "t-Value": model.tvalues,
    "P-Value": model.pvalues
})
summary_df = summary_df.round(4)

# Highlight significant p-values
def highlight_sig(val):
    return "color:green;" if val < 0.05 else "color:black;"

# Display styled summary
st.dataframe(summary_df.style.applymap(highlight_sig, subset=["P-Value"]))

# Display model metrics
st.markdown(f"""
**R-squared:** {model.rsquared:.3f}  
**Adj. R-squared:** {model.rsquared_adj:.3f}  
**AIC:** {model.aic:.1f}  
**BIC:** {model.bic:.1f}  
**F-statistic:** {model.fvalue:.2f} (p = {model.f_pvalue:.4f})  
""" )

# ─── Forecasting ─────────────────────────────────────────────────────────────────
st.subheader("🔮 Forecast Housing Starts")

# Prepare data for forecasting
valid = filtered.dropna()
X = valid[["Mortgage Rate", "Home Price Index"]]
y = valid["Housing Starts"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression forecast
test_model = LinearRegression()
test_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = test_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.markdown(f"**Test RMSE:** {rmse:.2f}")

# Predict next period
latest_input = filtered[["Mortgage Rate", "Home Price Index"]].iloc[-1:]
next_pred = test_model.predict(latest_input)[0]
st.markdown(f"**Forecasted Housing Starts (next period):** {next_pred:.0f}")