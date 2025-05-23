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
import streamlit as st  # ← make sure this is here

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

# ─── Load the data (this must come *after* the function above) ───────────────────
df = load_data()  # ← line 22 in your file

# ─── Sidebar Controls ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

# 1) Date range
start_date, end_date = st.sidebar.date_input(
    "Date Range",
    [df.index.min().date(), df.index.max().date()]
)
df = df.loc[start_date:end_date]

# 2) Threshold multiplier
threshold_mult = st.sidebar.slider(
    "Disruption Threshold = mean + σ ×",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1
)

# 3) Series selector
all_series = ["Housing Starts", "Mortgage Rate", "Home Price Index", "Pressure Index"]
selected = st.sidebar.multiselect(
    "Series to Plot",
    all_series,
    default=all_series[:-1]
)

# ─── Compute Composite Pressure Index & Flags ─────────────────────────────────
z_hs = (df["Housing Starts"] - df["Housing Starts"].mean()) / df["Housing Starts"].std()
z_mr = (df["Mortgage Rate"]  - df["Mortgage Rate"].mean())   / df["Mortgage Rate"].std()
z_hp = (df["Home Price Index"] - df["Home Price Index"].mean()) / df["Home Price Index"].std()

df["Pressure Index"]   = z_mr + z_hp - z_hs
threshold = df["Pressure Index"].mean() + threshold_mult * df["Pressure Index"].std()
df["Disruption Flag"]  = df["Pressure Index"] > threshold

# ─── Main Dashboard ───────────────────────────────────────────────────────────────
st.title("🏠 Housing Disruption Radar")
st.markdown(f"**Threshold** = mean + {threshold_mult:.1f} × σ → **{threshold:.2f}**")

# Plot the selected series
st.line_chart(df[selected])

# Show flagged disruption events
st.subheader("⚠️ Flagged Disruption Events")
st.dataframe(
    df[df["Disruption Flag"]][selected + ["Disruption Flag"]]
)

# ─── Regression Analysis ─────────────────────────────────────────────────────────
st.subheader("📈 Regression: Housing Starts ~ Mortgage Rate + Home Price Index")
X     = sm.add_constant(df[["Mortgage Rate","Home Price Index"]])
model = sm.OLS(df["Housing Starts"], X).fit()
st.text(model.summary().as_text())
