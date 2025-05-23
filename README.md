# 🏠 Housing Disruption Radar

**Real-time insights into pressure points in the U.S. housing market.**  
Powered by Streamlit, FRED, and economic modeling.
---

## 🔍 Overview

The **Housing Disruption Radar** is an interactive Streamlit dashboard that tracks early signs of stress in the U.S. housing market. By pulling real-time data from trusted economic indicators, it builds a composite **Housing Pressure Index** that helps flag periods of potential instability.

---

## 📊 Indicators Used

The app fetches live data from the Federal Reserve Economic Database (FRED):

- **🏗 Housing Starts** (`HOUST`)  
- **📈 30-Year Fixed Mortgage Rate** (`MORTGAGE30US`)  
- **🏘 U.S. National Home Price Index** (`CSUSHPISA`)

Each series is standardized (z-scored), then combined into a composite index to highlight housing pressure.

---

## ⚠️ Disruption Detection

A **Disruption Event** is flagged when the composite Housing Pressure Index exceeds one standard deviation above its historical mean. These thresholds help identify periods of potential overheating or volatility in the housing market.

---

## 🚀 Features

- 📈 Real-time line charts of all indicators  
- 🧮 Composite “Pressure Index” with z-score logic  
- ⚠️ Flags high-risk periods based on thresholds  
- 📋 Displays raw data and flagged dates  
- 📊 Built-in regression analysis (Housing Starts ~ Mortgage Rate + Home Prices)
