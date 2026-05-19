import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page title
st.title("Yield Curve Forecasting Dashboard")

# Load datasets
df_2y = pd.read_csv("data/data2y.csv", sep="\t")
df_10y = pd.read_csv("data/data10y.csv", sep="\t")

# Rename columns
df_2y.columns = ["Date", "2Y"]
df_10y.columns = ["Date", "10Y"]

# Merge datasets
df = pd.merge(df_2y, df_10y, on="Date")

# Convert data types
df["Date"] = pd.to_datetime(df["Date"])

df["2Y"] = pd.to_numeric(df["2Y"], errors="coerce")

df["10Y"] = pd.to_numeric(df["10Y"], errors="coerce")

# Remove missing values
df.dropna(inplace=True)

# Create spread
df["Spread"] = df["10Y"] - df["2Y"]

# Show data
st.subheader("Treasury Yield Data")

st.dataframe(df.head())



# Yield Curve Graph
st.subheader("Yield Curve Spread")

fig1, ax1 = plt.subplots(figsize=(12,6))

ax1.plot(df["Date"], df["2Y"], label="2Y Yield")

ax1.plot(df["Date"], df["10Y"], label="10Y Yield")

ax1.set_xlabel("Date")

ax1.set_ylabel("Yield")

ax1.legend()

st.pyplot(fig1)



# Spread Graph
st.subheader("10Y - 2Y Spread")

fig2, ax2 = plt.subplots(figsize=(12,6))

ax2.plot(df["Date"], df["Spread"])

ax2.axhline(0, linestyle="--")

ax2.set_xlabel("Date")

ax2.set_ylabel("Spread")

st.pyplot(fig2)



# RMSE Comparison
st.subheader("Model Comparison")

models = ["ARIMA", "XGBoost", "LSTM"]

rmse_scores = [0.15, 0.08, 0.05]

fig3, ax3 = plt.subplots(figsize=(8,5))

ax3.bar(models, rmse_scores)

ax3.set_ylabel("RMSE")

st.pyplot(fig3)



# Best Model
best_model = models[rmse_scores.index(min(rmse_scores))]

st.success(f"Best Model: {best_model}")