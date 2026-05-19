import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv("data/data10Y.csv", sep="\t")

# Rename columns
df.columns = ["Date", "Yield"]

# Convert types
df["Date"] = pd.to_datetime(df["Date"])
df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")

# Remove missing values
df.dropna(inplace=True)

# Train/Test Split
train_size = int(len(df) * 0.8)

train = df["Yield"][:train_size]
test = df["Yield"][train_size:]

# Build model
model = ARIMA(train, order=(5,1,0))

# Train model
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))

print("RMSE:", rmse)

# Plot
plt.figure(figsize=(12,6))

plt.plot(test.values, label="Actual")
plt.plot(forecast.values, label="Forecast")

plt.title("ARIMA Forecast vs Actual")

plt.xlabel("Time")
plt.ylabel("Yield")

plt.legend()

plt.show()