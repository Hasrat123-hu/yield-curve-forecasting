import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import numpy as np

# Load dataset
df = pd.read_csv("data/data10Y.csv", sep="\t")

# Rename columns
df.columns = ["Date", "Yield"]

# Convert data types
df["Date"] = pd.to_datetime(df["Date"])

df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")

# Remove missing values
df.dropna(inplace=True)

# Create lag features
df["Lag1"] = df["Yield"].shift(1)
df["Lag2"] = df["Yield"].shift(2)
df["Lag3"] = df["Yield"].shift(3)

# Remove NaN values after shifting
df.dropna(inplace=True)

# Features and target
X = df[["Lag1", "Lag2", "Lag3"]]

y = df["Yield"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# Build model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("RMSE:", rmse)

# Plot results
plt.figure(figsize=(12,6))

plt.plot(y_test.values, label="Actual Yield")

plt.plot(predictions, label="Predicted Yield")

plt.title("XGBoost Yield Forecast")

plt.xlabel("Time")

plt.ylabel("Yield")

plt.legend()

plt.show()