import pandas as pd
import matplotlib.pyplot as plt

# Read tab-separated files
df_2y = pd.read_csv("data/data2Y.csv", sep="\t")
df_10y = pd.read_csv("data/data10Y.csv", sep="\t")

# Rename columns
df_2y.columns = ["Date", "2Y"]
df_10y.columns = ["Date", "10Y"]

# Merge datasets
df = pd.merge(df_2y, df_10y, on="Date")

# Convert data types
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["2Y"] = pd.to_numeric(df["2Y"], errors="coerce")
df["10Y"] = pd.to_numeric(df["10Y"], errors="coerce")

# Remove missing values
df.dropna(inplace=True)

# Calculate spread
df["Spread"] = df["10Y"] - df["2Y"]

# Print data
print(df.head())

# Plot yields
plt.figure(figsize=(12,6))

plt.plot(df["Date"], df["2Y"], label="2Y Yield")
plt.plot(df["Date"], df["10Y"], label="10Y Yield")

plt.title("Treasury Yield Curve")
plt.xlabel("Date")
plt.ylabel("Yield")

plt.legend()

plt.show()

# Plot spread
plt.figure(figsize=(12,6))

plt.plot(df["Date"], df["Spread"])

plt.axhline(0, linestyle="--")

plt.title("10Y - 2Y Spread")
plt.xlabel("Date")
plt.ylabel("Spread")

plt.show()