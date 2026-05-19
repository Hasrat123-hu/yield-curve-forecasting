import matplotlib.pyplot as plt

# RMSE scores
models = ["ARIMA", "XGBoost", "LSTM"]

rmse_scores = [
    0.15,   # replace with your ARIMA RMSE
    0.08,   # replace with your XGBoost RMSE
    0.05    # replace with your LSTM RMSE
]

# Plot comparison
plt.figure(figsize=(8,5))

plt.bar(models, rmse_scores)

plt.title("Model RMSE Comparison")

plt.xlabel("Models")

plt.ylabel("RMSE")

plt.show()

# Best model
best_model = models[rmse_scores.index(min(rmse_scores))]

print("Best Model:", best_model)