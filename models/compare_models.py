import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from lstm_model import LSTMModel
from gru_model import GRUModel

# Load dataset
df = pd.read_csv(
    "datasets/sinr_dataset.csv"
)

sinr = df["SINR"].values.reshape(-1,1)

# Normalize
scaler = MinMaxScaler()

sinr_scaled = scaler.fit_transform(
    sinr
)

# Sliding window
window_size = 10

X = []
y = []

for i in range(
    len(sinr_scaled) - window_size
):

    X.append(
        sinr_scaled[i:i+window_size]
    )

    y.append(
        sinr_scaled[i+window_size]
    )

X = np.array(X)
y = np.array(y)

# Tensor conversion
X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

# =================================================
# LSTM MODEL
# =================================================

lstm_model = LSTMModel()

lstm_model.load_state_dict(
    torch.load(
        "models/lstm_weights.pth"
    )
)

lstm_model.eval()

with torch.no_grad():

    lstm_predictions = lstm_model(
        X_tensor
    )

lstm_predictions = lstm_predictions.numpy()

lstm_predictions = scaler.inverse_transform(
    lstm_predictions
)

# =================================================
# GRU MODEL
# =================================================

gru_model = GRUModel()

gru_model.load_state_dict(
    torch.load(
        "models/gru_weights.pth"
    )
)

gru_model.eval()

with torch.no_grad():

    gru_predictions = gru_model(
        X_tensor
    )

gru_predictions = gru_predictions.numpy()

gru_predictions = scaler.inverse_transform(
    gru_predictions
)

# =================================================
# ACTUAL VALUES
# =================================================

actual = scaler.inverse_transform(y)

# =================================================
# METRICS
# =================================================

# LSTM
lstm_mae = mean_absolute_error(
    actual,
    lstm_predictions
)

lstm_rmse = np.sqrt(
    mean_squared_error(
        actual,
        lstm_predictions
    )
)

# GRU
gru_mae = mean_absolute_error(
    actual,
    gru_predictions
)

gru_rmse = np.sqrt(
    mean_squared_error(
        actual,
        gru_predictions
    )
)

# =================================================
# PRINT RESULTS
# =================================================

print("\n===== MODEL COMPARISON =====\n")

print(f"LSTM MAE  : {lstm_mae:.4f}")

print(f"LSTM RMSE : {lstm_rmse:.4f}\n")

print(f"GRU MAE   : {gru_mae:.4f}")

print(f"GRU RMSE  : {gru_rmse:.4f}")

# =================================================
# PLOT
# =================================================

plt.figure(figsize=(14,6))

plt.plot(
    actual,
    label="Actual SINR",
    linewidth=2
)

plt.plot(
    lstm_predictions,
    label="LSTM Prediction"
)

plt.plot(
    gru_predictions,
    label="GRU Prediction"
)

plt.title(
    "LSTM vs GRU SINR Prediction"
)

plt.xlabel("Time")

plt.ylabel("SINR (dB)")

plt.legend()

plt.grid(True)

# Save graph
plt.savefig(
    "graphs/evaluation/lstm_vs_gru.png",
    dpi=300
)

plt.show()