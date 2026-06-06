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

# Load trained model
model = LSTMModel()

model.load_state_dict(
    torch.load(
        "models/lstm_weights.pth"
    )
)

model.eval()

# Prediction
with torch.no_grad():

    predictions = model(X_tensor)

# Convert back to original scale
predictions = predictions.numpy()

predictions = scaler.inverse_transform(
    predictions
)

actual = scaler.inverse_transform(y)

# Evaluation metrics
mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

# Plot
plt.figure(figsize=(12,6))

plt.plot(
    actual,
    label="Actual SINR"
)

plt.plot(
    predictions,
    label="Predicted SINR"
)

plt.title(
    "Model Evaluation"
)

plt.xlabel("Time")

plt.ylabel("SINR (dB)")

plt.legend()

plt.grid(True)

plt.savefig(
    "graphs/evaluation/model_evaluation.png",
    dpi=300
)

plt.show()