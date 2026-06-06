import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from lstm_model import LSTMModel

# Load dataset
df = pd.read_csv("datasets/sinr_dataset.csv")

sinr = df["SINR"].values.reshape(-1,1)

# Normalize
scaler = MinMaxScaler()

sinr_scaled = scaler.fit_transform(sinr)

# Sliding window
window_size = 10

X = []
y = []

for i in range(len(sinr_scaled) - window_size):

    X.append(sinr_scaled[i:i+window_size])

    y.append(sinr_scaled[i+window_size])

X = np.array(X)
y = np.array(y)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)

# Load model
model = LSTMModel()

model.load_state_dict(
    torch.load("models/lstm_weights.pth")
)

model.eval()

# Prediction
with torch.no_grad():

    predictions = model(X)

# Convert back to original scale
predictions = predictions.numpy()

predictions = scaler.inverse_transform(predictions)

actual = scaler.inverse_transform(y)

# Plot
plt.figure(figsize=(12,6))

plt.plot(actual, label="Actual SINR")

plt.plot(predictions, label="Predicted SINR")

plt.title("LSTM SINR Prediction")

plt.xlabel("Time")

plt.ylabel("SINR (dB)")

plt.legend()

plt.grid(True)

plt.savefig(
    "graphs/prediction/lstm_prediction.png",
    dpi=300
)

plt.show()