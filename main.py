import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from models.lstm_model import LSTMModel

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

# Tensor conversion
X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

# Load trained model
model = LSTMModel()

model.load_state_dict(
    torch.load("models/lstm_weights.pth")
)

model.eval()

# Prediction
with torch.no_grad():

    predictions = model(X_tensor)

# Convert back to original scale
predictions = predictions.numpy()

predicted_sinr = scaler.inverse_transform(predictions)

# Adaptive Modulation Logic
mcs_list = []

battery = 80

for sinr_value in predicted_sinr:

    sinr_db = sinr_value[0]

    # Energy-aware mode
    if battery < 30:

        if sinr_db < 15:
            mcs_list.append("BPSK")

        else:
            mcs_list.append("QPSK")

    # Normal mode
    else:

        if sinr_db < 8:
            mcs_list.append("BPSK")

        elif sinr_db < 15:
            mcs_list.append("QPSK")

        elif sinr_db < 22:
            mcs_list.append("16-QAM")

        else:
            mcs_list.append("64-QAM")

# Convert modulation labels to numbers
mapping = {
    "BPSK":1,
    "QPSK":2,
    "16-QAM":3,
    "64-QAM":4
}

mcs_numeric = [mapping[m] for m in mcs_list]

# Plot
plt.figure(figsize=(14,6))

plt.subplot(2,1,1)

plt.plot(
    predicted_sinr,
    label="Predicted SINR"
)

plt.title("AI Predicted SINR")

plt.ylabel("SINR (dB)")

plt.grid(True)

plt.legend()

# Modulation plot
plt.subplot(2,1,2)

plt.plot(
    mcs_numeric
)

plt.yticks(
    [1,2,3,4],
    ["BPSK","QPSK","16-QAM","64-QAM"]
)

plt.title("Adaptive Modulation Selection")

plt.xlabel("Time")

plt.ylabel("Modulation")

plt.grid(True)

plt.tight_layout()

plt.show()