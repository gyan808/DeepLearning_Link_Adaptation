import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from lstm_model import LSTMModel

# Load dataset
df = pd.read_csv("datasets/sinr_dataset.csv")

# Extract SINR column
sinr = df["SINR"].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()

sinr_scaled = scaler.fit_transform(sinr)

joblib.dump(
    scaler,
    "models/scaler.save"
)

# Sliding window
window_size = 10

X = []
y = []

for i in range(len(sinr_scaled) - window_size):

    X.append(sinr_scaled[i:i+window_size])

    y.append(sinr_scaled[i+window_size])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)

y = torch.tensor(y, dtype=torch.float32)

# Create dataset
dataset = TensorDataset(X, y)

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Initialize model
model = LSTMModel()

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training loop
epochs = 20

for epoch in range(epochs):

    for X_batch, y_batch in loader:

        # Forward pass
        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}"
    )

# Save model
torch.save(
    model.state_dict(),
    "models/lstm_weights.pth"
)

print("Model Trained Successfully")