import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

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
X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.float32
)

# Dataset
dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Model
model = GRUModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 20

for epoch in range(epochs):

    for X_batch, y_batch in loader:

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {loss.item():.6f}"
    )

# Save model
torch.save(
    model.state_dict(),
    "models/gru_weights.pth"
)

print("GRU Model Trained Successfully")