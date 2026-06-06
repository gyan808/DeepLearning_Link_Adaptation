import torch
import os
import time

from lstm_model import LSTMModel
from gru_model import GRUModel

# =================================================
# LOAD MODELS
# =================================================

lstm_model = LSTMModel()

lstm_model.load_state_dict(
    torch.load(
        "models/lstm_weights.pth"
    )
)

lstm_model.eval()

gru_model = GRUModel()

gru_model.load_state_dict(
    torch.load(
        "models/gru_weights.pth"
    )
)

gru_model.eval()

# =================================================
# DUMMY INPUT
# =================================================

dummy_input = torch.randn(
    1,
    10,
    1
)

# =================================================
# EXPORT LSTM TO ONNX
# =================================================

torch.onnx.export(
    lstm_model,
    dummy_input,
    "models/lstm_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)

print("\nLSTM ONNX Export Successful")

# =================================================
# EXPORT GRU TO ONNX
# =================================================

torch.onnx.export(
    gru_model,
    dummy_input,
    "models/gru_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)

print("GRU ONNX Export Successful")

# =================================================
# MODEL SIZE COMPARISON
# =================================================

lstm_pytorch_size = os.path.getsize(
    "models/lstm_weights.pth"
) / 1024

gru_pytorch_size = os.path.getsize(
    "models/gru_weights.pth"
) / 1024

lstm_onnx_size = os.path.getsize(
    "models/lstm_model.onnx"
) / 1024

gru_onnx_size = os.path.getsize(
    "models/gru_model.onnx"
) / 1024

print("\n===== MODEL SIZE COMPARISON =====\n")

print(
    f"LSTM PyTorch Size : "
    f"{lstm_pytorch_size:.2f} KB"
)

print(
    f"LSTM ONNX Size    : "
    f"{lstm_onnx_size:.2f} KB"
)

print()

print(
    f"GRU PyTorch Size  : "
    f"{gru_pytorch_size:.2f} KB"
)

print(
    f"GRU ONNX Size     : "
    f"{gru_onnx_size:.2f} KB"
)

# =================================================
# INFERENCE SPEED TEST
# =================================================

# LSTM Timing
start = time.time()

for _ in range(1000):

    with torch.no_grad():

        output = lstm_model(dummy_input)

end = time.time()

lstm_time = (
    (end - start) / 1000
) * 1000

# GRU Timing
start = time.time()

for _ in range(1000):

    with torch.no_grad():

        output = gru_model(dummy_input)

end = time.time()

gru_time = (
    (end - start) / 1000
) * 1000

print("\n===== INFERENCE SPEED =====\n")

print(
    f"LSTM Inference Time : "
    f"{lstm_time:.4f} ms"
)

print(
    f"GRU Inference Time  : "
    f"{gru_time:.4f} ms"
)