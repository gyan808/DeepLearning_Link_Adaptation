import matplotlib.pyplot as plt
import numpy as np

# =================================================
# MODEL COMPARISON METRICS
# =================================================

models = ["LSTM", "GRU"]

mae_values = [
    4.4739,
    4.4593
]

rmse_values = [
    5.9143,
    5.8331
]

# =================================================
# ENERGY METRICS
# =================================================

metrics = [
    "Power",
    "Throughput",
    "Bits/Joule"
]

values = [
    4.5,
    2.4,
    0.52
]

# =================================================
# HANDOVER METRICS
# =================================================

handover_metrics = [
    "Stable Handovers",
    "Reduced Switching",
    "Adaptive MCS"
]

handover_scores = [
    90,
    85,
    95
]

# =================================================
# CREATE FIGURE
# =================================================

plt.figure(figsize=(16,10))

# =================================================
# MAE COMPARISON
# =================================================

plt.subplot(2,2,1)

plt.bar(
    models,
    mae_values
)

plt.title("MAE Comparison")

plt.ylabel("MAE")

plt.grid(True)

# =================================================
# RMSE COMPARISON
# =================================================

plt.subplot(2,2,2)

plt.bar(
    models,
    rmse_values
)

plt.title("RMSE Comparison")

plt.ylabel("RMSE")

plt.grid(True)

# =================================================
# ENERGY EFFICIENCY
# =================================================

plt.subplot(2,2,3)

plt.bar(
    metrics,
    values
)

plt.title("Energy and Throughput Metrics")

plt.ylabel("Average Values")

plt.grid(True)

# =================================================
# HANDOVER PERFORMANCE
# =================================================

plt.subplot(2,2,4)

plt.bar(
    handover_metrics,
    handover_scores
)

plt.title("TN-NTN Handover Performance")

plt.ylabel("Performance Score")

plt.grid(True)

# =================================================
# FINALIZE
# =================================================

plt.suptitle(
    "Final AI-Based TN-NTN Benchmark Dashboard",
    fontsize=18
)

plt.tight_layout()

# Save graph
plt.savefig(
    "graphs/evaluation/final_dashboard.png",
    dpi=300
)

plt.show()