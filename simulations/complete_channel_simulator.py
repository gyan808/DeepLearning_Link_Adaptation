import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulation time
time = np.arange(0, 500)

# Satellite SINR curve
base_signal = 20 * np.sin(np.pi * time / 500)

# Noise
noise = np.random.normal(0, 1.5, 500)

# Rician fading
K = 5

los = np.sqrt(K / (K + 1))

scatter = np.sqrt(1 / (K + 1)) * (
    np.random.randn(500) + 1j * np.random.randn(500)
)

rician_fading = np.abs(los + scatter)

# Final signal
faded_signal = base_signal * rician_fading

sinr = faded_signal + noise

# Save dataset
df = pd.DataFrame({
    "Time": time,
    "SINR": sinr
})

df.to_csv(
    "datasets/sinr_dataset.csv",
    index=False
)

print("Dataset Generated Successfully")

# Plot
plt.figure(figsize=(12,6))

plt.plot(time, sinr)

plt.title("Realistic Satellite SINR Simulation")

plt.xlabel("Time")

plt.ylabel("SINR (dB)")

plt.grid(True)

plt.show()