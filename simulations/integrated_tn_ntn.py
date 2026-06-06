import numpy as np
import matplotlib.pyplot as plt

# Time axis
time = np.arange(0, 500)

# -------------------------
# TN NETWORK
# -------------------------

tn_signal = 15 + 3 * np.sin(
    2 * np.pi * time / 100
)

tn_noise = np.random.normal(
    0,
    1,
    500
)

tn_sinr = tn_signal + tn_noise

# -------------------------
# NTN NETWORK
# -------------------------

ntn_base = 20 * np.sin(
    np.pi * time / 500
)

ntn_noise = np.random.normal(
    0,
    2,
    500
)

K = 5

los = np.sqrt(K / (K + 1))

scatter = np.sqrt(1 / (K + 1)) * (
    np.random.randn(500)
    +
    1j * np.random.randn(500)
)

rician_fading = np.abs(
    los + scatter
)

ntn_sinr = (
    ntn_base * rician_fading
) + ntn_noise

# -------------------------
# NETWORK SELECTION
# -------------------------

selected_network = []

final_sinr = []

for tn, ntn in zip(
    tn_sinr,
    ntn_sinr
):

    if tn > ntn:

        selected_network.append("TN")

        final_sinr.append(tn)

    else:

        selected_network.append("NTN")

        final_sinr.append(ntn)

# -------------------------
# PLOTS
# -------------------------

plt.figure(figsize=(14,8))

# TN
plt.subplot(3,1,1)

plt.plot(tn_sinr)

plt.title("Terrestrial Network SINR")

plt.ylabel("SINR (dB)")

plt.grid(True)

# NTN
plt.subplot(3,1,2)

plt.plot(ntn_sinr)

plt.title("Non-Terrestrial Network SINR")

plt.ylabel("SINR (dB)")

plt.grid(True)

# Final selected network SINR
plt.subplot(3,1,3)

plt.plot(final_sinr)

plt.title("Integrated TN-NTN Selected SINR")

plt.xlabel("Time")

plt.ylabel("SINR (dB)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "graphs/simulation/tn_ntn_simulation.png",
    dpi=300
)

plt.show()