import numpy as np
import matplotlib.pyplot as plt

# =================================================
# TIME AXIS
# =================================================

time = np.arange(0, 500)

# =================================================
# TN NETWORK SIMULATION
# =================================================

tn_signal = 15 + 3 * np.sin(
    2 * np.pi * time / 100
)

tn_noise = np.random.normal(
    0,
    1,
    500
)

tn_sinr = tn_signal + tn_noise

# =================================================
# NTN NETWORK SIMULATION
# =================================================

ntn_base = 20 * np.sin(
    np.pi * time / 500
)

ntn_noise = np.random.normal(
    0,
    2,
    500
)

K = 5

los = np.sqrt(
    K / (K + 1)
)

scatter = np.sqrt(
    1 / (K + 1)
) * (
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

# =================================================
# AI-LIKE FUTURE PREDICTION
# =================================================

tn_predicted = np.convolve(
    tn_sinr,
    np.ones(10) / 10,
    mode='same'
)

ntn_predicted = np.convolve(
    ntn_sinr,
    np.ones(10) / 10,
    mode='same'
)

# =================================================
# SMART TN-NTN HANDOVER LOGIC
# =================================================

selected_network = []

final_sinr = []

battery = 80

# Initial network
current_network = "TN"

# Hysteresis margin (dB)
handover_margin = 3

# Minimum hold duration
hold_time = 10

hold_counter = 0

for tn, ntn in zip(
    tn_predicted,
    ntn_predicted
):

    hold_counter += 1

    # =============================================
    # ENERGY-AWARE MODE
    # =============================================

    if battery < 30:

        preferred_network = "TN"

    else:

        # =========================================
        # HYSTERESIS-BASED SWITCHING
        # =========================================

        if current_network == "TN":

            if ntn > tn + handover_margin:

                preferred_network = "NTN"

            else:

                preferred_network = "TN"

        else:

            if tn > ntn + handover_margin:

                preferred_network = "TN"

            else:

                preferred_network = "NTN"

    # =============================================
    # HOLD TIME LOGIC
    # =============================================

    if (
        preferred_network != current_network
        and hold_counter >= hold_time
    ):

        current_network = preferred_network

        hold_counter = 0

    # =============================================
    # STORE RESULTS
    # =============================================

    selected_network.append(
        current_network
    )

    if current_network == "TN":

        final_sinr.append(tn)

    else:

        final_sinr.append(ntn)

# =================================================
# ADAPTIVE MODULATION
# =================================================

mcs_selection = []

for sinr in final_sinr:

    if sinr < 8:

        mcs_selection.append("BPSK")

    elif sinr < 15:

        mcs_selection.append("QPSK")

    elif sinr < 22:

        mcs_selection.append("16-QAM")

    else:

        mcs_selection.append("64-QAM")

# =================================================
# NUMERIC CONVERSION
# =================================================

mapping = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 3,
    "64-QAM": 4
}

mcs_numeric = [
    mapping[m]
    for m in mcs_selection
]

# =================================================
# PLOTS
# =================================================

plt.figure(figsize=(14, 10))

# =================================================
# TN PREDICTION
# =================================================

plt.subplot(4, 1, 1)

plt.plot(
    tn_predicted,
    label="Predicted TN SINR"
)

plt.title("Predicted TN SINR")

plt.ylabel("SINR (dB)")

plt.grid(True)

plt.legend()

# =================================================
# NTN PREDICTION
# =================================================

plt.subplot(4, 1, 2)

plt.plot(
    ntn_predicted,
    label="Predicted NTN SINR"
)

plt.title("Predicted NTN SINR")

plt.ylabel("SINR (dB)")

plt.grid(True)

plt.legend()

# =================================================
# FINAL SELECTED NETWORK
# =================================================

plt.subplot(4, 1, 3)

plt.plot(
    final_sinr,
    label="Selected Network SINR"
)

plt.title(
    "Integrated AI-Based TN-NTN Selection"
)

plt.ylabel("SINR (dB)")

plt.grid(True)

plt.legend()

# =================================================
# ADAPTIVE MODULATION
# =================================================

plt.subplot(4, 1, 4)

plt.plot(mcs_numeric)

plt.yticks(
    [1, 2, 3, 4],
    ["BPSK", "QPSK", "16-QAM", "64-QAM"]
)

plt.title("Adaptive Modulation Selection")

plt.xlabel("Time")

plt.ylabel("Modulation")

plt.grid(True)

plt.tight_layout()

# =================================================
# SAVE GRAPH
# =================================================

plt.savefig(
    "graphs/integrated_system/final_integrated_system.png",
    dpi=300
)

plt.show()