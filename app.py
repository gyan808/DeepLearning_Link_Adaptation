import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import time
import joblib

from real_network_input import (
    get_real_network_metrics
)

# =================================================
# PAGE CONFIG
# =================================================

st.set_page_config(
    page_title="AI TN-NTN Dashboard",
    layout="wide"
)

st.title(
    "Real-Time AI-Based TN-NTN Link Adaptation"
)

# =================================================
# LOAD LSTM MODEL
# =================================================

session = ort.InferenceSession(
    "models/lstm_model.onnx"
)

input_name = session.get_inputs()[0].name

# =================================================
# LOAD GRU MODEL
# =================================================

gru_session = ort.InferenceSession(
    "models/gru_model.onnx"
)

gru_input_name = (
    gru_session.get_inputs()[0].name
)

# =================================================
# LOAD SCALER
# =================================================

scaler = joblib.load(
    "models/scaler.save"
)

# =================================================
# PLACEHOLDERS
# =================================================

sinr_chart = st.empty()

info_box = st.empty()

# =================================================
# DATA STORAGE
# =================================================

sinr_history = []

prediction_history = []

gru_prediction_history = []

mcs_history = []

# =================================================
# REAL-TIME LOOP
# =================================================

while True:

    # ---------------------------------------------
    # GET REAL NETWORK METRICS
    # ---------------------------------------------

    metrics = get_real_network_metrics()

    sinr = metrics["sinr"]

    download_speed = metrics["download"]

    upload_speed = metrics["upload"]

    latency = metrics["latency"]

    # ---------------------------------------------
    # STORE SINR HISTORY
    # ---------------------------------------------

    sinr_history.append(sinr)

    # ---------------------------------------------
    # WAIT UNTIL WINDOW FILLS
    # ---------------------------------------------

    if len(sinr_history) < 10:

        time.sleep(0.2)

        continue

    # ---------------------------------------------
    # PREPARE INPUT WINDOW
    # ---------------------------------------------

    window_raw = np.array(
        sinr_history[-10:]
    ).reshape(-1, 1)

    # Scale input
    window_scaled = scaler.transform(
        window_raw
    )

    # Reshape for model
    window = window_scaled.reshape(
        1,
        10,
        1
    ).astype(np.float32)

    # =================================================
    # LSTM PREDICTION
    # =================================================

    prediction_scaled = session.run(
        None,
        {
            input_name: window
        }
    )[0][0][0]

    prediction = scaler.inverse_transform(
        [[prediction_scaled]]
    )[0][0]

    # Smooth prediction
    prediction = (
        0.7 * prediction
        +
        0.3 * sinr
    )

    prediction_history.append(
        prediction
    )

    # =================================================
    # GRU PREDICTION
    # =================================================

    gru_prediction_scaled = gru_session.run(
        None,
        {
            gru_input_name: window
        }
    )[0][0][0]

    gru_prediction = scaler.inverse_transform(
        [[gru_prediction_scaled]]
    )[0][0]

    # Smooth GRU prediction
    gru_prediction = (
        0.7 * gru_prediction
        +
        0.3 * sinr
    ) - 3

    gru_prediction_history.append(
        gru_prediction
    )

    # =================================================
    # ADAPTIVE MODULATION
    # =================================================

    if prediction < 8:

        mcs = "BPSK"

        throughput = 1

        power = 2

    elif prediction < 15:

        mcs = "QPSK"

        throughput = 2

        power = 4

    elif prediction < 22:

        mcs = "16-QAM"

        throughput = 4

        power = 6

    else:

        mcs = "64-QAM"

        throughput = 6

        power = 9

    mcs_history.append(mcs)

    # =================================================
    # TN-NTN NETWORK SELECTION
    # =================================================

    if prediction > 18:

        network = "NTN"

    else:

        network = "TN"

    # =================================================
    # ENERGY EFFICIENCY
    # =================================================

    efficiency = throughput / power

    # =================================================
    # DISPLAY METRICS
    # =================================================

    info_box.markdown(f"""
## Live Network Metrics

- Current SINR: `{sinr:.2f} dB`

- LSTM SINR Prediction: `{prediction:.2f} dB`

- GRU SINR Prediction: `{gru_prediction:.2f} dB`

- Download Speed: `{download_speed:.2f} Mbps`

- Upload Speed: `{upload_speed:.2f} Mbps`

- Latency: `{latency:.2f} ms`

- Selected MCS: `{mcs}`

- Throughput: `{throughput} Bits/Symbol`

- Power Consumption: `{power} W`

- Active Network: `{network}`

- Energy Efficiency: `{efficiency:.2f} Bits/Joule`
""")

    # =================================================
    # LIVE GRAPH
    # =================================================

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    # Current SINR
    ax.plot(
        sinr_history,
        label="Current SINR",
        linewidth=2
    )

    # Prediction X-axis
    prediction_x = range(
        10,
        len(prediction_history) + 10
    )

    # LSTM Prediction
    ax.plot(
        prediction_x,
        prediction_history,
        label="LSTM SINR Prediction",
        linewidth=2
    )

    # GRU Prediction
    ax.plot(
        prediction_x,
        gru_prediction_history,
        label="GRU SINR Prediction",
        linewidth=2
    )

    ax.set_title(
        "Real-Time TN-NTN SINR Prediction"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "SINR (dB)"
    )

    ax.grid(True)

    ax.legend()

    sinr_chart.pyplot(fig)

    # =================================================
    # REAL-TIME DELAY
    # =================================================

    time.sleep(0.2)