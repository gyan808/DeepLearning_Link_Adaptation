import numpy as np

# -----------------------------------------
# GLOBAL STATE
# -----------------------------------------

previous_sinr = 15

time_step = 0

# -----------------------------------------
# REALISTIC NETWORK MODEL
# -----------------------------------------

def get_real_network_metrics():

    global previous_sinr
    global time_step

    # -------------------------------------
    # BASE CHANNEL VARIATION
    # -------------------------------------

    base_signal = (
        15
        + 3 * np.sin(
            2 * np.pi * time_step / 100
        )
    )

    # -------------------------------------
    # FAST FADING
    # -------------------------------------

    fast_fading = np.random.normal(
        0,
        1
    )

    # -------------------------------------
    # SHADOWING EFFECT
    # -------------------------------------

    shadowing = np.random.normal(
        0,
        0.5
    )

    # -------------------------------------
    # COMBINED CHANNEL
    # -------------------------------------

    base_signal = (
        base_signal
        + fast_fading
        + shadowing
    )

    # -------------------------------------
    # SMALL ADDITIONAL NOISE
    # -------------------------------------

    noise = np.random.normal(
        0,
        0.2
    )

    # -------------------------------------
    # TEMPORAL SMOOTHING
    # -------------------------------------

    sinr = (
        0.85 * previous_sinr
        +
        0.15 * (base_signal + noise)
    )

    # -------------------------------------
    # UPDATE STATES
    # -------------------------------------

    previous_sinr = sinr

    time_step += 1

    # -------------------------------------
    # ESTIMATED NETWORK METRICS
    # -------------------------------------

    latency = max(
        5,
        50 - sinr
    )

    download_speed = sinr * 2

    upload_speed = sinr * 0.8

    # -------------------------------------
    # RETURN METRICS
    # -------------------------------------

    return {

        "download":
        round(download_speed, 2),

        "upload":
        round(upload_speed, 2),

        "latency":
        round(latency, 2),

        "sinr":
        round(sinr, 2)
    }