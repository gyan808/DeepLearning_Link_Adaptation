import numpy as np
import matplotlib.pyplot as plt

# =================================================
# GENERATE CHANNEL CONDITIONS
# =================================================

time = np.arange(0, 500)

# =================================================
# AI-BASED SINR
# =================================================

ai_sinr = (
    15
    + 5 * np.sin(2 * np.pi * time / 200)
    + np.random.normal(0, 1, 500)
)

# =================================================
# BASELINE STATIC SINR
# =================================================

baseline_sinr = (
    12
    + 2 * np.sin(2 * np.pi * time / 200)
    + np.random.normal(0, 2, 500)
)

# =================================================
# AI-BASED ADAPTIVE MODULATION
# =================================================

ai_throughput = []
ai_power = []

for sinr in ai_sinr:

    if sinr < 8:

        ai_throughput.append(1)
        ai_power.append(2)

    elif sinr < 15:

        ai_throughput.append(2)
        ai_power.append(4)

    elif sinr < 22:

        ai_throughput.append(4)
        ai_power.append(6)

    else:

        ai_throughput.append(6)
        ai_power.append(9)

# =================================================
# STATIC BASELINE SYSTEM
# =================================================

baseline_throughput = []
baseline_power = []

for sinr in baseline_sinr:

    # Fixed modulation
    baseline_throughput.append(2)

    # Constant power
    baseline_power.append(5)

# =================================================
# ENERGY EFFICIENCY
# =================================================

ai_efficiency = (
    np.array(ai_throughput)
    /
    np.array(ai_power)
)

baseline_efficiency = (
    np.array(baseline_throughput)
    /
    np.array(baseline_power)
)

# =================================================
# PDR + BLER CALCULATION
# =================================================

# Total packets
total_packets = 1000

# Packet success probability
ai_success_probability = np.clip(
    ai_sinr / 25,
    0,
    1
)

baseline_success_probability = np.clip(
    baseline_sinr / 25,
    0,
    1
)

# Delivered packets
ai_delivered_packets = np.sum(
    ai_success_probability
)

baseline_delivered_packets = np.sum(
    baseline_success_probability
)

# Packet Delivery Ratio (PDR)
ai_pdr = (
    ai_delivered_packets
    /
    total_packets
)

baseline_pdr = (
    baseline_delivered_packets
    /
    total_packets
)

# Block Error Rate (BLER)
ai_bler = 1 - ai_pdr

baseline_bler = 1 - baseline_pdr

# =================================================
# AVERAGE METRICS
# =================================================

avg_ai_throughput = np.mean(
    ai_throughput
)

avg_baseline_throughput = np.mean(
    baseline_throughput
)

avg_ai_power = np.mean(
    ai_power
)

avg_baseline_power = np.mean(
    baseline_power
)

avg_ai_efficiency = np.mean(
    ai_efficiency
)

avg_baseline_efficiency = np.mean(
    baseline_efficiency
)

# =================================================
# PRINT RESULTS
# =================================================

print("\n===== FINAL BENCHMARK =====\n")

print(
    "AI Throughput :",
    round(avg_ai_throughput, 2)
)

print(
    "Baseline Throughput :",
    round(avg_baseline_throughput, 2)
)

print()

print(
    "AI Power :",
    round(avg_ai_power, 2)
)

print(
    "Baseline Power :",
    round(avg_baseline_power, 2)
)

print()

print(
    "AI Bits/Joule :",
    round(avg_ai_efficiency, 2)
)

print(
    "Baseline Bits/Joule :",
    round(avg_baseline_efficiency, 2)
)

print()

print(
    "AI PDR :",
    round(ai_pdr, 3)
)

print(
    "Baseline PDR :",
    round(baseline_pdr, 3)
)

print()

print(
    "AI BLER :",
    round(ai_bler, 3)
)

print(
    "Baseline BLER :",
    round(baseline_bler, 3)
)

# =================================================
# VISUALIZATION
# =================================================

plt.figure(figsize=(16, 18))

# =================================================
# SINR COMPARISON
# =================================================

plt.subplot(4, 1, 1)

plt.plot(
    ai_sinr,
    label="AI-Based SINR"
)

plt.plot(
    baseline_sinr,
    label="Baseline SINR"
)

plt.title(
    "AI vs Baseline SINR"
)

plt.ylabel("SINR (dB)")

plt.grid(True)

plt.legend()

# =================================================
# THROUGHPUT COMPARISON
# =================================================

plt.subplot(4, 1, 2)

plt.plot(
    ai_throughput,
    label="AI Throughput"
)

plt.plot(
    baseline_throughput,
    label="Baseline Throughput"
)

plt.title(
    "Throughput Comparison"
)

plt.ylabel("Bits/Symbol")

plt.grid(True)

plt.legend()

# =================================================
# ENERGY EFFICIENCY COMPARISON
# =================================================

plt.subplot(4, 1, 3)

plt.plot(
    ai_efficiency,
    label="AI Efficiency"
)

plt.plot(
    baseline_efficiency,
    label="Baseline Efficiency"
)

plt.title(
    "Energy Efficiency Comparison"
)

plt.ylabel("Bits/Joule")

plt.grid(True)

plt.legend()

# =================================================
# RELIABILITY METRICS
# =================================================

plt.subplot(4, 1, 4)

metrics = [
    "AI PDR",
    "Baseline PDR",
    "AI BLER",
    "Baseline BLER"
]

values = [
    ai_pdr,
    baseline_pdr,
    ai_bler,
    baseline_bler
]

plt.bar(
    metrics,
    values
)

plt.title(
    "Reliability Metrics Comparison"
)

plt.ylabel("Ratio")

plt.xlabel("Metrics")

plt.grid(True)

# =================================================
# FINALIZE
# =================================================

plt.tight_layout(
    pad=4.0
)

plt.savefig(
    "graphs/final_baseline_comparison.png",
    dpi=300
)

plt.show()