import numpy as np
import matplotlib.pyplot as plt

# Example modulation sequence
modulations = [
    "BPSK",
    "QPSK",
    "16-QAM",
    "64-QAM"
]

# Power consumption model (Watts)
power_cost = {
    "BPSK": 1,
    "QPSK": 2.5,
    "16-QAM": 5,
    "64-QAM": 9
}

# Throughput model (bits/symbol)
throughput = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4,
    "64-QAM": 6
}

# Generate random modulation sequence
np.random.seed(42)

mcs_sequence = np.random.choice(
    modulations,
    200
)

# Calculate energy + throughput
# Simulated SINR values
sinr_values = np.random.uniform(
    5,
    30,
    200
)

# Calculate energy + throughput
power_values = []

throughput_values = []

bits_per_joule = []

effective_throughput = []

for mcs, sinr in zip(
    mcs_sequence,
    sinr_values
):

    power = power_cost[mcs]

    raw_rate = throughput[mcs]

    # Link reliability factor
    reliability = min(
        sinr / 30,
        1
    )

    # Effective throughput
    rate = raw_rate * reliability

    efficiency = rate / power

    power_values.append(power)

    throughput_values.append(rate)

    bits_per_joule.append(efficiency)

    effective_throughput.append(rate)

# -------------------------------
# PLOTS
# -------------------------------

plt.figure(figsize=(14,10))

# Power
plt.subplot(3,1,1)

plt.plot(power_values)

plt.title("Power Consumption")

plt.ylabel("Power (W)")

plt.grid(True)

# Throughput
plt.subplot(3,1,2)

plt.plot(throughput_values)

plt.title("Throughput")

plt.ylabel("Bits/Symbol")

plt.grid(True)

# Energy efficiency
plt.subplot(3,1,3)

plt.plot(bits_per_joule)

plt.title("Energy Efficiency (Bits/Joule)")

plt.xlabel("Time")

plt.ylabel("Bits/Joule")

plt.grid(True)

plt.tight_layout()

# Save graph
plt.savefig(
    "graphs/evaluation/energy_efficiency.png",
    dpi=300
)

plt.show()

# Average statistics
print(
    f"Average Power Consumption: "
    f"{np.mean(power_values):.2f} W"
)

print(
    f"Average Throughput: "
    f"{np.mean(throughput_values):.2f} bits/symbol"
)

print(
    f"Average Bits/Joule: "
    f"{np.mean(bits_per_joule):.2f}"
)