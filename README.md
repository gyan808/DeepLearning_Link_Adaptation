# Real-Time AI-Based TN-NTN Link Adaptation Using Deep Learning

## Overview

This project implements a real-time AI-driven TN-NTN (Terrestrial Network / Non-Terrestrial Network) Link Adaptation System using deep learning models.

The system continuously monitors wireless channel conditions and predicts future SINR (Signal-to-Interference-plus-Noise Ratio) values using:

* LSTM Neural Network
* GRU Neural Network

Based on predicted SINR values, the system dynamically performs:

* Adaptive Modulation Selection
* Intelligent TN-NTN Switching
* Throughput Optimization
* Energy Efficiency Optimization

The project also includes:

* ONNX Runtime Deployment
* Real-Time Dashboard Visualization
* Wireless Channel Simulation
* Live AI Inference
* Comparative AI Benchmarking

---

# Key Features

## AI-Based SINR Prediction

* Real-time SINR prediction using LSTM
* Real-time SINR prediction using GRU
* Sliding window temporal prediction
* AI-assisted wireless communication optimization

---

## Real-Time Dashboard

* Live SINR monitoring
* Real-time graph visualization
* Interactive Streamlit dashboard
* Continuous AI inference display

---

## Dual Deep Learning Models

The system deploys:

* LSTM (Long Short-Term Memory)
* GRU (Gated Recurrent Unit)

Both models are compared in real time for:

* Prediction quality
* Temporal stability
* Communication performance

---

## ONNX Deployment

The trained models are exported to ONNX format for efficient deployment.

Benefits include:

* Faster inference
* Smaller model size
* Lightweight deployment
* Cross-platform compatibility

---

## Adaptive Modulation and Coding (AMC)

The system dynamically selects modulation schemes according to predicted SINR.

Supported Modulation Schemes:

| SINR Range | Modulation |
| ---------- | ---------- |
| < 8 dB     | BPSK       |
| 8–15 dB    | QPSK       |
| 15–22 dB   | 16-QAM     |
| > 22 dB    | 64-QAM     |

---

## TN-NTN Intelligent Switching

The system dynamically selects:

* TN (Terrestrial Network)
* NTN (Non-Terrestrial Network)

based on predicted wireless channel quality.

---

## Energy Efficiency Optimization

The system estimates:

* Power Consumption
* Throughput
* Bits/Joule Efficiency

for energy-aware communication.

---

## Wireless Channel Modeling

The project simulates realistic wireless channel conditions including:

* Fast Fading
* Shadow Fading
* Temporal Correlation
* Channel Noise
* Dynamic SINR Variation

---

## Real-Time Metrics

The dashboard displays:

* Current SINR
* LSTM Predicted SINR
* GRU Predicted SINR
* Download Speed
* Upload Speed
* Latency
* Selected MCS
* Throughput
* Power Consumption
* Energy Efficiency
* Active TN/NTN Network

---

# Technologies Used

* Python
* Streamlit
* NumPy
* Matplotlib
* PyTorch
* ONNX
* ONNX Runtime
* Scikit-Learn
* Joblib

---

# System Architecture

```text
Wireless Channel Model
        ↓
Real-Time SINR Stream
        ↓
LSTM + GRU Prediction
        ↓
Adaptive Modulation
        ↓
TN-NTN Selection
        ↓
Energy Optimization
        ↓
Live Dashboard Visualization
```

---

# Project Structure

```text
DeepLearning_Link_Adaptation/
│
├── app.py
├── real_network_input.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── lstm_model.onnx
│   ├── gru_model.onnx
│   └── scaler.save
│
├── graphs/
│
└── datasets/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/DeepLearning_Link_Adaptation.git

cd DeepLearning_Link_Adaptation
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
streamlit run app.py
```

---

# Deployment

The project can be deployed using:

* Streamlit Community Cloud
* Docker
* AWS
* Azure
* Google Cloud Platform

---

# Research Significance

This project demonstrates:

* AI-assisted wireless communication
* Real-time deep learning inference
* Intelligent link adaptation
* TN-NTN communication concepts
* AI-driven communication optimization
* Deployment-ready communication systems
* Future 6G communication concepts

---

# Current Capabilities

The system currently supports:

* Real-time SINR streaming
* Live LSTM inference
* Live GRU inference
* Real-time adaptive communication
* ONNX deployment
* Wireless environment simulation
* Real-time dashboard visualization

---

# Future Improvements

Potential future extensions include:

* SDR Hardware Integration
* Real 5G/6G Telemetry
* Satellite Communication APIs
* Reinforcement Learning-Based Adaptation
* Multi-User Communication Support
* Real NTN Datasets
* Edge AI Deployment

---

# Results

The system successfully demonstrates:

* Stable SINR prediction
* Real-time AI inference
* Adaptive communication behavior
* Smooth temporal channel prediction
* Efficient ONNX deployment
* Intelligent modulation selection

---

# Author

**Gyanendra Verma**

---

# License

This project is developed for educational, research, and internship purposes.
