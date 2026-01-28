# Efficient Neural Architectures for High-Frequency Forecasting

[![Rank 28- peacemaker](https://img.shields.io/badge/Rank-28%2F1000%2B-gold)](https://wundernn.io/wunder_challenge/finals)
[![Constraints](https://img.shields.io/badge/Constraint-20MB%20%7C%20CPU-red)]()

## 🚀 Project Overview
This repository contains the source code and ablation studies for my solution to the **Wunder Fund RNN Challenge**. The goal was to predict future market states based on anonymized high-frequency data features.

**The Challenge:**
*   **Data:** 500+ sequences of 32-dim financial features.
*   **Constraint:** The entire inference solution (code + weights) must fit under **20 MB** and run on **1 CPU core** within 60 minutes.
*   **Result:** Achieved **Rank 28** (Score 0.3873) on the private leaderboard.

## 💡 The Winning Solution: SE-Mish-Swarm
After benchmarking 42 different architectures (including Mamba, Transformers, and ResNets), the winning solution utilized a **Massive Ensemble of Quantized Deep Residual GRUs**.

### Key Innovations:
1.  **Mish Activation:** Replaced ReLU/Tanh to prevent dead gradients in deep (6-layer) Recurrent Networks.
2.  **Squeeze-and-Excitation (SE-Block):** Applied channel-wise attention at every timestep to dynamically weight feature importance based on volatility regimes.
3.  **INT8 Quantization Pipeline:** Developed a custom post-training quantization workflow that reduced model size by **72%**. This allowed ensembling **10 distinct models** within the 20MB limit, significantly reducing variance compared to a single FP32 model.
4.  **Ranger Optimizer:** Used RAdam + Lookahead to escape sharp local minima during training.

## 🔬 Ablation Study
I rigorously tested "State of the Art" sequence models. Most failed due to the high noise-to-signal ratio of financial data.

| Architecture | Score | Failure Mode |
| :--- | :--- | :--- |
| **SE-Mish-GRU (Winner)** | **0.3818** | - |
| Hybrid GRU+LSTM | 0.3797 | Complexity penalty |
| Transformer-XL | 0.3627 | Overfitting noise |
| Mamba-2 (SSM) | Failed | Unstable training (NaN) |
| WaveNet (CNN) | Timeout | Inference too slow on CPU |

## 🛠️ Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the interactive demo: `streamlit run demo/app.py`
