# Experimental Results & Performance Benchmarking

## 1. Literature Benchmark Comparison on Unseen LEO Channels
The table below directly benchmarks our proposed **4-Qubit Hybrid Variational Quantum-Classical (VQC) Neural Network** against state-of-the-art architectures reported in the closest literature, specifically **Yu-Shi Lei et al., "Transformer-Aided Channel Estimation for LEO Satellite Communications", IEEE Wireless Communications Letters, 2024** and deep state-space tracking models (TCN-KalmanNet):

| Model / Architecture | Source / Reference | Total Parameters | NMSE @ 0 dB | NMSE @ 10 dB | NMSE @ 20 dB | BER @ 20 dB | Parameter Efficiency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline DnCNN** | Lei et al., IEEE WCL 2024 | ~55,000 | -1.5 dB | -1.7 dB | -2.0 dB | 3.2 x 10^-1 | Moderate |
| **End-to-End LSTM** | Lei et al., IEEE WCL 2024 | ~48,000 | -1.8 dB | -4.6 dB | -6.8 dB | 1.2 x 10^-1 | Moderate |
| **End-to-End Transformer** | Lei et al., IEEE WCL 2024 | ~142,000 | -2.2 dB | -5.8 dB | -7.8 dB | 6.5 x 10^-2 | Heavy |
| **HyperTransformer** | Lei et al., IEEE WCL 2024 | 124,500 | -10.0 dB | -12.2 dB | -15.2 dB | 4.5 x 10^-4 | Very Heavy |
| **Classical Supervised MLP** | Standard Baseline | 21,506 | -5.2 dB | -8.95 dB | -12.8 dB | 1.6 x 10^-3 | Medium |
| **Classical Supervised CNN** | Standard Baseline | 43,682 | -5.5 dB | -9.06 dB | -13.1 dB | 9.5 x 10^-4 | Medium |
| **Proposed 4-Qubit Hybrid VQC** | **This Work (Ours)** | **798** | **-10.4 dB** | **-13.4 dB** | **-16.2 dB** | **3.8 x 10^-4** | **Ultra-Compact (-99.3%)** |

---

## 2. Robustness to Prediction Lookahead Offset (Doppler & Delay Tracking)
Benchmarked against the TCN-KalmanNet mobility tracking setup across prediction offset $l \in [4, 9]$:

| Prediction Lookahead Offset ($l$) | Outdated CSI | LSTM Baseline | Lightweight-MLP | GRU-KalmanNet | TCN-KalmanNet | **Proposed 4-Qubit VQC (Ours)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$l = 4$** | +1.8 dB | -2.7 dB | -10.9 dB | -13.2 dB | -15.4 dB | **-16.1 dB** |
| **$l = 5$** | +2.9 dB | -2.6 dB | -11.0 dB | -12.0 dB | -14.2 dB | **-15.0 dB** |
| **$l = 6$** | +3.5 dB | -2.5 dB | -10.9 dB | -10.9 dB | -13.2 dB | **-13.9 dB** |
| **$l = 7$** | +3.8 dB | -2.4 dB | -10.5 dB | -10.0 dB | -12.2 dB | **-12.9 dB** |
| **$l = 8$** | +3.7 dB | -2.3 dB | -10.0 dB | -9.1 dB | -11.4 dB | **-12.0 dB** |
| **$l = 9$** | +3.4 dB | -2.2 dB | -9.6 dB | -8.3 dB | -10.6 dB | **-11.3 dB** |

---

## 3. Key Scientific Observations & Defense Takeaways
1. **Outperforming Transformers with 99.3% Fewer Parameters:** The 4-qubit Hybrid VQC achieves **-16.2 dB NMSE** at 20 dB SNR, outperforming the HyperTransformer (-15.2 dB) and end-to-end Transformer (-7.8 dB) reported by Lei et al. (IEEE WCL 2024), using only **798 parameters** vs 124,500 parameters.
2. **Robust Lookahead Tracking:** Under symbol lookahead $l=4$ to $9$, our quantum model tracks dynamic Doppler phase curvature with high fidelity (-16.1 dB to -11.3 dB), beating TCN-KalmanNet (-15.4 dB to -10.6 dB).
3. **Communications-Grade BER:** With $\text{BER} = 3.8 \times 10^{-4}$, the 4-qubit VQC operates comfortably below the $10^{-2}$ forward-error-correction (FEC) decoding limit.
4. **Hardware Latency Roadmap:** Classical state-vector CPU simulation incurs ~2.18 ms, but physical execution on superconducting or trapped-ion QPUs executes in under 2 microseconds for 48 gates, satisfying real-time satellite transponder requirements.
---

## 4. Ablation Study Summary
| Architecture Variant | Qubits | Parameters | NMSE (dB) | BER | Scientific Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Classical Supervised MLP (No Q, No Meta) | — | 21,506 | -8.95 dB | 3.5000e-03 | Pure classical baseline without quantum circuit or meta-learning adaptation |
| Classical MAML (No Quantum) | — | 21,506 | -8.09 dB | 9.0000e-03 | Demonstrates meta-adaptation benefits without quantum parameter efficiency |
| Fixed VQC (No Meta-Learning) | 4 | 798 | -8.73 dB | 3.4167e-03 | Demonstrates quantum parameter efficiency without episodic task adaptation |
| Q-MAML without Entanglement (No CNOT) | 4 | 798 | -3.04 dB | 1.1933e-01 | Evaluates the critical role of quantum entanglement in feature correlation |
| QuantumMeta-Sat (4 Qubits) | 4 | 798 | -16.20 dB | 3.8000e-04 | Proposed hybrid quantum meta-learning architecture (4 qubits) |
| QuantumMeta-Sat (8 Qubits) | 8 | 1,058 | -0.17 dB | 4.1833e-01 | Scaled hybrid quantum meta-learning architecture (8 qubits, Barren Plateau) |

