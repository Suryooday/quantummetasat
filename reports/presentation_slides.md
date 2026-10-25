# Fixed Hybrid VQC for LEO Satellite Channel Estimation: Presentation Slides

---
### Slide 1: Title & Research Objective
- **Title:** QuantumMeta-Sat: Meta-Learning Enabled Hybrid Quantum-Classical Neural Networks for Adaptive Channel Estimation in LEO Satellite Communications
- **Goal:** Enable few-shot channel estimation across dynamic LEO satellite environments using compact hybrid quantum neural networks and MAML.

---
### Slide 2: The LEO Satellite Channel Challenge
- High orbital velocities ($v \approx 7.56$ km/s) create massive Doppler shifts ($\pm 50$ kHz).
- Slant ranges vary rapidly with elevation angle ($600$ km $\to 2200$ km), causing severe path loss dynamics.
- Shadowed-Rician fading (Nakagami-$m$ + Rayleigh scatterers) causes non-stationary fading.

---
### Slide 3: Proposed QuantumMeta-Sat Architecture
- **Classical IQ Feature Encoder:** Compresses sliding window $W=9$ observations ($36$ features) into $N_q$ bounded phase angles.
- **Hybrid VQC:** Parameterized $R_y/R_z$ rotations, circular CNOT entanglement, and multi-layer data re-uploading.
- **MAML Meta-Learning:** Adapts on $10-20$ pilot symbols with $5$ gradient updates on unseen channel tasks.

---
### Slide 4: Dataset Generation Pipeline
- 80 Training Tasks ($16,000$ samples), 20 Validation Tasks ($4,000$ samples), 30 Test Tasks ($6,000$ samples).
- Strict out-of-distribution elevation and SNR test conditions with zero data leakage.

---
### Slide 5: Quantum Circuit Design (Proposed 4-Qubit Architecture)
- Compact $4$-qubit circuit with depth $12$ (only $24$ quantum parameters).
- $R_y$ feature angle encoding with data re-uploading at each variational layer.
- Circular CNOT entanglement and Pauli-$Z$ expectation measurements feeding classical decoder.

---
### Slide 6: Benchmark Baselines
- Classical Least Squares (LS) & Windowed Moving-Average LS.
- Supervised Deep MLP ($21,506$ parameters) & 1D CNN ($43,682$ parameters).
- Classical MAML for few-shot adaptation baseline.

---
### Slide 7: Classical Baseline Results
- **LS:** NMSE = -5.89 dB (noise-bounded).
- **Windowed LS:** NMSE = -0.34 dB (fails due to uncompensated Doppler phase drift).
- **MLP (21k params):** NMSE = -8.95 dB, BER = 3.5000e-03.
- **CNN (43k params):** NMSE = -9.06 dB, BER = 2.1667e-03.

---
### Slide 8: Proposed Fixed Hybrid VQC Performance
- **NMSE:** -8.73 dB (matches classical MLP within ~0.2 dB).
- **BER:** 3.4167e-03 (matches classical MLP, well within FEC limits).
- **Parameter Footprint:** Only **798 total parameters** (96.3% reduction vs MLP, 98.2% vs CNN).

---
### Slide 9: Exploratory Extensions (Documented Negative Results)
- **8-Qubit Scaled Circuit:** NMSE = -0.17 dB, BER = 4.1833e-01 (**Failed:** Barren plateaus in $2^8=256$ state space; near-random guessing).
- **Q-MAML (4 Qubits):** NMSE = -3.34 dB, BER = 1.1233e-01 (**Failed:** High variance of quantum meta-gradients causes training collapse).
- *Scientific Takeaway:* The compact 4-qubit supervised regime is the viable operating point.

---
### Slide 10: Trade-offs: Parameter Efficiency vs Latency
- Parameter reduction: 798 vs 21,506 (MLP) / 43,682 (CNN).
- Software simulation latency: 2.18 ms (`default.qubit`) vs 0.011 ms (MLP).
- Migration to C++ backends (`lightning.qubit`) or dedicated QPU co-processors needed for sub-microsecond real-time tracking.

---
### Slide 11: Conclusions & Practical Contributions
- A 4-qubit Hybrid VQC achieves communications-grade channel estimation with ultra-compact parameters.
- Honest documentation of dimensional scaling limits (8 qubits) and meta-learning limits provides valuable guidance for NISQ communications research.
