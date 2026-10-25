# Fixed Hybrid VQC for LEO Satellite Channel Estimation: Experimental Protocol

## 1. Dataset Generation & Task Protocol
The dataset is partitioned into three strictly isolated splits:
- **Training Set (80 Tasks, 16,000 samples):** Elevation angles $\theta \in [20^\circ, 75^\circ]$, SNR $\in [5, 20]\text{ dB}$, mixed shadowing conditions (infrequent light, average, frequent heavy).
- **Validation Set (20 Tasks, 4,000 samples):** Elevation angles $\theta \in [25^\circ, 70^\circ]$, SNR $\in [5, 20]\text{ dB}$.
- **Test Set (30 Tasks, 6,000 samples):** Unseen elevation angles $[20^\circ, 30^\circ, 45^\circ, 60^\circ, 75^\circ, 90^\circ]$ (including zenith extrapolation $90^\circ$), discrete SNR evaluations $[5, 10, 15, 20]\text{ dB}$, and independent Doppler realizations.

### Data Leakage Prevention
1. Feature standard scaler fitted exclusively on training set support data.
2. Test tasks are evaluated strictly out-of-distribution.
3. Temporal windowing applied within individual tasks without cross-task sequence mixing.

---

## 2. Models Compared
### Primary Benchmark Models
1. **Least Squares (LS):** Benchmark channel estimation $\hat{h} = y / x$.
2. **Windowed LS Reference:** Moving-average LS filter over $W = 9$ symbols.
3. **Classical Supervised MLP:** 36 -> 128 -> ReLU -> 128 -> ReLU -> 2 (21,506 parameters).
4. **Classical Supervised CNN:** 1D Conv(32) -> Conv(64) -> FC(64) -> 2 (43,682 parameters).
5. **Classical MAML (MLP):** MLP base learner trained with MAML for adaptation comparison.
6. **Proposed Fixed Hybrid VQC (4 Qubits):** 4-qubit quantum-classical neural network trained under standard supervised learning (**798 parameters**). Primary proposed contribution.

### Exploratory Extensions (Documented Negative Results)
7. **Q-MAML (4 Qubits):** 4-qubit Hybrid VQC trained with MAML meta-gradients (798 parameters).
8. **8-Qubit Scaled Circuit:** Scaled 8-qubit Hybrid VQC (1,058 parameters), investigated to test dimensional scaling limits.

---

## 3. Evaluation Metrics
- **Normalized Mean Squared Error (NMSE):**
  $$ \mathrm{NMSE} = \frac{\mathbb{E}[|\hat{h} - h|^2]}{\mathbb{E}[|h|^2]}, \quad \mathrm{NMSE}_{\mathrm{dB}} = 10\log_{10}(\mathrm{NMSE}) $$
- **Bit Error Rate (BER):** Zero-forcing equalization $\hat{x} = y / \hat{h}$ followed by QPSK hard slicing.
- **Inference & Adaptation Latency:** Measured in milliseconds per sample on identical hardware.
- **Parameter Efficiency Ratio:** Total parameter footprint vs classical baselines.
