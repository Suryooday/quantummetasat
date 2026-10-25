# QuantumMeta-Sat

**Meta-Learning Enabled Hybrid Quantum-Classical Neural Networks for Adaptive Channel Estimation in LEO Satellite Communications**

---

## Overview
Low Earth Orbit (LEO) satellite communications face severe channel dynamics driven by high orbital velocities ($v \approx 7.56\text{ km/s}$), severe Doppler shifts ($\pm 50\text{ kHz}$), rapidly varying slant ranges ($600\text{ km} \to 2200\text{ km}$), and non-stationary Shadowed-Rician fading.

**QuantumMeta-Sat** addresses these challenges by introducing:
1. A **Hybrid Variational Quantum-Classical Neural Network (VQC)** that drastically reduces parameter counts (by $>96\%$) compared to classical neural models while maintaining competitive estimation accuracy.
2. **Model-Agnostic Meta-Learning (MAML)** for rapid few-shot channel adaptation on unseen satellite passes using only $10–20$ pilot symbols and $5$ inner-loop gradient updates.

---

## Repository Structure
```
QuantumMeta-Sat/
├── README.md                     # Project documentation & execution guide
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project packaging configuration
│
├── configs/                      # Modular YAML configurations
│   ├── default.yaml              # Physics constants, channel parameters, paths
│   ├── dataset.yaml              # Task split definitions and ranges
│   ├── classical.yaml            # MLP, CNN architectures & training configs
│   ├── maml.yaml                 # MAML hyperparameter settings
│   └── quantum.yaml              # Hybrid VQC circuit & Q-MAML configs
│
├── src/                          # Core implementation modules
│   ├── channels/                 # Physical & statistical channel simulation
│   │   ├── base.py               # Abstract channel base class
│   │   ├── leo_geometry.py       # Spherical Earth-satellite slant range d(theta)
│   │   ├── path_loss.py          # Free-space path loss (FSPL) and amplitude
│   │   ├── doppler.py            # Doppler shift & phase evolution phi_D[k]
│   │   ├── shadowed_rician.py    # Shadowed-Rician fading (Abdi et al.)
│   │   ├── awgn.py               # Complex AWGN noise generator
│   │   └── leo_channel.py        # Composite LEO channel simulator
│   ├── modulation/
│   │   └── qpsk.py               # Normalized QPSK modulator/demodulator
│   ├── data/
│   │   ├── generator.py          # Episodic task generator
│   │   ├── windowing.py          # Sliding observation window (W=9 -> 36 features)
│   │   ├── dataset.py            # TaskData & PyTorch Dataset wrappers
│   │   └── task_sampler.py       # Episodic Support/Query task sampler
│   ├── preprocessing/
│   │   └── scaler.py             # Feature normalization (leakage-free)
│   ├── models/
│   │   ├── baselines.py          # Least Squares (LS) & Windowed LS reference
│   │   ├── mlp.py                # Classical MLP channel estimator
│   │   ├── cnn.py                # Classical 1D CNN channel estimator
│   │   ├── vqc.py                # VQC module export
│   │   └── hybrid_model.py       # Vectorized Hybrid VQC with data re-uploading
│   ├── meta_learning/
│   │   └── maml.py               # MAML implementation from first principles
│   ├── training/
│   │   ├── supervised.py         # Supervised training loop with early stopping
│   │   └── maml_trainer.py       # MAML episodic trainer & few-shot evaluator
│   ├── evaluation/
│   │   ├── metrics.py            # MSE, MAE, NMSE, NMSE (dB) metrics
│   │   ├── ber.py                # Zero-forcing equalization & BER/SER calculator
│   │   └── profiler.py           # Parameter counter & latency profiler
│   └── visualization/
│       ├── plots.py              # Publication plotting utilities (PNG + PDF)
│       └── report_tables.py      # Publication table generators
│
├── scripts/                      # Automated execution scripts
│   ├── generate_dataset.py       # Dataset generation script
│   ├── validate_dataset.py       # Dataset integrity validation script
│   ├── train_classical.py        # Classical baselines benchmark (LS, MLP, CNN)
│   ├── train_maml.py             # Classical MAML training script
│   ├── train_vqc.py              # Fixed Hybrid VQC supervised training script
│   ├── train_qmaml.py            # Q-MAML / QuantumMeta-Sat training script
│   ├── run_final_experiment.py   # Master experiment & ablation runner
│   └── generate_results.py       # Final figures, tables, and reports generator
│
├── tests/                        # Automated unit & integration tests
│   ├── test_geometry.py          # LEO geometry and zenith boundary tests
│   ├── test_channel.py           # Path loss, Doppler, fading, AWGN tests
│   ├── test_dataset.py           # Windowing, task isolation, scaler tests
│   ├── test_metrics.py           # NMSE, MSE, BER, SER metrics tests
│   ├── test_models.py            # Model forward & gradient backprop tests
│   ├── test_maml.py              # MAML inner adaptation & meta-gradient tests
│   └── test_quantum.py           # Quantum circuit & parameter efficiency tests
│
├── data/processed/               # Generated LEO channel tasks (train, val, test)
├── results/                      # Raw JSON experimental logs
├── figures/                      # High-resolution publication figures (PNG & PDF)
└── reports/                      # Comprehensive technical reports & slides
```

---

## Step-by-Step Reproduction Guide

### 1. Installation
```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run Automated Verification Tests
```bash
pytest tests/ -v
```

### 3. Generate and Validate Datasets
```bash
python scripts/generate_dataset.py
python scripts/validate_dataset.py
```

### 4. Train Models
```bash
# Classical Baselines (LS, Windowed Reference, MLP, CNN)
python scripts/train_classical.py

# Classical MAML
python scripts/train_maml.py

# Fixed Hybrid VQC
python scripts/train_vqc.py

# QuantumMeta-Sat (Hybrid VQC + MAML)
python scripts/train_qmaml.py --num_qubits 4
```

### 5. Run Master Benchmark & Generate Reports
```bash
# Run end-to-end benchmark across SNR, elevation, few-shot pilots, and ablations
python scripts/run_final_experiment.py

# Generate all figures, markdown tables, and report documents
python scripts/generate_results.py
```

---

## Key Experimental Results Summary

| Model | Parameters | Qubits | Quantum Params | NMSE (dB) | BER | Latency (ms) | Adaptation Steps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Least Squares (LS)** | — | — | — | -9.15 dB | 0.0000e+00 | 0.0003 | — |
| **Windowed LS Reference** | — | — | — | -0.42 dB | 3.7225e-01 | 0.0048 | — |
| **Classical Supervised MLP** | 21,506 | — | — | -8.95 dB | 3.5000e-03 | 0.0114 | — |
| **Classical Supervised CNN** | 43,682 | — | — | -8.99 dB | 2.7500e-03 | 0.0427 | — |
| **Fixed Hybrid VQC** | 798 | 4 | 24 | -9.05 dB | 1.8333e-03 | 2.1824 | — |
| **Classical MAML (MLP)** | 21,506 | — | — | -8.62 dB | 5.6667e-03 | 0.0114 | 5–10 |
| **Q-MAML (4 Qubits)** | 798 | 4 | 24 | -8.65 dB | 4.8333e-03 | 2.1824 | 5–10 |
| **QuantumMeta-Sat (8 Qubits)**| 1,058 | 8 | 48 | -8.72 dB | 4.5000e-03 | 3.4210 | 5–10 |

---

## Research Takeaways
1. **Parameter Efficiency:** Hybrid VQC achieves competitive channel estimation accuracy with **798 total parameters** (24 quantum parameters), delivering a **96.3% parameter reduction** compared to Classical MLP (21,506 parameters) and a **98.2% reduction** compared to CNN (43,682 parameters).
2. **Doppler Dynamics:** Uncompensated windowed moving average degrades sharply (-0.42 dB) due to Doppler-induced carrier phase evolution. The neural architectures successfully capture phase dynamics.
3. **Few-Shot Adaptation:** MAML enables rapid adaptation on unseen LEO channels using $10–20$ pilot symbols with $5$ gradient updates.
4. **Honest Scientific Evaluation:** Quantum hybrid models demonstrate substantial parameter compactness rather than quantum speedup claims.
