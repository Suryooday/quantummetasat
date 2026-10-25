# QuantumMeta-Sat: Engineering Limitations & Future Directions

## 1. Scientific & Practical Limitations
1. **Simulation vs Hardware Execution:** Experiments were executed on classical state-vector quantum simulators (`default.qubit`). Physical Noisy Intermediate-Scale Quantum (NISQ) hardware introduces decoherence, gate infidelity, and readout errors.
2. **Inference Latency on Classical Processors:** Simulating quantum state evolution classically incurs higher latency (~2.1 ms) compared to highly optimized classical matrix multiplications (~0.01 ms). Hardware QPU acceleration would be required for real-time nanosecond-scale LEO tracking.
3. **Doppler Dynamics Approximation:** Doppler phase evolution was modeled across discrete bursts with configurable drift rates. Full multi-orbit dynamic Keplerian Doppler ephemeris integration remains a future extension point.
4. **Parameter Efficiency vs Absolute Advantage:** As explicitly emphasized in the proposal, quantum hybrid models demonstrate notable parameter compactness rather than claims of unconstrained quantum supremacy.

---

## 2. Reproducibility Instructions
To reproduce the entire experimental pipeline from scratch:
```bash
# 1. Generate LEO channel dataset
python scripts/generate_dataset.py

# 2. Validate dataset integrity
python scripts/validate_dataset.py

# 3. Train classical baselines (LS, Windowed LS, MLP, CNN)
python scripts/train_classical.py

# 4. Train Classical MAML
python scripts/train_maml.py

# 5. Train Fixed Hybrid VQC
python scripts/train_vqc.py

# 6. Train Q-MAML / QuantumMeta-Sat
python scripts/train_qmaml.py

# 7. Run full benchmark suite and generate all artifacts
python scripts/run_final_experiment.py
python scripts/generate_results.py
```
