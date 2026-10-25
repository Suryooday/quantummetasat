| Model                                | Params   | Qubits   | Quantum Params   |   NMSE |   NMSE (dB) |        BER |   Inference Latency (ms) | Adaptation Steps   |
|:-------------------------------------|:---------|:---------|:-----------------|-------:|------------:|-----------:|-------------------------:|:-------------------|
| Least Squares (LS)                   | —        | —        | —                | 0.2573 |       -5.89 | 0.00016667 |                   0.0003 | —                  |
| Windowed LS Reference                | —        | —        | —                | 0.9243 |       -0.34 | 0.37267    |                   0.0048 | —                  |
| Classical Supervised MLP             | 21,506   | —        | —                | 0.1274 |       -8.95 | 0.0035     |                   0.0114 | —                  |
| Classical Supervised CNN             | 43,682   | —        | —                | 0.1241 |       -9.06 | 0.0021667  |                   0.0427 | —                  |
| Classical MAML (MLP)                 | 21,506   | —        | —                | 0.1551 |       -8.09 | 0.009      |                   0.0114 | 5–10               |
| Proposed Fixed Hybrid VQC (4 Qubits) | 798      | 4        | 24               | 0.1339 |       -8.73 | 0.0034167  |                   2.1824 | —                  |