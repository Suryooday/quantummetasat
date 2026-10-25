# QuantumMeta-Sat: Methodology & Mathematical Formulations

## 1. Research Objective & System Model
The primary objective of **QuantumMeta-Sat** is to provide rapid, few-shot adaptive channel estimation for dynamic Low Earth Orbit (LEO) satellite communications using a hybrid variational quantum-classical neural network combined with Model-Agnostic Meta-Learning (MAML).

The received signal at discrete symbol index $k$ is given by:
$$ y[k] = h_{\mathrm{eff}}[k] x[k] + n[k] $$
where $x[k] \in \frac{1}{\sqrt{2}}\{1+j, 1-j, -1+j, -1-j\}$ is the normalized QPSK transmit symbol ($E[|x|^2] = 1$), $n[k] \sim \mathcal{CN}(0, P_n)$ is complex Additive White Gaussian Noise (AWGN), and $h_{\mathrm{eff}}[k]$ is the composite time-varying LEO channel.

---

## 2. LEO Physics & Orbital Geometry
### 2.1 Spherical Slant Range
Given Earth mean radius $R_e = 6,378,137\text{ m}$, satellite orbital altitude $h = 600,000\text{ m}$, and ground elevation angle $\theta \in [0^\circ, 90^\circ]$, the geometric slant range is modeled via spherical trigonometry:
$$ d(\theta) = \sqrt{ R_e^2\sin^2\theta + 2R_eh + h^2 } - R_e\sin\theta $$
- **Boundary Condition (Zenith):** At $\theta = 90^\circ$, $\sin\theta = 1 \implies d(90^\circ) = h = 600\text{ km}$.
- **Monotonicity:** $\frac{\partial d}{\partial \theta} < 0$, strictly decreasing from horizon to zenith.

### 2.2 Free-Space Path Loss (FSPL) & Amplitude
For carrier frequency $f_c = 2.0\text{ GHz}$ and speed of light $c = 2.99792458 \times 10^8\text{ m/s}$:
$$ \mathrm{FSPL}_{\mathrm{dB}} = 20\log_{10}\left( \frac{4\pi d f_c}{c} \right) $$
Amplitude attenuation normalized to zenith reference distance $h$:
$$ A(d) = \frac{h}{d(\theta)} $$

### 2.3 Shadowed-Rician Small-Scale Fading
Following the standard Land Mobile Satellite (LMS) formulation (Abdi et al., 2003):
$$ h_{\mathrm{small}} = \zeta e^{j\theta_0} + (X + jY) $$
where $\zeta^2 \sim \mathrm{Gamma}(m, \Omega/m)$ represents Nakagami-$m$ line-of-sight (LOS) shadowing, $\theta_0 \sim \mathcal{U}[0, 2\pi)$ is the random LOS phase, and $X, Y \sim \mathcal{N}(0, b_0)$ model diffuse Gaussian scatterers with total multipath power $2b_0$.

### 2.4 Doppler Frequency & Phase Evolution
Satellite orbital velocity $v_{\mathrm{sat}} \approx 7.56\text{ km/s}$ produces Doppler shifts up to $f_{D,\max} = \frac{v_{\mathrm{sat}}}{\lambda} \approx 50.4\text{ kHz}$.
The discrete-time Doppler phase evolution across symbol duration $T_s = 10^{-6}\text{ s}$ is:
$$ \phi_D[k] = 2\pi f_D k T_s + \phi_0 $$
Composite channel coefficient:
$$ h_{\mathrm{eff}}[k] = A(d(\theta)) \cdot h_{\mathrm{small}}[k] \cdot e^{j\phi_D[k]} $$

---

## 3. Sliding Window Feature Representation
For observation window size $W = 9$, the model extracts $9$ consecutive complex symbol pairs $(x, y)$:
$$ \mathbf{x}_{\mathrm{in}} = [\mathrm{Re}(x_{t-8}), \mathrm{Im}(x_{t-8}), \mathrm{Re}(y_{t-8}), \mathrm{Im}(y_{t-8}), \dots, \mathrm{Re}(x_t), \mathrm{Im}(x_t), \mathrm{Re}(y_t), \mathrm{Im}(y_t)]^T \in \mathbb{R}^{36} $$
Target prediction:
$$ \mathbf{y}_{\mathrm{target}} = [\mathrm{Re}(h_{\mathrm{eff}, t}), \mathrm{Im}(h_{\mathrm{eff}, t})]^T \in \mathbb{R}^2 $$

---

## 4. Hybrid Variational Quantum-Classical Neural Network
1. **Classical Encoder:** Compresses $\mathbb{R}^{36} \to \mathbb{R}^{N_q}$ bounded in $[-\pi, \pi]$ via $\tanh$ activation.
2. **Angle Encoding:** Rotates $N_q$ qubits using $R_y(\phi_q)$ gates.
3. **Variational Circuit Layers ($L = 3$):**
   - Data re-uploading at each layer ($R_y$ feature rotations).
   - Parameterized single-qubit rotations $R_y(\theta_{l,q,0})$ and $R_z(\theta_{l,q,1})$.
   - Circular CNOT entangling topology.
4. **Quantum Measurement:** Pauli-$Z$ expectation values $\langle Z_q \rangle \in [-1, 1]$.
5. **Classical Decoder:** Maps Pauli-$Z$ measurements $\mathbb{R}^{N_q} \to \mathbb{R}^2 \to [\hat{h}_r, \hat{h}_i]$.

---

## 5. Model-Agnostic Meta-Learning (MAML) Framework
- **Inner Loop (Task-Specific Adaptation on $K$ Pilots):**
  $$ \theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{T_i}(\mathcal{D}_i^S; \theta) $$
- **Outer Loop (Meta-Optimization across Tasks):**
  $$ \theta \leftarrow \theta - \beta \nabla_\theta \sum_{i} \mathcal{L}_{T_i}(\mathcal{D}_i^Q; \theta_i') $$
