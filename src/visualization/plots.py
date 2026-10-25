"""Publication-ready plotting utilities with high-resolution PNG and PDF exports."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional

# Set global publication styling
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.grid": True,
})

def save_fig(fig, output_dir: str, filename_base: str):
    """Save plot in both high-res PNG (300 DPI) and vector PDF formats."""
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{filename_base}.png")
    pdf_path = os.path.join(output_dir, f"{filename_base}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {png_path} and {pdf_path}")

def plot_leo_geometry(output_dir: str = "figures"):
    """Plot slant range d(theta) vs elevation angle."""
    from src.channels.leo_geometry import LEOGeometry
    geom = LEOGeometry(earth_radius_m=6378137.0, satellite_altitude_m=600000.0)
    elevations = np.linspace(0.0, 90.0, 200)
    ranges_km = geom.calculate_slant_range(elevations) / 1000.0
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(elevations, ranges_km, color="#1f77b4", label=r"Slant Range $d(\theta)$")
    ax.axhline(600.0, color="#d62728", linestyle=":", label="Zenith Altitude $h=600$ km")
    ax.axvline(20.0, color="#2ca02c", linestyle="--", label=r"Min Operational Elevation ($20^\circ$)")
    
    ax.set_xlabel("Elevation Angle $\\theta$ (degrees)")
    ax.set_ylabel("Slant Range (km)")
    ax.set_title("LEO Satellite Earth Slant Range vs Elevation")
    ax.set_xlim(0, 90)
    ax.legend(loc="upper right")
    save_fig(fig, output_dir, "leo_geometry_slant_range")

def plot_path_loss(output_dir: str = "figures"):
    """Plot Free-Space Path Loss and Amplitude vs elevation angle."""
    from src.channels.leo_geometry import LEOGeometry
    from src.channels.path_loss import PathLoss
    geom = LEOGeometry()
    pl = PathLoss(carrier_freq_hz=2.0e9, ref_distance_m=600000.0)
    
    elevations = np.linspace(10.0, 90.0, 200)
    ranges_m = geom.calculate_slant_range(elevations)
    fspl_db = pl.calculate_fspl_db(ranges_m)
    amp_norm = pl.calculate_amplitude(ranges_m, normalized=True)
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    color = "#1f77b4"
    ax1.set_xlabel("Elevation Angle $\\theta$ (degrees)")
    ax1.set_ylabel("FSPL (dB)", color=color)
    ax1.plot(elevations, fspl_db, color=color, label="FSPL (dB)")
    ax1.tick_params(axis="y", labelcolor=color)
    
    ax2 = ax1.twinx()
    color = "#ff7f0e"
    ax2.set_ylabel("Normalized Amplitude $A(d)$", color=color)
    ax2.plot(elevations, amp_norm, color=color, linestyle="--", label="Norm. Amplitude $A(d)$")
    ax2.tick_params(axis="y", labelcolor=color)
    
    plt.title("LEO Free-Space Path Loss and Channel Amplitude Attenuation")
    save_fig(fig, output_dir, "path_loss_vs_elevation")

def plot_channel_distributions(output_dir: str = "figures"):
    """Plot Shadowed-Rician channel fading amplitude distributions."""
    from src.channels.shadowed_rician import ShadowedRicianFading
    from src.utils.seed import set_seed
    set_seed(42)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    envs = [
        ("infrequent_light", "Light Shadowing (Open/Rural)", "#2ca02c"),
        ("average", "Average Shadowing (Suburban)", "#1f77b4"),
        ("frequent_heavy", "Heavy Shadowing (Urban/Dense)", "#d62728"),
    ]
    
    for env_key, label, color in envs:
        fading = ShadowedRicianFading.from_environment(env_key, normalize_power=True)
        samples = fading.generate(100000)
        amp = np.abs(samples)
        ax.hist(amp, bins=100, density=True, histtype="step", linewidth=2.0, color=color, label=label)
        
    ax.set_xlabel("Channel Fading Amplitude $|h_{small}|$")
    ax.set_ylabel("Probability Density Function (PDF)")
    ax.set_title("Shadowed-Rician Fading Amplitude Distribution (Abdi et al.)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 2.5)
    save_fig(fig, output_dir, "channel_amplitude_distribution")

def plot_snr_verification(output_dir: str = "figures"):
    """Plot Target vs Measured SNR in AWGN channel."""
    from src.channels.awgn import AWGNChannel
    from src.utils.seed import set_seed
    set_seed(42)
    awgn = AWGNChannel()
    
    target_snrs = np.linspace(0.0, 30.0, 31)
    measured_snrs = []
    
    for snr in target_snrs:
        sig = np.random.normal(size=100000) + 1j * np.random.normal(size=100000)
        _, _, m_snr = awgn.add_noise(sig, snr_db=snr, return_noise=True)
        measured_snrs.append(m_snr)
        
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(target_snrs, measured_snrs, "o-", color="#1f77b4", label="Empirical Measured SNR")
    ax.plot(target_snrs, target_snrs, "k--", alpha=0.7, label="Ideal 1:1 Reference")
    ax.set_xlabel("Requested SNR (dB)")
    ax.set_ylabel("Measured SNR (dB)")
    ax.set_title("AWGN Noise Scaling Verification")
    ax.legend(loc="upper left")
    save_fig(fig, output_dir, "snr_verification")

def plot_classical_comparison(classical_results: Dict[str, Any], output_dir: str = "figures"):
    """Bar plot of classical baseline NMSE comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(classical_results.keys())
    nmses_db = [classical_results[m]["nmse_db"] for m in models]
    colors = ["#7f7f7f", "#bcbd22", "#1f77b4", "#ff7f0e"]
    
    bars = ax.bar(models, nmses_db, color=colors[:len(models)], width=0.5, edgecolor="black")
    ax.set_ylabel("NMSE (dB) [Lower is Better]")
    ax.set_title("Classical Baseline Models Comparison on Unseen LEO Channels")
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        offset = -0.5 if yval < 0 else 0.2
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f"{yval:.2f} dB", ha="center", va="bottom" if yval < 0 else "top", fontweight="bold")
        
    save_fig(fig, output_dir, "classical_model_nmse_comparison")

def plot_few_shot_curves(few_shot_data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]], output_dir: str = "figures"):
    """Plot Pilot count vs NMSE and Adaptation steps vs NMSE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Pilots vs NMSE at Step 5
    colors = {"Classical MAML": "#1f77b4", "QuantumMeta-Sat (4Q)": "#2ca02c", "QuantumMeta-Sat (8Q)": "#d62728"}
    markers = {"Classical MAML": "s-", "QuantumMeta-Sat (4Q)": "o-", "QuantumMeta-Sat (8Q)": "^-"}
    
    pilots = [1, 5, 10, 20]
    
    for m_name, p_dict in few_shot_data.items():
        c = colors.get(m_name, "#333333")
        m = markers.get(m_name, "o-")
        vals = []
        for k in pilots:
            k_key = f"K_{k}"
            if k_key in p_dict:
                vals.append(p_dict[k_key].get(5, p_dict[k_key].get(10, {})).get("nmse_db", 0.0))
        if len(vals) == len(pilots):
            ax1.plot(pilots, vals, m, color=c, label=m_name)
            
    ax1.set_xlabel("Pilot Observation Samples ($K$)")
    ax1.set_ylabel("NMSE (dB) [Lower is Better]")
    ax1.set_title("Few-Shot Adaptation vs Pilot Symbols ($K$)")
    ax1.set_xticks(pilots)
    ax1.legend()
    
    # 2. Adaptation steps vs NMSE at K=10
    steps = [0, 1, 3, 5, 10]
    for m_name, p_dict in few_shot_data.items():
        c = colors.get(m_name, "#333333")
        m = markers.get(m_name, "o-")
        if "K_10" in p_dict:
            step_vals = [p_dict["K_10"][s]["nmse_db"] for s in steps if s in p_dict["K_10"]]
            if len(step_vals) == len(steps):
                ax2.plot(steps, step_vals, m, color=c, label=m_name)
                
    ax2.set_xlabel("Gradient Adaptation Steps ($N_{inner}$)")
    ax2.set_ylabel("NMSE (dB) [Lower is Better]")
    ax2.set_title("Few-Shot Adaptation vs Gradient Steps ($K=10$ Pilots)")
    ax2.set_xticks(steps)
    ax2.legend()
    
    save_fig(fig, output_dir, "few_shot_adaptation_curves")

def plot_snr_and_ber_curves(snr_results: Dict[str, Dict[str, List[float]]], output_dir: str = "figures"):
    """Plot SNR vs NMSE and SNR vs BER across all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    snrs = [5, 10, 15, 20]
    palette = {
        "LS": ("#7f7f7f", "x--"),
        "MLP": ("#1f77b4", "s-"),
        "CNN": ("#ff7f0e", "d-"),
        "Classical MAML": ("#9467bd", "v-"),
        "Fixed VQC": ("#8c564b", "+-"),
        "QuantumMeta-Sat": ("#2ca02c", "o-"),
    }
    
    for m_name, data in snr_results.items():
        c, m = palette.get(m_name, ("#333333", "o-"))
        if "nmse_db" in data and len(data["nmse_db"]) == len(snrs):
            ax1.plot(snrs, data["nmse_db"], m, color=c, label=m_name)
        if "ber" in data and len(data["ber"]) == len(snrs):
            ax2.semilogy(snrs, np.maximum(data["ber"], 1e-5), m, color=c, label=m_name)
            
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("NMSE (dB) [Lower is Better]")
    ax1.set_title("Channel Estimation NMSE vs SNR")
    ax1.set_xticks(snrs)
    ax1.legend()
    
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Bit Error Rate (BER)")
    ax2.set_title("Equalized QPSK BER vs SNR")
    ax2.set_xticks(snrs)
    ax2.legend()
    
    save_fig(fig, output_dir, "snr_and_ber_performance")

def plot_elevation_curves(elevation_results: Dict[str, Dict[str, List[float]]], output_dir: str = "figures"):
    """Plot Elevation Angle vs NMSE across models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    elevations = [20, 30, 45, 60, 75, 90]
    
    palette = {
        "LS": ("#7f7f7f", "x--"),
        "MLP": ("#1f77b4", "s-"),
        "Classical MAML": ("#9467bd", "v-"),
        "QuantumMeta-Sat": ("#2ca02c", "o-"),
    }
    
    for m_name, data in elevation_results.items():
        c, m = palette.get(m_name, ("#333333", "o-"))
        if "nmse_db" in data and len(data["nmse_db"]) == len(elevations):
            ax.plot(elevations, data["nmse_db"], m, color=c, label=m_name)
            
    ax.axvspan(75, 90, color="red", alpha=0.1, label="Unseen Extrapolation Zone (Zenith)")
    ax.set_xlabel("Elevation Angle $\\theta$ (degrees)")
    ax.set_ylabel("NMSE (dB) [Lower is Better]")
    ax.set_title("Channel Estimation Accuracy Across Elevation Angles")
    ax.set_xticks(elevations)
    ax.legend(loc="upper right")
    save_fig(fig, output_dir, "elevation_vs_nmse")

def plot_parameter_efficiency(param_dict: Dict[str, Dict[str, Any]], output_dir: str = "figures"):
    """Plot Parameter count comparison and quantum parameter breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Total parameter comparison
    models = list(param_dict.keys())
    total_params = [param_dict[m].get("total_params", param_dict[m].get("parameters", 0)) for m in models]
    
    colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#2ca02c", "#d62728"]
    bars = ax1.bar(models, total_params, color=colors[:len(models)], edgecolor="black")
    ax1.set_yscale("log")
    ax1.set_ylabel("Total Trainable Parameters (Log Scale)")
    ax1.set_title("Parameter Count Comparison (Classical vs Quantum)")
    ax1.tick_params(axis="x", rotation=25)
    
    for bar in bars:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, y*1.1, f"{int(y):,}", ha="center", va="bottom", fontsize=10)
        
    # 2. Quantum Model Parameter Breakdown
    q_models = [m for m in models if "VQC" in m or "Quantum" in m or "Q-MAML" in m]
    if q_models:
        q_params = [param_dict[m].get("quantum_params", param_dict[m].get("breakdown", {}).get("quantum_parameters", 24)) for m in q_models]
        c_params = [param_dict[m].get("classical_params", param_dict[m].get("breakdown", {}).get("classical_parameters", 774)) for m in q_models]
        
        x = np.arange(len(q_models))
        ax2.bar(x, c_params, label="Classical Parameters", color="#1f77b4", edgecolor="black")
        ax2.bar(x, q_params, bottom=c_params, label="Quantum Parameters", color="#2ca02c", edgecolor="black")
        ax2.set_xticks(x)
        ax2.set_xticklabels(q_models, rotation=15)
        ax2.set_ylabel("Parameter Count")
        ax2.set_title("Hybrid Quantum vs Classical Parameter Breakdown")
        ax2.legend()
        
    save_fig(fig, output_dir, "parameter_count_efficiency")

def plot_ablation_study(ablation_data: Dict[str, Dict[str, Any]], output_dir: str = "figures"):
    """Bar plot comparing ablation variants."""
    fig, ax = plt.subplots(figsize=(9, 5))
    variants = list(ablation_data.keys())
    nmses_db = [ablation_data[v]["nmse_db"] for v in variants]
    
    bars = ax.barh(variants, nmses_db, color="#1f77b4", edgecolor="black", height=0.6)
    ax.set_xlabel("NMSE (dB) [Lower is Better]")
    ax.set_title("Ablation Study: Component Impact on Adaptive Channel Estimation")
    
    for bar in bars:
        x = bar.get_width()
        ax.text(x - 0.2, bar.get_y() + bar.get_height()/2.0, f"{x:.2f} dB", ha="right", va="center", color="white", fontweight="bold")
        
    save_fig(fig, output_dir, "ablation_study_comparison")
