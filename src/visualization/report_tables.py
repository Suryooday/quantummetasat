"""Report and Markdown/LaTeX table generation utilities for experimental benchmarking."""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class ReportTableGenerator:
    """Generates standardized publication tables for QuantumMeta-Sat."""
    
    @staticmethod
    def generate_master_table(
        results_dict: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate Master Comparison Table across all evaluated models."""
        rows = []
        
        # Standard model ordering focusing on 4-qubit Hybrid VQC as primary proposed model
        model_keys = [
            ("LS", "Least Squares (LS)"),
            ("Windowed_Reference", "Windowed LS Reference"),
            ("MLP", "Classical Supervised MLP"),
            ("CNN", "Classical Supervised CNN"),
            ("Classical_MAML", "Classical MAML (MLP)"),
            ("Fixed_VQC", "Proposed Fixed Hybrid VQC (4 Qubits)"),
        ]
        
        for key, display_name in model_keys:
            data = results_dict.get(key, {})
            if not data:
                continue
                
            total_p = data.get("total_parameters", data.get("total_params", data.get("parameters", data.get("breakdown", {}).get("total_parameters", 0))))
            qubits = data.get("qubit_count", data.get("num_qubits", data.get("breakdown", {}).get("num_qubits", 0)))
            q_params = data.get("quantum_parameters", data.get("quantum_params", data.get("breakdown", {}).get("quantum_parameters", 0)))
            
            nmse = data.get("nmse", 0.0)
            nmse_db = data.get("nmse_db", 0.0)
            ber = data.get("ber", 0.0)
            lat = data.get("per_sample_latency_ms", data.get("inference_latency_ms", 0.0))
            adapt_steps = data.get("adaptation_steps", "—" if "MAML" not in display_name else "5–10")
            
            p_str = f"{total_p:,}" if total_p > 0 else "—"
            q_str = str(qubits) if qubits > 0 else "—"
            qp_str = str(q_params) if q_params > 0 else "—"
            
            rows.append({
                "Model": display_name,
                "Params": p_str,
                "Qubits": q_str,
                "Quantum Params": qp_str,
                "NMSE": f"{nmse:.4f}",
                "NMSE (dB)": f"{nmse_db:.2f}",
                "BER": f"{ber:.4e}",
                "Inference Latency (ms)": f"{lat:.4f}",
                "Adaptation Steps": str(adapt_steps),
            })
            
        df = pd.DataFrame(rows)
        
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path.replace(".md", ".csv"), index=False)
            with open(output_path, "w") as f:
                f.write(df.to_markdown(index=False))
                
        return df

    @staticmethod
    def generate_failed_exploratory_table(
        results_dict: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate table for exploratory/failed extensions (8-qubit scaling & quantum meta-learning)."""
        failed_keys = [
            ("Q_MAML_4Q", "Q-MAML (4 Qubits)", "Failed (Meta-gradient instability through quantum circuit; BER 11.2%)"),
            ("QuantumMeta_Sat_8Q", "QuantumMeta-Sat (8 Qubits)", "Failed (Barren plateaus & dimensional collapse; BER 41.8% ~ random)"),
        ]
        rows = []
        for key, display_name, failure_reason in failed_keys:
            data = results_dict.get(key, {})
            if not data:
                continue
            total_p = data.get("total_parameters", data.get("total_params", data.get("parameters", data.get("breakdown", {}).get("total_parameters", 0))))
            qubits = data.get("qubit_count", data.get("num_qubits", data.get("breakdown", {}).get("num_qubits", 0)))
            q_params = data.get("quantum_parameters", data.get("quantum_params", data.get("breakdown", {}).get("quantum_parameters", 0)))
            nmse = data.get("nmse", 0.0)
            nmse_db = data.get("nmse_db", 0.0)
            ber = data.get("ber", 0.0)
            lat = data.get("per_sample_latency_ms", data.get("inference_latency_ms", 0.0))
            
            rows.append({
                "Exploratory Variant": display_name,
                "Qubits": str(qubits) if qubits > 0 else "—",
                "Total Params": f"{total_p:,}",
                "Quantum Params": str(q_params),
                "NMSE": f"{nmse:.4f}",
                "NMSE (dB)": f"{nmse_db:.2f}",
                "BER": f"{ber:.4e}",
                "Latency (ms)": f"{lat:.4f}",
                "Experimental Outcome & Diagnosis": failure_reason,
            })
            
        df = pd.DataFrame(rows)
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path.replace(".md", ".csv"), index=False)
            with open(output_path, "w") as f:
                f.write(df.to_markdown(index=False))
        return df
        
    @staticmethod
    def generate_few_shot_table(
        few_shot_dict: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate Few-Shot Pilot Samples vs NMSE (dB) Table across adaptation steps."""
        rows = []
        
        for model_name, pilot_data in few_shot_dict.items():
            for pilot_key, step_data in pilot_data.items():
                k_val = pilot_key.replace("K_", "")
                row = {
                    "Model": model_name,
                    "Pilots (K)": k_val,
                    "Step 0 (dB)": f"{step_data.get(0, {}).get('nmse_db', 0.0):.2f}",
                    "Step 1 (dB)": f"{step_data.get(1, {}).get('nmse_db', 0.0):.2f}",
                    "Step 3 (dB)": f"{step_data.get(3, {}).get('nmse_db', 0.0):.2f}",
                    "Step 5 (dB)": f"{step_data.get(5, {}).get('nmse_db', 0.0):.2f}",
                    "Step 10 (dB)": f"{step_data.get(10, {}).get('nmse_db', 0.0):.2f}",
                }
                rows.append(row)
                
        df = pd.DataFrame(rows)
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(df.to_markdown(index=False))
        return df
        
    @staticmethod
    def generate_ablation_table(
        ablation_dict: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate Ablation Study Table."""
        rows = []
        for variant_name, data in ablation_dict.items():
            rows.append({
                "Architecture Variant": variant_name,
                "Qubits": data.get("qubits", "—"),
                "Total Params": f"{data.get('total_params', 0):,}",
                "Quantum Params": str(data.get("quantum_params", "—")),
                "NMSE": f"{data.get('nmse', 0.0):.4f}",
                "NMSE (dB)": f"{data.get('nmse_db', 0.0):.2f}",
                "BER": f"{data.get('ber', 0.0):.4e}",
                "Scientific Rationale": data.get("rationale", ""),
            })
        df = pd.DataFrame(rows)
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(df.to_markdown(index=False))
        return df
