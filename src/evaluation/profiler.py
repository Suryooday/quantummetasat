"""Computational profiling utilities: parameter counts, circuit depth, inference and adaptation latency."""
import time
import torch
import numpy as np
from typing import Dict, Any, Optional

class ModelProfiler:
    """Measures computational resources and latency for classical and quantum models."""
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """Count trainable, non-trainable, and total parameters of a PyTorch module."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total = trainable + non_trainable
        
        # Check if hybrid quantum model
        quantum_params = getattr(model, "num_quantum_params", 0)
        classical_params = total - quantum_params
        
        return {
            "total_params": int(total),
            "trainable_params": int(trainable),
            "classical_params": int(classical_params),
            "quantum_params": int(quantum_params),
            "qubit_count": int(getattr(model, "num_qubits", 0)),
            "circuit_depth": int(getattr(model, "circuit_depth", 0)),
        }
        
    @staticmethod
    def measure_inference_latency(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        num_warmup: int = 10,
        num_trials: int = 100,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """Measure average single-sample and batch inference latency in milliseconds."""
        model.eval()
        model.to(device)
        x = sample_input.to(device)
        
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)
                
            start_time = time.perf_counter()
            for _ in range(num_trials):
                _ = model(x)
            end_time = time.perf_counter()
            
        total_time_ms = (end_time - start_time) * 1000.0
        avg_batch_ms = total_time_ms / num_trials
        batch_size = sample_input.shape[0] if sample_input.ndim > 1 else 1
        per_sample_ms = avg_batch_ms / batch_size
        
        return {
            "batch_latency_ms": float(avg_batch_ms),
            "per_sample_latency_ms": float(per_sample_ms),
        }
