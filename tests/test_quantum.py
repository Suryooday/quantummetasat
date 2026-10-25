"""Unit tests for Quantum Circuit, QNode execution, parameter counting, and differentiability."""
import pytest
import torch
import numpy as np

from src.models.hybrid_model import HybridVQC
from src.evaluation.profiler import ModelProfiler

def test_quantum_circuit_qubit_scalability():
    """Verify hybrid model initialization for 4-qubit and 8-qubit circuits."""
    for n_qubits in [4, 8]:
        model = HybridVQC(input_dim=36, num_qubits=n_qubits, n_layers=2, hidden_encoder_dim=16, output_dim=2)
        breakdown = model.get_parameter_breakdown()
        
        assert breakdown["num_qubits"] == n_qubits
        assert breakdown["quantum_parameters"] == 2 * n_qubits * 2  # n_layers * n_qubits * 2 (Ry, Rz)
        assert breakdown["classical_parameters"] > 0
        assert breakdown["circuit_depth"] > 0
        
        x = torch.randn(2, 36)
        out = model(x)
        assert out.shape == (2, 2)
        assert torch.all(torch.isfinite(out))

def test_quantum_data_reuploading_and_entanglement():
    """Verify circuit with circular, linear, and no entanglement."""
    for ent in ["circular", "linear", "none"]:
        model = HybridVQC(input_dim=36, num_qubits=4, n_layers=2, entanglement=ent)
        x = torch.randn(3, 36)
        out = model(x)
        assert out.shape == (3, 2)
        
        # Test backpropagation to both classical encoder, decoder, and quantum parameters
        loss = out.pow(2).sum()
        loss.backward()
        assert model.quantum_weights.grad is not None
        assert model.encoder[0].weight.grad is not None
        assert model.decoder[0].weight.grad is not None

def test_quantum_parameter_efficiency_ratio():
    """Verify quantum/classical parameter breakdown calculation."""
    model = HybridVQC(input_dim=36, num_qubits=4, n_layers=3, hidden_encoder_dim=16, output_dim=2)
    stats = model.get_parameter_breakdown()
    
    total = stats["total_parameters"]
    q_p = stats["quantum_parameters"]
    c_p = stats["classical_parameters"]
    
    assert total == q_p + c_p
    assert 0.0 < stats["quantum_ratio"] < 1.0
    # Parameter efficiency: total parameters must be significantly lower than classical MLP (~21k)
    assert total < 2000, f"Hybrid VQC parameter count ({total}) should be compact."
