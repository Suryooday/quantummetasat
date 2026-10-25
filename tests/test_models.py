"""Unit tests for classical and quantum model architectures."""
import pytest
import torch
import numpy as np

from src.models.baselines import LSEstimator, WindowedLSEstimator
from src.models.mlp import ClassicalMLP
from src.models.cnn import ClassicalCNN
from src.models.hybrid_model import HybridVQC
from src.evaluation.profiler import ModelProfiler

def test_baseline_estimators():
    """Verify standard and windowed LS estimators."""
    ls = LSEstimator()
    w_ls = WindowedLSEstimator(window_size=9)
    
    # Feature shape (10, 36)
    x_features = np.random.normal(size=(10, 36)).astype(np.float32)
    
    ls_out = ls.estimate_from_features(x_features)
    assert ls_out.shape == (10, 2)
    assert np.all(np.isfinite(ls_out))
    
    w_ls_out = w_ls.estimate_from_features(x_features)
    assert w_ls_out.shape == (10, 2)
    assert np.all(np.isfinite(w_ls_out))

def test_classical_mlp_forward_and_grad():
    """Verify Classical MLP shapes and backprop."""
    mlp = ClassicalMLP(input_dim=36, hidden_dims=[128, 128], output_dim=2)
    prof = ModelProfiler.count_parameters(mlp)
    
    # Expected params: (36*128 + 128) + (128*128 + 128) + (128*2 + 2) = 4736 + 16512 + 258 = 21506
    assert prof["total_params"] == 21506
    
    x = torch.randn(8, 36, requires_grad=True)
    out = mlp(x)
    assert out.shape == (8, 2)
    
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_classical_cnn_forward_and_grad():
    """Verify Classical CNN shapes and backprop."""
    cnn = ClassicalCNN(input_channels=4, window_size=9, conv_channels=[32, 64], fc_hidden=64, output_dim=2)
    
    # Test flat input (8, 36)
    x_flat = torch.randn(8, 36, requires_grad=True)
    out = cnn(x_flat)
    assert out.shape == (8, 2)
    
    loss = out.sum()
    loss.backward()
    assert x_flat.grad is not None

def test_hybrid_vqc_forward_and_grad():
    """Verify Hybrid VQC forward execution and parameter gradients."""
    vqc = HybridVQC(input_dim=36, num_qubits=4, n_layers=2, hidden_encoder_dim=16, output_dim=2)
    breakdown = vqc.get_parameter_breakdown()
    
    assert breakdown["num_qubits"] == 4
    # Quantum parameters: 2 layers * 4 qubits * 2 rotations = 16
    assert breakdown["quantum_parameters"] == 16
    assert breakdown["total_parameters"] > 0
    
    x = torch.randn(4, 36)
    out = vqc(x)
    assert out.shape == (4, 2)
    
    loss = out.sum()
    loss.backward()
    assert vqc.quantum_weights.grad is not None
    assert torch.all(torch.isfinite(vqc.quantum_weights.grad))
