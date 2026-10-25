"""Unit tests for MAML inner-loop adaptation and outer-loop meta-learning."""
import pytest
import torch
import numpy as np

from src.models.mlp import ClassicalMLP
from src.meta_learning.maml import MAML
from src.data.dataset import TaskData

def test_maml_support_query_isolation():
    """Verify that sampled support and query sets are strictly disjoint."""
    N = 100
    X = np.arange(N * 36).reshape(N, 36)
    Y = np.zeros((N, 2))
    task = TaskData("test_task", X, Y)
    
    X_s, Y_s, X_q, Y_q = task.sample_support_query(k_support=10, q_query=30, shuffle=True)
    
    assert X_s.shape == (10, 36)
    assert X_q.shape == (30, 36)
    
    # Check disjoint row sets
    s_rows = set(tuple(row.tolist()) for row in X_s)
    q_rows = set(tuple(row.tolist()) for row in X_q)
    assert len(s_rows.intersection(q_rows)) == 0

def test_maml_inner_adaptation():
    """Verify that inner-loop adaptation updates parameters and reduces support loss."""
    mlp = ClassicalMLP(input_dim=36, hidden_dims=[32, 32], output_dim=2)
    maml = MAML(mlp, inner_lr=0.05, outer_lr=0.001)
    
    # Generate synthetic support batch
    X_s = torch.randn(10, 36)
    Y_s = torch.randn(10, 2)
    
    # Pre-adaptation loss
    pred_0 = maml.forward_with_params(X_s)
    loss_0 = maml.criterion(pred_0, Y_s).item()
    
    # 5 inner adaptation steps
    adapted_params = maml.adapt(X_s, Y_s, num_steps=5, create_graph=False)
    
    # Post-adaptation loss
    pred_5 = maml.forward_with_params(X_s, adapted_params)
    loss_5 = maml.criterion(pred_5, Y_s).item()
    
    # Parameter update must reduce support loss
    assert loss_5 < loss_0, f"Expected loss_5 ({loss_5}) < loss_0 ({loss_0})"

def test_maml_meta_step():
    """Verify outer-loop meta-gradient update modifies base model weights."""
    mlp = ClassicalMLP(input_dim=36, hidden_dims=[32, 32], output_dim=2)
    maml = MAML(mlp, inner_lr=0.01, outer_lr=0.01)
    
    initial_weights = [p.clone() for p in mlp.parameters()]
    
    # Create fake meta-batch with 2 tasks
    meta_batch = [
        (torch.randn(5, 36), torch.randn(5, 2), torch.randn(10, 36), torch.randn(10, 2), {}),
        (torch.randn(5, 36), torch.randn(5, 2), torch.randn(10, 36), torch.randn(10, 2), {}),
    ]
    
    loss_val = maml.meta_step(meta_batch, inner_steps=2)
    assert isinstance(loss_val, float)
    assert loss_val > 0.0
    
    # Check parameters were updated
    changed = False
    for init_p, current_p in zip(initial_weights, mlp.parameters()):
        if not torch.allclose(init_p, current_p):
            changed = True
            break
    assert changed, "Meta-step failed to update model parameters."
