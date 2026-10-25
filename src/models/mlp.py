"""Classical Multi-Layer Perceptron (MLP) channel estimator."""
import torch
import torch.nn as nn
from typing import List, Optional

class ClassicalMLP(nn.Module):
    """Classical MLP channel estimator.
    
    Architecture:
        Input (36) -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(2) -> [Re(h), Im(h)]
    """
    
    def __init__(
        self,
        input_dim: int = 36,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
            
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_d = h_d
            
        layers.append(nn.Linear(in_d, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.num_qubits = 0
        self.circuit_depth = 0
        self.num_quantum_params = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 36).
            
        Returns:
            Output predictions of shape (batch_size, 2).
        """
        return self.network(x)
