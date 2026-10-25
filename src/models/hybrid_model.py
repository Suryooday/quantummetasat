"""Hybrid Quantum-Classical Neural Network using Vectorized PennyLane and PyTorch."""
import math
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import pennylane as qml

class HybridVQC(nn.Module):
    """Vectorized Hybrid Quantum-Classical Neural Network with Data Re-uploading for LEO Channel Estimation.
    
    Architecture Pipeline:
        1. Classical Encoder: R^36 -> R^num_qubits (bounded in [-pi, pi] via tanh * pi).
        2. Angle Encoding: Encodes features as rotations on num_qubits.
        3. Variational Circuit Layers:
           - Parameterized Ry(theta) and Rz(theta) rotations
           - Entanglement layers (CNOT circular/linear topology)
           - Data re-uploading at each variational layer
        4. Expectation Measurements: Pauli-Z expectation values on all qubits.
        5. Classical Decoder: R^num_qubits -> R^2 -> [Re(h), Im(h)].
        
    Attributes:
        num_qubits (int): Number of qubits (default: 4).
        n_layers (int): Number of variational layers with data re-uploading.
        entanglement (str): Entanglement topology ('circular', 'linear', 'none').
    """
    
    def __init__(
        self,
        input_dim: int = 36,
        num_qubits: int = 4,
        n_layers: int = 3,
        hidden_encoder_dim: int = 16,
        output_dim: int = 2,
        entanglement: str = "circular",
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.output_dim = output_dim
        
        # 1. Classical Feature Encoder (reduces 36 inputs to num_qubits angles in [-pi, pi])
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_encoder_dim),
            nn.Tanh(),
            nn.Linear(hidden_encoder_dim, num_qubits),
            nn.Tanh()
        )
        
        # 2. Quantum Parameters: Ry and Rz angles per qubit per layer
        # Shape: (n_layers, num_qubits, 2)
        self.quantum_weights = nn.Parameter(
            (torch.rand(n_layers, num_qubits, 2) * 2.0 - 1.0) * math.pi
        )
        
        # Setup PennyLane Quantum Device and vectorized QNode
        self.dev = qml.device(device_name, wires=num_qubits)
        
        def _circuit(inputs, weights):
            # inputs shape: (batch_size, num_qubits)
            # weights shape: (n_layers, num_qubits, 2)
            for layer in range(self.n_layers):
                # Data re-uploading: encode input features into Ry rotations
                for q in range(self.num_qubits):
                    qml.RY(inputs[:, q], wires=q)
                    
                # Trainable variational single-qubit rotations (Ry and Rz)
                for q in range(self.num_qubits):
                    qml.RY(weights[layer, q, 0], wires=q)
                    qml.RZ(weights[layer, q, 1], wires=q)
                    
                # Entangling gates
                if self.entanglement == "circular" and self.num_qubits > 1:
                    for q in range(self.num_qubits):
                        qml.CNOT(wires=[q, (q + 1) % self.num_qubits])
                elif self.entanglement == "linear" and self.num_qubits > 1:
                    for q in range(self.num_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                # If entanglement == "none", no CNOT gates applied (for ablation)
                
            return [qml.expval(qml.PauliZ(q)) for q in range(self.num_qubits)]
            
        self.qnode = qml.QNode(_circuit, self.dev, interface="torch", diff_method="backprop")
        
        # 3. Classical Output Decoder: maps Pauli-Z expectation values to [Re(h), Im(h)]
        self.decoder = nn.Sequential(
            nn.Linear(num_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
        # Profiling statistics
        self.num_quantum_params = self.quantum_weights.numel()
        self.circuit_depth = self._calculate_circuit_depth()
        
    def _calculate_circuit_depth(self) -> int:
        """Estimate the quantum circuit depth."""
        gates_per_layer = 3 + (1 if self.entanglement != "none" else 0)
        return self.n_layers * gates_per_layer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classical encoder -> VQC -> classical decoder.
        
        Args:
            x: Input tensor of shape (batch_size, 36).
            
        Returns:
            Output channel estimates [Re(h), Im(h)] of shape (batch_size, 2).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        # 1. Classical Encoding -> scaled to [-pi, pi]
        encoded_angles = self.encoder(x) * math.pi  # shape (batch_size, num_qubits)
        
        # 2. Vectorized Quantum Circuit execution
        exp_vals = self.qnode(encoded_angles, self.quantum_weights)
        # exp_vals is a list of tensors of length num_qubits, each of shape (batch_size,)
        q_tensor = torch.stack(exp_vals, dim=1).to(dtype=x.dtype, device=x.device)  # shape (batch_size, num_qubits)
        
        # 3. Classical Decoding
        out = self.decoder(q_tensor)  # shape (batch_size, 2)
        return out
        
    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """Compute parameter efficiency breakdown."""
        total_p = sum(p.numel() for p in self.parameters())
        q_p = self.quantum_weights.numel()
        c_p = total_p - q_p
        return {
            "total_parameters": int(total_p),
            "quantum_parameters": int(q_p),
            "classical_parameters": int(c_p),
            "quantum_ratio": float(q_p / total_p) if total_p > 0 else 0.0,
            "num_qubits": self.num_qubits,
            "circuit_depth": self.circuit_depth,
            "n_layers": self.n_layers,
            "entanglement": self.entanglement,
        }
