"""Classical 1D Convolutional Neural Network (CNN) channel estimator."""
import torch
import torch.nn as nn
from typing import List, Optional

class ClassicalCNN(nn.Module):
    """Classical 1D CNN channel estimator.
    
    Processes the (4, W) sequential channel features via temporal 1D convolutions.
    
    Architecture:
        Input (batch, 4, W)
        -> Conv1d(4 -> 32, kernel=3, padding=1) -> ReLU
        -> Conv1d(32 -> 64, kernel=3, padding=1) -> ReLU
        -> Flatten
        -> Linear(64 * W -> 64) -> ReLU
        -> Linear(64 -> 2) -> [Re(h), Im(h)]
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        window_size: int = 9,
        conv_channels: Optional[List[int]] = None,
        fc_hidden: int = 64,
        output_dim: int = 2,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]
            
        self.input_channels = input_channels
        self.window_size = window_size
        
        self.conv1 = nn.Conv1d(input_channels, conv_channels[0], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        flattened_size = conv_channels[1] * window_size
        self.fc1 = nn.Linear(flattened_size, fc_hidden)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, output_dim)
        
        self.num_qubits = 0
        self.circuit_depth = 0
        self.num_quantum_params = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Supports both:
            - Flat input (batch, 36) -> reshaped automatically to (batch, 4, 9)
            - 3D input (batch, 4, 9)
        """
        if x.ndim == 2:
            batch_size = x.shape[0]
            # Reshape (batch, 9, 4) then permute to (batch, 4, 9)
            x_reshaped = x.view(batch_size, self.window_size, self.input_channels).permute(0, 2, 1)
        else:
            x_reshaped = x
            
        h = self.relu1(self.conv1(x_reshaped))
        h = self.relu2(self.conv2(h))
        h = h.reshape(h.size(0), -1)
        h = self.relu3(self.fc1(h))
        out = self.fc2(h)
        return out
