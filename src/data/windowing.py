"""Sliding window transformation for sequential channel observation features."""
import numpy as np
import torch
from typing import Tuple, Union

class ChannelWindowing:
    """Transforms raw transmitted/received symbol sequences into sliding observation windows.
    
    For a window size W (default W=9):
    Input at step i:
        [Re(x_{i-W+1}), Im(x_{i-W+1}), Re(y_{i-W+1}), Im(y_{i-W+1}), ..., Re(x_i), Im(x_i), Re(y_i), Im(y_i)]
        Flat dimension: W * 4 = 36 features.
        CNN 2D shape: (4, W) channels x time.
    Target at step i:
        [Re(h_{eff, i}), Im(h_{eff, i})] (2 features).
    """
    
    def __init__(self, window_size: int = 9):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.window_size = int(window_size)
        self.features_per_symbol = 4
        self.flat_feature_dim = self.window_size * self.features_per_symbol
        self.target_dim = 2
        
    def create_windows(
        self,
        x_symbols: np.ndarray,
        y_symbols: np.ndarray,
        h_channel: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create flattened sliding window features and target channel coefficients.
        
        Args:
            x_symbols: Transmitted complex symbols, shape (N,).
            y_symbols: Received complex noisy symbols, shape (N,).
            h_channel: True complex channel coefficients, shape (N,).
            
        Returns:
            X_windows: 2D array of shape (N - W + 1, W * 4).
            Y_targets: 2D array of shape (N - W + 1, 2) representing [Re(h), Im(h)].
        """
        N = len(x_symbols)
        W = self.window_size
        if N < W:
            raise ValueError(f"Sequence length N={N} must be >= window size W={W}.")
            
        num_windows = N - W + 1
        X_windows = np.zeros((num_windows, self.flat_feature_dim), dtype=np.float32)
        Y_targets = np.zeros((num_windows, self.target_dim), dtype=np.float32)
        
        # Raw symbol feature matrix: shape (N, 4) -> [Re(x), Im(x), Re(y), Im(y)]
        raw_features = np.column_stack([
            np.real(x_symbols),
            np.imag(x_symbols),
            np.real(y_symbols),
            np.imag(y_symbols),
        ]).astype(np.float32)
        
        for idx in range(num_windows):
            # Window slice of W consecutive symbols from idx to idx + W
            window_slice = raw_features[idx : idx + W, :]  # shape (W, 4)
            X_windows[idx, :] = window_slice.flatten()     # shape (4*W,)
            
            # Target is the channel corresponding to the current (last) symbol in the window
            target_h = h_channel[idx + W - 1]
            Y_targets[idx, :] = [np.real(target_h), np.imag(target_h)]
            
        return X_windows, Y_targets
        
    def to_cnn_format(self, X_flat: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Reshape flat (N, 36) features to CNN format (N, 4, 9)."""
        if isinstance(X_flat, torch.Tensor):
            N = X_flat.shape[0]
            # Reshape (N, W, 4) then transpose to (N, 4, W)
            return X_flat.view(N, self.window_size, self.features_per_symbol).permute(0, 2, 1)
        else:
            N = len(X_flat)
            reshaped = X_flat.reshape(N, self.window_size, self.features_per_symbol)
            return np.transpose(reshaped, (0, 2, 1))
