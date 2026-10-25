"""Classical non-parametric channel estimation baselines: Standard LS and Windowed Reference."""
import numpy as np
import torch
from typing import Tuple, Union

class LSEstimator:
    """Standard Least Squares (LS) single-symbol channel estimator:
    
        h_hat_LS = y / x
        
    Given QPSK constellation x with |x|^2 = 1, h_hat_LS = y * conj(x).
    """
    
    def estimate(
        self,
        x_symbols: np.ndarray,
        y_symbols: np.ndarray
    ) -> np.ndarray:
        """Estimate complex channel coefficients via LS.
        
        Args:
            x_symbols: Transmitted pilot symbols (complex).
            y_symbols: Received noisy symbols (complex).
            
        Returns:
            Complex channel estimates h_hat, shape (N,).
        """
        x = np.asarray(x_symbols, dtype=np.complex128)
        y = np.asarray(y_symbols, dtype=np.complex128)
        return y / x
        
    def estimate_from_features(self, X_windows: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Extract current symbol pair from window features and compute LS estimate."""
        if isinstance(X_windows, torch.Tensor):
            X_arr = X_windows.detach().cpu().numpy()
        else:
            X_arr = np.asarray(X_windows)
            
        # Last 4 features in window: [Re(x_t), Im(x_t), Re(y_t), Im(y_t)]
        re_x = X_arr[:, -4]
        im_x = X_arr[:, -3]
        re_y = X_arr[:, -2]
        im_y = X_arr[:, -1]
        
        x = re_x + 1j * im_x
        y = re_y + 1j * im_y
        
        h_hat_c = y / x
        return np.column_stack([np.real(h_hat_c), np.imag(h_hat_c)]).astype(np.float32)


class WindowedLSEstimator:
    """Windowed Moving-Average LS Reference Estimator.
    
    Averages individual LS estimates over the sliding observation window W to filter AWGN noise:
        h_hat_windowed = (1/W) * sum_{k=0}^{W-1} (y_{t-k} / x_{t-k})
    """
    
    def __init__(self, window_size: int = 9):
        self.window_size = window_size
        
    def estimate_from_features(self, X_windows: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Compute window-averaged LS estimate across the W past observation symbols."""
        if isinstance(X_windows, torch.Tensor):
            X_arr = X_windows.detach().cpu().numpy()
        else:
            X_arr = np.asarray(X_windows)
            
        N = len(X_arr)
        W = self.window_size
        
        h_estimates = np.zeros((N, 2), dtype=np.float32)
        
        for i in range(N):
            window_flat = X_arr[i]  # shape (4*W,)
            window_symbols = window_flat.reshape(W, 4)
            
            re_x = window_symbols[:, 0]
            im_x = window_symbols[:, 1]
            re_y = window_symbols[:, 2]
            im_y = window_symbols[:, 3]
            
            x_w = re_x + 1j * im_x
            y_w = re_y + 1j * im_y
            
            ls_w = y_w / x_w  # shape (W,)
            avg_ls = np.mean(ls_w)
            
            h_estimates[i, 0] = float(np.real(avg_ls))
            h_estimates[i, 1] = float(np.imag(avg_ls))
            
        return h_estimates
