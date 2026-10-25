"""BER and SER evaluation via zero-forcing channel equalization and QPSK detection."""
import numpy as np
from typing import Dict, Tuple, Union
from src.modulation.qpsk import QPSKModulator

class BERCalculator:
    """Calculates BER and SER by equalizing received symbols with estimated channel coefficients.
    
    Zero-Forcing equalization:
        x_hat = y / h_hat
    """
    
    def __init__(self):
        self.modulator = QPSKModulator()
        
    def equalize_and_decode(
        self,
        y_symbols: np.ndarray,
        h_estimated: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        tx_symbols: np.ndarray,
        eps: float = 1e-8,
    ) -> Dict[str, float]:
        """Equalize received symbols using estimated channel and compute BER and SER.
        
        Args:
            y_symbols: Received noisy complex symbols, shape (N,).
            h_estimated: Estimated channel coefficients, either complex array (N,) or real (N, 2).
            tx_symbols: Transmitted ground-truth QPSK symbols, shape (N,).
            eps: Small regularization term to prevent division by near-zero channel estimates.
            
        Returns:
            Dictionary with 'ber', 'ser', and 'evm_percent'.
        """
        if isinstance(h_estimated, np.ndarray) and h_estimated.ndim == 2 and h_estimated.shape[1] == 2:
            h_c = h_estimated[:, 0] + 1j * h_estimated[:, 1]
        else:
            h_c = np.asarray(h_estimated, dtype=np.complex128)
            
        # Regularized zero-forcing equalization: x_hat = y * conj(h) / (|h|^2 + eps)
        h_power = np.abs(h_c)**2
        equalized_x = (y_symbols * np.conj(h_c)) / (h_power + eps)
        
        ber = self.modulator.compute_ber(tx_symbols, equalized_x)
        ser = self.modulator.compute_ser(tx_symbols, equalized_x)
        
        # Error Vector Magnitude (EVM)
        evm = np.sqrt(np.mean(np.abs(equalized_x - tx_symbols)**2) / np.mean(np.abs(tx_symbols)**2)) * 100.0
        
        return {
            "ber": float(ber),
            "ser": float(ser),
            "evm_percent": float(evm),
        }
