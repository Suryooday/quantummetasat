"""Normalized QPSK (Quadrature Phase Shift Keying) modulator and demodulator."""
import numpy as np
from typing import Tuple, Union

class QPSKModulator:
    """Normalized Gray-coded QPSK Modulator / Demodulator.
    
    Constellation symbols:
        x in { (+1 + 1j)/sqrt(2), (+1 - 1j)/sqrt(2), (-1 + 1j)/sqrt(2), (-1 - 1j)/sqrt(2) }
        Normalized average symbol energy: E[|x|^2] = 1.0
    """
    
    CONSTELLATION = np.array([
        (1.0 + 1.0j) / np.sqrt(2.0),   # 00 -> (+1, +1)
        (1.0 - 1.0j) / np.sqrt(2.0),   # 01 -> (+1, -1)
        (-1.0 + 1.0j) / np.sqrt(2.0),  # 10 -> (-1, +1)
        (-1.0 - 1.0j) / np.sqrt(2.0),  # 11 -> (-1, -1)
    ], dtype=np.complex128)
    
    # Bit mapping: (b0, b1) -> index
    # 00: 0, 01: 1, 10: 2, 11: 3
    
    def __init__(self):
        pass
        
    def generate_symbols(self, num_symbols: int) -> np.ndarray:
        """Generate random normalized QPSK complex symbols.
        
        Args:
            num_symbols: Number of symbols to generate.
            
        Returns:
            1D complex NumPy array of QPSK symbols with E[|x|^2] = 1.0.
        """
        indices = np.random.randint(0, 4, size=num_symbols)
        return self.CONSTELLATION[indices]
        
    def modulate_bits(self, bit_array: np.ndarray) -> np.ndarray:
        """Modulate binary stream into QPSK symbols using Gray mapping.
        
        Args:
            bit_array: 1D array of bits (0 or 1), length must be even.
            
        Returns:
            1D complex NumPy array of modulated symbols.
        """
        bits = np.asarray(bit_array, dtype=np.int32)
        if len(bits) % 2 != 0:
            raise ValueError("Bit stream length must be a multiple of 2 for QPSK.")
            
        b0 = bits[0::2]
        b1 = bits[1::2]
        
        # Real part: b0=0 -> +1/sqrt(2), b0=1 -> -1/sqrt(2)
        real_part = (1.0 - 2.0 * b0) / np.sqrt(2.0)
        # Imag part: b1=0 -> +1/sqrt(2), b1=1 -> -1/sqrt(2)
        imag_part = (1.0 - 2.0 * b1) / np.sqrt(2.0)
        
        return real_part + 1j * imag_part
        
    def demodulate_to_symbols(self, received_equalized: np.ndarray) -> np.ndarray:
        """Hard-decision slicer mapping equalized complex values to the nearest QPSK constellation point.
        
        Args:
            received_equalized: Equalized complex samples.
            
        Returns:
            Hard-decision sliced QPSK symbols from the constellation.
        """
        rx = np.asarray(received_equalized)
        sliced_real = np.where(np.real(rx) >= 0.0, 1.0, -1.0) / np.sqrt(2.0)
        sliced_imag = np.where(np.imag(rx) >= 0.0, 1.0, -1.0) / np.sqrt(2.0)
        return sliced_real + 1j * sliced_imag
        
    def demodulate_to_bits(self, received_equalized: np.ndarray) -> np.ndarray:
        """Demodulate equalized complex symbols into binary bit stream.
        
        Args:
            received_equalized: Equalized complex samples.
            
        Returns:
            1D array of recovered bits (0 or 1).
        """
        rx = np.asarray(received_equalized)
        num_symbols = len(rx)
        bits = np.zeros(num_symbols * 2, dtype=np.int32)
        
        # Real >= 0 -> bit 0 = 0, else 1
        bits[0::2] = np.where(np.real(rx) >= 0.0, 0, 1)
        # Imag >= 0 -> bit 1 = 0, else 1
        bits[1::2] = np.where(np.imag(rx) >= 0.0, 0, 1)
        
        return bits
        
    def compute_ser(self, tx_symbols: np.ndarray, rx_symbols_equalized: np.ndarray) -> float:
        """Compute Symbol Error Rate (SER)."""
        decided_symbols = self.demodulate_to_symbols(rx_symbols_equalized)
        # Symbols match if both real and imag parts have the same signs
        errors = np.sum((np.sign(np.real(tx_symbols)) != np.sign(np.real(decided_symbols))) |
                        (np.sign(np.imag(tx_symbols)) != np.sign(np.imag(decided_symbols))))
        return float(errors / len(tx_symbols))
        
    def compute_ber(self, tx_symbols: np.ndarray, rx_symbols_equalized: np.ndarray) -> float:
        """Compute Bit Error Rate (BER) directly from transmitted and equalized symbols."""
        tx_bits = self.demodulate_to_bits(tx_symbols)
        rx_bits = self.demodulate_to_bits(rx_symbols_equalized)
        bit_errors = np.sum(tx_bits != rx_bits)
        return float(bit_errors / len(tx_bits))
