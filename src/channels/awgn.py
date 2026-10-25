"""Additive White Gaussian Noise (AWGN) channel and SNR verification."""
import numpy as np
from typing import Tuple, Union

class AWGNChannel:
    """Complex Additive White Gaussian Noise (AWGN) channel generator.
    
    Implements:
        y = s + n
        P_s = E[|s|^2]
        P_n = P_s / (10^(SNR_dB / 10))
        n ~ CN(0, P_n) = N(0, P_n/2) + j * N(0, P_n/2)
    """
    
    def __init__(self):
        pass
        
    def add_noise(
        self,
        signal: np.ndarray,
        snr_db: float,
        return_noise: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, float]]:
        """Add complex AWGN noise to an input signal for a target SNR.
        
        Args:
            signal: Complex or real 1D NumPy array representing transmitted/received faded signal s.
            snr_db: Target Signal-to-Noise Ratio in decibels.
            return_noise: If True, also return noise vector and measured SNR.
            
        Returns:
            noisy_signal: Noisy received signal y = s + n.
            (Optional) noise: Generated noise vector n.
            (Optional) measured_snr_db: Empirically verified SNR in dB.
        """
        signal_arr = np.asarray(signal)
        num_samples = len(signal_arr)
        if num_samples == 0:
            raise ValueError("Input signal is empty.")
            
        # Empirical signal power P_s = (1/N) * sum(|s_i|^2)
        signal_power = np.mean(np.abs(signal_arr)**2)
        if signal_power == 0:
            raise ValueError("Signal power is zero.")
            
        snr_linear = 10.0**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Complex noise with variance noise_power (noise_power / 2 per real and imag component)
        sigma = np.sqrt(noise_power / 2.0)
        noise_real = np.random.normal(0.0, sigma, size=num_samples)
        noise_imag = np.random.normal(0.0, sigma, size=num_samples)
        noise = noise_real + 1j * noise_imag
        
        noisy_signal = signal_arr + noise
        
        if return_noise:
            empirical_noise_power = np.mean(np.abs(noise)**2)
            measured_snr_db = 10.0 * np.log10(signal_power / empirical_noise_power)
            return noisy_signal, noise, float(measured_snr_db)
            
        return noisy_signal
        
    @staticmethod
    def measure_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
        """Compute the empirical SNR in dB between a clean signal and noise."""
        p_signal = np.mean(np.abs(signal)**2)
        p_noise = np.mean(np.abs(noise)**2)
        if p_noise <= 0:
            return float('inf')
        return float(10.0 * np.log10(p_signal / p_noise))
