"""Doppler shift and phase evolution modeling for LEO satellite dynamics."""
import numpy as np
from typing import Optional, Union

class DopplerModel:
    """LEO Doppler shift and time-domain phase evolution model.
    
    Distinguishes:
        1. Doppler frequency shift f_D (Hz).
        2. Doppler-induced discrete-time phase evolution phi_D[k] = 2 * pi * f_D * k * T_s + phi_0.
        
    Attributes:
        carrier_freq_hz (float): Carrier frequency in Hz.
        sampling_rate_hz (float): System symbol/sampling rate in Hz (1/T_s).
        orbital_velocity_m_s (float): Satellite orbital speed in m/s (~7.56 km/s for ~600 km LEO).
        speed_of_light (float): Speed of light in m/s.
    """
    
    def __init__(
        self,
        carrier_freq_hz: float = 2.0e9,
        sampling_rate_hz: float = 1.0e6,
        orbital_velocity_m_s: float = 7560.0,
        speed_of_light: float = 299792458.0,
    ):
        self.carrier_freq_hz = float(carrier_freq_hz)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.orbital_velocity_m_s = float(orbital_velocity_m_s)
        self.speed_of_light = float(speed_of_light)
        self.symbol_period_s = 1.0 / self.sampling_rate_hz
        self.wavelength = self.speed_of_light / self.carrier_freq_hz
        self.max_doppler_hz = (self.orbital_velocity_m_s / self.speed_of_light) * self.carrier_freq_hz
        
    def calculate_doppler_frequency(self, elevation_deg: float) -> float:
        """Calculate approximate geometric Doppler shift for given elevation angle.
        
        Args:
            elevation_deg: Elevation angle in degrees.
            
        Returns:
            Doppler frequency shift in Hz.
        """
        elev_rad = np.radians(elevation_deg)
        # Approximate geometric line-of-sight component: v_los = v_sat * cos(theta)
        f_d = (self.orbital_velocity_m_s * np.cos(elev_rad) / self.speed_of_light) * self.carrier_freq_hz
        return float(f_d)
        
    def generate_phase_evolution(
        self,
        num_samples: int,
        doppler_freq_hz: Optional[float] = None,
        initial_phase_rad: Optional[float] = None,
        drift_rate_hz_s: float = 0.0,
    ) -> np.ndarray:
        """Generate Doppler phase evolution vector phi_D[k] over discrete symbol indices.
        
        Args:
            num_samples: Number of symbols in the transmission burst/window.
            doppler_freq_hz: Specific Doppler shift in Hz. If None, sampled uniformly in [-max_doppler, max_doppler].
            initial_phase_rad: Initial random phase offset phi_0. If None, sampled uniformly in [0, 2*pi).
            drift_rate_hz_s: Doppler rate of change df_d/dt in Hz/s.
            
        Returns:
            1D NumPy array of phase values phi_D[k] in radians, shape (num_samples,).
        """
        if doppler_freq_hz is None:
            doppler_freq_hz = np.random.uniform(-self.max_doppler_hz, self.max_doppler_hz)
        if initial_phase_rad is None:
            initial_phase_rad = np.random.uniform(0.0, 2.0 * np.pi)
            
        k_indices = np.arange(num_samples, dtype=np.float64)
        t_seconds = k_indices * self.symbol_period_s
        
        # phi(t) = 2*pi*(f_d*t + 0.5 * drift * t^2) + phi_0
        instantaneous_phase = 2.0 * np.pi * (doppler_freq_hz * t_seconds + 0.5 * drift_rate_hz_s * (t_seconds**2)) + initial_phase_rad
        return instantaneous_phase
        
    def generate_complex_carrier(
        self,
        num_samples: int,
        doppler_freq_hz: Optional[float] = None,
        initial_phase_rad: Optional[float] = None,
    ) -> np.ndarray:
        """Generate complex exponential Doppler phasor e^(j * phi_D[k])."""
        phase = self.generate_phase_evolution(num_samples, doppler_freq_hz, initial_phase_rad)
        return np.exp(1j * phase)
