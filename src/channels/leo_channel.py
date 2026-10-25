"""Composite LEO satellite channel model combining orbital geometry, path loss, Shadowed-Rician fading, Doppler phase, and AWGN."""
import numpy as np
from typing import Dict, Optional, Tuple, Union

from .base import BaseChannel
from .leo_geometry import LEOGeometry
from .path_loss import PathLoss
from .doppler import DopplerModel
from .shadowed_rician import ShadowedRicianFading
from .awgn import AWGNChannel

class LEOChannel(BaseChannel):
    """Composite physical-statistical LEO satellite channel simulator.
    
    Implements:
        h_eff[k] = A(d(theta)) * h_small[k] * exp(j * phi_D[k])
        y[k] = h_eff[k] * x[k] + n[k]
        
    where:
        theta: Elevation angle (deg)
        d(theta): Spherical Earth-satellite slant range (m)
        A(d): Free-space path loss amplitude factor (normalized or physical)
        h_small[k]: Shadowed-Rician small-scale fading
        phi_D[k]: Doppler frequency induced phase evolution
        n[k]: Additive White Gaussian Noise
        
    Attributes:
        geometry (LEOGeometry): Spherical orbital geometry model.
        path_loss (PathLoss): Path loss and amplitude model.
        doppler (DopplerModel): Doppler frequency and phase evolution model.
        awgn (AWGNChannel): AWGN noise generator.
        normalize_path_loss (bool): Whether amplitude A(d) is normalized to zenith reference.
    """
    
    def __init__(
        self,
        earth_radius_m: float = 6378137.0,
        satellite_altitude_m: float = 600000.0,
        carrier_freq_hz: float = 2.0e9,
        sampling_rate_hz: float = 1.0e6,
        orbital_velocity_m_s: float = 7560.0,
        normalize_path_loss: bool = True,
    ):
        self.geometry = LEOGeometry(earth_radius_m=earth_radius_m, satellite_altitude_m=satellite_altitude_m)
        self.path_loss = PathLoss(
            carrier_freq_hz=carrier_freq_hz,
            ref_distance_m=satellite_altitude_m,
        )
        self.doppler = DopplerModel(
            carrier_freq_hz=carrier_freq_hz,
            sampling_rate_hz=sampling_rate_hz,
            orbital_velocity_m_s=orbital_velocity_m_s,
        )
        self.awgn = AWGNChannel()
        self.normalize_path_loss = normalize_path_loss
        
    def generate_coefficients(
        self,
        num_samples: int,
        elevation_deg: float,
        environment: str = "average",
        doppler_freq_hz: Optional[float] = None,
        initial_phase_rad: Optional[float] = None,
        custom_fading_params: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate composite channel coefficients h_eff for a given channel state.
        
        Args:
            num_samples: Number of symbols in the transmission burst.
            elevation_deg: Satellite elevation angle in degrees [0, 90].
            environment: Shadowing environment name ('frequent_heavy', 'average', 'infrequent_light').
            doppler_freq_hz: Doppler frequency shift in Hz. If None, derived from elevation.
            initial_phase_rad: Initial phase offset. If None, randomly sampled.
            custom_fading_params: Optional dict with (b0, m, omega) to override standard environments.
            
        Returns:
            h_eff: Complex channel coefficient array, shape (num_samples,).
            metadata: Dictionary of underlying physical channel parameters.
        """
        # 1. Slant range d(theta)
        slant_range_m = self.geometry.calculate_slant_range(elevation_deg)
        
        # 2. Path loss amplitude A(d)
        amplitude_A = self.path_loss.calculate_amplitude(slant_range_m, normalized=self.normalize_path_loss)
        
        # 3. Small-scale Shadowed-Rician fading
        if custom_fading_params is not None:
            fading_gen = ShadowedRicianFading(**custom_fading_params, normalize_power=True)
        else:
            fading_gen = ShadowedRicianFading.from_environment(environment, normalize_power=True)
        h_small = fading_gen.generate(num_samples)
        
        # 4. Doppler phase evolution
        if doppler_freq_hz is None:
            # Derived from orbital geometryLOS projection
            doppler_freq_hz = self.doppler.calculate_doppler_frequency(elevation_deg)
        doppler_phasor = self.doppler.generate_complex_carrier(
            num_samples, doppler_freq_hz=doppler_freq_hz, initial_phase_rad=initial_phase_rad
        )
        
        # 5. Composite channel coefficient: h_eff = A(d) * h_small * exp(j * phi_D)
        h_eff = amplitude_A * h_small * doppler_phasor
        
        metadata = {
            "elevation_deg": float(elevation_deg),
            "slant_range_m": float(slant_range_m),
            "amplitude_A": float(amplitude_A),
            "fspl_db": float(self.path_loss.calculate_fspl_db(slant_range_m)),
            "doppler_freq_hz": float(doppler_freq_hz),
            "environment": environment,
        }
        
        return h_eff, metadata
        
    def generate(self, num_samples: int, elevation_deg: float = 45.0, **kwargs) -> np.ndarray:
        """Base channel interface wrapper."""
        h_eff, _ = self.generate_coefficients(num_samples, elevation_deg, **kwargs)
        return h_eff
        
    def transmit(
        self,
        x_symbols: np.ndarray,
        elevation_deg: float,
        snr_db: float,
        environment: str = "average",
        doppler_freq_hz: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Simulate full transmission: y = h_eff * x + n.
        
        Args:
            x_symbols: Modulated complex symbols (e.g. QPSK), shape (num_samples,).
            elevation_deg: Elevation angle in degrees.
            snr_db: Target SNR in decibels.
            environment: Shadowing condition.
            doppler_freq_hz: Optional Doppler shift.
            
        Returns:
            y: Received noisy symbols, shape (num_samples,).
            h_eff: True complex channel coefficients, shape (num_samples,).
            meta: Channel and transmission metadata.
        """
        num_samples = len(x_symbols)
        h_eff, meta = self.generate_coefficients(
            num_samples, elevation_deg, environment=environment, doppler_freq_hz=doppler_freq_hz
        )
        
        # Faded noiseless signal
        faded_signal = h_eff * x_symbols
        
        # Add complex AWGN
        y, noise, measured_snr_db = self.awgn.add_noise(faded_signal, snr_db=snr_db, return_noise=True)
        
        meta["snr_db_target"] = float(snr_db)
        meta["snr_db_measured"] = float(measured_snr_db)
        
        return y, h_eff, meta
