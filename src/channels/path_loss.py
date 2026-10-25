"""Free-space path loss (FSPL) and amplitude attenuation formulation."""
import numpy as np
from typing import Union

class PathLoss:
    """Free-space path loss model for satellite communication links.
    
    Implements:
        FSPL_dB = 20 * log10( (4 * pi * d * f_c) / c )
        A(d) = 10^(-FSPL_dB / 20) = c / (4 * pi * d * f_c)
        
    Attributes:
        carrier_freq_hz (float): Carrier frequency in Hertz (e.g. 2.0 GHz).
        speed_of_light (float): Propagation speed in m/s (299,792,458 m/s).
        ref_distance_m (float): Reference distance (e.g. zenith altitude h) for normalized amplitude.
    """
    
    def __init__(
        self,
        carrier_freq_hz: float = 2.0e9,
        speed_of_light: float = 299792458.0,
        ref_distance_m: float = 600000.0,
    ):
        if carrier_freq_hz <= 0:
            raise ValueError(f"Carrier frequency must be positive, got {carrier_freq_hz}")
        self.carrier_freq_hz = float(carrier_freq_hz)
        self.speed_of_light = float(speed_of_light)
        self.ref_distance_m = float(ref_distance_m)
        self.wavelength = self.speed_of_light / self.carrier_freq_hz
        
    def calculate_fspl_db(self, distance_m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute free-space path loss in decibels.
        
        Args:
            distance_m: Propagation distance / slant range in meters.
            
        Returns:
            Path loss in dB.
        """
        dist = np.asarray(distance_m, dtype=np.float64)
        if np.any(dist <= 0):
            raise ValueError("Distance must be strictly positive.")
            
        # 20 * log10( (4 * pi * d * f_c) / c ) = 20 * log10( (4 * pi * d) / lambda )
        fspl_db = 20.0 * np.log10((4.0 * np.pi * dist * self.carrier_freq_hz) / self.speed_of_light)
        if np.isscalar(distance_m):
            return float(fspl_db)
        return fspl_db
        
    def calculate_amplitude(
        self,
        distance_m: Union[float, np.ndarray],
        normalized: bool = True
    ) -> Union[float, np.ndarray]:
        """Compute path-loss amplitude attenuation factor A(d).
        
        Args:
            distance_m: Slant range distance in meters.
            normalized: If True, normalize by zenith reference distance (A_norm(d) = ref_dist / d)
                        to ensure numerical stability in neural network estimators.
                        If False, compute physical amplitude c / (4 * pi * d * f_c).
                        
        Returns:
            Amplitude attenuation factor A(d).
        """
        dist = np.asarray(distance_m, dtype=np.float64)
        if np.any(dist <= 0):
            raise ValueError("Distance must be strictly positive.")
            
        if normalized:
            # Normalized amplitude relative to zenith reference distance
            amp = self.ref_distance_m / dist
        else:
            # Physical electromagnetic field attenuation
            amp = self.speed_of_light / (4.0 * np.pi * dist * self.carrier_freq_hz)
            
        if np.isscalar(distance_m):
            return float(amp)
        return amp
        
    def validate_monotonicity(self, d_min: float = 600000.0, d_max: float = 2500000.0, num_points: int = 100) -> bool:
        """Verify that FSPL increases monotonically with distance, and amplitude decreases."""
        distances = np.linspace(d_min, d_max, num_points)
        fspl = self.calculate_fspl_db(distances)
        amp = self.calculate_amplitude(distances, normalized=True)
        return bool(np.all(np.diff(fspl) > 0) and np.all(np.diff(amp) < 0))
