"""Shadowed-Rician fading distribution for satellite communications (Abdi et al. parameterization)."""
import numpy as np
from typing import Dict, Optional, Tuple
from .base import BaseChannel

class ShadowedRicianFading(BaseChannel):
    """Shadowed-Rician statistical channel model for LEO satellite links.
    
    The channel envelope is composed of a Nakagami-m shadowed line-of-sight (LOS) component
    plus Rayleigh/Gaussian distributed multipath scatterers:
        h_small = z * e^(j * theta_0) + (X + j * Y)
        
    where:
        z ~ Nakagami(m, Omega), with z^2 ~ Gamma(shape=m, scale=Omega/m)
        theta_0 ~ Uniform(0, 2*pi)
        X, Y ~ Normal(0, sqrt(b0))
        Total average power: E[|h_small|^2] = Omega + 2 * b0
        
    Attributes:
        b0 (float): Average power of the multipath scatter component (half power per real/imag dimension).
        m (float): Nakagami fading severity parameter (m >= 0.5).
        omega (float): Average power of the line-of-sight component.
        normalize_power (bool): If True, normalize so that E[|h_small|^2] = 1.0.
    """
    
    # Standard Abdi et al. benchmark parameter sets for LMS channels
    STANDARD_ENVIRONMENTS: Dict[str, Dict[str, float]] = {
        "frequent_heavy": {
            "b0": 0.063,
            "m": 0.739,
            "omega": 8.97e-4,
            "description": "Frequent Heavy Shadowing (dense urban/foliage, severely obstructed LOS)"
        },
        "average": {
            "b0": 0.126,
            "m": 10.1,
            "omega": 0.835,
            "description": "Average Shadowing (suburban/moderate foliage)"
        },
        "infrequent_light": {
            "b0": 0.158,
            "m": 19.4,
            "omega": 1.29,
            "description": "Infrequent Light Shadowing (open rural / near unobstructed LOS)"
        }
    }
    
    def __init__(
        self,
        b0: float = 0.126,
        m: float = 10.1,
        omega: float = 0.835,
        normalize_power: bool = True,
    ):
        if b0 <= 0 or m <= 0 or omega <= 0:
            raise ValueError(f"Parameters b0, m, omega must be strictly positive. Got b0={b0}, m={m}, omega={omega}")
        self.b0 = float(b0)
        self.m = float(m)
        self.omega = float(omega)
        self.normalize_power = normalize_power
        self.total_power = self.omega + 2.0 * self.b0
        
    @classmethod
    def from_environment(cls, env_name: str, normalize_power: bool = True) -> "ShadowedRicianFading":
        """Factory method to instantiate standard shadowing environments."""
        if env_name not in cls.STANDARD_ENVIRONMENTS:
            raise ValueError(
                f"Unknown environment '{env_name}'. Choose from: {list(cls.STANDARD_ENVIRONMENTS.keys())}"
            )
        params = cls.STANDARD_ENVIRONMENTS[env_name]
        return cls(b0=params["b0"], m=params["m"], omega=params["omega"], normalize_power=normalize_power)
        
    def generate(self, num_samples: int, **kwargs) -> np.ndarray:
        """Generate complex channel fading coefficients h_small.
        
        Args:
            num_samples: Number of independent fading realizations to sample.
            
        Returns:
            Complex 1D NumPy array of channel fading coefficients, shape (num_samples,).
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
            
        # 1. Sample LOS power from Gamma distribution: shape = m, scale = Omega / m
        # (Gamma(shape=k, scale=theta) has mean k*theta = Omega)
        z_squared = np.random.gamma(shape=self.m, scale=self.omega / self.m, size=num_samples)
        z = np.sqrt(np.maximum(z_squared, 0.0))
        
        # 2. Sample LOS phase uniformly over [0, 2*pi)
        theta_0 = np.random.uniform(0.0, 2.0 * np.pi, size=num_samples)
        los_component = z * np.exp(1j * theta_0)
        
        # 3. Sample diffuse multipath components X, Y ~ N(0, b0)
        x = np.random.normal(0.0, np.sqrt(self.b0), size=num_samples)
        y = np.random.normal(0.0, np.sqrt(self.b0), size=num_samples)
        multipath_component = x + 1j * y
        
        # 4. Composite small-scale fading
        h_small = los_component + multipath_component
        
        if self.normalize_power:
            h_small = h_small / np.sqrt(self.total_power)
            
        return h_small
