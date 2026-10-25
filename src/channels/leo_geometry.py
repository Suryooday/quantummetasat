"""LEO Earth-satellite orbital geometry and slant-range formulation."""
import numpy as np
from typing import Union

class LEOGeometry:
    """Earth-satellite spherical geometry model for Low Earth Orbit (LEO) communications.
    
    Implements the exact spherical trigonometry slant-range relation:
        d(theta) = sqrt(R_e^2 * sin^2(theta) + 2 * R_e * h + h^2) - R_e * sin(theta)
        
    Attributes:
        earth_radius_m (float): Mean radius of Earth (WGS-84 standard: 6,378,137 m).
        satellite_altitude_m (float): Satellite orbital altitude above Earth surface in meters.
    """
    
    def __init__(self, earth_radius_m: float = 6378137.0, satellite_altitude_m: float = 600000.0):
        if earth_radius_m <= 0:
            raise ValueError(f"Earth radius must be positive, got {earth_radius_m}")
        if satellite_altitude_m <= 0:
            raise ValueError(f"Satellite altitude must be positive, got {satellite_altitude_m}")
        self.earth_radius_m = float(earth_radius_m)
        self.satellite_altitude_m = float(satellite_altitude_m)
        
    def calculate_slant_range(self, elevation_deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the slant range distance d(theta) between ground station and satellite.
        
        Args:
            elevation_deg: Elevation angle(s) in degrees in range [0, 90].
            
        Returns:
            Slant range distance in meters.
        """
        elev_arr = np.asarray(elevation_deg, dtype=np.float64)
        if np.any(elev_arr < 0.0) or np.any(elev_arr > 90.0):
            raise ValueError("Elevation angle must be physically bounded between 0 and 90 degrees.")
            
        theta_rad = np.radians(elev_arr)
        sin_theta = np.sin(theta_rad)
        
        re = self.earth_radius_m
        h = self.satellite_altitude_m
        
        term_under_sqrt = (re * sin_theta)**2 + 2.0 * re * h + h**2
        d = np.sqrt(term_under_sqrt) - re * sin_theta
        
        if np.isscalar(elevation_deg):
            return float(d)
        return d
        
    def validate_zenith(self, tolerance_m: float = 1e-3) -> bool:
        """Verify that at zenith (90 deg), slant range equals satellite altitude h."""
        d_zenith = self.calculate_slant_range(90.0)
        return abs(d_zenith - self.satellite_altitude_m) < tolerance_m
        
    def validate_monotonicity(self, num_points: int = 100) -> bool:
        """Verify that slant range strictly decreases as elevation angle increases from 0 to 90 deg."""
        elevations = np.linspace(0.0, 90.0, num_points)
        ranges = self.calculate_slant_range(elevations)
        diffs = np.diff(ranges)
        return np.all(diffs < 0)
