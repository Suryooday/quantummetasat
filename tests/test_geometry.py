"""Unit tests for LEO Earth-satellite orbital geometry and slant-range formulation."""
import pytest
import numpy as np
from src.channels.leo_geometry import LEOGeometry

def test_leo_geometry_zenith():
    """Verify that at zenith (90 deg elevation), slant range equals satellite altitude h."""
    altitude = 600000.0  # 600 km
    earth_radius = 6378137.0
    geom = LEOGeometry(earth_radius_m=earth_radius, satellite_altitude_m=altitude)
    
    d_zenith = geom.calculate_slant_range(90.0)
    assert np.isclose(d_zenith, altitude, atol=1e-3), f"Expected {altitude}, got {d_zenith}"
    assert geom.validate_zenith(tolerance_m=1e-3)

def test_leo_geometry_monotonicity():
    """Verify that slant range strictly decreases as elevation angle increases."""
    geom = LEOGeometry(earth_radius_m=6378137.0, satellite_altitude_m=600000.0)
    assert geom.validate_monotonicity(num_points=180)
    
    elevations = np.array([0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    ranges = geom.calculate_slant_range(elevations)
    
    # Check strict descending order
    assert np.all(np.diff(ranges) < 0.0)
    
    # Check positive and finite
    assert np.all(ranges > 0)
    assert np.all(np.isfinite(ranges))
    
    # Check horizon range is greater than altitude
    assert ranges[0] > geom.satellite_altitude_m

def test_leo_geometry_invalid_inputs():
    """Verify bounds checking and input validation."""
    geom = LEOGeometry()
    with pytest.raises(ValueError):
        geom.calculate_slant_range(-5.0)
    with pytest.raises(ValueError):
        geom.calculate_slant_range(95.0)
    with pytest.raises(ValueError):
        LEOGeometry(earth_radius_m=-100)
    with pytest.raises(ValueError):
        LEOGeometry(satellite_altitude_m=0)
