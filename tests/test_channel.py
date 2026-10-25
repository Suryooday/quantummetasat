"""Unit tests for physical channels, path loss, Doppler, Shadowed-Rician fading, AWGN, and QPSK modulation."""
import pytest
import numpy as np
from src.channels.path_loss import PathLoss
from src.channels.doppler import DopplerModel
from src.channels.shadowed_rician import ShadowedRicianFading
from src.channels.awgn import AWGNChannel
from src.channels.leo_channel import LEOChannel
from src.modulation.qpsk import QPSKModulator
from src.utils.seed import set_seed

def test_path_loss_physics():
    """Verify free-space path loss scaling and monotonicity."""
    pl = PathLoss(carrier_freq_hz=2.0e9, ref_distance_m=600000.0)
    
    # Check monotonicity
    assert pl.validate_monotonicity()
    
    # Check doubling distance increases FSPL by ~6.02 dB (20*log10(2))
    d1 = 600000.0
    d2 = 1200000.0
    fspl1 = pl.calculate_fspl_db(d1)
    fspl2 = pl.calculate_fspl_db(d2)
    assert np.isclose(fspl2 - fspl1, 20.0 * np.log10(2.0), atol=1e-3)
    
    # Check normalized amplitude at reference distance is 1.0
    amp_ref = pl.calculate_amplitude(d1, normalized=True)
    assert np.isclose(amp_ref, 1.0, atol=1e-5)
    
    # Check amplitude at double distance is 0.5
    amp_double = pl.calculate_amplitude(d2, normalized=True)
    assert np.isclose(amp_double, 0.5, atol=1e-5)

def test_doppler_physics():
    """Verify Doppler calculations and phase continuity."""
    set_seed(42)
    doppler = DopplerModel(carrier_freq_hz=2.0e9, sampling_rate_hz=1.0e6, orbital_velocity_m_s=7560.0)
    
    # Max Doppler check: v/c * fc = (7560 / 3e8) * 2e9 ~ 50400 Hz
    expected_max = (7560.0 / 299792458.0) * 2.0e9
    assert np.isclose(doppler.max_doppler_hz, expected_max, atol=1.0)
    
    # Complex carrier modulus must be unity
    phasor = doppler.generate_complex_carrier(num_samples=1000, doppler_freq_hz=20000.0, initial_phase_rad=0.0)
    assert np.allclose(np.abs(phasor), 1.0, atol=1e-6)

def test_shadowed_rician_fading():
    """Verify Shadowed-Rician fading statistics and normalization."""
    set_seed(42)
    for env in ["frequent_heavy", "average", "infrequent_light"]:
        fading = ShadowedRicianFading.from_environment(env, normalize_power=True)
        samples = fading.generate(num_samples=200000)
        
        # Verify finite values and correct dtype
        assert np.all(np.isfinite(samples))
        assert np.iscomplexobj(samples)
        
        # Verify normalized average power E[|h_small|^2] ~ 1.0
        avg_power = np.mean(np.abs(samples)**2)
        assert np.isclose(avg_power, 1.0, atol=0.02), f"Environment {env} failed power normalization: {avg_power}"

def test_awgn_snr_accuracy():
    """Verify AWGN noise scaling matches requested SNR."""
    set_seed(42)
    awgn = AWGNChannel()
    
    for target_snr in [5.0, 10.0, 15.0, 20.0]:
        signal = np.random.normal(0, 1, 500000) + 1j * np.random.normal(0, 1, 500000)
        noisy_signal, noise, measured_snr = awgn.add_noise(signal, snr_db=target_snr, return_noise=True)
        
        assert np.isclose(measured_snr, target_snr, atol=0.1), f"Target {target_snr} dB, got {measured_snr} dB"

def test_qpsk_modulation():
    """Verify QPSK constellation power normalization, modulation and bit error rate."""
    set_seed(42)
    qpsk = QPSKModulator()
    
    symbols = qpsk.generate_symbols(10000)
    
    # E[|x|^2] must be exactly 1.0
    avg_power = np.mean(np.abs(symbols)**2)
    assert np.isclose(avg_power, 1.0, atol=1e-6)
    
    # Verify bit round-trip under noiseless condition
    bits = np.random.randint(0, 2, size=1000)
    tx_symbols = qpsk.modulate_bits(bits)
    rx_bits = qpsk.demodulate_to_bits(tx_symbols)
    assert np.array_equal(bits, rx_bits)
    
    ber = qpsk.compute_ber(tx_symbols, tx_symbols)
    ser = qpsk.compute_ser(tx_symbols, tx_symbols)
    assert ber == 0.0
    assert ser == 0.0

def test_composite_leo_channel_transmission():
    """Verify end-to-end LEO channel transmission pipeline."""
    set_seed(42)
    leo = LEOChannel()
    qpsk = QPSKModulator()
    
    tx_symbols = qpsk.generate_symbols(500)
    y, h_eff, meta = leo.transmit(
        tx_symbols,
        elevation_deg=45.0,
        snr_db=15.0,
        environment="average"
    )
    
    assert len(y) == 500
    assert len(h_eff) == 500
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(h_eff))
    assert "slant_range_m" in meta
    assert "fspl_db" in meta
    assert np.isclose(meta["snr_db_target"], 15.0)
    assert np.isclose(meta["snr_db_measured"], 15.0, atol=0.5)
