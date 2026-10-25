"""Unit tests for NMSE, MSE, MAE, BER, and SER metrics."""
import pytest
import numpy as np
import torch
from src.evaluation.metrics import calculate_mse, calculate_mae, calculate_nmse, calculate_nmse_db, compute_all_metrics
from src.evaluation.ber import BERCalculator
from src.modulation.qpsk import QPSKModulator

def test_nmse_perfect_prediction():
    """Verify that perfect prediction gives zero MSE and zero NMSE."""
    h_true = np.array([[1.0, 0.5], [-0.5, 0.2]], dtype=np.float32)
    h_pred = np.array([[1.0, 0.5], [-0.5, 0.2]], dtype=np.float32)
    
    assert calculate_mse(h_true, h_pred) == 0.0
    assert calculate_mae(h_true, h_pred) == 0.0
    assert calculate_nmse(h_true, h_pred) == 0.0
    assert calculate_nmse_db(h_true, h_pred) == -120.0

def test_nmse_known_values():
    """Verify NMSE calculation on synthetic known power."""
    # True power = 1^2 + 1^2 = 2.0
    # Error = 0.5^2 + 0.5^2 = 0.5
    # NMSE = 0.5 / 2.0 = 0.25 (-6.02 dB)
    h_true = np.array([[1.0, 1.0]], dtype=np.float32)
    h_pred = np.array([[1.5, 1.5]], dtype=np.float32)
    
    nmse = calculate_nmse(h_true, h_pred)
    assert np.isclose(nmse, 0.25, atol=1e-5)
    nmse_db = calculate_nmse_db(h_true, h_pred)
    assert np.isclose(nmse_db, 10.0 * np.log10(0.25), atol=1e-4)

def test_ber_ser_perfect_channel():
    """Verify BER is zero when channel is known perfectly in high SNR."""
    mod = QPSKModulator()
    ber_calc = BERCalculator()
    
    tx = mod.generate_symbols(1000)
    h = np.ones(1000, dtype=np.complex128) * (0.8 + 0.6j)
    y = h * tx  # noiseless
    
    res = ber_calc.equalize_and_decode(y, h, tx)
    assert res["ber"] == 0.0
    assert res["ser"] == 0.0
    assert res["evm_percent"] < 1e-4
