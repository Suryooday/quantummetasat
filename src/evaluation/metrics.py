"""Standard statistical regression and channel estimation error metrics."""
import numpy as np
import torch
from typing import Dict, Union

def calculate_mse(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """Mean Squared Error: E[|y_pred - y_true|^2]."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    return float(np.mean((y_pred - y_true)**2))

def calculate_mae(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """Mean Absolute Error: E[|y_pred - y_true|]."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    return float(np.mean(np.abs(y_pred - y_true)))

def calculate_nmse(
    h_true: Union[np.ndarray, torch.Tensor],
    h_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Normalized Mean Squared Error: E[|h_pred - h_true|^2] / E[|h_true|^2].
    
    Supports:
        1. 2D real tensor/array of shape (N, 2) where col 0 is Re and col 1 is Im.
        2. 1D complex array of shape (N,).
    """
    if isinstance(h_true, torch.Tensor):
        h_true = h_true.detach().cpu().numpy()
    if isinstance(h_pred, torch.Tensor):
        h_pred = h_pred.detach().cpu().numpy()
        
    if np.iscomplexobj(h_true) or np.iscomplexobj(h_pred):
        h_true_c = np.asarray(h_true, dtype=np.complex128)
        h_pred_c = np.asarray(h_pred, dtype=np.complex128)
        err_sq = np.abs(h_pred_c - h_true_c)**2
        power = np.abs(h_true_c)**2
    else:
        # Array of shape (N, 2)
        err_sq = (h_pred[:, 0] - h_true[:, 0])**2 + (h_pred[:, 1] - h_true[:, 1])**2
        power = h_true[:, 0]**2 + h_true[:, 1]**2
        
    mean_err = np.mean(err_sq)
    mean_power = np.mean(power)
    if mean_power <= 1e-12:
        return 1.0
    return float(mean_err / mean_power)

def calculate_nmse_db(
    h_true: Union[np.ndarray, torch.Tensor],
    h_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Normalized Mean Squared Error in decibels: 10 * log10(NMSE)."""
    nmse = calculate_nmse(h_true, h_pred)
    if nmse <= 1e-12:
        return -120.0
    return float(10.0 * np.log10(nmse))

def compute_all_metrics(
    h_true: Union[np.ndarray, torch.Tensor],
    h_pred: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """Compute comprehensive dictionary of estimation metrics."""
    mse = calculate_mse(h_true, h_pred)
    mae = calculate_mae(h_true, h_pred)
    nmse = calculate_nmse(h_true, h_pred)
    nmse_db = calculate_nmse_db(h_true, h_pred)
    return {
        "mse": mse,
        "mae": mae,
        "nmse": nmse,
        "nmse_db": nmse_db,
    }
