"""Feature normalization and scaling utilities with strict leakage prevention."""
import numpy as np
import torch
from typing import Optional, Union

class ChannelFeatureScaler:
    """Standard scaler for channel observation features.
    
    Fits mean and standard deviation strictly on training data to prevent data leakage.
    
    Attributes:
        mean (np.ndarray): Mean vector computed across features.
        std (np.ndarray): Standard deviation vector computed across features.
        is_fitted (bool): Whether the scaler has been fitted.
    """
    
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        
    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "ChannelFeatureScaler":
        """Fit scaler parameters on training feature data ONLY.
        
        Args:
            X: Input feature array or tensor of shape (N, feature_dim).
        """
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy()
        else:
            X_arr = np.asarray(X)
            
        self.mean = np.mean(X_arr, axis=0, keepdims=True)
        self.std = np.std(X_arr, axis=0, keepdims=True)
        # Avoid zero division
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        self.is_fitted = True
        return self
        
    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Transform features using fitted statistics.
        
        Args:
            X: Input features of shape (N, feature_dim).
            
        Returns:
            Normalized feature array.
        """
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("ChannelFeatureScaler must be fitted before transforming data.")
            
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy()
        else:
            X_arr = np.asarray(X)
            
        return (X_arr - self.mean) / self.std
        
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Fit scaler and transform features."""
        return self.fit(X).transform(X)
        
    def to_dict(self) -> dict:
        """Serialize scaler statistics."""
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "is_fitted": self.is_fitted,
        }
        
    @classmethod
    def from_dict(cls, d: dict) -> "ChannelFeatureScaler":
        """Deserialize scaler statistics."""
        scaler = cls()
        if d.get("is_fitted", False):
            scaler.mean = np.array(d["mean"])
            scaler.std = np.array(d["std"])
            scaler.is_fitted = True
        return scaler
