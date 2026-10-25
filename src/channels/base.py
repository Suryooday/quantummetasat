"""Abstract base class for channel components."""
from abc import ABC, abstractmethod
import numpy as np

class BaseChannel(ABC):
    """Base interface for physical and statistical channel components."""
    
    @abstractmethod
    def generate(self, num_samples: int, **kwargs) -> np.ndarray:
        """Generate channel coefficients or impairments.
        
        Args:
            num_samples: Number of discrete symbol samples to simulate.
            **kwargs: Additional contextual parameters.
            
        Returns:
            NumPy array of channel coefficients (complex or real).
        """
        pass
