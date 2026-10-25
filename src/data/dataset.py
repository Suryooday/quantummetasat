"""PyTorch Dataset implementations for supervised pooling and episodic meta-learning."""
import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any

class TaskData:
    """Holds data and metadata for an individual channel condition task."""
    
    def __init__(
        self,
        task_id: str,
        X: np.ndarray,
        Y: np.ndarray,
        x_raw: Optional[np.ndarray] = None,
        y_raw: Optional[np.ndarray] = None,
        h_raw: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.h_raw = h_raw
        self.metadata = metadata or {}
        
    def __len__(self) -> int:
        return len(self.X)
        
    def sample_support_query(
        self,
        k_support: int,
        q_query: int,
        shuffle: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample disjoint support and query subsets for meta-learning.
        
        Args:
            k_support: Number of pilot support samples.
            q_query: Number of evaluation query samples.
            shuffle: Whether to sample randomly or sequentially from the task sequence.
            
        Returns:
            X_s, Y_s, X_q, Y_q tensors.
        """
        total_samples = len(self.X)
        if k_support + q_query > total_samples:
            raise ValueError(
                f"Requested k_support ({k_support}) + q_query ({q_query}) > total samples ({total_samples})"
            )
            
        if shuffle:
            indices = np.random.permutation(total_samples)
        else:
            indices = np.arange(total_samples)
            
        supp_idx = indices[:k_support]
        query_idx = indices[k_support : k_support + q_query]
        
        X_s, Y_s = self.X[supp_idx], self.Y[supp_idx]
        X_q, Y_q = self.X[query_idx], self.Y[query_idx]
        return X_s, Y_s, X_q, Y_q


class SupervisedChannelDataset(Dataset):
    """Presents a flattened, pooled dataset across multiple channel tasks for standard supervised training."""
    
    def __init__(self, tasks: List[TaskData], scaler=None):
        self.tasks = tasks
        self.scaler = scaler
        
        # Concatenate all task features and targets
        all_X = torch.cat([t.X for t in tasks], dim=0)
        all_Y = torch.cat([t.Y for t in tasks], dim=0)
        
        if self.scaler is not None:
            if not self.scaler.is_fitted:
                self.scaler.fit(all_X)
            scaled_X_np = self.scaler.transform(all_X)
            self.X = torch.tensor(scaled_X_np, dtype=torch.float32)
        else:
            self.X = all_X
            
        self.Y = all_Y
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


class MetaChannelDataset:
    """Collection of channel tasks for episodic few-shot meta-learning."""
    
    def __init__(self, tasks: List[TaskData], scaler=None):
        self.tasks = tasks
        self.scaler = scaler
        
        # Apply scaler to all task data if provided and fitted
        if self.scaler is not None and self.scaler.is_fitted:
            for t in self.tasks:
                scaled_np = self.scaler.transform(t.X)
                t.X = torch.tensor(scaled_np, dtype=torch.float32)
                
    def __len__(self) -> int:
        return len(self.tasks)
        
    def __getitem__(self, idx: int) -> TaskData:
        return self.tasks[idx]
        
    @classmethod
    def load_from_dir(cls, split_dir: str, scaler=None) -> "MetaChannelDataset":
        """Load all tasks from a directory containing saved NPZ files and manifest."""
        manifest_path = os.path.join(split_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found in {split_dir}")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        tasks = []
        for entry in manifest:
            npz_path = entry["file_path"]
            if not os.path.isabs(npz_path):
                npz_path = os.path.join(split_dir, os.path.basename(npz_path))
            data = np.load(npz_path)
            
            task = TaskData(
                task_id=entry["task_id"],
                X=data["X"],
                Y=data["Y"],
                x_raw=data["x_raw"],
                y_raw=data["y_raw"],
                h_raw=data["h_raw"],
                metadata=entry["metadata"],
            )
            tasks.append(task)
            
        return cls(tasks, scaler=scaler)
