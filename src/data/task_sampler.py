"""Episodic task sampler for meta-learning (MAML) training and evaluation."""
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from .dataset import MetaChannelDataset, TaskData

class EpisodicTaskSampler:
    """Samples episodic batches of tasks with disjoint Support and Query sets.
    
    For each meta-iteration:
        1. Sample B tasks {T_1, ..., T_B} from MetaChannelDataset.
        2. For each task T_i, extract Support set D_i^S (K pilots) and Query set D_i^Q (Q samples).
    """
    
    def __init__(
        self,
        meta_dataset: MetaChannelDataset,
        batch_size: int = 4,
        k_support: int = 10,
        q_query: int = 30,
        shuffle_tasks: bool = True,
    ):
        self.meta_dataset = meta_dataset
        self.batch_size = min(batch_size, len(meta_dataset))
        self.k_support = k_support
        self.q_query = q_query
        self.shuffle_tasks = shuffle_tasks
        
    def sample_meta_batch(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """Sample a batch of meta-learning tasks.
        
        Returns:
            List of tuples: (X_support, Y_support, X_query, Y_query, metadata) for each task in batch.
        """
        num_tasks = len(self.meta_dataset)
        if self.shuffle_tasks:
            task_indices = np.random.choice(num_tasks, size=self.batch_size, replace=False)
        else:
            task_indices = np.arange(self.batch_size)
            
        meta_batch = []
        for idx in task_indices:
            task = self.meta_dataset[idx]
            x_s, y_s, x_q, y_q = task.sample_support_query(self.k_support, self.q_query, shuffle=True)
            meta_batch.append((x_s, y_s, x_q, y_q, task.metadata))
            
        return meta_batch
