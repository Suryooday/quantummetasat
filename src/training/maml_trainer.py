"""Trainer class for MAML and Q-MAML episodic optimization."""
import os
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from src.meta_learning.maml import MAML
from src.data.dataset import MetaChannelDataset, TaskData
from src.data.task_sampler import EpisodicTaskSampler
from src.evaluation.metrics import calculate_nmse, calculate_nmse_db, compute_all_metrics
from src.evaluation.ber import BERCalculator

class MAMLTrainer:
    """Orchestrates meta-training, episodic validation, and few-shot evaluation for MAML models."""
    
    def __init__(
        self,
        maml: MAML,
        meta_lr_scheduler=None,
    ):
        self.maml = maml
        self.meta_lr_scheduler = meta_lr_scheduler
        self.ber_calc = BERCalculator()
        
    def train(
        self,
        train_dataset: MetaChannelDataset,
        val_dataset: Optional[MetaChannelDataset] = None,
        meta_epochs: int = 50,
        tasks_per_epoch: int = 20,
        meta_batch_size: int = 4,
        k_support: int = 10,
        q_query: int = 30,
        inner_steps: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Meta-train the model across episodic channel tasks."""
        sampler = EpisodicTaskSampler(
            train_dataset,
            batch_size=meta_batch_size,
            k_support=k_support,
            q_query=q_query,
            shuffle_tasks=True,
        )
        
        history = {"meta_loss": [], "val_nmse_db": []}
        start_time = time.time()
        best_val_nmse = float("inf")
        best_weights = None
        
        for epoch in range(1, meta_epochs + 1):
            epoch_loss = 0.0
            
            for _ in range(tasks_per_epoch):
                meta_batch = sampler.sample_meta_batch()
                step_loss = self.maml.meta_step(meta_batch, inner_steps=inner_steps)
                epoch_loss += step_loss
                
            avg_loss = epoch_loss / tasks_per_epoch
            history["meta_loss"].append(avg_loss)
            
            # Validation every 5 epochs or on the final epoch
            val_nmse_db = 0.0
            if val_dataset is not None and (epoch % 5 == 0 or epoch == meta_epochs):
                val_res = self.evaluate_few_shot(
                    val_dataset.tasks,
                    k_support=k_support,
                    q_query=q_query,
                    adaptation_steps=[inner_steps],
                )
                val_nmse = val_res[inner_steps]["nmse"]
                val_nmse_db = val_res[inner_steps]["nmse_db"]
                history["val_nmse_db"].append(val_nmse_db)
                
                if val_nmse < best_val_nmse:
                    best_val_nmse = val_nmse
                    best_weights = {k: v.cpu().clone() for k, v in self.maml.model.state_dict().items()}
                    
            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == meta_epochs):
                val_str = f" | Val NMSE (step={inner_steps}): {val_nmse_db:.2f} dB" if val_dataset is not None and (epoch % 5 == 0 or epoch == meta_epochs) else ""
                print(f"Meta-Epoch {epoch:03d}/{meta_epochs:03d} | Meta-Loss: {avg_loss:.6f}{val_str}")
                
        if best_weights is not None:
            self.maml.model.load_state_dict(best_weights)
            
        history["total_meta_train_time_s"] = time.time() - start_time
        return history
        
    def evaluate_few_shot(
        self,
        test_tasks: List[TaskData],
        k_support: int = 10,
        q_query: int = 50,
        adaptation_steps: List[int] = [0, 1, 3, 5, 10],
        inner_lr: Optional[float] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate few-shot adaptation performance on unseen test tasks across multiple adaptation steps."""
        results_by_step = {s: {"nmses": [], "bers": []} for s in adaptation_steps}
        
        for task in test_tasks:
            x_s, y_s, x_q, y_q = task.sample_support_query(k_support, q_query, shuffle=False)
            
            for steps in adaptation_steps:
                with torch.enable_grad():
                    adapted_params = self.maml.adapt(
                        x_s.to(self.maml.device),
                        y_s.to(self.maml.device),
                        num_steps=steps,
                        inner_lr=inner_lr,
                        create_graph=False,
                    )
                    
                with torch.no_grad():
                    pred_q = self.maml.forward_with_params(x_q.to(self.maml.device), adapted_params).cpu().numpy()
                    true_q = y_q.numpy()
                    
                    nmse = calculate_nmse(true_q, pred_q)
                    results_by_step[steps]["nmses"].append(nmse)
                    
                    if task.x_raw is not None and task.y_raw is not None:
                        # Raw symbol alignment
                        W = 9
                        y_q_raw = task.y_raw[W - 1 + k_support : W - 1 + k_support + q_query]
                        x_q_raw = task.x_raw[W - 1 + k_support : W - 1 + k_support + q_query]
                        ber_res = self.ber_calc.equalize_and_decode(y_q_raw, pred_q, x_q_raw)
                        results_by_step[steps]["bers"].append(ber_res["ber"])
                        
        summary = {}
        for steps in adaptation_steps:
            mean_nmse = float(np.mean(results_by_step[steps]["nmses"]))
            mean_nmse_db = float(10.0 * np.log10(mean_nmse)) if mean_nmse > 1e-12 else -120.0
            mean_ber = float(np.mean(results_by_step[steps]["bers"])) if results_by_step[steps]["bers"] else 0.0
            summary[steps] = {
                "nmse": mean_nmse,
                "nmse_db": mean_nmse_db,
                "ber": mean_ber,
            }
            
        return summary
