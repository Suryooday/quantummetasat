"""Supervised training engine for classical and baseline models."""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from src.data.dataset import SupervisedChannelDataset, MetaChannelDataset, TaskData
from src.evaluation.metrics import calculate_nmse, calculate_nmse_db, compute_all_metrics
from src.evaluation.ber import BERCalculator
from src.evaluation.profiler import ModelProfiler

class SupervisedTrainer:
    """Trains and evaluates neural channel estimators in standard supervised learning regimes."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.ber_calc = BERCalculator()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(batch_x)
            loss = self.criterion(pred, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / max(num_batches, 1)
        
    def evaluate_dataloader(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a DataLoader."""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.numpy())
                
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        
        return compute_all_metrics(y_true, y_pred)
        
    def evaluate_tasks(
        self,
        tasks: List[TaskData],
        scaler=None,
    ) -> Dict[str, float]:
        """Evaluate model across isolated tasks and compute aggregate NMSE and BER."""
        self.model.eval()
        task_nmses = []
        task_bers = []
        
        with torch.no_grad():
            for task in tasks:
                x_in = task.X
                if scaler is not None and scaler.is_fitted:
                    x_in = torch.tensor(scaler.transform(x_in), dtype=torch.float32)
                x_in = x_in.to(self.device)
                
                pred = self.model(x_in).cpu().numpy()
                true = task.Y.numpy()
                
                nmse = calculate_nmse(true, pred)
                task_nmses.append(nmse)
                
                # Compute BER if raw signals are present
                if task.x_raw is not None and task.y_raw is not None:
                    # Alignment with window output
                    W = 9
                    y_aligned = task.y_raw[W - 1 :]
                    x_aligned = task.x_raw[W - 1 :]
                    ber_res = self.ber_calc.equalize_and_decode(y_aligned, pred, x_aligned)
                    task_bers.append(ber_res["ber"])
                    
        mean_nmse = float(np.mean(task_nmses))
        mean_nmse_db = float(10.0 * np.log10(mean_nmse)) if mean_nmse > 1e-12 else -120.0
        mean_ber = float(np.mean(task_bers)) if task_bers else 0.0
        
        return {
            "nmse": mean_nmse,
            "nmse_db": mean_nmse_db,
            "ber": mean_ber,
        }
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping and history tracking."""
        history = {"train_loss": [], "val_loss": [], "val_nmse_db": []}
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            
            val_loss = train_loss
            val_nmse_db = 0.0
            if val_loader is not None:
                val_metrics = self.evaluate_dataloader(val_loader)
                val_loss = val_metrics["mse"]
                val_nmse_db = val_metrics["nmse_db"]
                history["val_loss"].append(val_loss)
                history["val_nmse_db"].append(val_nmse_db)
                
            if verbose and (epoch % 5 == 0 or epoch == epochs or epoch == 1):
                print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE: {train_loss:.6f} | Val NMSE: {val_nmse_db:.2f} dB")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}.")
                    break
                    
        total_time = time.time() - start_time
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            
        history["total_training_time_s"] = total_time
        return history
