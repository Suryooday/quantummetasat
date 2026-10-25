"""Script to train and benchmark classical baseline models: LS, Windowed Reference, MLP, and CNN."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import get_full_config
from src.utils.seed import set_seed
from src.preprocessing.scaler import ChannelFeatureScaler
from src.data.dataset import MetaChannelDataset, SupervisedChannelDataset
from src.models.baselines import LSEstimator, WindowedLSEstimator
from src.models.mlp import ClassicalMLP
from src.models.cnn import ClassicalCNN
from src.training.supervised import SupervisedTrainer
from src.evaluation.metrics import calculate_nmse, calculate_nmse_db, compute_all_metrics
from src.evaluation.ber import BERCalculator
from src.evaluation.profiler import ModelProfiler

def evaluate_baseline_on_tasks(estimator_func, tasks, is_windowed=False):
    """Evaluate non-parametric LS/Windowed baseline on test tasks."""
    ber_calc = BERCalculator()
    task_nmses = []
    task_bers = []
    
    start_t = time.perf_counter()
    for task in tasks:
        x_in = task.X.numpy()
        pred = estimator_func(x_in)
        true = task.Y.numpy()
        
        nmse = calculate_nmse(true, pred)
        task_nmses.append(nmse)
        
        if task.x_raw is not None and task.y_raw is not None:
            W = 9
            y_aligned = task.y_raw[W - 1 :]
            x_aligned = task.x_raw[W - 1 :]
            ber_res = ber_calc.equalize_and_decode(y_aligned, pred, x_aligned)
            task_bers.append(ber_res["ber"])
            
    latency_ms = ((time.perf_counter() - start_t) / (len(tasks) * len(tasks[0].X))) * 1000.0
    
    mean_nmse = float(np.mean(task_nmses))
    mean_nmse_db = float(10.0 * np.log10(mean_nmse)) if mean_nmse > 1e-12 else -120.0
    mean_ber = float(np.mean(task_bers)) if task_bers else 0.0
    
    return {
        "nmse": mean_nmse,
        "nmse_db": mean_nmse_db,
        "ber": mean_ber,
        "per_sample_latency_ms": float(latency_ms),
        "total_params": 0,
        "trainable_params": 0,
    }

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate classical channel estimation models.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/classical", help="Directory to save experiment results")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs for neural models")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "train"))
    val_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "val"))
    test_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "test"))
    
    # 1. Fit scaler on training data ONLY to prevent leakage
    scaler = ChannelFeatureScaler()
    train_ds = SupervisedChannelDataset(train_meta.tasks, scaler=scaler)
    val_ds = SupervisedChannelDataset(val_meta.tasks, scaler=scaler)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    results = {}
    
    # -------------------------------------------------------------
    # 1. Least Squares (LS) Baseline
    # -------------------------------------------------------------
    print("\n[1/4] Evaluating Least Squares (LS) Baseline...")
    ls_model = LSEstimator()
    ls_results = evaluate_baseline_on_tasks(ls_model.estimate_from_features, test_meta.tasks)
    results["LS"] = ls_results
    print(f"  LS -> NMSE: {ls_results['nmse']:.4f} ({ls_results['nmse_db']:.2f} dB) | BER: {ls_results['ber']:.4e} | Latency: {ls_results['per_sample_latency_ms']:.4f} ms")
    
    # -------------------------------------------------------------
    # 2. Windowed LS Reference
    # -------------------------------------------------------------
    print("\n[2/4] Evaluating Windowed Reference Baseline...")
    w_ls_model = WindowedLSEstimator(window_size=9)
    w_ls_results = evaluate_baseline_on_tasks(w_ls_model.estimate_from_features, test_meta.tasks, is_windowed=True)
    results["Windowed_Reference"] = w_ls_results
    print(f"  Windowed Reference -> NMSE: {w_ls_results['nmse']:.4f} ({w_ls_results['nmse_db']:.2f} dB) | BER: {w_ls_results['ber']:.4e} | Latency: {w_ls_results['per_sample_latency_ms']:.4f} ms")
    
    # -------------------------------------------------------------
    # 3. Classical Supervised MLP
    # -------------------------------------------------------------
    print("\n[3/4] Training Classical MLP...")
    mlp = ClassicalMLP(input_dim=36, hidden_dims=[128, 128], output_dim=2)
    mlp_trainer = SupervisedTrainer(mlp, learning_rate=1e-3, weight_decay=1e-5)
    mlp_history = mlp_trainer.fit(train_loader, val_loader, epochs=args.epochs, early_stopping_patience=10, verbose=True)
    
    # Evaluate MLP on test tasks (using scaler)
    mlp_eval = mlp_trainer.evaluate_tasks(test_meta.tasks, scaler=scaler)
    mlp_prof = ModelProfiler.count_parameters(mlp)
    sample_tensor = torch.randn(1, 36)
    mlp_latency = ModelProfiler.measure_inference_latency(mlp, sample_tensor)
    
    results["MLP"] = {
        **mlp_eval,
        **mlp_prof,
        **mlp_latency,
        "training_time_s": mlp_history["total_training_time_s"],
    }
    torch.save(mlp.state_dict(), os.path.join(args.output_dir, "mlp_best.pt"))
    print(f"  MLP -> NMSE: {mlp_eval['nmse']:.4f} ({mlp_eval['nmse_db']:.2f} dB) | BER: {mlp_eval['ber']:.4e} | Params: {mlp_prof['total_params']} | Latency: {mlp_latency['per_sample_latency_ms']:.4f} ms")
    
    # -------------------------------------------------------------
    # 4. Classical Supervised CNN
    # -------------------------------------------------------------
    print("\n[4/4] Training Classical 1D CNN...")
    cnn = ClassicalCNN(input_channels=4, window_size=9, conv_channels=[32, 64], fc_hidden=64, output_dim=2)
    cnn_trainer = SupervisedTrainer(cnn, learning_rate=1e-3, weight_decay=1e-5)
    cnn_history = cnn_trainer.fit(train_loader, val_loader, epochs=args.epochs, early_stopping_patience=10, verbose=True)
    
    cnn_eval = cnn_trainer.evaluate_tasks(test_meta.tasks, scaler=scaler)
    cnn_prof = ModelProfiler.count_parameters(cnn)
    cnn_latency = ModelProfiler.measure_inference_latency(cnn, sample_tensor)
    
    results["CNN"] = {
        **cnn_eval,
        **cnn_prof,
        **cnn_latency,
        "training_time_s": cnn_history["total_training_time_s"],
    }
    torch.save(cnn.state_dict(), os.path.join(args.output_dir, "cnn_best.pt"))
    print(f"  CNN -> NMSE: {cnn_eval['nmse']:.4f} ({cnn_eval['nmse_db']:.2f} dB) | BER: {cnn_eval['ber']:.4e} | Params: {cnn_prof['total_params']} | Latency: {cnn_latency['per_sample_latency_ms']:.4f} ms")
    
    # Save master summary JSON
    with open(os.path.join(args.output_dir, "classical_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n==========================================================================================")
    print("CLASSICAL BASELINE COMPARISON ON UNSEEN TEST TASKS")
    print("==========================================================================================")
    print(f"{'Model':<20} | {'Parameters':<12} | {'NMSE':<10} | {'NMSE (dB)':<12} | {'BER':<12} | {'Latency (ms)':<12}")
    print("-" * 90)
    for model_name, res in results.items():
        params = res.get("total_params", 0)
        p_str = f"{params:,}" if params > 0 else "—"
        lat = res.get("per_sample_latency_ms", 0.0)
        print(f"{model_name:<20} | {p_str:<12} | {res['nmse']:<10.4f} | {res['nmse_db']:<12.2f} | {res['ber']:<12.4e} | {lat:<12.4f}")
    print("==========================================================================================\n")

if __name__ == "__main__":
    main()
