"""Script to train and benchmark the Fixed Hybrid VQC baseline (Supervised without MAML)."""
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
from src.models.hybrid_model import HybridVQC
from src.training.supervised import SupervisedTrainer
from src.evaluation.profiler import ModelProfiler

def main():
    parser = argparse.ArgumentParser(description="Train Fixed Hybrid VQC baseline.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/vqc", help="Directory to save experiment results")
    parser.add_argument("--num_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of variational layers")
    parser.add_argument("--epochs", type=int, default=25, help="Supervised training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading datasets for Fixed VQC...")
    train_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "train"))
    val_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "val"))
    test_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "test"))
    
    # Fit scaler on train tasks
    scaler = ChannelFeatureScaler()
    train_ds = SupervisedChannelDataset(train_meta.tasks, scaler=scaler)
    val_ds = SupervisedChannelDataset(val_meta.tasks, scaler=scaler)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nInitializing Hybrid VQC ({args.num_qubits} qubits, {args.n_layers} layers)...")
    vqc = HybridVQC(
        input_dim=36,
        num_qubits=args.num_qubits,
        n_layers=args.n_layers,
        hidden_encoder_dim=16,
        output_dim=2,
        entanglement="circular",
    )
    
    breakdown = vqc.get_parameter_breakdown()
    print("Quantum Parameter Efficiency Breakdown:")
    print(f"  • Total Parameters: {breakdown['total_parameters']}")
    print(f"  • Quantum Parameters: {breakdown['quantum_parameters']}")
    print(f"  • Classical Parameters: {breakdown['classical_parameters']}")
    print(f"  • Quantum Parameter Ratio: {breakdown['quantum_ratio']*100:.2f}%")
    print(f"  • Circuit Depth: {breakdown['circuit_depth']}")
    
    trainer = SupervisedTrainer(vqc, learning_rate=args.learning_rate, weight_decay=1e-5)
    print("\nTraining Fixed VQC with Supervised Learning...")
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs, early_stopping_patience=8, verbose=True)
    
    # Evaluate on unseen test tasks
    test_eval = trainer.evaluate_tasks(test_meta.tasks, scaler=scaler)
    sample_tensor = torch.randn(1, 36)
    latency_stats = ModelProfiler.measure_inference_latency(vqc, sample_tensor, num_trials=50)
    
    results = {
        "model_name": "Fixed VQC",
        **test_eval,
        **breakdown,
        **latency_stats,
        "training_time_s": history["total_training_time_s"],
    }
    
    torch.save(vqc.state_dict(), os.path.join(args.output_dir, "vqc_fixed_best.pt"))
    with open(os.path.join(args.output_dir, "vqc_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n==========================================================================================")
    print("FIXED VQC EVALUATION ON UNSEEN TEST TASKS")
    print("==========================================================================================")
    print(f"NMSE: {test_eval['nmse']:.4f} ({test_eval['nmse_db']:.2f} dB) | BER: {test_eval['ber']:.4e} | Params: {breakdown['total_parameters']} (Q={breakdown['quantum_parameters']}) | Latency: {latency_stats['per_sample_latency_ms']:.4f} ms")
    print("==========================================================================================\n")

if __name__ == "__main__":
    main()
