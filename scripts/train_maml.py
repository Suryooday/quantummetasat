"""Script to train and benchmark Classical MAML on episodic LEO channel tasks."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import argparse
import numpy as np
import torch

from src.utils.config import get_full_config
from src.utils.seed import set_seed
from src.preprocessing.scaler import ChannelFeatureScaler
from src.data.dataset import MetaChannelDataset, SupervisedChannelDataset
from src.models.mlp import ClassicalMLP
from src.meta_learning.maml import MAML
from src.training.maml_trainer import MAMLTrainer
from src.evaluation.profiler import ModelProfiler

def main():
    parser = argparse.ArgumentParser(description="Train Classical MAML on episodic channel tasks.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/maml", help="Directory to save experiment results")
    parser.add_argument("--meta_epochs", type=int, default=40, help="Outer meta-training epochs")
    parser.add_argument("--tasks_per_epoch", type=int, default=25, help="Tasks per epoch")
    parser.add_argument("--meta_batch_size", type=int, default=4, help="Meta batch size")
    parser.add_argument("--k_support", type=int, default=10, help="Support set pilot size")
    parser.add_argument("--q_query", type=int, default=30, help="Query set evaluation size")
    parser.add_argument("--inner_lr", type=float, default=0.01, help="Inner-loop learning rate alpha")
    parser.add_argument("--outer_lr", type=float, default=0.001, help="Outer-loop learning rate beta")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading datasets for MAML...")
    train_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "train"))
    val_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "val"))
    test_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "test"))
    
    # Fit scaler on train tasks
    scaler = ChannelFeatureScaler()
    _ = SupervisedChannelDataset(train_meta.tasks, scaler=scaler)
    
    # Scale dataset tasks
    for t in train_meta.tasks:
        t.X = torch.tensor(scaler.transform(t.X), dtype=torch.float32)
    for t in val_meta.tasks:
        t.X = torch.tensor(scaler.transform(t.X), dtype=torch.float32)
    for t in test_meta.tasks:
        t.X = torch.tensor(scaler.transform(t.X), dtype=torch.float32)
        
    print("\nInitializing Classical MAML Model (MLP base learner)...")
    base_mlp = ClassicalMLP(input_dim=36, hidden_dims=[128, 128], output_dim=2)
    maml = MAML(base_mlp, inner_lr=args.inner_lr, outer_lr=args.outer_lr, first_order=False)
    trainer = MAMLTrainer(maml)
    
    print("Meta-Training Classical MAML...")
    history = trainer.train(
        train_dataset=train_meta,
        val_dataset=val_meta,
        meta_epochs=args.meta_epochs,
        tasks_per_epoch=args.tasks_per_epoch,
        meta_batch_size=args.meta_batch_size,
        k_support=args.k_support,
        q_query=args.q_query,
        inner_steps=5,
        verbose=True,
    )
    
    # Save meta-trained weights
    torch.save(base_mlp.state_dict(), os.path.join(args.output_dir, "maml_mlp_best.pt"))
    
    print("\nEvaluating Few-Shot Adaptation across Pilot Counts and Gradient Steps on UNSEEN Test Tasks...")
    eval_results = {}
    pilot_configs = [5, 10, 20]
    step_configs = [0, 1, 3, 5, 10]
    
    for k in pilot_configs:
        print(f"\n--- Evaluating K = {k} Pilots ---")
        step_res = trainer.evaluate_few_shot(
            test_tasks=test_meta.tasks,
            k_support=k,
            q_query=50,
            adaptation_steps=step_configs,
            inner_lr=args.inner_lr,
        )
        eval_results[f"K_{k}"] = step_res
        for s, metrics in step_res.items():
            print(f"  Step {s:02d} -> NMSE: {metrics['nmse']:.4f} ({metrics['nmse_db']:.2f} dB) | BER: {metrics['ber']:.4e}")
            
    # Measure adaptation latency
    x_s_samp, y_s_samp, _, _ = test_meta.tasks[0].sample_support_query(10, 10)
    t0 = time.perf_counter()
    _ = maml.adapt(x_s_samp, y_s_samp, num_steps=5)
    adapt_latency_ms = (time.perf_counter() - t0) * 1000.0
    
    prof = ModelProfiler.count_parameters(base_mlp)
    
    final_payload = {
        "model_name": "Classical MAML",
        "parameters": prof["total_params"],
        "training_time_s": history["total_meta_train_time_s"],
        "adaptation_latency_ms": adapt_latency_ms,
        "few_shot_results": eval_results,
        "history": history,
    }
    
    with open(os.path.join(args.output_dir, "maml_results.json"), "w") as f:
        json.dump(final_payload, f, indent=2)
        
    print("\nClassical MAML training & evaluation finished successfully.")

if __name__ == "__main__":
    main()
