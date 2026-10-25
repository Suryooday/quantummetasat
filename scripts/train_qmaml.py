"""Script to meta-train and benchmark Q-MAML / QuantumMeta-Sat (Hybrid VQC + MAML)."""
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
from src.models.hybrid_model import HybridVQC
from src.meta_learning.maml import MAML
from src.training.maml_trainer import MAMLTrainer
from src.evaluation.profiler import ModelProfiler

def main():
    parser = argparse.ArgumentParser(description="Train Q-MAML / QuantumMeta-Sat.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/qmaml", help="Directory to save experiment results")
    parser.add_argument("--num_qubits", type=int, default=4, help="Number of qubits (4 or 8)")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of variational layers")
    parser.add_argument("--meta_epochs", type=int, default=30, help="Outer meta-training epochs")
    parser.add_argument("--tasks_per_epoch", type=int, default=20, help="Tasks per epoch")
    parser.add_argument("--meta_batch_size", type=int, default=4, help="Meta batch size")
    parser.add_argument("--k_support", type=int, default=10, help="Support set pilot size")
    parser.add_argument("--q_query", type=int, default=30, help="Query set evaluation size")
    parser.add_argument("--inner_lr", type=float, default=0.02, help="Inner-loop learning rate alpha")
    parser.add_argument("--outer_lr", type=float, default=0.002, help="Outer-loop learning rate beta")
    parser.add_argument("--inner_steps_train", type=int, default=3, help="Inner-loop steps during meta-training")
    parser.add_argument("--first_order", action="store_true", help="Use first-order MAML approximation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading datasets for Q-MAML...")
    train_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "train"))
    val_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "val"))
    test_meta = MetaChannelDataset.load_from_dir(os.path.join(args.data_dir, "test"))
    
    # Fit scaler on train tasks
    scaler = ChannelFeatureScaler()
    _ = SupervisedChannelDataset(train_meta.tasks, scaler=scaler)
    
    for t in train_meta.tasks:
        t.X = torch.tensor(scaler.transform(t.X), dtype=torch.float32)
    for t in val_meta.tasks:
        t.X = torch.tensor(scaler.transform(t.X), dtype=torch.float32)
    for t in test_meta.tasks:
        t.X = torch.tensor(scaler.transform(t.X), dtype=torch.float32)
        
    print(f"\nInitializing Hybrid VQC Base Learner ({args.num_qubits} qubits, {args.n_layers} layers)...")
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
    
    qmaml = MAML(vqc, inner_lr=args.inner_lr, outer_lr=args.outer_lr, first_order=args.first_order)
    trainer = MAMLTrainer(qmaml)
    
    print("\nMeta-Training Q-MAML...")
    history = trainer.train(
        train_dataset=train_meta,
        val_dataset=val_meta,
        meta_epochs=args.meta_epochs,
        tasks_per_epoch=args.tasks_per_epoch,
        meta_batch_size=args.meta_batch_size,
        k_support=args.k_support,
        q_query=args.q_query,
        inner_steps=args.inner_steps_train,
        verbose=True,
    )
    
    model_tag = f"qmaml_{args.num_qubits}q_best.pt"
    torch.save(vqc.state_dict(), os.path.join(args.output_dir, model_tag))
    
    print("\nEvaluating Q-MAML Few-Shot Adaptation across Pilot Counts and Gradient Steps on UNSEEN Test Tasks...")
    eval_results = {}
    pilot_configs = [1, 5, 10, 20]
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
    _ = qmaml.adapt(x_s_samp, y_s_samp, num_steps=5)
    adapt_latency_ms = (time.perf_counter() - t0) * 1000.0
    
    sample_tensor = torch.randn(1, 36)
    inference_latency = ModelProfiler.measure_inference_latency(vqc, sample_tensor, num_trials=50)
    
    final_payload = {
        "model_name": f"QuantumMeta-Sat ({args.num_qubits} Qubits)",
        "num_qubits": args.num_qubits,
        "n_layers": args.n_layers,
        "breakdown": breakdown,
        "training_time_s": history["total_meta_train_time_s"],
        "adaptation_latency_ms": adapt_latency_ms,
        "inference_latency_ms": inference_latency["per_sample_latency_ms"],
        "few_shot_results": eval_results,
        "history": history,
    }
    
    out_file = f"qmaml_{args.num_qubits}q_results.json"
    with open(os.path.join(args.output_dir, out_file), "w") as f:
        json.dump(final_payload, f, indent=2)
        
    print(f"\nQ-MAML ({args.num_qubits} qubits) training & evaluation finished successfully.")

if __name__ == "__main__":
    main()
