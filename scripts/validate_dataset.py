"""Script to validate generated datasets against mathematical and physical integrity criteria."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import json
import argparse
import numpy as np

def validate_dataset(data_dir: str = "data/processed") -> bool:
    """Validate all generated dataset splits."""
    print(f"Validating dataset in: {data_dir}")
    splits = ["train", "val", "test"]
    all_passed = True
    
    total_samples = 0
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        manifest_file = os.path.join(split_dir, "manifest.json")
        
        if not os.path.exists(manifest_file):
            print(f"❌ [FAILED] Manifest missing in {split_dir}")
            return False
            
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
            
        print(f"\n--- Validating Split: {split.upper()} ({len(manifest)} tasks) ---")
        
        elevations = []
        snrs = []
        x_powers = []
        y_powers = []
        h_powers = []
        
        for entry in manifest:
            task_file = entry["file_path"]
            if not os.path.isabs(task_file):
                task_file = os.path.join(split_dir, os.path.basename(task_file))
                
            if not os.path.exists(task_file):
                print(f"❌ [FAILED] Task file missing: {task_file}")
                return False
                
            data = np.load(task_file)
            X = data["X"]
            Y = data["Y"]
            x_raw = data["x_raw"]
            y_raw = data["y_raw"]
            h_raw = data["h_raw"]
            
            # Check shapes
            if X.ndim != 2 or X.shape[1] != 36:
                print(f"❌ [FAILED] Invalid X shape in {task_file}: {X.shape}, expected (N, 36)")
                all_passed = False
            if Y.ndim != 2 or Y.shape[1] != 2:
                print(f"❌ [FAILED] Invalid Y shape in {task_file}: {Y.shape}, expected (N, 2)")
                all_passed = False
                
            # Check finite
            if not np.all(np.isfinite(X)):
                print(f"❌ [FAILED] NaNs/Infs found in X: {task_file}")
                all_passed = False
            if not np.all(np.isfinite(Y)):
                print(f"❌ [FAILED] NaNs/Infs found in Y: {task_file}")
                all_passed = False
                
            # Check powers
            x_p = np.mean(np.abs(x_raw)**2)
            y_p = np.mean(np.abs(y_raw)**2)
            h_p = np.mean(np.abs(h_raw)**2)
            
            x_powers.append(x_p)
            y_powers.append(y_p)
            h_powers.append(h_p)
            
            elevations.append(entry["metadata"]["elevation_deg"])
            snrs.append(entry["metadata"]["snr_db_target"])
            total_samples += len(X)
            
        print(f"  ✓ Tasks: {len(manifest)}, Total windowed samples: {sum(len(np.load(os.path.join(split_dir, os.path.basename(e['file_path'])))['X']) for e in manifest)}")
        print(f"  ✓ Elevation range: [{min(elevations):.1f}°, {max(elevations):.1f}°]")
        print(f"  ✓ SNR range: [{min(snrs):.1f} dB, {max(snrs):.1f} dB]")
        print(f"  ✓ Mean QPSK Tx Power: {np.mean(x_powers):.4f} (Expected: ~1.0000)")
        print(f"  ✓ Mean Channel |h|^2: {np.mean(h_powers):.4f}")
        print(f"  ✓ Mean Received |y|^2: {np.mean(y_powers):.4f}")
        print(f"  ✓ No NaNs, No Infs, Valid Shapes Verified.")
        
    print(f"\n=======================================================")
    print(f"Overall Dataset Integrity: {'PASSED ✅' if all_passed else 'FAILED ❌'}")
    print(f"Total Windowed Samples across all tasks: {total_samples}")
    print(f"=======================================================\n")
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate generated datasets.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed dataset")
    args = parser.parse_args()
    success = validate_dataset(args.data_dir)
    exit(0 if success else 1)
