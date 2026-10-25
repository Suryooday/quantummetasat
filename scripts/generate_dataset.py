"""Script to generate the standardized train, validation, and test datasets for QuantumMeta-Sat."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from src.utils.config import get_full_config
from src.data.generator import ChannelTaskGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate QuantumMeta-Sat datasets.")
    parser.add_argument("--config_dir", type=str, default="configs", help="Path to configuration directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for generated data")
    args = parser.parse_args()
    
    cfg = get_full_config(args.config_dir)
    ds_cfg = cfg.get("dataset_config", {}).get("dataset", {})
    physics_cfg = cfg.get("physics", {})
    
    generator = ChannelTaskGenerator(
        window_size=cfg.get("data", {}).get("window_size", 9),
        earth_radius_m=physics_cfg.get("earth_radius_m", 6378137.0),
        satellite_altitude_m=physics_cfg.get("satellite_altitude_m", 600000.0),
        carrier_freq_hz=physics_cfg.get("carrier_freq_hz", 2.0e9),
        sampling_rate_hz=physics_cfg.get("sampling_rate_hz", 1.0e6),
        normalize_path_loss=cfg.get("channel", {}).get("normalize_path_loss_amplitude", True),
    )
    
    # 1. Train Split
    train_cfg = ds_cfg.get("train", {})
    print(f"Generating {train_cfg.get('num_tasks', 80)} training tasks...")
    train_tasks = generator.generate_task_split(
        num_tasks=train_cfg.get("num_tasks", 80),
        split_name="train",
        elevation_range=tuple(train_cfg.get("elevation_deg_range", [20.0, 75.0])),
        snr_range=tuple(train_cfg.get("snr_db_range", [5.0, 20.0])),
        environments=train_cfg.get("environments", ["infrequent_light", "average", "frequent_heavy"]),
        doppler_freq_range=tuple(train_cfg.get("doppler_freq_range_hz", [-40000.0, 40000.0])),
        samples_per_task=train_cfg.get("samples_per_task", 200),
        seed=train_cfg.get("seed", 1001),
    )
    
    # 2. Validation Split
    val_cfg = ds_cfg.get("val", {})
    print(f"Generating {val_cfg.get('num_tasks', 20)} validation tasks...")
    val_tasks = generator.generate_task_split(
        num_tasks=val_cfg.get("num_tasks", 20),
        split_name="val",
        elevation_range=tuple(val_cfg.get("elevation_deg_range", [25.0, 70.0])),
        snr_range=tuple(val_cfg.get("snr_db_range", [5.0, 20.0])),
        environments=val_cfg.get("environments", ["infrequent_light", "average", "frequent_heavy"]),
        doppler_freq_range=tuple(val_cfg.get("doppler_freq_range_hz", [-35000.0, 35000.0])),
        samples_per_task=val_cfg.get("samples_per_task", 200),
        seed=val_cfg.get("seed", 2001),
    )
    
    # 3. Test Split
    test_cfg = ds_cfg.get("test", {})
    print(f"Generating {test_cfg.get('num_tasks', 30)} test tasks (unseen elevations and SNR test points)...")
    test_tasks = generator.generate_task_split(
        num_tasks=test_cfg.get("num_tasks", 30),
        split_name="test",
        elevation_list=test_cfg.get("elevation_deg_list", [20.0, 30.0, 45.0, 60.0, 75.0, 90.0]),
        snr_list=test_cfg.get("snr_db_list", [5.0, 10.0, 15.0, 20.0]),
        environments=test_cfg.get("environments", ["infrequent_light", "average", "frequent_heavy"]),
        doppler_freq_range=tuple(test_cfg.get("doppler_freq_range_hz", [-48000.0, 48000.0])),
        samples_per_task=test_cfg.get("samples_per_task", 200),
        seed=test_cfg.get("seed", 3001),
    )
    
    generator.save_dataset(args.output_dir, train_tasks, val_tasks, test_tasks)
    print("Dataset generation completed successfully.")

if __name__ == "__main__":
    main()
