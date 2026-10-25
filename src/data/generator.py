"""Synthetic LEO channel dataset and task generator with physics-grounded parameterizations."""
import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

from src.channels.leo_channel import LEOChannel
from src.modulation.qpsk import QPSKModulator
from src.data.windowing import ChannelWindowing
from src.utils.seed import set_seed

class ChannelTaskGenerator:
    """Generates episodic channel estimation tasks for supervised and meta-learning pipelines.
    
    Each task T_i represents a distinct LEO pass / propagation state characterized by:
        - Elevation angle theta (deg)
        - SNR (dB)
        - Fading condition (frequent_heavy, average, infrequent_light)
        - Doppler frequency shift f_D (Hz)
        - Unique channel realization
    """
    
    def __init__(
        self,
        window_size: int = 9,
        earth_radius_m: float = 6378137.0,
        satellite_altitude_m: float = 600000.0,
        carrier_freq_hz: float = 2.0e9,
        sampling_rate_hz: float = 1.0e6,
        normalize_path_loss: bool = True,
    ):
        self.window_size = window_size
        self.windowing = ChannelWindowing(window_size=window_size)
        self.channel_sim = LEOChannel(
            earth_radius_m=earth_radius_m,
            satellite_altitude_m=satellite_altitude_m,
            carrier_freq_hz=carrier_freq_hz,
            sampling_rate_hz=sampling_rate_hz,
            normalize_path_loss=normalize_path_loss,
        )
        self.modulator = QPSKModulator()
        
    def generate_single_task(
        self,
        task_id: str,
        num_symbols: int = 208,  # Yields 200 windowed samples with W=9
        elevation_deg: float = 45.0,
        snr_db: float = 15.0,
        environment: str = "average",
        doppler_freq_hz: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a single channel estimation task.
        
        Args:
            task_id: Unique string identifier for the task.
            num_symbols: Total symbol sequence length.
            elevation_deg: Elevation angle.
            snr_db: Target SNR.
            environment: Shadowing environment.
            doppler_freq_hz: Doppler frequency.
            
        Returns:
            Dictionary containing:
                - task_id: str
                - X_windows: np.ndarray (N_samples, 36)
                - Y_targets: np.ndarray (N_samples, 2)
                - x_raw: np.ndarray (num_symbols,)
                - y_raw: np.ndarray (num_symbols,)
                - h_raw: np.ndarray (num_symbols,)
                - metadata: dict of channel properties
        """
        # 1. Transmit QPSK symbols
        x_symbols = self.modulator.generate_symbols(num_symbols)
        
        # 2. Transmit through LEO channel
        y_symbols, h_eff, meta = self.channel_sim.transmit(
            x_symbols,
            elevation_deg=elevation_deg,
            snr_db=snr_db,
            environment=environment,
            doppler_freq_hz=doppler_freq_hz,
        )
        
        # 3. Create sliding observation windows
        X_windows, Y_targets = self.windowing.create_windows(x_symbols, y_symbols, h_eff)
        
        task_dict = {
            "task_id": task_id,
            "X": X_windows,
            "Y": Y_targets,
            "x_raw": x_symbols,
            "y_raw": y_symbols,
            "h_raw": h_eff,
            "metadata": meta,
        }
        return task_dict
        
    def generate_task_split(
        self,
        num_tasks: int,
        split_name: str,
        elevation_range: Optional[Tuple[float, float]] = None,
        elevation_list: Optional[List[float]] = None,
        snr_range: Optional[Tuple[float, float]] = None,
        snr_list: Optional[List[float]] = None,
        environments: Optional[List[str]] = None,
        doppler_freq_range: Optional[Tuple[float, float]] = None,
        samples_per_task: int = 200,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Generate a complete split (train, val, or test) of isolated tasks."""
        set_seed(seed)
        if environments is None:
            environments = ["infrequent_light", "average", "frequent_heavy"]
            
        raw_symbols_needed = samples_per_task + self.window_size - 1
        tasks = []
        
        for i in range(num_tasks):
            task_id = f"{split_name}_task_{i:04d}"
            
            # Elevation selection
            if elevation_list is not None:
                elevation_deg = float(elevation_list[i % len(elevation_list)])
            elif elevation_range is not None:
                elevation_deg = float(np.random.uniform(elevation_range[0], elevation_range[1]))
            else:
                elevation_deg = 45.0
                
            # SNR selection
            if snr_list is not None:
                snr_db = float(snr_list[i % len(snr_list)])
            elif snr_range is not None:
                snr_db = float(np.random.uniform(snr_range[0], snr_range[1]))
            else:
                snr_db = 15.0
                
            # Environment selection
            env = environments[i % len(environments)]
            
            # Doppler selection
            if doppler_freq_range is not None:
                doppler_freq = float(np.random.uniform(doppler_freq_range[0], doppler_freq_range[1]))
            else:
                doppler_freq = None
                
            task = self.generate_single_task(
                task_id=task_id,
                num_symbols=raw_symbols_needed,
                elevation_deg=elevation_deg,
                snr_db=snr_db,
                environment=env,
                doppler_freq_hz=doppler_freq,
            )
            tasks.append(task)
            
        return tasks
        
    def save_dataset(
        self,
        output_dir: str,
        train_tasks: List[Dict[str, Any]],
        val_tasks: List[Dict[str, Any]],
        test_tasks: List[Dict[str, Any]],
    ) -> None:
        """Serialize dataset tasks to disk in compressed NPZ and JSON metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, tasks in [("train", train_tasks), ("val", val_tasks), ("test", test_tasks)]:
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            manifest = []
            for task in tasks:
                t_id = task["task_id"]
                file_path = os.path.join(split_dir, f"{t_id}.npz")
                np.savez_compressed(
                    file_path,
                    X=task["X"],
                    Y=task["Y"],
                    x_raw=task["x_raw"],
                    y_raw=task["y_raw"],
                    h_raw=task["h_raw"],
                )
                manifest_entry = {
                    "task_id": t_id,
                    "file_path": file_path,
                    "num_samples": len(task["X"]),
                    "metadata": task["metadata"],
                }
                manifest.append(manifest_entry)
                
            with open(os.path.join(split_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
                
        print(f"Dataset successfully saved to {output_dir}: "
              f"{len(train_tasks)} train, {len(val_tasks)} val, {len(test_tasks)} test tasks.")
