"""Unit tests for dataset generation, windowing, task sampling, and leakage verification."""
import os
import shutil
import pytest
import numpy as np
import torch

from src.data.generator import ChannelTaskGenerator
from src.data.windowing import ChannelWindowing
from src.data.dataset import TaskData, SupervisedChannelDataset, MetaChannelDataset
from src.data.task_sampler import EpisodicTaskSampler
from src.preprocessing.scaler import ChannelFeatureScaler

def test_windowing_shapes_and_values():
    """Verify windowing transformation shapes and target alignment."""
    windowing = ChannelWindowing(window_size=9)
    N = 100
    x = (np.random.choice([1, -1], size=N) + 1j * np.random.choice([1, -1], size=N)) / np.sqrt(2)
    y = x * 0.8 + 0.1 * (np.random.normal(size=N) + 1j * np.random.normal(size=N))
    h = np.ones(N, dtype=np.complex128) * (0.8 + 0.2j)
    
    X_win, Y_targets = windowing.create_windows(x, y, h)
    
    # Expected num windows: N - W + 1 = 100 - 9 + 1 = 92
    assert X_win.shape == (92, 36)
    assert Y_targets.shape == (92, 2)
    assert np.all(np.isfinite(X_win))
    assert np.all(np.isfinite(Y_targets))
    
    # Verify CNN formatting shape
    X_cnn = windowing.to_cnn_format(X_win)
    assert X_cnn.shape == (92, 4, 9)

def test_task_generator_and_isolation(tmp_path):
    """Verify task generation, serialization, and isolation between splits."""
    gen = ChannelTaskGenerator(window_size=9)
    
    train_tasks = gen.generate_task_split(
        num_tasks=5,
        split_name="train",
        elevation_range=(20.0, 70.0),
        snr_range=(5.0, 20.0),
        samples_per_task=50,
        seed=101,
    )
    
    test_tasks = gen.generate_task_split(
        num_tasks=3,
        split_name="test",
        elevation_list=[30.0, 60.0, 90.0],
        snr_list=[5.0, 15.0, 20.0],
        samples_per_task=50,
        seed=202,
    )
    
    assert len(train_tasks) == 5
    assert len(test_tasks) == 3
    
    # Verify task data shapes
    for t in train_tasks:
        assert t["X"].shape == (50, 36)
        assert t["Y"].shape == (50, 2)
        assert not np.isnan(t["X"]).any()
        assert not np.isnan(t["Y"]).any()
        
    # Verify serialization and loading
    out_dir = str(tmp_path / "test_data")
    gen.save_dataset(out_dir, train_tasks, [], test_tasks)
    
    train_meta_ds = MetaChannelDataset.load_from_dir(os.path.join(out_dir, "train"))
    assert len(train_meta_ds) == 5
    
    # Verify episodic sampler
    sampler = EpisodicTaskSampler(train_meta_ds, batch_size=2, k_support=5, q_query=15)
    meta_batch = sampler.sample_meta_batch()
    assert len(meta_batch) == 2
    for x_s, y_s, x_q, y_q, meta in meta_batch:
        assert x_s.shape == (5, 36)
        assert y_s.shape == (5, 2)
        assert x_q.shape == (15, 36)
        assert y_q.shape == (15, 2)

def test_feature_scaler_no_leakage():
    """Verify that feature scaler fits only on train data and normalizes correctly."""
    scaler = ChannelFeatureScaler()
    train_X = np.random.normal(loc=5.0, scale=2.0, size=(100, 36))
    test_X = np.random.normal(loc=5.0, scale=2.0, size=(50, 36))
    
    scaler.fit(train_X)
    assert scaler.is_fitted
    
    scaled_train = scaler.transform(train_X)
    scaled_test = scaler.transform(test_X)
    
    # Train data mean should be ~0, std ~1
    assert np.allclose(np.mean(scaled_train, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.std(scaled_train, axis=0), 1.0, atol=1e-5)
    
    # Test data should not change scaler parameters
    old_mean = scaler.mean.copy()
    _ = scaler.transform(test_X)
    assert np.array_equal(old_mean, scaler.mean)
