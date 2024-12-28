"""
Tests for utility functions
"""

import pytest
import torch
import tempfile
from pathlib import Path
from ..utils import (
    setup_device,
    validate_model_inputs,
    measure_time,
    save_metrics,
    load_metrics,
    get_memory_stats,
    optimize_tensor_memory,
    batch_encode_texts,
    validate_outputs
)

def test_setup_device():
    """Test device setup"""
    device = setup_device(require_gpu=False)
    assert isinstance(device, torch.device)
    assert device.type in ['cuda', 'cpu']
    
    # Test GPU requirement
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            setup_device(require_gpu=True)

def test_validate_model_inputs():
    """Test input validation"""
    # Valid input
    valid_input = torch.randint(0, 1000, (2, 10))
    validate_model_inputs(valid_input)
    
    # Invalid shape
    invalid_shape = torch.randint(0, 1000, (2, 10, 5))
    with pytest.raises(ValueError):
        validate_model_inputs(invalid_shape)
    
    # Invalid type
    invalid_type = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(ValueError):
        validate_model_inputs(invalid_type)
    
    # Test length constraints
    validate_model_inputs(valid_input, min_length=5, max_length=15)
    
    with pytest.raises(ValueError):
        validate_model_inputs(valid_input, min_length=20)
    
    with pytest.raises(ValueError):
        validate_model_inputs(valid_input, max_length=5)

@measure_time
def dummy_function():
    """Dummy function for testing measure_time decorator"""
    return "test"

def test_measure_time():
    """Test time measurement decorator"""
    result = dummy_function()
    assert result == "test"

def test_metrics_save_load():
    """Test metrics saving and loading"""
    metrics = {
        'loss': 0.5,
        'accuracy': 0.95,
        'nested': {
            'value': 42
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        
        # Test saving
        save_metrics(metrics, save_dir)
        assert (save_dir / "metrics.json").exists()
        
        # Test loading
        loaded_metrics = load_metrics(save_dir)
        assert loaded_metrics == metrics
        
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            load_metrics(save_dir, "nonexistent.json")

def test_get_memory_stats():
    """Test memory statistics"""
    stats = get_memory_stats()
    assert isinstance(stats, dict)
    assert 'cpu_allocated' in stats
    assert 'cpu_cached' in stats
    
    if torch.cuda.is_available():
        assert 'gpu_allocated' in stats
        assert 'gpu_cached' in stats

def test_optimize_tensor_memory():
    """Test tensor memory optimization"""
    # Test float32 to float16 conversion
    tensor = torch.randn(100, 100, dtype=torch.float32)
    optimized = optimize_tensor_memory(tensor)
    assert optimized.dtype == torch.float16
    
    # Test explicit dtype
    tensor = torch.randn(100, 100)
    optimized = optimize_tensor_memory(tensor, dtype=torch.float64)
    assert optimized.dtype == torch.float64
    
    # Test contiguous memory
    tensor = torch.randn(100, 100).transpose(0, 1)
    assert not tensor.is_contiguous()
    optimized = optimize_tensor_memory(tensor)
    assert optimized.is_contiguous()

def test_batch_encode_texts(mock_tokenizer):
    """Test batch text encoding"""
    texts = [
        "First text",
        "Second text",
        "Third text",
        "Fourth text"
    ]
    
    # Test with default batch size
    batches = batch_encode_texts(texts, mock_tokenizer)
    assert isinstance(batches, list)
    assert all(isinstance(batch, torch.Tensor) for batch in batches)
    
    # Test with custom batch size
    batches = batch_encode_texts(texts, mock_tokenizer, batch_size=2)
    assert len(batches) == 2
    
    # Test with max length
    batches = batch_encode_texts(texts, mock_tokenizer, max_length=10)
    assert all(batch.size(1) <= 10 for batch in batches)

def test_validate_outputs(mock_tokenizer):
    """Test output validation"""
    # Create sample outputs
    baseline = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mtp = torch.tensor([[1, 2, 3], [4, 5, 7]])
    
    # Test validation
    similarity, metrics = validate_outputs(baseline, mtp, mock_tokenizer)
    
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1
    assert isinstance(metrics, dict)
    assert 'avg_token_overlap' in metrics
    assert 'avg_length_diff' in metrics

@pytest.mark.slow
def test_large_batch_processing():
    """Test processing of large batches"""
    # Create large tensor
    large_tensor = torch.randn(1000, 1000)
    
    # Test memory optimization
    optimized = optimize_tensor_memory(large_tensor)
    assert optimized.dtype == torch.float16
    assert optimized.is_contiguous()
    
    # Test memory stats
    stats_before = get_memory_stats()
    del optimized
    stats_after = get_memory_stats()
    
    if torch.cuda.is_available():
        assert stats_before['gpu_allocated'] >= stats_after['gpu_allocated']

@pytest.mark.gpu
def test_gpu_specific_operations():
    """Test GPU-specific operations"""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    device = setup_device(require_gpu=True)
    assert device.type == 'cuda'
    
    # Test GPU memory handling
    tensor = torch.randn(100, 100, device=device)
    stats = get_memory_stats()
    assert stats['gpu_allocated'] > 0
