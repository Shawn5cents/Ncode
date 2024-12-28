"""
Utility functions for the MTP Accelerator module
"""

import torch
import time
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_device(require_gpu: bool = False) -> torch.device:
    """
    Set up and validate compute device
    
    Args:
        require_gpu: Whether GPU is required
        
    Returns:
        torch.device: Selected compute device
        
    Raises:
        RuntimeError: If GPU is required but not available
    """
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required but not available")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return device

def validate_model_inputs(
    input_ids: torch.Tensor,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> None:
    """
    Validate model input tensors
    
    Args:
        input_ids: Input token IDs
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        
    Raises:
        ValueError: If inputs don't meet requirements
    """
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("input_ids must be a torch.Tensor")
    
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D (batch_size, seq_len), got {input_ids.dim()}D")
    
    seq_len = input_ids.size(1)
    if min_length and seq_len < min_length:
        raise ValueError(f"Sequence length {seq_len} below minimum {min_length}")
    
    if max_length and seq_len > max_length:
        raise ValueError(f"Sequence length {seq_len} above maximum {max_length}")

def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper

def save_metrics(
    metrics: Dict[str, Any],
    save_dir: Path,
    filename: str = "metrics.json"
) -> None:
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        save_dir: Directory to save metrics
        filename: Name of metrics file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {save_path}")

def load_metrics(
    save_dir: Path,
    filename: str = "metrics.json"
) -> Dict[str, Any]:
    """
    Load metrics from JSON file
    
    Args:
        save_dir: Directory containing metrics
        filename: Name of metrics file
        
    Returns:
        Dictionary of metrics
    """
    load_path = save_dir / filename
    
    if not load_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {load_path}")
    
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    logger.info(f"Metrics loaded from {load_path}")
    return metrics

def get_memory_stats() -> Dict[str, float]:
    """
    Get current memory usage statistics
    
    Returns:
        Dictionary of memory statistics in GB
    """
    stats = {
        'cpu_allocated': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        'cpu_cached': torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        stats.update({
            'gpu_allocated': torch.cuda.memory_allocated() / 1e9,
            'gpu_cached': torch.cuda.memory_reserved() / 1e9
        })
    
    return stats

def optimize_tensor_memory(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Optimize tensor memory usage
    
    Args:
        tensor: Input tensor
        dtype: Target dtype for optimization
        
    Returns:
        Optimized tensor
    """
    if dtype is None:
        # Use lower precision for efficiency if possible
        if tensor.dtype == torch.float32:
            dtype = torch.float16
    
    # Move to contiguous memory
    tensor = tensor.contiguous()
    
    # Convert dtype if specified
    if dtype is not None:
        tensor = tensor.to(dtype)
    
    return tensor

def batch_encode_texts(
    texts: List[str],
    tokenizer: Any,
    batch_size: int = 32,
    max_length: Optional[int] = None,
    device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    """
    Encode texts in batches to manage memory
    
    Args:
        texts: List of input texts
        tokenizer: Tokenizer instance
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        device: Target device for tensors
        
    Returns:
        List of encoded tensors
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoded_batches = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True if max_length else False,
            max_length=max_length,
            return_tensors="pt"
        )
        encoded_batches.append(encodings['input_ids'].to(device))
    
    return encoded_batches

@measure_time
def validate_outputs(
    baseline_outputs: torch.Tensor,
    mtp_outputs: torch.Tensor,
    tokenizer: Any
) -> Tuple[float, Dict[str, float]]:
    """
    Validate and compare model outputs
    
    Args:
        baseline_outputs: Outputs from baseline model
        mtp_outputs: Outputs from MTP model
        tokenizer: Tokenizer for decoding
        
    Returns:
        Tuple of (similarity_score, metrics_dict)
    """
    # Decode outputs
    baseline_texts = [tokenizer.decode(output) for output in baseline_outputs]
    mtp_texts = [tokenizer.decode(output) for output in mtp_outputs]
    
    # Calculate metrics
    metrics = {}
    
    # Length difference
    length_diffs = [len(m) - len(b) for m, b in zip(mtp_texts, baseline_texts)]
    metrics['avg_length_diff'] = sum(length_diffs) / len(length_diffs)
    
    # Token overlap
    overlaps = []
    for baseline, mtp in zip(baseline_outputs, mtp_outputs):
        overlap = len(set(baseline.tolist()) & set(mtp.tolist()))
        overlaps.append(overlap / max(len(baseline), len(mtp)))
    metrics['avg_token_overlap'] = sum(overlaps) / len(overlaps)
    
    # Overall similarity score (0-1)
    similarity_score = metrics['avg_token_overlap'] * (
        1 - abs(metrics['avg_length_diff']) / max(
            max(len(t) for t in baseline_texts),
            max(len(t) for t in mtp_texts)
        )
    )
    
    return similarity_score, metrics
