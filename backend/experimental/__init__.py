"""
Multi-Token Prediction (MTP) Accelerator

This experimental module provides implementations for:
1. MTP training objective for improved model performance
2. Speculative decoding for faster inference
3. Performance tracking and benchmarking utilities

Example usage:
    from backend.experimental import MTPTrainer, SpeculativeDecoder, PerformanceTracker
    
    # Initialize components
    trainer = MTPTrainer(model, tokenizer, mtp_sequence_length=4)
    decoder = SpeculativeDecoder(model, tokenizer, draft_steps=4)
    tracker = PerformanceTracker()
    
    # Training with MTP objective
    loss = trainer.train_step(input_ids, optimizer)
    
    # Inference with speculative decoding
    output = decoder.generate(input_ids, max_length=100)
    
    # Track and compare performance
    summary = tracker.get_summary()
"""

from .mtp_accelerator import MTPTrainer, SpeculativeDecoder, PerformanceTracker

__all__ = [
    'MTPTrainer',
    'SpeculativeDecoder',
    'PerformanceTracker'
]

__version__ = '0.1.0'
