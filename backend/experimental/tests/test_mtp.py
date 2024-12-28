import pytest
import torch
from ..mtp_accelerator import MTPTrainer, SpeculativeDecoder, PerformanceTracker

class MockModel:
    def __init__(self):
        self.vocab_size = 32000
        
    def __call__(self, input_ids):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        # Mock logits with random values
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        return type('ModelOutput', (), {'logits': logits})()

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()

def test_mtp_trainer_initialization():
    model = MockModel()
    tokenizer = MockTokenizer()
    trainer = MTPTrainer(model, tokenizer, mtp_sequence_length=4)
    
    assert trainer.model == model
    assert trainer.tokenizer == tokenizer
    assert trainer.mtp_sequence_length == 4

def test_prepare_mtp_batch():
    model = MockModel()
    tokenizer = MockTokenizer()
    trainer = MTPTrainer(model, tokenizer, mtp_sequence_length=2)
    
    # Create sample input
    batch_size = 2
    seq_len = 5
    input_ids = torch.randint(1, 1000, (batch_size, seq_len))
    
    # Get MTP inputs and targets
    mtp_inputs, mtp_targets = trainer.prepare_mtp_batch(input_ids)
    
    # Check shapes
    max_pos = seq_len - trainer.mtp_sequence_length
    assert mtp_inputs.size() == (max_pos, batch_size, seq_len)
    assert mtp_targets.size() == (batch_size * max_pos, trainer.mtp_sequence_length)

def test_compute_mtp_loss():
    model = MockModel()
    tokenizer = MockTokenizer()
    trainer = MTPTrainer(model, tokenizer, mtp_sequence_length=2)
    
    # Create sample inputs
    batch_size = 2
    max_pos = 3
    vocab_size = 1000
    logits = torch.randn(batch_size * max_pos, 2, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size * max_pos, 2))
    
    # Compute loss
    loss = trainer.compute_mtp_loss(logits, targets)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor

def test_speculative_decoder():
    model = MockModel()
    tokenizer = MockTokenizer()
    decoder = SpeculativeDecoder(model, tokenizer, draft_steps=2)
    
    # Create sample input
    input_ids = torch.randint(1, 1000, (1, 5))
    
    # Generate tokens
    generated = decoder.generate(input_ids, max_length=10)
    
    assert isinstance(generated, list)
    assert all(isinstance(token, int) for token in generated)

def test_performance_tracker():
    tracker = PerformanceTracker()
    
    # Track training metrics
    tracker.track_training('baseline', epoch=1, loss=2.5, duration=10.0)
    tracker.track_training('mtp', epoch=1, loss=2.0, duration=12.0)
    
    # Track inference metrics
    tracker.track_inference('baseline', tokens_generated=100, duration=5.0)
    tracker.track_inference('mtp', tokens_generated=100, duration=3.0)
    
    # Get summary
    summary = tracker.get_summary()
    
    assert 'training' in summary
    assert 'inference' in summary
    assert isinstance(summary['training']['loss_reduction'], str)
    assert isinstance(summary['inference']['speedup'], str)

def test_mtp_end_to_end():
    model = MockModel()
    tokenizer = MockTokenizer()
    trainer = MTPTrainer(model, tokenizer, mtp_sequence_length=2)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters()) if hasattr(model, 'parameters') else None
    
    # Create sample input
    input_ids = torch.randint(1, 1000, (2, 5))
    
    # Perform training step
    if optimizer is not None:
        loss = trainer.train_step(input_ids, optimizer)
        assert isinstance(loss, float)
