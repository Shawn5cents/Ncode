"""
Test configuration and shared fixtures for MTP accelerator tests
"""

import pytest
import torch
from typing import Dict, Any

class MockModel:
    """Mock model for testing"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __call__(self, input_ids: torch.Tensor) -> Any:
        """Mock forward pass"""
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        # Generate random logits
        logits = torch.randn(batch_size, seq_len, self.vocab_size, device=self.device)
        return type('ModelOutput', (), {'logits': logits})()
    
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Mock generation"""
        batch_size = input_ids.size(0)
        max_length = kwargs.get('max_length', input_ids.size(1) + 20)
        # Generate random token sequence
        return torch.randint(
            1, self.vocab_size, 
            (batch_size, max_length),
            device=self.device
        )
    
    def parameters(self):
        """Mock parameters for optimizer"""
        return [torch.randn(100, 100, requires_grad=True)]
    
    def train(self):
        """Mock train mode"""
        pass
    
    def eval(self):
        """Mock eval mode"""
        pass

class MockTokenizer:
    """Mock tokenizer for testing"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
    
    def __call__(self, texts, **kwargs) -> Dict[str, torch.Tensor]:
        """Mock tokenization"""
        if isinstance(texts, str):
            texts = [texts]
        
        max_length = kwargs.get('max_length', 64)
        # Generate random token IDs
        batch_size = len(texts)
        input_ids = torch.randint(1, self.vocab_size, (batch_size, max_length))
        
        if kwargs.get('padding', False):
            # Add padding
            attention_mask = torch.ones_like(input_ids)
            attention_mask[:, -5:] = 0  # Last 5 tokens are padding
            input_ids[:, -5:] = self.pad_token_id
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        return {'input_ids': input_ids}
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Mock decoding"""
        return f"Decoded text for {len(token_ids)} tokens"

@pytest.fixture
def mock_model():
    """Fixture providing a mock model"""
    return MockModel()

@pytest.fixture
def mock_tokenizer():
    """Fixture providing a mock tokenizer"""
    return MockTokenizer()

@pytest.fixture
def sample_input_ids(mock_tokenizer):
    """Fixture providing sample input IDs"""
    texts = [
        "The quick brown fox",
        "In a world where technology",
        "Once upon a time"
    ]
    return mock_tokenizer(texts, padding=True)['input_ids']

@pytest.fixture
def optimizer(mock_model):
    """Fixture providing an optimizer"""
    return torch.optim.Adam(mock_model.parameters())

@pytest.fixture
def device():
    """Fixture providing the compute device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Skip slow tests unless explicitly requested
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip GPU tests if no GPU available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

def pytest_addoption(parser):
    """Add custom pytest options"""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
