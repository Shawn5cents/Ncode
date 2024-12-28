import pytest
import pytest_asyncio
import logging
import tempfile
import shutil
import asyncio
import sys
from pathlib import Path

# Configure asyncio for Windows if needed
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@pytest_asyncio.fixture(scope="function")
async def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def test_data_dir(temp_dir):
    """Create test data directory structure"""
    data_dir = Path(temp_dir)
    
    # Create subdirectories
    (data_dir / "conversations").mkdir(parents=True)
    (data_dir / "knowledge_base").mkdir(parents=True)
    (data_dir / "training_data" / "planning").mkdir(parents=True)
    (data_dir / "training_data" / "coding").mkdir(parents=True)
    (data_dir / "model_versions").mkdir(parents=True)
    (data_dir / "backups").mkdir(parents=True)
    
    yield data_dir

@pytest.fixture(scope="function")
def mock_ollama_response():
    """Mock Ollama API response"""
    return {
        "model": "test_model",
        "response": "Test response",
        "done": True
    }

@pytest.fixture(scope="function")
def mock_storage_metrics():
    """Mock storage metrics"""
    return {
        "total_space": 100 * 1024**3,  # 100GB
        "used_space": 60 * 1024**3,    # 60GB
        "free_space": 40 * 1024**3,    # 40GB
        "usage_percent": 60.0
    }

@pytest.fixture(scope="function")
def mock_version_data():
    """Mock version data"""
    return {
        "name": "test_model_v1",
        "base_model": "base_model",
        "created_at": "2024-01-01T00:00:00",
        "training_samples": 100,
        "metrics": {"accuracy": 0.95},
        "status": "success"
    }

@pytest.fixture(scope="function")
def mock_training_data():
    """Mock training data"""
    return {
        "prompt": "Test prompt with sufficient length for validation",
        "response": "Test response with sufficient length to pass validation checks",
        "timestamp": "2024-01-01T00:00:00",
        "metadata": {"test": True}
    }
