"""
Test suite for the Multi-Token Prediction (MTP) Accelerator

This package contains tests for:
1. MTP training objective
2. Speculative decoding
3. Performance tracking
4. Integration tests

Test organization:
- conftest.py: Shared fixtures and configuration
- test_mtp.py: Core functionality tests

Usage:
    # Run all tests
    pytest backend/experimental/tests/

    # Run specific test file
    pytest backend/experimental/tests/test_mtp.py

    # Run specific test
    pytest backend/experimental/tests/test_mtp.py::test_mtp_end_to_end

    # Run with coverage
    pytest --cov=backend.experimental backend/experimental/tests/

    # Skip slow tests
    pytest -m "not slow" backend/experimental/tests/

    # Skip GPU tests
    pytest -m "not gpu" backend/experimental/tests/

Available test markers:
- slow: marks tests as slow (deselect with '-m "not slow"')
- gpu: marks tests that require GPU (deselect with '-m "not gpu"')

Configuration options:
--runslow: Include slow tests in test run
"""

# Version should match the main package
from ... import __version__

# Import test utilities for easier access
from .conftest import MockModel, MockTokenizer
