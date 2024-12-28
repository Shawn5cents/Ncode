# Contributing to MTP Accelerator

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. Install dependencies:
```bash
pip install -r backend/experimental/requirements.txt
```

2. Run tests to verify setup:
```bash
pytest backend/experimental/tests/
```

## Development Process

1. **Fork and Clone**
   - Fork the repository
   - Clone your fork locally
   - Add upstream remote: `git remote add upstream [repository-url]`

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Development Guidelines**
   - Follow PEP 8 style guide
   - Add docstrings to all functions/classes
   - Include type hints
   - Keep functions focused and modular
   - Use meaningful variable names

4. **Testing**
   - Add tests for new functionality
   - Run full test suite before submitting
   ```bash
   # Run all tests
   pytest backend/experimental/tests/
   
   # Run specific test file
   pytest backend/experimental/tests/test_mtp.py
   
   # Run with coverage
   pytest --cov=backend.experimental backend/experimental/tests/
   ```

5. **Performance Testing**
   - Run benchmarks for performance-critical changes
   ```bash
   python -m backend.experimental.benchmark
   ```
   - Include benchmark results in PR

## Code Organization

```
backend/experimental/
├── __init__.py           # Package initialization
├── mtp_accelerator.py    # Core implementation
├── utils.py             # Utility functions
├── benchmark.py         # Performance benchmarks
├── example.py          # Usage examples
├── requirements.txt    # Dependencies
├── tests/             # Test suite
│   ├── __init__.py
│   ├── conftest.py    # Test configuration
│   ├── test_mtp.py    # Core tests
│   └── test_utils.py  # Utility tests
└── .github/           # GitHub templates
```

## Pull Request Process

1. **Update Documentation**
   - Update README.md if needed
   - Add docstrings for new code
   - Update example.py if adding features

2. **Run Quality Checks**
   ```bash
   # Format code
   black backend/experimental/
   
   # Sort imports
   isort backend/experimental/
   
   # Run linter
   pylint backend/experimental/
   
   # Type checking
   mypy backend/experimental/
   ```

3. **Create Pull Request**
   - Fill out PR template completely
   - Link related issues
   - Include test results
   - Add benchmark results if relevant

4. **Review Process**
   - Address reviewer comments
   - Keep PR focused and manageable
   - Maintain clean commit history

## Testing Guidelines

1. **Unit Tests**
   - Test each function/class independently
   - Use appropriate fixtures
   - Mock external dependencies
   - Cover edge cases

2. **Integration Tests**
   - Test component interactions
   - Verify end-to-end workflows
   - Test with realistic data

3. **Performance Tests**
   - Benchmark critical operations
   - Compare against baseline
   - Document performance impact

## Best Practices

1. **Code Style**
   - Clear, readable code
   - Consistent formatting
   - Descriptive names
   - Proper error handling

2. **Documentation**
   - Clear docstrings
   - Usage examples
   - Implementation notes
   - Performance considerations

3. **Testing**
   - Comprehensive test coverage
   - Fast test execution
   - Deterministic tests
   - Meaningful assertions

4. **Performance**
   - Memory efficient
   - GPU-aware
   - Batch processing
   - Resource cleanup

## Getting Help

- Open an issue for bugs
- Start a discussion for features
- Ask questions in discussions
- Review existing PRs and issues

## License

By contributing, you agree that your contributions will be licensed under the project's license.
