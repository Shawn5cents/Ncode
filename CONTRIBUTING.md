# Contributing to Ncode

Thank you for your interest in contributing to Ncode! This project builds upon and is inspired by excellent open-source projects like [Aider](https://github.com/paul-gauthier/aider) and [DeepSeek Coder](https://github.com/deepseek-ai/DeepSeek-Coder). We aim to maintain compatibility and respect for these inspirations while adding our own innovations.

## Design Philosophy

Ncode combines two powerful approaches:
1. **Architect Mode** (inspired by Aider):
   - Separation of planning and implementation
   - Clear architectural thinking before coding
   - Streaming interaction model

2. **MTP Mode** (inspired by DeepSeek):
   - Multi-token prediction for efficiency
   - Combined planning and implementation
   - Optimized generation pipeline

## Getting Started

1. Fork and clone:
```bash
git clone https://github.com/yourusername/ncode.git
cd ncode
```

2. Set up environment:
```bash
python -m venv ncode_venv
source ncode_venv/bin/activate  # Linux/Mac
# or
ncode_venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development Guidelines

### Code Style
- Follow PEP 8
- Use type hints
- Document with docstrings
- Keep functions focused

### Testing
```bash
pytest tests/
pytest --cov=backend tests/
flake8 backend/
black backend/ --check
```

### Commit Messages
Format: `type(scope): description`

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Testing
- chore: Maintenance

### Pull Request Process

1. Create feature branch:
```bash
git checkout -b feature/your-feature
```

2. Make changes:
- Write code
- Add tests
- Update docs

3. Test thoroughly:
- Unit tests
- Integration tests
- GPU/CPU compatibility

4. Submit PR:
- Clear description
- Reference issues
- Update documentation

## Project Structure

```
ncode/
├── backend/           # Core implementation
│   ├── app.py        # Main application
│   ├── cli_client.py # CLI interface
│   └── utils/        # Utilities
├── docs/             # Documentation
├── tests/            # Test suite
└── experimental/     # New features
```

## Feature Guidelines

### Standard Features
- Maintain compatibility with core architecture
- Follow established patterns
- Consider both GPU and CPU users
- Document thoroughly

### Experimental Features
- Use experimental/ directory
- Include benchmarks
- Document limitations
- Plan mainline path

## Compatibility Notes

### Aider Compatibility
When working with architect mode:
- Maintain planning/implementation separation
- Follow similar prompt structures
- Keep streaming output format
- Support similar CLI patterns

### DeepSeek Compatibility
When working with MTP mode:
- Follow token prediction patterns
- Maintain generation efficiency
- Support model compatibility
- Consider pipeline optimizations

## Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for design changes
- Update CHANGELOG.md for all changes
- Keep examples current

## Questions?

- Open an issue for discussion
- Check existing documentation
- Review similar features in Aider/DeepSeek

## License

This project is under the MIT License. Contributions will be under the same license.

Thank you for helping make Ncode better while respecting and building upon its open-source inspirations!
