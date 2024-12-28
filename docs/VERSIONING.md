# Versioning Strategy

## Version Format

Ncode follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- MAJOR: Incompatible API changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

Current Version: 1.0.0

## Version Files

Version information is maintained in:
- VERSION file (root directory)
- setup.py
- Documentation files

## Compatibility Guarantees

### Core Features
- Standard parallel generation mode
- GPU/CPU support
- Model hot-swapping
- CLI interface
- Resource management

### Aider Compatibility
We maintain compatibility with Aider-inspired features:
- Planning/implementation separation
- Architectural planning phase
- Streaming output format
- CLI interaction patterns

### DeepSeek Compatibility
We maintain compatibility with DeepSeek-inspired features:
- MTP token generation
- Combined output mode
- Generation efficiency
- Pipeline optimizations

## Version History

### 1.0.0
- Initial stable release
- MTP support
- GPU optimization
- Parallel generation
- Full documentation

## Breaking Changes Policy

1. Major Version Changes:
   - API modifications
   - Core behavior changes
   - Compatibility breaks

2. Minor Version Changes:
   - New features
   - Performance improvements
   - Extended functionality

3. Patch Version Changes:
   - Bug fixes
   - Documentation updates
   - Minor improvements

## Deprecation Policy

1. Feature Deprecation:
   - Minimum one major version notice
   - Clear migration path
   - Documentation updates

2. API Changes:
   - Backward compatibility maintained in minor versions
   - Clear upgrade guides
   - Migration tools when possible

## Release Process

1. Version Update:
   - Update VERSION file
   - Update setup.py
   - Update documentation

2. Testing:
   - Full test suite
   - GPU/CPU verification
   - Documentation check

3. Release:
   - Tag version
   - Update changelog
   - Release notes

## Version Checking

Code can check version:
```python
with open('VERSION') as f:
    version = f.read().strip()
```

Or via package:
```python
from ncode import __version__
```

## Future Versions

Planned improvements:
1. Enhanced MTP capabilities
2. Extended model support
3. Performance optimizations
4. Additional generation modes

## Version Support

- Major versions: 12 months
- Security fixes: 18 months
- Documentation: Maintained for all supported versions
