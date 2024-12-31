# Changelog

## [2.0.0] - YYYY-MM-DD

### Major Changes
- [Add major changes here]

### Features
- [Add new features here]

### Fixes
- [Add bug fixes here]

### Breaking Changes
- [List any breaking changes here]

## [1.2.1] - 2024-12-30

### Fixed
- StopIteration handling in async token streaming
- Ready state indicator in CLI after command completion
- Error propagation in async futures chain

### Improved
- Multi-paragraph summary generation with continuation indicators
- CLI command documentation for new features
- Error handling documentation

## [1.2.0] - 2024-07-18

### Added
- Code summary feature with `code-summary` command
  - Generates code-focused summaries with examples
  - Uses coding model for specialized output
  - Includes syntax highlighting and code formatting
  - Automatically saves as .py files when prompted

## [1.1.1] - 2024-07-16

### Added
- Async content streaming with proper error handling
- Enhanced model initialization with async verification
- Improved test coverage for async operations
- Better async cleanup handling in tests

### Improved
- Model client fixture in tests with proper mocking
- Error handling for streaming operations
- Async resource cleanup
- Test structure and organization

### Fixed
- Context manager usage in streaming operations
- Model verification in test environment
- Async cleanup in test fixtures
- Error handling in streaming tests

## [1.1.0] - 2024-07-15

### Added
- Enhanced error handling with detailed context
- Improved async/sync context handling
- Comprehensive resource management system
- GPU memory monitoring and optimization
- Robust import handling with fallbacks

### Improved
- Error messages with detailed context information
- Async warmup operations for better initialization
- Model instance cleanup and resource management
- GPU memory management with automatic cleanup
- Import handling with graceful degradation

### Fixed
- Async/sync context issues in model loading
- Resource leaks in model instances
- GPU memory management during high usage
- Import handling for optional features
- Error recovery procedures

## [1.0.0] - 2023-12-27

### Added
- Multi-Token Prediction (MTP) experimental mode
  - Toggle with `mtp` command in CLI
  - Single model generation for both planning and implementation
  - Improved efficiency for certain tasks
- Parallel generation with separate planning and coding models
- GPU acceleration with automatic layer optimization
- Model hot-swapping with `switch` command
- Rich console output with progress indicators
- Error handling and graceful fallback to CPU

### Core Features
- Local model execution with llama.cpp
- Support for multiple model types (planning and coding)
- Streaming token generation
- Asynchronous parallel processing
- Dynamic context management
- Resource cleanup and memory optimization

### Technical Details
- GPU memory detection and automatic configuration
- Configurable context sizes based on available resources
- Thread-safe model loading and initialization
- Proper cleanup of model resources on exit

### Acknowledgments

#### Special Thanks
- michael5cents and Louisce5cents for inspiration and support

#### Development Environment
- Visual Studio Code as the primary development environment
- Cline extension for VS Code for enhanced development workflow

#### AI Assistants
- Claude 3.5 Sonnet
- Google Gemini 2.0 Experimental
- DeepSeek v3

#### MCP Integrations
- Anthropic MCP Server
