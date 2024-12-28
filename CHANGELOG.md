# Changelog

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
