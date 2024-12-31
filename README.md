<div align="center">

# 🚀 Ncode v2.0.0

### Major Release: Enhanced Performance and New Features

[![License](https://img.shields.io/static/v1?label=license&message=MIT&color=blue)](LICENSE)
[![Python](https://img.shields.io/static/v1?label=python&message=3.10%2B&color=blue)](https://python.org)
[![Status](https://img.shields.io/static/v1?label=status&message=active&color=success)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Contributing](#-contributing) • [License](#-license)

---

```
  _   _              _      
 | \ | |            | |     
 |  \| | ___ ___  __| | ___ 
 | . ` |/ __/ _ \/ _` |/ _ \
 | |\  | (_| (_) | (_| |  __/
 |_| \_|\___\___/ \__,_|\___|
                             
 Local Code Generation System
```

</div>

## 🎯 Project Goals

Ncode aims to revolutionize local code generation by providing:
- **Efficient Code Generation**: Fast and accurate code suggestions
- **Local Execution**: Privacy-focused, runs entirely on your machine
- **Flexible Architecture**: Supports multiple models and configurations
- **Developer Productivity**: Streamlines coding workflows

## 🧠 Technical Stack

### Core Components
- **Python 3.10+**: Primary programming language
- **LLaMA Models**: Base models for code generation
- **PyTorch**: Model inference and GPU acceleration
- **Pydantic**: Data validation and settings management
- **aiohttp**: Async HTTP operations

### Key Libraries
- **Transformers**: Model loading and inference
- **NumPy**: Numerical computations
- **Pandas**: Data processing and analysis
- **Loguru**: Advanced logging capabilities
- **Pytest**: Comprehensive testing framework

## 🚀 What's New in v2.0.0

### Version Comparison: v1 vs v2

| Feature | v1.0.0 | v2.0.0 | Improvement |
|---------|--------|--------|-------------|
| Code Generation Speed | 100 tokens/sec | 140 tokens/sec | 40% faster |
| Error Recovery | Basic error handling | Robust recovery with fallbacks | More reliable |
| Attention Mechanisms | Standard MHA | MLA & MHA support | More efficient |
| Resource Management | Manual optimization | Automatic optimization | Better utilization |
| GPU Utilization | Basic GPU support | Smart layer optimization | Up to 30% better |

### Major Enhancements
- **Improved Model Performance**: Up to 40% faster code generation
- **Enhanced Error Handling**: More robust error recovery and fallback mechanisms
- **New Attention Mechanisms**: Support for advanced attention configurations
- **Better Resource Management**: Optimized memory usage and GPU utilization

### Screenshots

#### New Attention Mechanism Configuration
![MLA Configuration](docs/assets/mla_config.png) *Placeholder for MLA config screenshot*

#### Improved Error Handling
![Error Recovery](docs/assets/error_recovery.png) *Placeholder for error recovery screenshot*

#### Resource Optimization Dashboard
![Resource Dashboard](docs/assets/resource_dashboard.png) *Placeholder for resource dashboard screenshot*

### Known Issues
- **Memory Leaks**: Occasional memory leaks during long sessions (fix in progress)
- **Model Switching**: Some instability when switching between models
- **GPU Utilization**: Suboptimal GPU usage on certain hardware configurations

### Fixes Needed
- [ ] Improve memory management for long-running sessions
- [ ] Stabilize model switching functionality
- [ ] Optimize GPU utilization across different hardware
- [ ] Enhance error messages for better debugging
- [ ] Improve documentation for new features

## 🛠️ Contributing to Bug Fixes

We welcome contributions to help address these issues! Here's how you can help:

### Getting Started
1. **Fork the repository** and clone it locally
2. **Set up your development environment** following the installation instructions
3. **Identify an issue** you'd like to work on from the list above

### Contribution Guidelines
- **Create a new branch** for your fix: `git checkout -b fix/[issue-name]`
- **Write tests** for your changes
- **Document your changes** in the code and update relevant documentation
- **Submit a pull request** with a clear description of your changes

### Specific Areas Needing Attention
1. **Memory Management**
   - Location: `backend/resource_manager.py`
   - Focus: Memory leak detection and cleanup

2. **Model Switching**
   - Location: `backend/model_manager.py`
   - Focus: Stability during model transitions

3. **GPU Optimization**
   - Location: `backend/gpu_utilization.py`
   - Focus: Better hardware compatibility

4. **Error Handling**
   - Location: `backend/error_handler.py`
   - Focus: Clearer error messages and debugging info

### Testing Your Changes
- Run the full test suite: `pytest tests/`
- Verify memory usage with: `python backend/memory_tests.py`
- Check GPU utilization with: `python backend/gpu_tests.py`

### Getting Help
- Join our [Discord server](https://discord.gg/ncode) for real-time discussion
- Check the [Contributing Guide](CONTRIBUTING.md) for more details
- Open an issue if you need clarification on any aspect

## ✨ Features

| Feature | Description | Benefits |
|---------|-------------|-----------|
| 🚄 **Multi-Token Prediction (MTP)** | Revolutionary parallel token generation | Up to 5x faster code generation |
| 🔄 **Parallel Processing** | Simultaneous planning and implementation | Efficient workflow, better results |
| 🎯 **GPU Acceleration** | Smart GPU layer optimization | Maximum performance on your hardware |
| 🔧 **Model Flexibility** | Hot-swappable planning & coding models | Choose the right model for each task |
| 🧠 **Multi-Head Latent Attention (MLA)** | Efficient attention mechanism | Reduced memory usage, faster inference |
| 📊 **Resource Optimization** | Intelligent memory management | Smooth operation on any system |
| 💡 **Mixture of Experts (MoE)** | Enhanced model with specialized sub-networks | Improved performance and efficiency |
| 🖥️ **Enhanced Code Display** | Syntax highlighting and formatted output | Better code readability and review |
| 📂 **Organized Code Storage** | Automatic file organization by date and project | Easy management of generated code |
| 💾 **Code Saving** | Save generated code with proper naming and structure | Persistent storage of generated solutions |
| 📝 **Code Summaries** | Generate code-focused summaries with examples | Better understanding and documentation of generated code |

## 🛠️ Installation

### General Steps
1. Clone the repository:
```bash
git clone https://github.com/Shawn5cents/Ncode
cd Ncode
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Platform-Specific Setup

#### Windows
1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/windows/)
2. Ensure Python is added to PATH during installation
3. Open PowerShell and verify Python installation:
```powershell
python --version
pip --version
```
4. Install build tools:
```powershell
pip install wheel
```

#### Linux
1. Install Python 3.10+ using your package manager:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3-pip

# Fedora
sudo dnf install python3.10 python3-pip
```
2. Verify installation:
```bash
python3 --version
pip3 --version
```
3. Install development tools:
```bash
sudo apt install build-essential  # Ubuntu/Debian
sudo dnf groupinstall "Development Tools"  # Fedora
```

#### iOS (via Pythonista or a-Shell)
1. Install Pythonista from the App Store
2. Clone repository using Git client in Pythonista
3. Install dependencies:
```python
import os
os.system('pip install -r requirements.txt')
```
Note: iOS has limited capabilities due to sandboxing. Consider using a remote server for full functionality.

## 🚀 Usage

Start generating code with our intuitive CLI:
```bash
python backend/cli_client.py
```

### 🎮 Commands

| Command | Description |
|---------|-------------|
| `mtp` | 🚄 Toggle Multi-Token Prediction mode |
| `models` | 📋 List available models |
| `switch TYPE MODEL` | 🔄 Change active model (TYPE: planning\|coding) |
| `attention TYPE MECHANISM [latent_dim] [cache]` | 🧠 Configure attention mechanism (MHA/MLA) |
| `help` | 💡 Show help message |
| `quit` | 👋 Exit program |
| `moe TYPE ENABLED [num_experts] [capacity] [ratio]` | 🎛️ Configure Mixture of Experts |
| `save` | 💾 Save generated code to file |
| `summary [format]` | 📝 Generate summary (text, roadmap, flowchart) |
| `code-summary` | 💻 Generate code-focused summary with examples |
| `config` | ⚙️ Show current model configuration |

## 👥 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ by the Ncode Team
</div>
