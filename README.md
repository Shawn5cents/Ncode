<div align="center">

# 🚀 Ncode v1.0.0

### AI-Powered Code Generation, Reimagined

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

Transform your ideas into code with Ncode - a powerful local code generation system that combines LLaMA's intelligence with blazing-fast performance. Experience the future of coding with our unique dual-model approach: one for planning, one for implementation.

## ✨ Features

| Feature | Description | Benefits |
|---------|-------------|-----------|
| 🚄 **Multi-Token Prediction (MTP)** | Revolutionary parallel token generation | Up to 5x faster code generation |
| 🔄 **Parallel Processing** | Simultaneous planning and implementation | Efficient workflow, better results |
| 🎯 **GPU Acceleration** | Smart GPU layer optimization | Maximum performance on your hardware |
| 🔧 **Model Flexibility** | Hot-swappable planning & coding models | Choose the right model for each task |
| 📊 **Resource Optimization** | Intelligent memory management | Smooth operation on any system |

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Shawn5cents/Ncode
cd Ncode
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
# Create models directory
mkdir -p models

# Download recommended models
# Planning: Mistral-7B-Instruct
# Coding: CodeLlama-7B-Instruct
```

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
| `help` | 💡 Show help message |
| `quit` | 👋 Exit program |

### 💫 Example: Creating a Web Server

```python
> create a fast http server with rate limiting

[Planning] Designing architecture...
✓ Selected FastAPI framework
✓ Added rate limiting middleware
✓ Included error handling

[Coding] Implementing solution...

from fastapi import FastAPI, Request
from fastapi.middleware.throttling import ThrottlingMiddleware
import uvicorn

app = FastAPI(title="Fast HTTP Server")

# Rate limiting middleware
app.add_middleware(
    ThrottlingMiddleware,
    rate_limit=100,  # requests
    time_window=60   # seconds
)

@app.get("/")
async def root():
    return {"message": "Welcome to the rate-limited server!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

✨ Generated server.py with rate limiting!
```

## 🏗️ Architecture

Our dual-model architecture ensures both speed and quality:

- 🧠 **LocalModelClient**: Smart engine managing model operations
- ⚡ **Parallel Generation**: Asynchronous planning and implementation
- 🚄 **MTP Mode**: Experimental single-model generation
- 🎯 **GPU Optimization**: Automatic layer configuration
- 📊 **Resource Management**: Dynamic context and cleanup

## 🔧 Technical Details

- 🎮 GPU memory-based optimization
- 🔒 Thread-safe model loading
- 📡 Streaming token generation
- 🧹 Proper resource cleanup
- 🛡️ Error handling with CPU fallback

## 👥 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🙏 Acknowledgments

### Special Thanks
- **michael5cents**: For inspiration and invaluable support
- **Louisce5cents**: For crucial support in shaping the project

### Project Inspirations
- [Aider](https://github.com/paul-gauthier/aider): Inspiration for architect mode
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder): Inspiration for MTP

See [CREDITS.md](docs/CREDITS.md) for a complete list of acknowledgments.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ by the Ncode Team
</div>
