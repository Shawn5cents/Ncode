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
| 🧠 **Multi-Head Latent Attention (MLA)** | Efficient attention mechanism | Reduced memory usage, faster inference |
| 📊 **Resource Optimization** | Intelligent memory management | Smooth operation on any system |
| 💡 **Mixture of Experts (MoE)** | Enhanced model with specialized sub-networks | Improved performance and efficiency |
| 🖥️ **Enhanced Code Display** | Syntax highlighting and formatted output | Better code readability and review |
| 📂 **Organized Code Storage** | Automatic file organization by date and project | Easy management of generated code |
| 💾 **Code Saving** | Save generated code with proper naming and structure | Persistent storage of generated solutions |

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

3. Download and Setup Models:

Ncode requires two GGUF-quantized models for optimal performance. Follow these steps carefully:

#### Step 1: Create Models Directory
```bash
mkdir -p models
cd models
```

#### Step 2: Download Required Models

**Planning Model (Choose one):**
- [Mistral-7B-Instruct-v0.2-Q4_K_M](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf) (4-bit, 4.37GB, Recommended)
- [Mistral-7B-Instruct-v0.2-Q5_K_M](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf) (5-bit, 5.36GB, Higher quality)

**Coding Model (Choose one):**
- [CodeLlama-7B-Instruct-Q4_K_M](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf) (4-bit, 4.24GB, Recommended)
- [CodeLlama-7B-Instruct-Q5_K_M](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q5_K_M.gguf) (5-bit, 5.21GB, Higher quality)

**Windows Users:**
1. Open PowerShell in the models directory
2. Run these commands one at a time:
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf" -OutFile "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf" -OutFile "codellama-7b-instruct.Q4_K_M.gguf"
```

**Linux/Mac Users:**
```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
wget https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf
```

#### Step 3: Verify Models
After downloading, your models directory should contain:
- mistral-7b-instruct-v0.2.Q4_K_M.gguf
- codellama-7b-instruct.Q4_K_M.gguf

If you see errors about missing models, double-check that:
1. The files are in the models directory
2. The filenames match exactly
3. The files are complete (check file sizes)

**System Requirements:**
- Disk: ~9GB for both 4-bit models, ~11GB for both 5-bit models
- RAM: Minimum 16GB recommended
- GPU: Optional, but recommended for faster generation (8GB VRAM minimum for 4-bit models)

#### Alternative Models

Ncode supports various model configurations:

##### Local GGUF Models
You can use any GGUF-quantized model that follows instruction format:
- [Llama-2-7B-Chat](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- [Neural-Chat-7B](https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF)
- [Deepseek Coder](https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF)
- [WizardCoder](https://huggingface.co/TheBloke/WizardCoder-Python-7B-V1.0-GGUF)

Simply download your chosen model and place it in the `models/` directory.

##### API-Based Models
Ncode also supports API-based models through environment variables:

1. OpenAI API:
```bash
# Add to your .env file
OPENAI_API_KEY=your_api_key
PLANNING_MODEL=gpt-4
CODING_MODEL=gpt-4-turbo
```

2. Anthropic API:
```bash
# Add to your .env file
ANTHROPIC_API_KEY=your_api_key
PLANNING_MODEL=claude-3-opus
CODING_MODEL=claude-3-sonnet
```

3. Custom API Endpoint:
```bash
# Add to your .env file
CUSTOM_API_KEY=your_api_key
CUSTOM_API_URL=https://your-api-endpoint
PLANNING_MODEL=your_planning_model
CODING_MODEL=your_coding_model
```

To use API models:
1. Create a `.env` file in the project root
2. Add your API configuration
3. Install API dependencies:
```bash
pip install python-dotenv openai anthropic
```
4. Start Ncode normally - it will automatically detect and use the configured API models

**Note:** When using API models, the system requirements are much lower since models run on the provider's servers.

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
| `moe [on\|off]` | 🎛️ Enable or disable Mixture of Experts |
| `save` | 💾 Save generated code to file |

### 💫 Example: Configuring MLA

```bash
# Switch planning model to MLA with custom settings
> attention planning mla 128 true
[green]Switched planning to MLA with latent_dim=128, cache=true[/green]

# Switch coding model to MHA
> attention coding mha
[green]Switched coding to MHA[/green]
```

### 💫 Example: Creating and Saving Code

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

> save server.py
[green]Code saved to storage/generated_code/2024-01-01/default/server.py[/green]
```

## 🏗️ Architecture

Our dual-model architecture ensures both speed and quality:

- 🧠 **LocalModelClient**: Smart engine managing model operations
- ⚡ **Parallel Generation**: Asynchronous planning and implementation
- 🚄 **MTP Mode**: Experimental single-model generation
- 🧠 **MLA Support**: Multi-Head Latent Attention for efficient inference
- 🎯 **GPU Optimization**: Automatic layer configuration
- 📊 **Resource Management**: Dynamic context and cleanup
- 💡 **MoE Support**: Integration of Mixture of Experts models for enhanced performance.

## 🔧 Technical Details

- 🎮 GPU memory-based optimization
- 🔒 Thread-safe model loading
- 📡 Streaming token generation
- 🧹 Proper resource cleanup
- 🛡️ Error handling with CPU fallback
- 🧠 MLA with configurable latent dimensions and caching

## 👥 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🙏 Acknowledgments

### Special Thanks
- **michael5cents**: For inspiration and invaluable support
- **Louisce5cents**: For crucial support in shaping the project

### Project Inspirations
- [Aider](https://github.com/paul-gauthier/aider): Inspiration for architect mode
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder): Inspiration for MTP and MLA

See [CREDITS.md](docs/CREDITS.md) for a complete list of acknowledgments.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ by the Ncode Team
</div>
