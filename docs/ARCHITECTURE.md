# Ncode Architecture (v1.0.0)

## System Overview

Ncode is a local code generation system that uses LLaMA-based models to generate both high-level plans and implementation code. The system supports two primary modes of operation:

1. **Standard Mode**: Parallel generation using separate planning and coding models
2. **MTP Mode**: Experimental Multi-Token Prediction using a single model

## Core Components

### LocalModelClient (backend/app.py)
- Central orchestrator for model operations
- Manages model loading and resource allocation
- Handles GPU optimization and context sizing
- Provides streaming token generation
- Implements both standard and MTP generation modes

### CLI Interface (backend/cli_client.py)
- Interactive command-line interface
- Supports model switching and MTP toggling
- Rich console output with progress indicators
- Streaming output display
- Command history and help system

## Generation Modes

### Standard Parallel Generation
```
User Input → Planning Model → Implementation Model → Output
              ↓               ↓
         Architecture    Code Generation
              ↓               ↓
         Stream Output   Stream Output
```

### MTP (Multi-Token Prediction)
```
User Input → Single Model → Combined Output
              ↓
    Architecture + Implementation
              ↓
        Stream Output
```

## Resource Management

### GPU Optimization
- Automatic GPU detection
- Memory-based layer configuration
- Dynamic batch size adjustment
- Fallback to CPU when needed

### Context Management
- Dynamic context sizing
- Thread-safe model loading
- Proper resource cleanup
- Memory optimization

## Model Configuration

### Planning Models
- Default: mistral-7b-instruct-v0.2.Q4_K_M.gguf
- Alternatives:
  - llama-2-7b-chat.Q4_K_M.gguf
  - mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf

### Coding Models
- Default: codellama-7b-instruct.Q4_K_M.gguf
- Alternatives:
  - codellama-13b-instruct.Q4_K_M.gguf
  - deepseek-coder-6.7b-instruct.Q4_K_M.gguf

## Error Handling

### Graceful Degradation
1. GPU Error → CPU Fallback
2. Model Load Error → Informative Messages
3. Generation Error → Clean Error States

### Resource Cleanup
- Automatic model unloading
- Memory release on exit
- Thread cleanup
- Context reset

## Performance Optimization

### GPU Acceleration
- Layer optimization based on memory
- Batch size tuning
- Context size adjustment
- Efficient token generation

### Memory Management
- Dynamic allocation
- Resource pooling
- Garbage collection
- Context reuse

## Future Considerations

### Planned Improvements
- Model fine-tuning support
- Additional generation modes
- Enhanced GPU utilization
- Extended model compatibility

### Experimental Features
- MTP mode refinements
- Alternative architectures
- Performance optimizations
- Model combinations

## Acknowledgments & Design Inspirations

### Architect Mode
The planning and implementation separation was inspired by Aider's approach to code generation:
- Two-phase generation process
- Architectural planning before implementation
- Streaming output design
- CLI interaction patterns

### Multi-Token Prediction (MTP)
The MTP implementation draws from DeepSeek Coder's approach:
- Single model for combined generation
- Efficient token prediction
- Reduced context switching
- Streamlined generation pipeline

These inspirations helped shape Ncode's hybrid approach, combining the best aspects of both systems while adding our own optimizations for local execution and resource management.
