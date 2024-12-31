# Ncode Architecture (v1.0.0)

## System Overview

Ncode is a local code generation system that uses LLaMA-based models to generate both high-level plans and implementation code. The system supports multiple attention mechanisms and generation modes:

1. **Standard Mode**: Parallel generation using separate planning and coding models
2. **MTP Mode**: Experimental Multi-Token Prediction using a single model
3. **MLA Mode**: Multi-Head Latent Attention for efficient inference

## Core Components

### DeepSeek-V3 MoE Architecture
Ncode implements the DeepSeek-V3 inspired Mixture of Experts (MoE) architecture with several key enhancements:

#### Architecture Overview
- **Finer-grained experts**: Increased specialization through sub-experts
- **Shared and routed experts**: Combination of shared and specialized processing
- **Auxiliary-loss-free load balancing**: Improved expert utilization without additional loss terms
- **Dynamic bias adjustment**: Adaptive expert routing based on utilization
- **Complementary sequence-wise auxiliary loss**: Enhanced training stability

#### Key Features
- **Sub-expert specialization**: Each expert contains multiple sub-experts for finer-grained processing
- **Shared expert processing**: Common processing path for all inputs
- **Dynamic routing**: Adaptive expert selection based on input characteristics
- **Load balancing**: Automatic expert utilization optimization
- **Auxiliary loss tracking**: Monitoring of complementary sequence-wise loss

#### Integration
- **Version tracking**: Full integration with version manager
- **Configuration management**: Detailed MoE configuration tracking
- **Utilization monitoring**: Expert and sub-expert utilization metrics
- **Performance analysis**: Version comparison and impact assessment

#### Performance Benefits
- **Efficient capacity utilization**: Better use of model parameters
- **Improved specialization**: Enhanced task-specific processing
- **Balanced computation**: Optimized expert utilization
- **Stable training**: Reduced loss fluctuations
- **Scalable architecture**: Supports growing model complexity

#### Configuration Options
- **Number of experts**: Controls model capacity
- **Expert capacity**: Determines processing capability
- **Sub-expert count**: Specifies specialization level
- **Shared expert ratio**: Balances shared vs specialized processing
- **Auxiliary loss weight**: Controls loss contribution
- **Routing strategy**: Determines expert selection approach

#### Validation Rules
- Expert capacity must be greater than sub-expert capacity
- Shared expert ratio must be between 0 and 1
- Auxiliary loss weight must be non-negative
- Routing strategy must be valid (top_k, random, learned)
- Sub-expert count must be positive

### LocalModelClient (backend/app.py)
- Central orchestrator for model operations
- Manages model loading and resource allocation
- Handles GPU optimization and context sizing
- Provides streaming token generation
- Implements standard, MTP, and MLA generation modes
- Supports attention mechanism configuration

### CLI Interface (backend/cli_client.py)
- Interactive command-line interface
- Supports model switching and attention configuration
- Rich console output with progress indicators
- Streaming output display
- Command history and help system
- Attention mechanism control:
  - Toggle between MHA and MLA
  - Configure MLA parameters (latent_dim, cache_enabled)
  - View current attention configuration

### MoE Version Tracking (backend/version_manager.py)

#### Configuration Management
- Tracks MoE (Mixture of Experts) configuration changes across versions
- Supports configuration parameters:
  - Number of experts (positive integer)
  - Expert capacity (positive integer)
  - Routing strategy (top_k, random, learned)
  - Expert dropout rate (0.0 to 1.0)
  - Load balancing weight (0.0 to 1.0)
  - Noise standard deviation (non-negative float)
  - K-value for top_k routing (positive integer)

#### Validation Rules
- Enforces strict validation of MoE parameters:
  - Type checking for all configuration values
  - Value range validation
  - Dependency validation (e.g., k-value required for top_k routing)
  - Cross-parameter consistency checks
  - Automatic validation during version registration

#### Utilization Tracking
- Monitors expert utilization metrics:
  - Expert usage percentages (per-expert utilization)
  - Load balance factor (0.0 to 1.0)
  - Total tokens processed
  - Expert dropout statistics
  - Routing decision distribution
  - Load balancing effectiveness

#### Version Comparison
- Provides detailed comparison between MoE versions:
  - Configuration differences with impact analysis
  - Utilization trend analysis
  - Similarity scoring (0.0 to 1.0)
  - Change impact assessment
  - Performance delta analysis
  - Expert utilization comparison

#### Version Management
- Integrates with model version history:
  - Automatic tracking of MoE configuration changes
  - Rollback support for MoE configurations
  - Cleanup of old MoE versions
  - Version-specific MoE statistics
  - Historical trend analysis

#### Error Handling
- Robust error handling for MoE operations:
  - Validation error reporting
  - Configuration conflict resolution
  - Version comparison error handling
  - Utilization tracking error recovery
  - Rollback safety checks

#### Performance Considerations
- Optimized MoE tracking performance:
  - Efficient configuration storage
  - Fast version comparison
  - Lightweight utilization tracking
  - Minimal impact on model performance
  - Scalable version history management

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

### MLA (Multi-Head Latent Attention)
```
User Input → Compressed Attention → Efficient Generation → Output
              ↓                     ↓
         Latent Space         Optimized Inference
              ↓                     ↓
         Stream Output        Reduced Memory Usage
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

### Attention Mechanisms
- Multi-Head Attention (MHA) - Standard attention
- Multi-Head Latent Attention (MLA) - Compressed attention
  - Configurable latent dimensions
  - KV cache optimization
  - Rotary Positional Embedding (RoPE)
  - CLI control:
    - Toggle on/off with 'mla' command
    - Applies to both planning and coding models
    - Status feedback in console

## Model Configuration

### Planning Models
- Default: codellama-7b-instruct.Q4_K_M.gguf
- Alternatives:
  - mistral-7b-instruct-v0.2.Q4_K_M.gguf

### Coding Models
- Default: deepseek-coder-6.7b-instruct-Q4_K_M.gguf
- Alternatives:
  - codellama-7b-instruct.Q4_K_M.gguf
  - codellama-13b-instruct.Q4_K_M.gguf
  - mistral-7b-instruct-v0.2.Q4_K_M.gguf

## Error Handling

### Graceful Degradation
1. GPU Error → CPU Fallback
2. Model Load Error → Informative Messages
3. Generation Error → Clean Error States
4. Attention Mechanism Error → Fallback to MHA

### Resource Cleanup
- Automatic model unloading
- Memory release on exit
- Thread cleanup
- Context reset
- Attention cache clearing

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
- MLA compression

### Attention Optimization
- Latent dimension configuration
- KV cache management
- RoPE integration
- Efficient attention computation

## Future Considerations

### Planned Improvements
- Model fine-tuning support
- Additional generation modes
- Enhanced GPU utilization
- Extended model compatibility
- MLA performance benchmarks

### Experimental Features
- MTP mode refinements
- Alternative architectures
- Performance optimizations
- Model combinations
- Advanced attention mechanisms

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

### Multi-Head Latent Attention (MLA)
The MLA implementation is inspired by DeepSeek-V3's attention mechanism:
- Low-rank joint compression
- Rotary Positional Embedding
- Efficient KV caching
- Memory optimization

These inspirations helped shape Ncode's hybrid approach, combining the best aspects of multiple systems while adding our own optimizations for local execution and resource management.

## MCP Integration

The Model Context Protocol (MCP) provides tools for enhancing generation with context and examples. The main tools include:

### Core Tools
- **enhance_generation**: Enhances generation with context and examples
  - Inputs: prompt (string), mode (planning/coding), context_type (fast/comprehensive)
- **get_context**: Gets relevant context for generation
  - Inputs: mode (planning/coding), keywords (optional array of strings)

### Management Tools
- **manage_connections**: Manage MCP server connections
  - Inputs: action (connect/disconnect/status), server (optional string)
- **monitor_usage**: Monitor MCP tool usage statistics
  - Inputs: time_range (optional day/week/month), tool_name (optional string)
- **validate_responses**: Validate MCP tool responses
  - Inputs: tool_name (string), response (object)
- **manage_configurations**: Manage MCP tool configurations
  - Inputs: action (get/set), tool_name (optional string), config (optional object)

### Usage
MCP tools can be accessed through the CLI interface using the 'mcp' command:
- `mcp enhance [prompt] [mode]`: Enhance generation with context
- `mcp context [mode]`: Get generation context
- `mcp manage [action]`: Manage MCP connections
- `mcp monitor`: View usage statistics
- `mcp validate [tool]`: Validate tool responses
- `mcp config [action]`: Manage tool configurations

### Error Handling
- Invalid inputs return detailed error messages
- Failed tool calls return None to allow graceful degradation
- Connection errors automatically retry with exponential backoff
- Validation errors include specific failure details

### Performance
- Tool calls are cached for 5 minutes
- Background initialization for faster first use
- Concurrent tool registration
- Automatic cache cleanup

## Hybrid GPU/CPU Resource Management

### Overview
The system intelligently utilizes both GPU and CPU resources to optimize performance while maintaining reliability. The hybrid approach ensures efficient resource utilization across different hardware configurations.

### GPU Utilization
1. **Automatic Detection**: System detects available GPU hardware and memory
2. **Layer Allocation**: Optimizes number of GPU layers based on available VRAM
3. **Batch Sizing**: Adjusts batch size according to GPU memory capacity
4. **Context Management**: Dynamically sets context size based on GPU capabilities

### CPU Optimization
1. **RAM Monitoring**: Tracks available system memory for optimal allocation
2. **Thread Management**: Adjusts thread count based on CPU cores and RAM
3. **Batch Sizing**: Sets appropriate batch sizes for CPU inference
4. **Memory Efficiency**: Enables use_mlock for better memory management

### Fallback Mechanism
1. **GPU Failure Detection**: Automatically detects CUDA errors
2. **Graceful Degradation**: Falls back to CPU mode with reduced settings
3. **Resource Adjustment**: Adjusts batch size and thread count for CPU
4. **Error Recovery**: Attempts to recover GPU functionality periodically

### Resource Monitoring
1. **Real-time Tracking**: Monitors GPU and CPU utilization
2. **Dynamic Adjustment**: Adjusts settings based on current load
3. **Performance Logging**: Tracks resource usage for optimization
4. **Threshold Alerts**: Warns when resource limits are approached

### Configuration Parameters
1. **GPU Layers**: Number of layers to offload to GPU
2. **CPU Threads**: Number of threads for CPU processing
3. **Batch Size**: Processing batch size for both GPU and CPU
4. **Context Size**: Maximum context length based on resources
5. **Memory Locking**: use_mlock setting for memory efficiency

### Performance Benefits
1. **Optimal Resource Utilization**: Maximizes use of available hardware
2. **Reliable Operation**: Graceful degradation ensures continued operation
3. **Adaptive Performance**: Adjusts to changing resource availability
4. **Memory Efficiency**: Minimizes memory usage while maintaining performance
5. **Scalability**: Works across different hardware configurations
