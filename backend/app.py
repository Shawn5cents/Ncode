import logging
import os
import time
import asyncio
from pathlib import Path
from threading import Lock

import torch
from llama_cpp import Llama
from backend.experimental.mla_attention import MultiHeadLatentAttention
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.text import Text

from backend.utils.error_handler import create_error_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalModelClient:
    def __init__(self):
        self.models_dir = Path("models")
        self.available_models = {
            'planning': {
                'default': "wizardcoder-python-7b-v1.0.Q4_K_M.gguf",
                'alternatives': [
                    "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    "codellama-7b-instruct.Q4_K_M.gguf"
                ],
                'attention': 'mha',  # Default to Multi-Head Attention
                'mla_config': {
                    'latent_dim': 64,
                    'cache_enabled': True
                },
                'moe_config': {
                    'enabled': False,
                    'num_experts': 8,
                    'expert_capacity': 64,
                    'shared_expert_ratio': 0.25
                }
            },
            'coding': {
                'default': "deepseek-coder-6.7b-instruct-Q4_K_M.gguf",
                'alternatives': [
                    "codellama-7b-instruct.Q4_K_M.gguf",
                    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
                ],
                'attention': 'mha',  # Default to Multi-Head Attention
                'mla_config': {
                    'latent_dim': 64,
                    'cache_enabled': True
                },
                'moe_config': {
                    'enabled': False,
                    'num_experts': 8,
                    'expert_capacity': 64,
                    'shared_expert_ratio': 0.25
                }
            }
        }
        self.planner_model = self.available_models['planning']['default']
        self.coder_model = self.available_models['coding']['default']
        self._shutdown = False
        self._cleanup_registered = False
        self.context_sizes = {
            'small': 4096,   # For simple generations
            'medium': 8192,  # Default size
            'large': 16384   # For complex generations
        }
        self.console = Console()
        self.gpu_available = torch.cuda.is_available()
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory if self.gpu_available else 0
        self.n_gpu_layers = self._calculate_gpu_layers()
        self.initialized = False
        self.init_lock = Lock()
        self.models_loaded = {
            self.planner_model: False,
            self.coder_model: False
        }
        self.model_instances = {}
        self.model_load_times = {}
        self.generation_metrics = {}
        self.loading_errors = {}

        os.makedirs(self.models_dir, exist_ok=True)
        self._start_model_loading()
        logger.info("Created LocalModelClient instance")

    def _calculate_gpu_layers(self) -> int:
        if not self.gpu_available:
            logger.info("No GPU available, using CPU only")
            return 0
            
        try:
            gpu_mem_gb = self.gpu_memory / (1024 * 1024 * 1024)
            logger.info(f"GPU detected with {gpu_mem_gb:.1f}GB memory")
            
            # Minimal GPU layer allocation for 4GB VRAM
            layers = 1
                
            logger.info(f"Using {layers} GPU layer (minimal allocation)")
            return layers
            
        except Exception as e:
            logger.error(f"Error detecting GPU: {str(e)}")
            return 0

    def _start_model_loading(self):
        import threading
        import atexit
        
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True

        def load_model(model_name, progress):
            with progress:
                task = progress.add_task("Loading model...", total=100)
                try:
                    start_time = time.time()
                    model_path = str(self.models_dir / model_name)
                    
                    # Initialize loading errors
                    self.loading_errors[model_name] = None

                    model_file = Path(model_path)
                    if not model_file.exists():
                        error_msg = f"Model file not found: {model_path}"
                        self.loading_errors[model_name] = error_msg
                        logger.error(error_msg)
                        return
                    elif model_file.stat().st_size == 0:
                        error_msg = f"Model file is empty: {model_path}"
                        self.loading_errors[model_name] = error_msg
                        logger.error(error_msg)
                        return

                    # Verify model file integrity and compatibility
                    try:
                        with open(model_path, 'rb') as f:
                            # Check GGUF header
                            header = f.read(4)
                            if header != b'gguf':
                                error_msg = f"Invalid model file format: {model_path}"
                                self.loading_errors[model_name] = error_msg
                                logger.error(error_msg)
                                return
                            
                            # Check file size is within expected ranges
                            file_size = model_file.stat().st_size
                            if file_size < 100 * 1024 * 1024:  # Less than 100MB
                                error_msg = f"Model file is too small: {model_path} ({file_size / (1024 * 1024):.2f} MB)"
                                self.loading_errors[model_name] = error_msg
                                logger.error(error_msg)
                                return
                            elif file_size > 10 * 1024 * 1024 * 1024:  # More than 10GB
                                error_msg = f"Model file is too large: {model_path} ({file_size / (1024 * 1024 * 1024):.2f} GB)"
                                self.loading_errors[model_name] = error_msg
                                logger.error(error_msg)
                                return
                            
                            # Read version and architecture info
                            f.seek(4)  # Skip GGUF header
                            version = int.from_bytes(f.read(4), 'little')
                            if version < 1 or version > 3:  # Supported GGUF versions
                                error_msg = f"Unsupported GGUF version {version} in {model_path}"
                                self.loading_errors[model_name] = error_msg
                                logger.error(error_msg)
                                return
                            
                            # Read architecture string length
                            arch_len = int.from_bytes(f.read(4), 'little')
                            if arch_len > 256:  # Sanity check
                                error_msg = f"Invalid architecture string length in {model_path}"
                                self.loading_errors[model_name] = error_msg
                                logger.error(error_msg)
                                return
                            
                            # Read architecture string
                            architecture = f.read(arch_len).decode('utf-8')
                            supported_archs = ['llama', 'mistral', 'deepseek']
                            if not any(arch in architecture.lower() for arch in supported_archs):
                                error_msg = f"Unsupported model architecture '{architecture}' in {model_path}"
                                self.loading_errors[model_name] = error_msg
                                logger.error(error_msg)
                                return
                            
                            # Check quantization type
                            f.seek(4 + 4 + 4 + arch_len)  # Skip to quantization type
                            quant_type_len = int.from_bytes(f.read(4), 'little')
                            if quant_type_len > 256:  # Sanity check
                                error_msg = f"Invalid quantization type length in {model_path}"
                                self.loading_errors[model_name] = error_msg
                                logger.error(error_msg)
                                return
                            
                            quant_type = f.read(quant_type_len).decode('utf-8')
                            supported_quants = ['Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_0']
                            if not any(quant in quant_type for quant in supported_quants):
                                error_msg = f"Unsupported quantization type '{quant_type}' in {model_path}"
                                self.loading_errors[model_name] = error_msg
                                logger.error(error_msg)
                                return
                            
                            logger.info(f"Model validation passed: {model_path} (Arch: {architecture}, Quant: {quant_type}, Size: {file_size / (1024 * 1024):.2f} MB)")
                            
                    except Exception as e:
                        error_msg = f"Error validating model file: {str(e)}"
                        self.loading_errors[model_name] = error_msg
                        logger.error(error_msg)
                        return

                    progress.update(task, advance=20, description="Initializing...")
                    logger.info(f"Loading model from: {model_path} (Size: {model_file.stat().st_size / (1024 * 1024):.2f} MB)")

                    context_size = 8192
                    if self.gpu_available:
                        gpu_mem_gb = self.gpu_memory / (1024 * 1024 * 1024)
                        if gpu_mem_gb >= 24:
                            context_size = 12288
                    
                    batch_size = 32
                    if self.gpu_available and gpu_mem_gb >= 12:
                        batch_size = 64
                    
                    # Initialize model with appropriate attention mechanism
                    # Get available CPU RAM in GB
                    import psutil
                    cpu_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
                    
                    # Calculate optimal thread count based on CPU cores and RAM
                    cpu_cores = os.cpu_count() or 4
                    max_threads = min(cpu_cores, 8)
                    if cpu_ram_gb < 8:  # Less than 8GB RAM
                        max_threads = min(cpu_cores, 4)
                    elif cpu_ram_gb < 16:  # Less than 16GB RAM
                        max_threads = min(cpu_cores, 6)
                        
                    # Adjust batch size based on available RAM
                    if cpu_ram_gb < 8:
                        batch_size = 16
                    elif cpu_ram_gb < 16:
                        batch_size = 24
                    else:
                        batch_size = 32
                        
                    model_config = {
                        'model_path': model_path,
                        'n_ctx': context_size,
                        'n_threads': max_threads,
                        'n_gpu_layers': self.n_gpu_layers,
                        'n_batch': batch_size,
                        'chat_format': "llama-2",
                        'embedding': False,
                        'use_mlock': True,  # Enable memory locking for better performance
                        'rope_scaling_type': 1,
                        'low_vram': cpu_ram_gb < 8  # Enable low VRAM mode if RAM is limited
                    }
                    
                    # If MLA is enabled, configure attention
                    model_type = 'planning' if model_name in self.available_models['planning']['alternatives'] + [self.available_models['planning']['default']] else 'coding'
                    if self.available_models[model_type]['attention'] == 'mla':
                        mla_config = self.available_models[model_type]['mla_config']
                        model_config['attention'] = MultiHeadLatentAttention(
                            embed_dim=4096,  # Default embedding dimension
                            num_heads=8,     # Default number of attention heads
                            latent_dim=mla_config['latent_dim'],
                            max_seq_len=context_size
                        )
                        if mla_config['cache_enabled']:
                            model_config['attention'].enable_cache()
                    
                    # If MoE is enabled, configure experts
                    if self.available_models[model_type]['moe_config']['enabled']:
                        moe_config = self.available_models[model_type]['moe_config']
                        from backend.experimental.deepseek_moe import DeepSeekMoE
                        model_config['moe'] = DeepSeekMoE(
                            num_experts=moe_config['num_experts'],
                            expert_capacity=moe_config['expert_capacity'],
                            hidden_size=4096,  # Default hidden size
                            shared_expert_ratio=moe_config['shared_expert_ratio']
                        )
                            
                    try:
                        self.model_instances[model_name] = Llama(**model_config)
                        
                        progress.update(task, advance=60)
                        self.model_load_times[model_name] = time.time() - start_time
                        self.models_loaded[model_name] = True
                        self.loading_errors[model_name] = None

                        gpu_info = f" (GPU Layers: {self.n_gpu_layers})" if self.gpu_available else " (CPU)"
                        progress.update(task, advance=20, description=f"Loaded {gpu_info}")
                        logger.info(f"Loaded {model_name} in {self.model_load_times[model_name]:.2f}s {gpu_info}")
                    except Exception as e:
                        error_msg = f"Failed to initialize model: {str(e)}"
                        self.loading_errors[model_name] = error_msg
                        logger.error(error_msg)
                        raise

                except Exception as e:
                    error_msg = str(e)
                    if "CUDA" in error_msg:
                        logger.error(f"CUDA error loading {model_name}. Falling back to CPU. Error: {error_msg}")
                        try:
                            self.model_instances[model_name] = Llama(
                                model_path=model_path,
                                n_ctx=4096,
                                n_threads=min(os.cpu_count() or 4, 6),
                                n_gpu_layers=0,
                                n_batch=16,
                                chat_format="llama-2",
                                embedding=False,
                                use_mlock=False
                            )
                            self.model_load_times[model_name] = time.time() - start_time
                            self.models_loaded[model_name] = True
                            self.loading_errors[model_name] = None
                            logger.info(f"Loaded {model_name} in CPU mode in {self.model_load_times[model_name]:.2f}s")
                        except Exception as cpu_e:
                            error_msg = f"Failed to load {model_name} in CPU mode: {str(cpu_e)}"
                            self.loading_errors[model_name] = error_msg
                            logger.error(error_msg)
                    else:
                        self.loading_errors[model_name] = error_msg
                        logger.error(f"Failed to load {model_name}: {error_msg}")

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        for model in [self.planner_model, self.coder_model]:
            thread = threading.Thread(target=load_model, args=(model, progress))
            thread.daemon = True
            thread.start()

    def cleanup(self):
        if self._shutdown:
            return
            
        self._shutdown = True
        logger.info("Cleaning up model resources...")
        
        for model_name, model_instance in self.model_instances.items():
            try:
                if hasattr(model_instance, 'close'):
                    model_instance.close()
                    logger.info(f"Closed model instance: {model_name}")
            except Exception as e:
                logger.error(f"Error closing model {model_name}: {str(e)}")
        
        self.model_instances.clear()
        logger.info("Model resources cleaned up")

    def set_model(self, model_type: str, model_name: str) -> bool:
        """Change the active model for planning or coding."""
        if model_type not in ['planning', 'coding']:
            raise ValueError("Model type must be 'planning' or 'coding'")
            
        available = self.available_models[model_type]
        if model_name not in [available['default']] + available['alternatives']:
            raise ValueError(f"Unknown model: {model_name}")
            
        if model_type == 'planning':
            if self.planner_model != model_name:
                self.planner_model = model_name
                if model_name in self.models_loaded:
                    del self.models_loaded[model_name]
                return True
        else:
            if self.coder_model != model_name:
                self.coder_model = model_name
                if model_name in self.models_loaded:
                    del self.models_loaded[model_name]
                return True
        return False

    def set_attention(self, model_type: str, attention_type: str, latent_dim: int = 64, cache_enabled: bool = True) -> bool:
        """Change the attention mechanism for a model type."""
        if model_type not in ['planning', 'coding']:
            raise ValueError("Model type must be 'planning' or 'coding'")
            
        if attention_type not in ['mha', 'mla']:
            raise ValueError("Attention type must be 'mha' or 'mla'")
            
        if self.available_models[model_type]['attention'] != attention_type:
            self.available_models[model_type]['attention'] = attention_type
            if attention_type == 'mla':
                self.available_models[model_type]['mla_config'] = {
                    'latent_dim': latent_dim,
                    'cache_enabled': cache_enabled
                }
            # Reload model to apply new attention mechanism
            if model_type == 'planning':
                if self.planner_model in self.models_loaded:
                    del self.models_loaded[self.planner_model]
            else:
                if self.coder_model in self.models_loaded:
                    del self.models_loaded[self.coder_model]
            return True
        return False

    def set_moe(self, model_type: str, enabled: bool, num_experts: int = 8, expert_capacity: int = 64, shared_expert_ratio: float = 0.25) -> bool:
        """Configure Mixture of Experts settings for a model type."""
        if model_type not in ['planning', 'coding']:
            raise ValueError("Model type must be 'planning' or 'coding'")
            
        current_config = self.available_models[model_type]['moe_config']
        if (current_config['enabled'] != enabled or
            current_config['num_experts'] != num_experts or
            current_config['expert_capacity'] != expert_capacity or
            current_config['shared_expert_ratio'] != shared_expert_ratio):
            
            self.available_models[model_type]['moe_config'] = {
                'enabled': enabled,
                'num_experts': num_experts,
                'expert_capacity': expert_capacity,
                'shared_expert_ratio': shared_expert_ratio
            }
            
            # Reload model to apply new MoE configuration
            if model_type == 'planning':
                if self.planner_model in self.models_loaded:
                    del self.models_loaded[self.planner_model]
            else:
                if self.coder_model in self.models_loaded:
                    del self.models_loaded[self.coder_model]
            return True
        return False

    def get_attention_config(self, model_type: str) -> dict:
        """Get the current attention configuration for a model type."""
        if model_type not in ['planning', 'coding']:
            raise ValueError("Model type must be 'planning' or 'coding'")
            
        config = {
            'type': self.available_models[model_type]['attention']
        }
        if config['type'] == 'mla':
            config.update(self.available_models[model_type]['mla_config'])
        return config

    def list_available_models(self, model_type: str = None) -> dict:
        """List available models for planning and/or coding."""
        if model_type:
            if model_type not in ['planning', 'coding']:
                raise ValueError("Model type must be 'planning' or 'coding'")
            return {model_type: self.available_models[model_type]}
        return self.available_models

    async def generate_plan(self, prompt: str, stream_callback=None) -> str:
        """Generate a high-level plan for the given prompt."""
        architect_prompt = f"""[INST] You are a planning assistant. Create a high-level plan for this task:

Task: {prompt}

Break down the solution into clear steps. Focus on architecture and design decisions. Do not write any code - just describe what needs to be done.
[/INST]"""
        return await self.generate_content(self.planner_model, architect_prompt, stream_callback)

    async def generate_implementation(self, prompt: str, plan: str, stream_callback=None) -> str:
        """Generate implementation code based on the plan."""
        implementation_prompt = f"""[INST] You are a coding assistant. Based on this plan, implement a solution:

Plan:
{plan}

Original Task: {prompt}

Provide a complete, working implementation with clear code examples and explanations. Include any necessary setup or configuration steps.
[/INST]"""
        return await self.generate_content(self.coder_model, implementation_prompt, stream_callback)

    async def generate_parallel(self, prompt: str, plan_callback=None, code_callback=None, use_mtp=False):
        """Generate plan and implementation in parallel with optional MTP mode."""
        if use_mtp:
            # MTP mode - generate multiple tokens per prediction
            return await self._generate_mtp(prompt, plan_callback, code_callback)
        
        # Standard parallel generation
        plan_buffer = []
        code_generation_started = False
        
        async def plan_stream_handler(text: str):
            plan_buffer.append(text)
            if plan_callback:
                await plan_callback(text)

        async def implementation_handler():
            nonlocal code_generation_started
            current_plan = ""
            buffer_pos = 0
            
            while True:
                # Check if we have new plan content
                if len(plan_buffer) > buffer_pos:
                    # Get new content
                    new_content = ''.join(plan_buffer[buffer_pos:])
                    current_plan += new_content
                    buffer_pos = len(plan_buffer)

                    # If we have enough context, start or continue implementation
                    if len(current_plan.split()) >= 10 and not code_generation_started:  # Start after 10 words
                        code_generation_started = True
                        implementation_prompt = f"""[INST] You are a coding assistant. Implement this solution following the evolving plan:

Plan (in progress):
{current_plan}

Original Task: {prompt}

Write the code implementation following the architectural guidance from the plan. Focus on writing clean, working code.
[/INST]"""
                        
                        if code_callback:
                            await code_callback("\n[Starting implementation based on current plan...]\n")
                        
                        async for code_token in self.generate_content_stream(self.coder_model, implementation_prompt):
                            if code_callback:
                                await code_callback(code_token)
                        
                        break  # Exit after first implementation attempt

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

        # Start plan generation
        plan_task = asyncio.create_task(self.generate_plan(prompt, plan_stream_handler))
        
        # Start implementation monitoring
        implementation_task = asyncio.create_task(implementation_handler())
        
        # Wait for both tasks to complete
        await asyncio.gather(plan_task, implementation_task)

    async def _generate_mtp(self, prompt: str, plan_callback=None, code_callback=None):
        """Multi-Token Prediction generation mode."""
        # Generate multiple tokens per prediction
        mtp_prompt = f"""[INST] You are an AI assistant using Multi-Token Prediction. 
Generate both the high-level plan and implementation code for this task:

Task: {prompt}

First, create a detailed plan. Then, provide the complete implementation code.
[/INST]"""
        
        async def mtp_stream_handler(text: str):
            if plan_callback:
                await plan_callback(text)
            if code_callback:
                await code_callback(text)

        return await self.generate_content(self.planner_model, mtp_prompt, mtp_stream_handler)

    async def generate_content_stream(self, model_name: str, prompt: str):
        """Generate content using the specified model and yield tokens."""
        if not self.models_loaded.get(model_name, False):
            if model_name in self.loading_errors:
                error_msg = self.loading_errors[model_name]
                raise Exception(f"Model {model_name} failed to load: {error_msg}")
            raise Exception(f"Model {model_name} is still loading. Please wait.")

        if model_name not in self.model_instances:
            raise Exception(f"Model {model_name} instance not found")

        llama = self.model_instances[model_name]
        try:
            temperature = 0.2 if model_name == self.coder_model else 0.7
            
            for token in llama.create_completion(
                prompt,
                max_tokens=1024,
                stop=["</s>", "</task>"],
                temperature=temperature,
                top_p=0.1,
                repeat_penalty=1.2,
                mirostat_mode=0,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                stream=True
            ):
                if token and 'choices' in token and token['choices']:
                    text = token['choices'][0].get('text', '')
                    if text:
                        yield text

        except Exception as e:
            logger.error(f"Error generating content with {model_name}: {e}")
            raise Exception(f"Generation error with {model_name}: {str(e)}")

    async def generate_content(self, model_name: str, prompt: str, stream_callback=None) -> str:
        """Generate content using the specified model."""
        full_response = ""
        async for token in self.generate_content_stream(model_name, prompt):
            full_response += token
            if stream_callback:
                await stream_callback(token)
        return full_response

local_model_client = LocalModelClient()
