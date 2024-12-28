import logging
import os
import time
import asyncio
from pathlib import Path
from threading import Lock

import torch
from llama_cpp import Llama
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
                'default': "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                'alternatives': [
                    "llama-2-7b-chat.Q4_K_M.gguf",
                    "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
                ]
            },
            'coding': {
                'default': "codellama-7b-instruct.Q4_K_M.gguf",
                'alternatives': [
                    "codellama-13b-instruct.Q4_K_M.gguf",
                    "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
                ]
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
            
            if gpu_mem_gb >= 24:
                layers = 20
            elif gpu_mem_gb >= 12:
                layers = 15
            elif gpu_mem_gb >= 6:
                layers = 8
            else:
                layers = 4
                
            logger.info(f"Using {layers} GPU layers")
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

                    if not Path(model_path).exists():
                        error_msg = f"Model file not found: {model_path}"
                        self.loading_errors[model_name] = error_msg
                        logger.error(error_msg)
                        return

                    progress.update(task, advance=20, description="Initializing...")

                    context_size = 8192
                    if self.gpu_available:
                        gpu_mem_gb = self.gpu_memory / (1024 * 1024 * 1024)
                        if gpu_mem_gb >= 24:
                            context_size = 12288
                    
                    batch_size = 32
                    if self.gpu_available and gpu_mem_gb >= 12:
                        batch_size = 64
                    
                    self.model_instances[model_name] = Llama(
                        model_path=model_path,
                        n_ctx=context_size,
                        n_threads=min(os.cpu_count() or 4, 8),
                        n_gpu_layers=self.n_gpu_layers,
                        n_batch=batch_size,
                        chat_format="llama-2",
                        embedding=False,
                        use_mlock=False,
                        rope_scaling_type=1
                    )

                    progress.update(task, advance=60)
                    self.model_load_times[model_name] = time.time() - start_time
                    self.models_loaded[model_name] = True
                    self.loading_errors[model_name] = None

                    gpu_info = f" (GPU Layers: {self.n_gpu_layers})" if self.gpu_available else " (CPU)"
                    progress.update(task, advance=20, description=f"Loaded {gpu_info}")
                    logger.info(f"Loaded {model_name} in {self.model_load_times[model_name]:.2f}s {gpu_info}")

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
