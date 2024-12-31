import argparse
import asyncio
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.syntax import Syntax

from backend.app import local_model_client

class EditFormat(Enum):
    DIFF = "diff"
    WHOLE = "whole"
    INLINE = "inline"

class CLICommand(Enum):
    QUIT = "quit"
    LIST_MODELS = "models"
    SWITCH_MODEL = "switch"
    HELP = "help"
    TOGGLE_MTP = "mtp"
    SWITCH_ATTENTION = "attention"
    TOGGLE_MLA = "mla"
    MCP = "mcp"

class CLIClient:
    def add_model_options(self):
        """Add model-related command line options"""
        self.parser.add_argument(
            '--planning-model',
            type=str,
            default='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
            help='Path to planning model file'
        )
        self.parser.add_argument(
            '--coding-model',
            type=str,
            default='codellama-7b-instruct.Q4_K_M.gguf',
            help='Path to coding model file'
        )
        self.parser.add_argument(
            '--moe-enabled',
            type=bool,
            default=False,
            help='Enable Mixture of Experts architecture'
        )
        self.parser.add_argument(
            '--num-experts',
            type=int,
            default=8,
            help='Number of experts in MoE'
        )
        self.parser.add_argument(
            '--expert-capacity',
            type=int,
            default=64,
            help='Capacity of each expert'
        )
        self.parser.add_argument(
            '--shared-expert-ratio',
            type=float,
            default=0.25,
            help='Ratio of shared expert capacity'
        )

    def show_available_models(self, model_type=None):
        """Display available models."""
        models = self.model_client.list_available_models(model_type)
        for type_name, type_models in models.items():
            self.console.print(f"\n[bold blue]{type_name.title()} Models:[/bold blue]")
            self.console.print(f"  Current: {type_models['default']}")
            if type_models['alternatives']:
                self.console.print("  Alternatives:")
                for model in type_models['alternatives']:
                    self.console.print(f"    - {model}")

    def switch_model(self, args: str):
        """Switch to a different model."""
        try:
            model_type, model_name = args.split(' ', 1)
            if self.model_client.set_model(model_type, model_name):
                self.console.print(f"[green]Switched {model_type} model to: {model_name}[/green]")
            else:
                self.console.print("[yellow]Model is already active[/yellow]")
        except ValueError as e:
            self.console.print("[red]Error: Invalid command format. Use 'switch planning|coding model_name'[/red]")
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")

    def switch_attention(self, args: str):
        """Switch to a different attention mechanism with optional parameters."""
        try:
            parts = args.split()
            model_type = parts[0]
            attention_type = parts[1]
            
            # Handle MLA-specific parameters
            if attention_type == 'mla':
                latent_dim = 64  # Default
                cache_enabled = True  # Default
                
                if len(parts) > 2:
                    latent_dim = int(parts[2])
                if len(parts) > 3:
                    cache_enabled = parts[3].lower() == 'true'
                
                if self.model_client.set_attention(model_type, attention_type, latent_dim, cache_enabled):
                    self.console.print(f"[green]Switched {model_type} to MLA with latent_dim={latent_dim}, cache={cache_enabled}[/green]")
                else:
                    self.console.print(f"[yellow]{model_type} is already using MLA[/yellow]")
            else:
                if self.model_client.set_attention(model_type, attention_type):
                    self.console.print(f"[green]Switched {model_type} to MHA[/green]")
                else:
                    self.console.print(f"[yellow]{model_type} is already using MHA[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")
            self.console.print("[yellow]Usage: attention planning|coding mha|mla [latent_dim] [cache_enabled][/yellow]")

    def show_attention_config(self):
        """Display current attention configuration."""
        try:
            self.console.print("\n[bold blue]Attention Configuration:[/bold blue]")
            for model_type in ['planning', 'coding']:
                config = self.model_client.get_attention_config(model_type)
                self.console.print(f"  {model_type.title()}:")
                self.console.print(f"    Type: {config['type'].upper()}")
                if config['type'] == 'mla':
                    self.console.print(f"    Latent Dimension: {config['latent_dim']}")
                    self.console.print(f"    Cache Enabled: {config['cache_enabled']}")
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")

    def set_moe(self, args: str):
        """Configure Mixture of Experts settings."""
        try:
            if not args:
                raise ValueError("Missing arguments")
                
            parts = args.split()
            if len(parts) < 2:
                raise ValueError("Insufficient arguments")
                
            model_type = parts[0]
            if model_type not in ['planning', 'coding']:
                raise ValueError("Model type must be 'planning' or 'coding'")
                
            enabled = parts[1].lower() == 'true'
            
            # Handle MoE parameters with validation
            num_experts = 8  # Default
            expert_capacity = 64  # Default
            shared_expert_ratio = 0.25  # Default
            
            if len(parts) > 2:
                num_experts = int(parts[2])
                if num_experts < 1:
                    raise ValueError("Number of experts must be positive")
                    
            if len(parts) > 3:
                expert_capacity = int(parts[3])
                if expert_capacity < 1:
                    raise ValueError("Expert capacity must be positive")
                    
            if len(parts) > 4:
                shared_expert_ratio = float(parts[4])
                if not 0 <= shared_expert_ratio <= 1:
                    raise ValueError("Shared expert ratio must be between 0 and 1")
            
            if self.model_client.set_moe(model_type, enabled, num_experts, expert_capacity, shared_expert_ratio):
                self.console.print(f"[green]Updated {model_type} MoE configuration[/green]")
            else:
                self.console.print(f"[yellow]{model_type} MoE configuration unchanged[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")
            self.console.print("[yellow]Usage: moe planning|coding enabled [num_experts] [expert_capacity] [shared_expert_ratio][/yellow]")
            self.console.print("[yellow]Example: moe planning true 8 64 0.25[/yellow]")

    def show_help(self):
        """Display available commands."""
        self.console.print("\n[bold blue]Available Commands:[/bold blue]")
        self.console.print("  models              - List available models")
        self.console.print("  switch TYPE MODEL   - Switch to a different model (TYPE: planning|coding)")
        self.console.print("  attention TYPE MECHANISM [latent_dim] [cache] - Switch attention mechanism")
        self.console.print("  moe TYPE ENABLED [num_experts] [capacity] [ratio] - Configure Mixture of Experts")
        self.console.print("  mtp                - Toggle Multi-Token Prediction experimental mode")
        self.console.print("  mla                - Toggle Multi-Head Latent Attention")
        self.console.print("  mcp                - Manage MCP tools and connections")
        self.console.print("  help               - Show this help message")
        self.console.print("  quit               - Exit the program")
        self.console.print("\nExample: switch planning mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf")
        self.console.print("Example: attention planning mla 128 true")
        self.console.print("Example: moe coding true 8 64 0.25")
        self.console.print("Example: attention coding mha")
        self.console.print("Example: mcp enhance [prompt] [mode] - Enhance generation with context\n")

    def __init__(self, edit_format=EditFormat.DIFF):
        self.console = Console()
        self.edit_format = edit_format
        self.running = False
        self.model_client = local_model_client
        self.plan_buffer = ""
        self.code_buffer = ""
        self.last_newline = True
        self.code_block_started = False
        self.current_section = None
        self.use_mtp = False
        self.commands = {
            "quit": CLICommand.QUIT,
            "models": CLICommand.LIST_MODELS,
            "switch": CLICommand.SWITCH_MODEL,
            "help": CLICommand.HELP,
            "mtp": CLICommand.TOGGLE_MTP,
            "attention": CLICommand.SWITCH_ATTENTION,
            "moe": "moe",
            "mla": CLICommand.TOGGLE_MLA,
            "mcp": CLICommand.MCP
        }

    def flush_buffers(self):
        """Flush buffers showing instruction-implementation pairs with enhanced display."""
        # Handle planning instruction
        if self.plan_buffer:
            if not self.plan_buffer.isspace():
                content = self.plan_buffer.strip()
                # Get current planning model name
                planning_model = self.model_client.list_available_models('planning')['planning']['default']
                
                # Enhanced display with syntax highlighting
                self.console.print("\n" + Panel.fit(
                    Text(content, style="green"),
                    title=f"[bold magenta]{planning_model}[/bold magenta]",
                    subtitle="Planning Instruction",
                    border_style="blue"
                ))
            self.plan_buffer = ""
        
        # Handle corresponding code implementation
        if self.code_buffer:
            if not self.code_buffer.isspace():
                content = self.code_buffer.strip()
                # Get current coding model name
                coding_model = self.model_client.list_available_models('coding')['coding']['default']
                
                # Enhanced code display with syntax highlighting
                self.console.print(Panel.fit(
                    Syntax(content, "python", theme="monokai", line_numbers=True),
                    title=f"[bold yellow]{coding_model}[/bold yellow]",
                    subtitle="Code Implementation",
                    border_style="blue"
                ))
            self.code_buffer = ""

    async def plan_stream_handler(self, text: str):
        """Handle planning model output as single-line instructions."""
        self.plan_buffer += text
        # Flush on complete instruction (end of sentence or line)
        if text in ['.', '!', '?', '\n'] and len(self.plan_buffer.strip()) > 0:
            self.flush_buffers()

    async def code_stream_handler(self, text: str):
        """Handle coding model output as implementation of last instruction."""
        self.code_buffer += text
        # Flush when code implementation is complete
        if text.endswith(('}\n', ';\n', 'end\n')) or \
           (text.strip() and len(self.code_buffer.strip().split('\n')) > 1):
            self.flush_buffers()

    async def process_prompt(self, prompt: str):
        """Process a user prompt with enhanced output handling."""
        try:
            # Enhanced processing display
            self.console.print(Panel.fit(
                Text("Processing Request", style="bold blue"),
                border_style="blue"
            ))
            self.code_block_started = False
            self.current_section = None

            # Reset buffers
            self.plan_buffer = ""
            self.code_buffer = ""

            if self.use_mtp:
                self.console.print(Panel.fit(
                    Text("Using Multi-Token Prediction (experimental)", style="bold yellow"),
                    border_style="yellow"
                ))
                await local_model_client.generate_parallel(
                    prompt,
                    self.plan_stream_handler,
                    self.code_stream_handler,
                    use_mtp=True
                )
            else:
                await local_model_client.generate_parallel(
                    prompt,
                    self.plan_stream_handler,
                    self.code_stream_handler
                )

            # Flush any remaining content
            self.flush_buffers()
            
            # Prompt for saving the output
            save = self.console.input("[bold]Save this output? (y/n): [/bold]").lower()
            if save == 'y':
                filename = self.console.input("[bold]Enter filename: [/bold]")
                await self.save_output(filename)
                self.console.print(f"[green]Output saved to {filename}[/green]")
            
            self.console.print("\n")  # Add space after generation

        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")

    async def save_output(self, filename: str):
        """Save the current buffers to a file."""
        try:
            if not filename.endswith('.py'):
                filename += '.py'
                
            # Combine plan and code buffers
            content = ""
            if self.plan_buffer.strip():
                content += f"# Planning:\n{self.plan_buffer.strip()}\n\n"
            if self.code_buffer.strip():
                content += f"# Implementation:\n{self.code_buffer.strip()}"
                
            # Save using storage manager
            await local_model_client.storage_manager.save_generated_code(filename, content)
            
        except Exception as e:
            self.console.print(f"[red]Error saving file: {str(e)}[/red]")

    async def run(self, prompt=None):
        """Run the CLI client."""
        self.running = True
        
        try:
            if prompt:
                await self.process_prompt(prompt)
            else:
                self.console.print("[bold blue]Ncode CLI[/bold blue]")
                self.console.print("Type 'help' for available commands\n")
                
                while self.running:
                    user_input = self.console.input("[bold]Enter command or prompt: [/bold]")
                    command = user_input.lower().strip()

                    if command == CLICommand.QUIT.value:
                        break
                    elif command == CLICommand.LIST_MODELS.value:
                        self.show_available_models()
                    elif command == CLICommand.HELP.value:
                        self.show_help()
                    elif command == CLICommand.TOGGLE_MTP.value:
                        self.use_mtp = not self.use_mtp
                        status = "[green]enabled[/green]" if self.use_mtp else "[red]disabled[/red]"
                        self.console.print(f"[bold blue]MTP is now {status}[/bold blue]")
                    elif command.startswith(CLICommand.SWITCH_MODEL.value):
                        self.switch_model(user_input[len(CLICommand.SWITCH_MODEL.value):].strip())
                    elif command.startswith(CLICommand.SWITCH_ATTENTION.value):
                        self.switch_attention(user_input[len(CLICommand.SWITCH_ATTENTION.value):].strip())
                    elif command.startswith("moe"):
                        self.set_moe(user_input[len("moe"):].strip())
                    elif command == CLICommand.TOGGLE_MLA.value:
                        # Toggle MLA for both planning and coding models
                        current_config_planning = self.model_client.get_attention_config('planning')
                        current_config_coding = self.model_client.get_attention_config('coding')
                        
                        new_type = 'mha' if current_config_planning['type'] == 'mla' else 'mla'
                        
                        # Set for both model types
                        self.model_client.set_attention('planning', new_type)
                        self.model_client.set_attention('coding', new_type)
                        
                        status = "[green]enabled[/green]" if new_type == 'mla' else "[red]disabled[/red]"
                        self.console.print(f"[bold blue]MLA is now {status} for both planning and coding models[/bold blue]")
                    elif command.startswith(CLICommand.MCP.value):
                        # Handle MCP commands
                        mcp_command = user_input[len(CLICommand.MCP.value):].strip()
                        if mcp_command.startswith("enhance"):
                            # Handle enhance_generation command
                            try:
                                parts = mcp_command.split()
                                prompt = " ".join(parts[1:-1])
                                mode = parts[-1]
                                if mode not in ["planning", "coding"]:
                                    raise ValueError("Mode must be 'planning' or 'coding'")
                                
                                # Call MCP enhance_generation
                                result = await self.model_client.mcp_integration.enhance_generation(prompt, mode)
                                if result:
                                    self.console.print("[bold green]Enhanced Generation:[/bold green]")
                                    self.console.print(result)
                                else:
                                    self.console.print("[yellow]No enhancement available[/yellow]")
                            except Exception as e:
                                self.console.print(f"[red]Error: {str(e)}[/red]")
                                self.console.print("[yellow]Usage: mcp enhance [prompt] [planning|coding][/yellow]")
                        else:
                            self.console.print("[yellow]Unknown MCP command. Use 'mcp enhance'[/yellow]")
                    else:
                        await self.process_prompt(user_input)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Shutting down...[/yellow]")
        finally:
            self.running = False

def main():
    parser = argparse.ArgumentParser(description="Ncode CLI")
    parser.add_argument("--prompt", help="Direct prompt to process")
    parser.add_argument(
        "--edit-format",
        choices=["diff", "whole", "inline"],
        default="diff",
        help="Format for code edits",
    )
    parser.add_argument(
        '--moe-enabled',
        type=bool,
        default=False,
        help='Enable Mixture of Experts architecture'
    )
    parser.add_argument(
        '--num-experts',
        type=int,
        default=8,
        help='Number of experts in MoE'
    )
    parser.add_argument(
        '--expert-capacity',
        type=int,
        default=64,
        help='Capacity of each expert'
    )
    parser.add_argument(
        '--shared-expert-ratio',
        type=float,
        default=0.25,
        help='Ratio of shared expert capacity'
    )
    args = parser.parse_args()

    client = CLIClient(edit_format=EditFormat(args.edit_format))
    asyncio.run(client.run(args.prompt))

if __name__ == "__main__":
    main()
