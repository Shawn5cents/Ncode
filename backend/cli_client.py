import argparse
import asyncio
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

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

class CLIClient:
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

    def show_help(self):
        """Display available commands."""
        self.console.print("\n[bold blue]Available Commands:[/bold blue]")
        self.console.print("  models              - List available models")
        self.console.print("  switch TYPE MODEL   - Switch to a different model (TYPE: planning|coding)")
        self.console.print("  mtp                - Toggle Multi-Token Prediction experimental mode")
        self.console.print("  help               - Show this help message")
        self.console.print("  quit               - Exit the program")
        self.console.print("\nExample: switch planning mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf\n")

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

    def flush_buffers(self):
        """Flush buffers showing instruction-implementation pairs with model identifiers."""
        # Handle planning instruction
        if self.plan_buffer:
            if not self.plan_buffer.isspace():
                content = self.plan_buffer.strip()
                # Get current planning model name
                planning_model = self.model_client.list_available_models('planning')['planning']['default']
                # Print instruction with model ID
                self.console.print(f"\n[bold magenta]{planning_model}:[/bold magenta]", end=" ")
                self.console.print(Text(content, style="green"))
            self.plan_buffer = ""
        
        # Handle corresponding code implementation
        if self.code_buffer:
            if not self.code_buffer.isspace():
                content = self.code_buffer.strip()
                # Get current coding model name
                coding_model = self.model_client.list_available_models('coding')['coding']['default']
                # Print implementation with model ID
                self.console.print(f"[bold yellow]{coding_model}:[/bold yellow]")
                self.console.print("```python")
                self.console.print(content, style="yellow")
                self.console.print("```")
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
        """Process a user prompt with optional Multi-Token Prediction."""
        try:
            self.console.print("\n[bold blue]Processing Request[/bold blue]")
            self.code_block_started = False
            self.current_section = None

            # Reset buffers
            self.plan_buffer = ""
            self.code_buffer = ""

            if self.use_mtp:
                self.console.print("[bold yellow]Using Multi-Token Prediction (experimental)[/bold yellow]")
                # Placeholder for MTP-enabled generation
                await local_model_client.generate_parallel(
                    prompt,
                    self.plan_stream_handler,
                    self.code_stream_handler,
                    use_mtp=True  # Assuming a flag in the client
                )
            else:
                # Standard parallel generation
                await local_model_client.generate_parallel(
                    prompt,
                    self.plan_stream_handler,
                    self.code_stream_handler
                )

            # Flush any remaining content
            self.flush_buffers()
            print("\n")  # Add space after generation

        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")

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
    args = parser.parse_args()

    client = CLIClient(edit_format=EditFormat(args.edit_format))
    asyncio.run(client.run(args.prompt))

if __name__ == "__main__":
    main()
