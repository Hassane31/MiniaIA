"""Rich terminal UI. The only module that prints or asks the human anything."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .agent import AgentUI
from .command_executor import ExecResult
from .config import Config
from .ollama_client import OllamaStatus
from .security import Assessment
from .tools import ToolResult

BANNER = r"""  __  __ _       _       ___    _
 |  \/  (_)     (_)     |_ _|  / \
 | |\/| | |_ __  _  __ _ | |  / _ \
 | |  | | | '_ \| |/ _` || | / ___ \
 |_|  |_|_|_| |_|_|\__,_|___/_/   \_\
"""

HELP_TEXT = """[bold]Talk to me in English, French or Arabic.[/] Examples:
  [cyan]how much disk space is left?[/]
  [cyan]quels ports sont ouverts sur mon PC ?[/]
  [cyan]why is my laptop slow?[/]  [cyan]what is using port 8080?[/]  [cyan]is ssh running?[/]
  [cyan]is my internet working?[/]  [cyan]is my battery charging?[/]

[bold]Slash commands[/]
  [cyan]/help[/]    show this help
  [cyan]/status[/]  connection, model, paths
  [cyan]/tools[/]   list the built-in system/network tools
  [cyan]/clear[/]   forget the conversation
  [cyan]/exit[/]    quit (Ctrl+D also works)

[bold]Safety[/]: read-only commands run automatically. Anything privileged, destructive,
sensitive or unknown asks you first. Some things are always blocked."""

_STATUS_STYLE = {"ok": "green", "failed": "red", "timeout": "yellow", "error": "red",
                 "blocked": "yellow", "denied": "yellow"}


_CATEGORY_STYLE = {"System Monitor": "green", "Network Investigator": "blue", "File Reader": "magenta",
                   "File Actions": "yellow", "User Management": "red"}


class RichUI(AgentUI):
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._status: Any = None            # the live spinner, if any (stopped while asking for approval)

    # ------------------------------------------------------------- static views
    def show_banner(self, config: Config, status: OllamaStatus) -> None:
        c = self.console
        c.print(Text(BANNER, style="bold cyan"), end="")
        c.print("\n       [bold]Your Personal Linux Assistant[/]  [dim]v" + __version__ + "[/]\n")
        c.print(f"  Model:   [bold]{escape(config.model)}[/]")
        c.print(f"  Runtime: Ollama [dim]({escape(config.ollama_host)})[/]")
        c.print("  Status:  " + self._status_markup(status) + "\n")
        c.print("  [dim]Type a request, or /help[/]\n")

    @staticmethod
    def _status_markup(status: OllamaStatus) -> str:
        if not status.reachable:
            return f"[bold red]Disconnected[/] [dim]({escape(status.error or 'unknown error')})[/]"
        if not status.model_available:
            return "[bold yellow]Connected, but model not installed[/]"
        return "[bold green]Connected[/]"

    def show_help(self) -> None:
        self.console.print(Panel(HELP_TEXT, title="Help", border_style="cyan"))

    def show_status(self, config: Config, status: OllamaStatus, cwd: str, history_len: int) -> None:
        t = Table.grid(padding=(0, 2))
        t.add_row("Ollama", self._status_markup(status) + (f" [dim]v{status.version}[/]" if status.version else ""))
        t.add_row("Host", escape(config.ollama_host))
        t.add_row("Model", escape(config.model))
        if status.reachable:
            t.add_row("Installed", escape(", ".join(status.models) or "(none)"))
        t.add_row("Working dir", escape(cwd))
        t.add_row("Command log", escape(str(config.log_path)))
        t.add_row("Limits", f"{config.command_timeout:.0f}s per command, {config.max_agent_steps} steps per request")
        t.add_row("Memory", f"{history_len} messages")
        self.console.print(Panel(t, title="Status", border_style="cyan"))
        if status.reachable and not status.model_available:
            self.console.print(f"[yellow]Install the model with:[/] ollama pull {escape(config.model)}")
        if not status.reachable:
            self.console.print("[yellow]Start Ollama with:[/] sudo systemctl start ollama   [dim](or: ollama serve)[/]")

    def clear(self) -> None:
        self.console.clear()

    # ------------------------------------------------------------- agent hooks
    @contextmanager
    def thinking(self, phase: str = "thinking"):
        label = "analyzing results" if phase == "analyzing" else "thinking"
        count = 0
        with self.console.status(f"[dim]{label}...[/]", spinner="dots") as spinner:
            self._status = spinner

            def on_token(chunk: str) -> None:
                nonlocal count
                count += 1
                spinner.update(f"[dim]{label}... {count} tokens[/]")
            try:
                yield on_token
            finally:
                self._status = None

    @contextmanager
    def activity(self, label: str):
        with self.console.status(f"[dim]{escape(label)}...[/]", spinner="line") as spinner:
            self._status = spinner
            try:
                yield
            finally:
                self._status = None

    def show_tool_call(self, name: str, args: dict) -> None:
        shown = ", ".join(f"{k}={v}" for k, v in args.items())
        self.console.print(f"[bold blue]*[/] Calling tool [bold]{escape(name)}[/]([dim]{escape(shown)}[/])")

    def show_tool_result(self, result: ToolResult) -> None:
        if result.ok:
            style = _CATEGORY_STYLE.get(result.category, "white")
            self.console.print(Panel(Text((result.display or result.text).rstrip()), title=Text(f"[{result.category}]"), title_align="left",
                                     border_style=style, subtitle=f"{result.duration:.1f}s", subtitle_align="right"))
        else:
            self.console.print(Panel(Text(result.error), title=Text(f"[{result.tool}: {result.status}]"),
                                     title_align="left", border_style="yellow"))

    def show_command(self, command: str) -> None:
        self.console.print(f"[bold magenta]>[/] [bold]{escape(command)}[/]")

    def show_result(self, result: ExecResult) -> None:
        style = _STATUS_STYLE.get(result.status, "white")
        body = Text()
        out = result.stdout.rstrip()
        if out:
            lines = out.splitlines()
            shown = lines[:60]
            body.append("\n".join(shown))
            if len(lines) > len(shown):
                body.append(f"\n... ({len(lines) - len(shown)} more lines)", style="dim")
        if result.stderr.strip():
            body.append(("\n" if len(body) else "") + result.stderr.rstrip(), style="red")
        if not len(body):
            body.append(result.status if not result.ran else "(no output)", style="dim")
        sub = result.status + (f" · exit {result.exit_code}" if result.exit_code is not None else "")
        if result.duration:
            sub += f" · {result.duration:.1f}s"
        self.console.print(Panel(body, border_style=style, subtitle=sub, subtitle_align="right"))

    def show_answer(self, text: str) -> None:
        self.console.print(Panel(Markdown(text), title=Text("[AI Analysis]"), title_align="left", border_style="cyan"))

    def show_recommendation(self, text: str) -> None:
        self.console.print(Panel(Markdown(text), title=Text("[Recommendation]"), title_align="left", border_style="green"))

    def show_tools(self, tools) -> None:
        t = Table.grid(padding=(0, 2))
        for name in tools.names():
            spec = tools.get(name)
            t.add_row(f"[bold]{escape(name)}[/]", Text(spec.category, style="dim"), escape(spec.description))
        self.console.print(Panel(t, title="Built-in tools", border_style="cyan"))

    def show_notice(self, text: str) -> None:
        self.console.print(f"[yellow]{escape(text)}[/]")

    def show_error(self, text: str) -> None:
        self.console.print(f"[bold red]Error:[/] {escape(text)}")

    # ---------------------------------------------------------------- approval
    def ask_approval(self, command: str, assessment: Assessment) -> bool:
        strong = assessment.strong_confirmation
        body = Text()
        body.append(command + "\n", style="bold")
        body.append("\nType: ", style="dim")
        body.append(", ".join(assessment.categories), style="bold red" if strong else "bold yellow")
        for reason in assessment.reasons:
            body.append(f"\n - {reason}")
        expanded = " | ".join(" ".join(seg) for seg in assessment.segments)
        if expanded and expanded != command and len(expanded) < 600:
            body.append(f"\n\nWill actually run: {expanded}", style="dim")
        if self._status:                      # a spinner would garble the prompt
            self._status.stop()
        self.console.print(Panel(body, title="Waiting for approval", border_style="red" if strong else "yellow"))
        # escape(): Rich would otherwise treat "[y/N]" as a markup tag and silently drop it
        prompt = escape("Type 'yes' to allow it, anything else cancels: " if strong else "Allow it? [y/N]: ")
        try:
            answer = self.console.input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return False
        finally:
            if self._status:
                self._status.start()
        return answer == "yes" if strong else answer in ("y", "yes")
