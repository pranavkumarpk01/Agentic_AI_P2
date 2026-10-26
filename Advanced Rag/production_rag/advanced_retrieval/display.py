"""
display.py - small helpers that make terminal output easy to read in class.
Nothing RAG-specific lives here; it only prints things nicely.
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Pause between steps when running live in a terminal.
# Add  --no-pause  to the command to run straight through.
PAUSE = sys.stdin.isatty() and "--no-pause" not in sys.argv


def title(text):
    console.print()
    console.print(Panel(f"[bold]{text}[/bold]", style="bold magenta", expand=False))


def step(number, text):
    console.print()
    console.rule(f"[bold cyan]STEP {number}: {text}")


def note(text, style="blue"):
    console.print(Panel(text, border_style=style, expand=False))


def problem(text):
    note(f"[bold red]PROBLEM[/bold red]  {text}", style="red")


def fixed(text):
    note(f"[bold green]FIX[/bold green]  {text}", style="green")


def pause():
    if PAUSE:
        console.input("\n[dim]Press Enter to continue...[/dim]")


def short(text, limit=90):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def table(title_text, columns, rows):
    """Print a table. columns = list of header names, rows = list of lists."""
    t = Table(title=title_text, show_lines=True, title_justify="left")
    for col in columns:
        t.add_column(col, overflow="fold")
    for row in rows:
        t.add_row(*[str(cell) for cell in row])
    console.print(t)
