"""inferhost command-line interface."""
from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from inferhost import __version__
from inferhost.core import binaries, configs, hf, paths, processes, probe, quant, registry
from inferhost.core.logs import log_path, tail
from inferhost.settings import settings

app = typer.Typer(
    name="inferhost",
    help="Run any Hugging Face model on your own GPU. No configs, no YAML.",
    no_args_is_help=False,
    add_completion=False,
)
gateway_app = typer.Typer(name="gateway", help="Manage the LiteLLM OpenAI-compatible gateway.")
app.add_typer(gateway_app, name="gateway")

console = Console()


# ---- helpers ----

def _resolve_model_filename(repo_id: str, prefer_quant: Optional[str]) -> hf.GgufFile:
    files = hf.list_ggufs(repo_id)
    if not files:
        raise typer.BadParameter(f"No .gguf files found in {repo_id}")
    if prefer_quant:
        for f in files:
            if f.quant and f.quant.upper() == prefer_quant.upper():
                return f
        console.print(f"[yellow]Requested quant {prefer_quant!r} not found; auto-picking.[/yellow]")
    vram = probe.probe().primary_vram_gib
    target = vram if vram > 0 else 8.0  # CPU fallback budget; user can override later
    pick = quant.pick_best(files, target)
    return pick or files[0]


def _add_model_to_registry(repo_id: str, prefer_quant: Optional[str], ctx: Optional[int]) -> registry.Model:
    paths.ensure_dirs()
    pick = _resolve_model_filename(repo_id, prefer_quant)
    console.print(f"Selected: [bold]{pick.filename}[/bold]  ({pick.quant or '?'}, {pick.size_gib} GiB)")
    console.print(f"Downloading from {repo_id} ...")
    local = hf.download_gguf(repo_id, pick.filename)
    reg = registry.load()
    name = hf.normalize_name(repo_id)
    if pick.quant:
        name = f"{name}-{pick.quant.lower().replace('_', '-')}"
    s = settings()
    model = registry.Model(
        name=name,
        repo_id=repo_id,
        filename=pick.filename,
        quant=pick.quant,
        ctx=ctx or s.default_ctx,
        port=reg.next_port(s.swap_port),
        size_gib=pick.size_gib,
        local_path=str(local),
    )
    reg.add(model)
    registry.save(reg)
    configs.write_all(reg)
    return model


# ---- commands ----

@app.command()
def install(
    skip_binaries: Annotated[bool, typer.Option("--skip-binaries", help="Just create dirs.")] = False,
) -> None:
    """First-time setup: download llama.cpp + llama-swap binaries, create dirs."""
    paths.ensure_dirs()
    console.print(f"[green]Created[/green] {paths.data_dir()}")
    console.print(f"[green]Created[/green] {paths.config_dir()}")
    if skip_binaries:
        return
    console.print("Fetching llama-server (llama.cpp) ...")
    server = binaries.install_llama_server()
    console.print(f"  → {server.path} ({server.version})")
    console.print("Fetching llama-swap ...")
    swap = binaries.install_llama_swap()
    console.print(f"  → {swap.path} ({swap.version})")
    console.print("[bold green]Install complete.[/bold green]")


@app.command()
def doctor() -> None:
    """Show environment summary: binaries, GPU, config paths."""
    pr = probe.probe()
    bins = binaries.installed_versions()
    table = Table(title="inferhost doctor", show_header=False, expand=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Version", __version__)
    table.add_row("OS / arch", f"{pr.os} / {pr.arch}")
    table.add_row("RAM", f"{pr.ram_gib} GiB")
    if pr.gpus:
        for g in pr.gpus:
            table.add_row(f"GPU {g.index}", f"{g.name} — {g.vram_total_gib} GiB ({g.vram_free_gib} free)")
    else:
        table.add_row("GPU", "none detected")
    table.add_row("Data dir", str(paths.data_dir()))
    table.add_row("Config dir", str(paths.config_dir()))
    table.add_row("llama-server", "installed" if bins["llama-server"] else "[red]missing[/red] (run `inferhost install`)")
    table.add_row("llama-swap", "installed" if bins["llama-swap"] else "[red]missing[/red] (run `inferhost install`)")
    table.add_row("litellm gateway", "available" if processes.gateway_available() else "not installed (optional)")
    console.print(table)
    for note in pr.notes:
        console.print(f"[yellow]Note:[/yellow] {note}")


@app.command()
def serve(
    repo_id: str = typer.Argument(..., help="Hugging Face repo id, e.g. Qwen/Qwen2.5-7B-Instruct-GGUF"),
    quant_pref: Annotated[Optional[str], typer.Option("--quant", help="Preferred quant (e.g. Q4_K_M)")] = None,
    ctx: Annotated[Optional[int], typer.Option("--ctx", help="Context length")] = None,
) -> None:
    """Add a Hugging Face model and start serving it (one-command path)."""
    model = _add_model_to_registry(repo_id, quant_pref, ctx)
    console.print(f"[green]Added[/green] {model.name}")
    st = processes.start_swap()
    console.print(f"[green]llama-swap[/green] {'running' if st.running else 'failed to start'} on port {st.port}")
    base = f"http://localhost:{st.port}/v1"
    console.print(Panel.fit(
        f"OpenAI-compatible endpoint:\n  [bold cyan]{base}[/bold cyan]\n\n"
        f"Try it:\n"
        f"  curl -s {base}/chat/completions \\\n"
        f"    -H 'Content-Type: application/json' \\\n"
        f"    -d '{{\"model\":\"{model.name}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'",
        title="Ready",
    ))


@app.command()
def add(
    repo_id: str = typer.Argument(...),
    quant_pref: Annotated[Optional[str], typer.Option("--quant")] = None,
    ctx: Annotated[Optional[int], typer.Option("--ctx")] = None,
) -> None:
    """Register a model without starting llama-swap."""
    model = _add_model_to_registry(repo_id, quant_pref, ctx)
    console.print(f"[green]Added[/green] {model.name}")


@app.command()
def start(
    name: Annotated[Optional[str], typer.Argument()] = None,
) -> None:
    """Start llama-swap (which lazy-spawns model backends on first request)."""
    if name is not None:
        console.print(
            "[yellow]Note:[/yellow] llama-swap loads models lazily on first request; "
            "starting the daemon serves all registered models."
        )
    reg = registry.load()
    configs.write_all(reg)
    st = processes.start_swap()
    console.print(f"llama-swap: {'running' if st.running else 'stopped'} (pid {st.pid}, port {st.port})")


@app.command()
def stop(
    all_: Annotated[bool, typer.Option("--all", help="Also stop the LiteLLM gateway.")] = False,
) -> None:
    """Stop llama-swap (and optionally the gateway)."""
    processes.stop_swap()
    if all_:
        processes.stop_gateway()
    console.print("Stopped.")


@app.command()
def restart() -> None:
    """Restart llama-swap with the current config."""
    processes.stop_swap()
    reg = registry.load()
    configs.write_all(reg)
    st = processes.start_swap()
    console.print(f"llama-swap: {'running' if st.running else 'failed'} (port {st.port})")


@app.command()
def ls() -> None:
    """List registered models and daemon status."""
    reg = registry.load()
    if not reg.models:
        console.print("No models registered. Try: [bold]inferhost serve <hf_repo_id>[/bold]")
        return
    swap = processes.swap_status()
    table = Table(title="Models", expand=False)
    table.add_column("name", style="bold cyan")
    table.add_column("repo")
    table.add_column("quant")
    table.add_column("size")
    table.add_column("ctx")
    table.add_column("port")
    for m in reg.models:
        table.add_row(m.name, m.repo_id, m.quant or "-", f"{m.size_gib} GiB", str(m.ctx), str(m.port))
    console.print(table)
    console.print(
        f"\nllama-swap: {'[green]running[/green]' if swap.running else '[red]stopped[/red]'}  "
        f"endpoint: http://localhost:{swap.port}/v1"
    )


@app.command()
def rm(name: str) -> None:
    """Remove a model from the registry. Does not delete the GGUF file from HF cache."""
    reg = registry.load()
    if not reg.remove(name):
        raise typer.BadParameter(f"No model named {name!r}")
    registry.save(reg)
    configs.write_all(reg)
    console.print(f"Removed {name}")


@app.command()
def logs(
    name: Annotated[str, typer.Argument(help="Model name, or 'swap' / 'gateway'")] = "swap",
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
    n: Annotated[int, typer.Option("--lines", "-n")] = 200,
) -> None:
    """Show logs."""
    path = log_path(name)
    if not path.exists():
        console.print(f"[yellow]No log file at {path}[/yellow]")
        return
    if follow:
        from inferhost.core.logs import follow as follow_log
        try:
            for line in follow_log(path):
                console.print(line, markup=False, highlight=False)
        except KeyboardInterrupt:
            return
    else:
        for line in tail(path, n):
            console.print(line, markup=False, highlight=False)


@app.command()
def status() -> None:
    """Show daemon status table."""
    swap = processes.swap_status()
    gw = processes.gateway_status()
    table = Table(title="Daemon status", expand=False)
    table.add_column("name", style="bold")
    table.add_column("status")
    table.add_column("pid")
    table.add_column("port")
    table.add_row(swap.name, "running" if swap.running else "stopped", str(swap.pid or "-"), str(swap.port or "-"))
    table.add_row(gw.name, "running" if gw.running else "stopped", str(gw.pid or "-"), str(gw.port or "-"))
    console.print(table)


@gateway_app.command("start")
def gateway_start() -> None:
    """Start the LiteLLM unified gateway."""
    if not processes.gateway_available():
        console.print(
            "[red]litellm not installed.[/red] Install with: "
            "pip install 'inferhost[gateway]'"
        )
        raise typer.Exit(1)
    st = processes.start_gateway()
    console.print(f"litellm gateway: {'running' if st.running else 'failed'} on port {st.port}")
    if st.running:
        console.print(f"  → http://localhost:{st.port}/v1")


@gateway_app.command("stop")
def gateway_stop() -> None:
    """Stop the LiteLLM gateway."""
    processes.stop_gateway()
    console.print("Gateway stopped.")


@app.command()
def tui() -> None:
    """Launch the interactive dashboard."""
    from inferhost.tui.app import run_tui
    run_tui()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    if version:
        console.print(f"inferhost {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":
    app()
