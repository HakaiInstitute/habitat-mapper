"""Habitat-Mapper CLI for managing and using segmentation models."""

from __future__ import annotations

import io
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import humanize
from cyclopts import App, Parameter
from cyclopts.types import File, PositiveInt  # noqa: TC002
from loguru import logger
from pydantic import TypeAdapter, ValidationError
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.traceback import install

from habitat_mapper.accelerators import DEVICE_ENV_VAR, Device, find_accelerators, select_accelerators
from habitat_mapper.cache import model_cache
from habitat_mapper.config import BandOrder, KernelSize, TileSize
from habitat_mapper.registry import model_registry
from habitat_mapper.utils import is_url

if TYPE_CHECKING:
    from collections.abc import Callable

    from habitat_mapper.config import ModelConfig

app = App(name="hab", help="Habitat segmentation of geospatial imagery with AI.")
console = Console()


def _log_formatter(record: dict) -> str:
    """Log message formatter.

    Returns:
        Formatting string for log messages logged with Rich console
    """
    color_map = {
        "TRACE": "dim blue",
        "DEBUG": "cyan",
        "INFO": "bold",
        "SUCCESS": "green",
        "WARNING": "yellow",
        "ERROR": "bold red",
        "CRITICAL": "bold white on red",
    }
    lvl_color = color_map.get(record["level"].name, "cyan")
    # Escape the message so text in square brackets, like `habitat-mapper[cuda]`, isn't read as Rich markup
    record["extra"]["escaped_message"] = escape(record["message"])
    return (
        f"[not bold cyan]{{time:YYYY-MM-DD HH:mm:ss}}[/not bold cyan] | "
        f"[{lvl_color}]{{extra[escaped_message]}}[/{lvl_color}]"
    )


DEBUG = bool(os.getenv("DEBUG", False))

# Hide debug and trace messages unless DEBUG is set
LOG_LEVEL = "TRACE" if DEBUG else "INFO"

# Setup Rich logger handler
logger.remove()
logger.add(
    console.print,
    level=LOG_LEVEL,
    format=_log_formatter,
    colorize=None,
)

# Set by main(), so that errors give a failing exit code from the command line but not when called from Python
_exit_on_error = False


def _on_error(_: BaseException) -> None:
    if _exit_on_error and not DEBUG:
        sys.exit(1)


# Decorator to log error messages
logger_catch = logger.catch(
    message="Program interrupted due to {record[exception].type.__name__}: '{record[exception].value}'",
    reraise=DEBUG,
    onerror=_on_error,
)

# Install rich traceback formatting
if DEBUG:
    install()


def _follows(rule: object) -> Callable[[type, object], None]:
    """Make a cyclopts validator from one of the option rules in habitat_mapper.config, so both check the same things.

    Returns:
        The validator.
    """
    adapter = TypeAdapter(rule)

    def validator(type_: type, value: object) -> None:
        if value is None:
            return
        try:
            adapter.validate_python(value)
        except ValidationError as e:
            raise TypeError(e.errors()[0]["msg"].removeprefix("Value error, ")) from None

    return validator


def _existing_model_validator(type_: type, value: str) -> None:
    if value not in model_registry.list_model_names():
        raise TypeError("Specified model is not a valid option.")


def _existing_image_validator(type_: type, value: Path | str) -> None:
    """Validate that input path exists and is either a file (including a zipped .SAFE) or SAFE directory.

    Args:
        type_: The type being validated (unused, required by cyclopts)
        value: The path to validate

    Raises:
        TypeError: If path doesn't exist or is not a valid image input
    """
    path = Path(value) if isinstance(value, str) else value
    if not path.exists():
        raise TypeError(f"Input path does not exist: {path}")

    # Accept SAFE directories or any file
    if path.is_dir():
        if not path.name.endswith(".SAFE"):
            raise TypeError(f"Directory must be a Sentinel-2 SAFE format (name ends with .SAFE), got: {path.name}")
    elif not path.is_file():
        raise TypeError(f"Input must be a file or SAFE directory, got: {path}")


def _cache_status(cfg: ModelConfig) -> str:
    if not any(is_url(dep) for dep in cfg.dependencies):
        return "[green]Local[/green]"
    return "[green]Cached[/green]" if model_cache.is_cached(cfg) else "[yellow]Available[/yellow]"


@app.command
@logger_catch
def models() -> None:
    """List all available models with their latest revisions."""
    table = Table(title="[bold green]Available Models[/bold green]")
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column("Revision", style="magenta")
    table.add_column("Description", style="white")
    table.add_column("Status", style="green")

    for model_name in model_registry.list_model_names():
        cfg = model_registry.config(model_name)
        table.add_row(
            f"{model_name}{' [i][red](beta)[/red][/i]' if cfg.beta else ''}",
            cfg.revision,
            cfg.description or "No description",
            _cache_status(cfg),
        )

    console.print(table)


@app.command
@logger_catch
def revisions(
    model_name: Annotated[
        str,
        Parameter(
            help="The name of the model to run. Run `hab models` to see options",
            validator=_existing_model_validator,
            name=["--model", "-m"],
        ),
    ],
) -> None:
    """List all available revisions for a specific model."""
    table = Table(title=f"[bold green]Revisions for {model_name}[/bold green]")
    table.add_column("Revision", style="magenta")
    table.add_column("Latest", style="bright_yellow", justify="center")
    table.add_column("Description", style="white")
    table.add_column("Status", style="green")

    latest_revision = model_registry.get_latest_revision(model_name)
    for revision in model_registry.revisions(model_name):
        cfg = model_registry.config(model_name, revision)
        table.add_row(
            revision,
            "✓" if revision == latest_revision else "",
            cfg.description or "No description",
            _cache_status(cfg),
        )

    console.print(table)


@app.command
@logger_catch
def devices() -> None:
    """List the hardware Habitat-Mapper can run models on, in the order `--device auto` tries them."""
    auto = select_accelerators("auto")

    table = Table(title="[bold green]Available Devices[/bold green]")
    table.add_column("Device", style="cyan", no_wrap=True)
    table.add_column("Hardware", style="white")
    table.add_column("Execution Provider", style="magenta")
    table.add_column("Auto", style="green")
    for accelerator in find_accelerators():
        order = auto.index(accelerator) + 1 if accelerator in auto else None
        table.add_row(
            accelerator.device,
            accelerator.hardware,
            accelerator.ep_name,
            f"{order}" if order else "[dim]only with --device[/dim]",
        )
    console.print(table)

    packages = []
    for package in ("onnxruntime", "onnxruntime-gpu", "onnxruntime-ep-webgpu"):
        try:
            packages.append(f"{package} {version(package)}")
        except PackageNotFoundError:
            pass
    console.print(f"[dim]Installed: {', '.join(packages)}[/dim]")
    if device := os.environ.get(DEVICE_ENV_VAR):
        console.print(f"[dim]{DEVICE_ENV_VAR}={device}[/dim]")


@app.command
@logger_catch
def clean() -> None:
    """Clear the Habitat-Mapper model cache to free up space. Models will be re-downloaded as needed."""
    files = model_cache.files()

    if not files:
        panel = Panel(
            "[yellow]Model cache is already empty.[/yellow]",
            title="[bold blue]Cache Status[/bold blue]",
            border_style="blue",
        )
        console.print(panel)
        return

    total_size = sum(f.stat().st_size for f in files)

    # 3. Build a nice table to show the user
    table = Table(
        title=f"[bold red]Cache Contents ({humanize.naturalsize(total_size)})[/bold red]",
        box=box.SIMPLE,
        show_header=True,
    )
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Location", style="dim")
    table.add_column("Size", justify="right", style="green")

    # Limit displayed rows to avoid flooding terminal if there are 100s of files
    max_display = 15
    for i, file in enumerate(files):
        if i >= max_display:
            remaining = len(files) - max_display
            table.add_row(
                f"[italic]... and {remaining} more files[/italic]",
                "",
                humanize.naturalsize(sum(f.stat().st_size for f in files[i:])),
            )
            break

        # clean up path for display (replace /home/user with ~)
        try:
            display_path = str(file.parent).replace(str(Path.home()), "~")
        except ValueError:
            display_path = str(file.parent)

        table.add_row(file.name, display_path, humanize.naturalsize(file.stat().st_size))

    console.print(table)
    console.print()  # Add a little breathing room

    # 4. Confirm and Execute
    if not Confirm.ask(
        "[yellow]Are you sure you want to clear the model cache?[/yellow]",
        default=False,
    ):
        logger.warning("Model cache clearing aborted.")
        return

    with console.status("[bold red]Deleting files...[/bold red]"):
        for file, error in model_cache.clear():
            logger.warning(f"Could not delete {file}: {error}")

    panel = Panel(
        f"[green]✓ Cleared model cache.\nFreed {humanize.naturalsize(total_size)}[/green]",
        title="[bold green]Cache Cleared[/bold green]",
        border_style="green",
    )
    console.print(panel)


@app.command
@logger_catch
def segment(
    model_name: Annotated[
        str,
        Parameter(
            help="The name of the model to run. Run `hab models` to see options",
            validator=_existing_model_validator,
            name=["--model", "-m"],
        ),
    ],
    img_path: Annotated[
        Path,
        Parameter(
            help="Path to input raster file (GeoTIFF, etc.), or Sentinel-2 SAFE directory or its .zip download",
            validator=_existing_image_validator,
            name=["--input", "-i"],
        ),
    ],
    output_path: Annotated[
        File,
        Parameter(
            help="Path to the output raster that will be created",
            name=["--output", "-o"],
        ),
    ],
    revision: Annotated[
        str,
        Parameter(
            help="The revision of the model to use. Run `hab revisions <model_name>` to see options",
            name=["--revision", "--rev"],
        ),
    ] = "latest",
    batch_size: Annotated[
        PositiveInt,
        Parameter(
            help="Batch size for processing",
        ),
    ] = 1,
    crop_size: Annotated[
        int | None,
        Parameter(
            help="Tile size for processing (must be even). Defaults to the 1024 or to the size required by the model",
            validator=_follows(TileSize),
            name=["--crop-size", "-z"],
        ),
    ] = None,
    blur_kernel_size: Annotated[
        int,
        Parameter(
            help="Size of median blur kernel (must be odd, 0 to disable)",
            validator=_follows(KernelSize),
            name=["--blur"],
        ),
    ] = 0,
    morph_kernel_size: Annotated[
        int,
        Parameter(
            help="Size of morphological kernel (must be odd, 0 to disable)",
            validator=_follows(KernelSize),
            name=["--morph"],
        ),
    ] = 0,
    band_order: Annotated[
        list[int] | None,
        Parameter(
            help="Band reordering flag for rearranging bands into RGB(+NIR) order when necessary.",
            validator=_follows(BandOrder),
            name=["--band", "-b"],
            negative="",
        ),
    ] = None,
    device: Annotated[
        Device | None,
        Parameter(
            help=(
                f"Hardware to run the model on. `auto` uses the fastest available device. Run `hab devices` to see "
                f"what is available. Defaults to the {DEVICE_ENV_VAR} environment variable, or `auto`"
            ),
        ),
    ] = None,
    quiet: Annotated[
        bool,
        Parameter(
            name=["--quiet", "-q"],
            negative="--no-quiet",
            help="Suppress progress output and non-error log messages (errors are still shown).",
        ),
    ] = False,
) -> None:
    """Apply a segmentation model to an input raster and save the output."""
    if quiet:
        logger.remove()
        logger.add(console.print, level="ERROR", format=_log_formatter, colorize=None)

    try:
        # An unknown revision raises, so that `hab segment` exits with an error code scripts can detect
        cfg = model_registry.config(model_name, revision)
        model = cfg.model_cls(cfg)

        if not quiet:
            # Show processing info in a panel
            info_panel = Panel(
                f"[blue]Model:[/blue] {model_name}\n"
                f"[blue]Revision:[/blue] {cfg.revision}\n"
                f"[blue]Input:[/blue] {img_path}\n"
                f"[blue]Output:[/blue] {Path(output_path).expanduser()}",
                title="[bold blue]Processing Configuration[/bold blue]",
                border_style="blue",
            )
            console.print(info_panel)

        model.process(
            img_path=img_path,
            output_path=Path(output_path).expanduser(),
            batch_size=batch_size,
            crop_size=crop_size,
            blur_kernel_size=blur_kernel_size,
            morph_kernel_size=morph_kernel_size,
            band_order=band_order,
            quiet=quiet,
            device=device,
        )

        if not quiet:
            # Show a success message
            success_panel = Panel(
                f"[green]✓ Successfully processed {img_path.name}\n"
                f"Output saved to: {Path(output_path).expanduser()}[/green]",
                title="[bold green]Processing Complete[/bold green]",
                border_style="green",
            )
            console.print(success_panel)
    finally:
        if quiet:
            logger.remove()
            logger.add(console.print, level=LOG_LEVEL, format=_log_formatter, colorize=None)


def main() -> None:
    """Run the command line interface."""
    global _exit_on_error  # noqa: PLW0603
    _exit_on_error = True
    # Output piped on Windows is encoded with the ANSI code page, which can't encode symbols like ✓. Replace them
    # instead of failing
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    app()


if __name__ == "__main__":
    main()
