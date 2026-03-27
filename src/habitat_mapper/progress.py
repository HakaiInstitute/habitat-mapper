"""Progress reporting abstractions for decoupled output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    from types import TracebackType


class ProgressTask(Protocol):
    """Handle to a single in-progress task."""

    def update(self, advance: int = 1) -> None:
        """Advance the task by the given amount."""
        ...


class ProgressReporter(Protocol):
    """Context manager that creates and tracks progress tasks."""

    def __enter__(self) -> Self:
        """Enter the progress context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the progress context."""
        ...

    def add_task(self, description: str, total: int | None = None) -> ProgressTask:
        """Register a new task and return a handle for updating it."""
        ...


class _NullTask:
    def update(self, advance: int = 1) -> None:
        pass


class NullProgressReporter:
    """No-op reporter that produces no output. Use for quiet mode."""

    def __enter__(self) -> Self:
        """Enter the progress context.

        Returns:
            self
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the progress context."""

    def add_task(self, description: str, total: int | None = None) -> _NullTask:
        """Return a no-op task handle."""
        return _NullTask()


class _RichTask:
    def __init__(self, progress: object, task_id: int) -> None:
        self._progress = progress
        self._task_id = task_id

    def update(self, advance: int = 1) -> None:
        self._progress.update(self._task_id, advance=advance, refresh=True)  # type: ignore[attr-defined]


class RichProgressReporter:
    """Rich-backed progress reporter with a default bar layout."""

    def __init__(self) -> None:
        """Instantiate the underlying Rich Progress bar."""
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

    def __enter__(self) -> Self:
        """Enter the progress context.

        Returns:
            self
        """
        self._progress.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the progress context."""
        self._progress.__exit__(exc_type, exc_val, exc_tb)

    def add_task(self, description: str, total: int | None = None) -> _RichTask:
        """Register a new task and return a handle for updating it.

        Returns:
            A task handle wrapping the Rich task ID.
        """
        task_id = self._progress.add_task(description, total=total)
        return _RichTask(self._progress, task_id)
