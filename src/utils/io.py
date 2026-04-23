"""I/O helpers for filesystem operations."""

from pathlib import Path


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
