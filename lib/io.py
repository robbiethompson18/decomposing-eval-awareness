"""Generic file I/O helpers."""

import json
from typing import Any


def read_file(filepath: str) -> str:
    """Read text file contents."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_responses(response_file: str) -> tuple[list[dict[str, Any]], str]:
    """Load samples and model name from a response JSON file.

    Args:
        response_file: Path to response JSON file

    Returns:
        (samples, model) tuple
    """
    with open(response_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", []), data.get("model", "unknown")
