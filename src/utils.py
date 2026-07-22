from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_FILE = RESULTS_DIR / "experiments.log"
LOGGER_NAME = "gliner_reproduction"


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it unchanged."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_results_dir() -> Path:
    """Return the results directory, creating it on demand."""
    return ensure_directory(RESULTS_DIR)


def configure_logging(log_file: Path | None = None) -> logging.Logger:
    """
    Configure a shared project logger.

    The logger writes both to stdout and to ``results/experiments.log`` so that
    experiment metadata remains auditable across runs.
    """
    ensure_results_dir()

    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file or LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def serialize_labels(labels: Sequence[str]) -> str:
    """Persist label prompts as JSON for CSV reproducibility."""
    return json.dumps(list(labels), ensure_ascii=True)


def write_rows_to_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write experiment rows to CSV, overwriting previous contents."""
    ensure_directory(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text to disk, creating parent directories if needed."""
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")


def format_float(value: Any) -> str:
    """Render numeric values compactly for console tables."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Create a compact plain-text table without extra dependencies."""
    if not rows:
        return ""

    widths = {
        column: max(
            len(column),
            *(len(format_float(row.get(column, ""))) for row in rows),
        )
        for column in columns
    }

    def render(row: Mapping[str, Any]) -> str:
        return " | ".join(
            format_float(row.get(column, "")).ljust(widths[column])
            for column in columns
        )

    header = render({column: column for column in columns})
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = "\n".join(render(row) for row in rows)
    return f"{header}\n{separator}\n{body}"


def key_value_lines(items: Iterable[tuple[str, Any]]) -> str:
    """Render metadata blocks such as experiment summaries."""
    return "\n".join(f"{key}: {value}" for key, value in items)
