"""
Logging configuration for experiment mode.

Configures Python logging to:
- Suppress verbose utils logs in terminal output
- Preserve full DEBUG logs in debug.log file
- Keep user-friendly print statements visible
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_experiment_logging(
    run_dir: Path,
    level: str = "INFO",
    suppress_terminal: bool = True
) -> None:
    """
    Configure logging for experiment mode.
    
    Args:
        run_dir: Run directory for log files
        level: Base logging level (default: INFO)
        suppress_terminal: Whether to suppress verbose utils logs in terminal
    """
    # Ensure run directory exists
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file paths
    debug_log = run_dir / "debug.log"
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler: DEBUG level, captures everything
    file_handler = logging.FileHandler(debug_log, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Terminal handler: configurable level
    if suppress_terminal:
        # Only show WARNING and above for most loggers
        terminal_handler = logging.StreamHandler(sys.stdout)
        terminal_handler.setLevel(logging.WARNING)
        terminal_formatter = logging.Formatter("%(levelname)s: %(message)s")
        terminal_handler.setFormatter(terminal_formatter)
        root_logger.addHandler(terminal_handler)
        
        # Suppress specific verbose utils loggers
        verbose_loggers = [
            "utils.system_init",
            "utils.database",
            "utils.redis_client",
            "utils.redis_df",
            "utils.tools",
            "utils.data_prep",
            "monitoring.metrics",
            "monitoring.tracing",
            "monitoring.llm",
        ]
        
        for logger_name in verbose_loggers:
            logger_obj = logging.getLogger(logger_name)
            logger_obj.setLevel(logging.WARNING)
            # Prevent propagation to root logger for terminal output
            logger_obj.propagate = True
    else:
        # Show INFO level in terminal
        terminal_handler = logging.StreamHandler(sys.stdout)
        terminal_handler.setLevel(getattr(logging, level.upper()))
        terminal_formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s"
        )
        terminal_handler.setFormatter(terminal_formatter)
        root_logger.addHandler(terminal_handler)
    
    # Keep experiment tracking logs visible (INFO level)
    experiment_logger = logging.getLogger("monitoring.experiment")
    experiment_logger.setLevel(logging.INFO)
    
    logging.info(f"Experiment logging configured: debug_log={debug_log}")


def suppress_utils_logging() -> None:
    """
    Suppress verbose utils logging for terminal output only.
    This is a lightweight version for non-experiment mode.
    """
    verbose_loggers = [
        "utils.system_init",
        "utils.database",
        "utils.redis_client",
        "utils.redis_df",
        "utils.tools",
        "utils.data_prep.metadata",
        "utils.data_prep.related_tables",
    ]
    
    for logger_name in verbose_loggers:
        logger_obj = logging.getLogger(logger_name)
        # Only show WARNING and above
        logger_obj.setLevel(logging.WARNING)

