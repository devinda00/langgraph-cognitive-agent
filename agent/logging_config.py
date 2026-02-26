# agent/logging_config.py
"""
Logging configuration for the cognitive agent.
Agent logs go to a file (agent.log) so they can be viewed in a separate terminal
using: tail -f agent.log
"""
import logging
import os
import sys

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent.log")


def setup_logging():
    """
    Set up logging so that:
    - Agent internal logs go to agent.log (viewable via `tail -f agent.log`)
    - Only critical errors go to stderr (so the CLI stays clean)
    """
    # Clear the log file on startup
    with open(LOG_FILE, "w") as f:
        f.write("")

    logger = logging.getLogger("agent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — all agent logs
    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # Stderr handler — only critical errors
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.CRITICAL)
    stderr_handler.setFormatter(file_fmt)
    logger.addHandler(stderr_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'agent' namespace."""
    return logging.getLogger(f"agent.{name}")
