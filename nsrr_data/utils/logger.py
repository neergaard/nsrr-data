import logging
from rich.logging import RichHandler
from typing import Optional


def get_logger(save_dir: Optional[str] = None):
    logger = logging.getLogger("nsrr_data")

    if not logger.hasHandlers():
        # Add handlers
        # shell_handler = logging.StreamHandler()
        shell_handler = RichHandler()
        if save_dir is not None:
            file_handler = logging.FileHandler(save_dir)

        # Set output levels
        logger.setLevel(logging.INFO)
        shell_handler.setLevel(logging.INFO)
        if save_dir is not None:
            file_handler.setLevel(logging.INFO)

        # Set formatter
        fmt_shell = "%(message)s"
        if save_dir is not None:
            fmt_file = "%(asctime)s | %(levelname)s | [%(filename)s:%(funcName)s:%(lineno)d] | %(message)s"

        # Attach the formatters
        shell_formatter = logging.Formatter(fmt_shell)
        if save_dir is not None:
            file_formatter = logging.Formatter(fmt_file)

        # Hook everything
        shell_handler.setFormatter(shell_formatter)
        if save_dir is not None:
            file_handler.setFormatter(file_formatter)
        logger.addHandler(shell_handler)
        if save_dir is not None:
            logger.addHandler(file_handler)

    return logger
