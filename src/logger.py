"""
Centralized logger for the ophthalmology assistant.

Usage in any module:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("...")
    log.error("...", exc_info=True)

Log file: logs/app.log (rotating, max 5 MB × 3 backups)
Console:  WARNING and above
"""
import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")
_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    _configured = True

    os.makedirs(_LOG_DIR, exist_ok=True)

    root = logging.getLogger("ophthal")
    root.setLevel(logging.DEBUG)

    # File handler — DEBUG and above, rotating
    fh = RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(fh)

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    _configure()
    # Strip "src." prefix so log names are short
    short = name.replace("src.", "")
    return logging.getLogger(f"ophthal.{short}")
