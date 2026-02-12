"""QR-X structured logging with audit trail and debug tracing."""

import functools
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone

# Custom AUDIT level (between WARNING=30 and ERROR=40)
AUDIT = 35
logging.addLevelName(AUDIT, "AUDIT")


def _truncate(value: object, max_len: int = 80) -> str:
    """Truncate a string for safe logging."""
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


class JsonFormatter(logging.Formatter):
    """Outputs one JSON object per line for machine parsing."""

    def format(self, record):
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "src": record.name,
        }
        if hasattr(record, "event"):
            entry["event"] = record.event
        if hasattr(record, "duration_ms"):
            entry["duration_ms"] = round(record.duration_ms, 2)
        if hasattr(record, "ctx"):
            entry["ctx"] = record.ctx
        if record.getMessage() and not hasattr(record, "event"):
            entry["msg"] = record.getMessage()
        if record.exc_info and record.exc_info[1]:
            entry["traceback"] = traceback.format_exception(*record.exc_info)
        return json.dumps(entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable colored console output."""

    COLORS = {
        "DEBUG": "\033[36m",    # cyan
        "INFO": "\033[32m",     # green
        "AUDIT": "\033[35m",    # magenta
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",    # red
    }
    RESET = "\033[0m"

    def format(self, record):
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        color = self.COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:5s}{self.RESET}"
        src = f"[{record.name}]"

        parts = [ts, level, src]

        if hasattr(record, "event"):
            parts.append(record.event)

        if hasattr(record, "duration_ms"):
            parts.append(f"({record.duration_ms:.1f}ms)")

        if hasattr(record, "ctx") and record.ctx:
            ctx_str = " ".join(f"{k}={_truncate(v)}" for k, v in record.ctx.items())
            parts.append(ctx_str)
        elif record.getMessage() and not hasattr(record, "event"):
            parts.append(record.getMessage())

        if record.exc_info and record.exc_info[1]:
            parts.append(f"\n{''.join(traceback.format_exception(*record.exc_info))}")

        return " ".join(parts)


def setup_logging(level: str = "INFO", log_file: str | None = None, json_format: bool = False):
    """Configure the root qrx logger.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, AUDIT).
        log_file: If set, write JSON logs to this file path.
        json_format: If True, use JSON format on console too.
    """
    root = logging.getLogger("qrx")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(JsonFormatter() if json_format else ConsoleFormatter())
    root.addHandler(console)

    # File handler (always JSON)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)


def get_logger(module_name: str) -> logging.Logger:
    """Get a logger scoped under the qrx namespace."""
    return logging.getLogger(f"qrx.{module_name}")


def audit(event: str, logger: logging.Logger | None = None, **context):
    """Emit an AUDIT-level structured log entry.

    Args:
        event: Machine-readable event tag (e.g., "url.shortened").
        logger: Logger to use. Defaults to qrx root.
        **context: Key-value pairs for the event context.
    """
    log = logger or logging.getLogger("qrx")
    record = log.makeRecord(
        name=log.name,
        level=AUDIT,
        fn="",
        lno=0,
        msg="",
        args=(),
        exc_info=None,
    )
    record.event = event
    record.ctx = context
    log.handle(record)


def trace(func=None, *, logger_name: str | None = None):
    """Decorator that auto-logs function entry/exit with timing.

    - DEBUG on entry with arguments
    - INFO on exit with duration
    - ERROR on exception with traceback and duration
    """
    def decorator(fn):
        _logger_name = logger_name or fn.__module__.replace("qrx.", "")
        log = get_logger(_logger_name)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fn_name = fn.__name__

            # Log entry at DEBUG (skip if not enabled to avoid arg formatting cost)
            if log.isEnabledFor(logging.DEBUG):
                safe_args = []
                for a in args:
                    s = repr(a)
                    if len(s) > 100 or "Image" in type(a).__name__:
                        safe_args.append(f"<{type(a).__name__}>")
                    else:
                        safe_args.append(_truncate(s, 80))
                safe_kwargs = {k: _truncate(repr(v), 80) for k, v in kwargs.items()}
                record = log.makeRecord(
                    name=log.name, level=logging.DEBUG, fn="", lno=0,
                    msg="", args=(), exc_info=None,
                )
                record.event = f"{fn_name}.enter"
                record.ctx = {"args": safe_args, "kwargs": safe_kwargs}
                log.handle(record)

            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000

                # Log exit at INFO
                result_summary = type(result).__name__
                if isinstance(result, (str, int, float, bool)):
                    result_summary = _truncate(repr(result), 80)
                elif isinstance(result, (list, tuple)):
                    result_summary = f"{type(result).__name__}[{len(result)}]"
                elif isinstance(result, dict):
                    result_summary = f"dict[{len(result)} keys]"

                record = log.makeRecord(
                    name=log.name, level=logging.INFO, fn="", lno=0,
                    msg="", args=(), exc_info=None,
                )
                record.event = f"{fn_name}.done"
                record.duration_ms = elapsed
                record.ctx = {"result": result_summary}
                log.handle(record)

                return result
            except Exception:
                elapsed = (time.perf_counter() - start) * 1000
                record = log.makeRecord(
                    name=log.name, level=logging.ERROR, fn="", lno=0,
                    msg="", args=(), exc_info=sys.exc_info(),
                )
                record.event = f"{fn_name}.error"
                record.duration_ms = elapsed
                record.ctx = {"function": fn_name}
                log.handle(record)
                raise

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
