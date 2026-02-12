# Add Logging — Audit & Debug Trace

Add structured logging with audit trail and debug tracing to the specified module or file.

## Requirements

Every log entry MUST include:
- **ISO 8601 timestamp** with milliseconds (`2026-02-12T14:32:01.123Z`)
- **Log level**: `DEBUG`, `INFO`, `WARN`, `ERROR`, `AUDIT`
- **Module/source**: file name and function name (e.g., `shorturl.shorten`)
- **Event name**: a short machine-readable event tag (e.g., `url.shortened`, `qr.generated`, `scan.verified`)
- **Duration**: elapsed time in ms for any operation that takes measurable time
- **Context**: relevant key-value pairs (url, key, version, ecc, mask, result, error, etc.)

## Log Format

Use Python `logging` module with a structured JSON formatter for machine parsing, plus a human-readable console formatter for development.

```python
# JSON line format (for files):
{"ts":"2026-02-12T14:32:01.123Z","level":"INFO","src":"shorturl.shorten","event":"url.shortened","duration_ms":2.1,"ctx":{"url":"https://...","key":"x8Y2"}}

# Console format (for dev):
14:32:01.123 INFO  [shorturl.shorten] url.shortened (2.1ms) url=https://... key=x8Y2
```

## Implementation Pattern

1. **Create `qrx/logging.py`** with:
   - `setup_logging(level, log_file, json_format)` — configure root logger
   - `get_logger(module_name)` — return a logger with the structured formatter
   - `@trace` decorator — auto-logs function entry/exit with args, return value, duration, and any exceptions
   - `audit(event, **context)` — shorthand for AUDIT-level structured log entry
   - Custom `AUDIT` log level (between WARNING and ERROR, value 35)

2. **Add `@trace` decorator** to all public functions in the target module. The decorator should:
   - Log `DEBUG` on function entry with arguments (redact sensitive values)
   - Log `INFO` on function exit with return value summary and duration
   - Log `ERROR` on exception with full traceback, duration, and arguments
   - Be zero-overhead when DEBUG logging is disabled

3. **Add `audit()` calls** at key business events:
   - URL shortened → `audit("url.shortened", url=..., key=..., mode=...)`
   - QR generated → `audit("qr.generated", url=..., version=..., ecc=..., mask=...)`
   - Scan verified → `audit("scan.verified", decoder=..., success=..., time_ms=...)`
   - Stress test completed → `audit("stress.completed", pass_rate=..., total=...)`
   - Any error → `audit("error", operation=..., error=..., traceback=...)`

4. **Wire into CLI** (`qrx/cli.py`):
   - Add `--verbose` / `-V` flag → sets DEBUG level
   - Add `--log-file` flag → writes JSON logs to file
   - Default level: INFO with console formatter

## What to log vs what NOT to log

**DO log:**
- All function entry/exit in public API (via @trace)
- Business events (shortened, generated, verified, etc.)
- Performance timings (every operation's duration_ms)
- Errors with full context and traceback
- Configuration changes

**DO NOT log:**
- Raw image data or large binary blobs
- Full URL destinations (truncate to first 80 chars)
- Internal loop iterations (log summary counts instead)

## Applying to target module

When the user specifies a file or module:
1. Read the file
2. Import and call `setup_logging()` at module level
3. Add `@trace` to all public functions
4. Add `audit()` calls at key business events
5. Ensure existing functionality is unchanged — logging is additive only
