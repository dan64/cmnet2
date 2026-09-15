"""
-------------------------------------------------------------------------------
Author: dan64
Date: 2026-09-15
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Minimal stand-in for print() shared between trainer.py and
train_dinov3.py: prints to the console WITH a timestamp (unchanged behavior)
AND writes the same line to a log file, with an immediate flush after each
line - WITHOUT depending on an external tee (PowerShell/`| ForEach-Object`,
run_session.bat).

Why not the tee: verified with a real manual test (not just in theory) that
when train_dinov3.py is launched through a PowerShell pipe (`python ... 2>&1
| ForEach-Object {...}`, exactly the pattern of
run_session.bat/run_session-shutdown.bat) a Ctrl+C pressed in the console is
NO LONGER reliably delivered to the python.exe process - the SIGINT handler
(train_dinov3.py, _signal_handler) stops firing. Launching python DIRECTLY
in the console (no pipe in between) makes Ctrl+C work correctly. To have
BOTH (reliable Ctrl+C + persistent file log) the Python process itself must
write the file, not an external wrapper around its output pipe.

If init_log_file() is never called, log() behaves like a plain print() with
only the timestamp prepended (no file) - this lets diagnostic scripts/probes
that import trainer.py (training/verify/*.py, all CPU/GPU-only, never
launched through train_dinov3.py) keep working unchanged, without having to
initialize anything.
"""
import builtins
import time

_log_file = None


def init_log_file(path):
    """Opens (append, no truncation) the log file for this session - UTF-8
    WITHOUT BOM (Tee-Object produced UTF-16LE with BOM, awkward to grep with
    standard tools - avoided here from the start, `newline='\\n'` so that
    Python on Windows does not translate to CRLF on its own, same behavior as
    the old tee)."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
    _log_file = open(path, "a", encoding="utf-8", newline="\n")


def close_log_file():
    """Explicitly closes the log file, if open - not mandatory (the file is
    still flushed line by line thanks to the flush() in log(), an unclean
    process exit loses nothing already written), but good hygiene at session
    end."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def log(*args, **kwargs):
    """Drop-in replacement for print() (same signature - args/sep/end)
    PREPENDING a timestamp, same behavior as the print() redefined locally in
    trainer.py/train_dinov3.py before this change - now centralized in one
    place, and it ALSO writes to file (if init_log_file() was called), with
    an immediate flush after each line (the exact same effect of the old
    PowerShell tee '$w.WriteLine($_); $w.Flush()', but from inside the Python
    process - no external pipe, Ctrl+C delivered normally)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{ts}]", *args, **kwargs)
    if _log_file is not None:
        # Reuses the same builtins.print(), only redirected to file (via
        # 'file=' in kwargs) instead of rebuilding the line by hand with
        # sep.join() - an earlier attempt that did so produced a result
        # DIFFERENT from the console one every time the caller passed an
        # explicit 'sep' (the timestamp was joined with that sep only on the
        # console side, never on the file side) - bug found and fixed before
        # committing, no leftover.
        file_kwargs = dict(kwargs)
        file_kwargs["file"] = _log_file
        builtins.print(f"[{ts}]", *args, **file_kwargs)
        _log_file.flush()
