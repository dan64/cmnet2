"""
-------------------------------------------------------------------------------
Author: dan64
Date: 2026-09-11
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Activity watchdog for train_dinov3.py. A separate, lightweight thread that
periodically writes to a dedicated file (heartbeat.log) the current phase
and how long it has been since any progress was recorded - so a stall
(training OR validation, wherever it happens) is visible by comparing the
time of the last line with the current time, without having to wait for
the next checkpoint_every cadence.

ProgressState is the single shared point between the main training loop
(train_dinov3.py) and the per-video loop inside do_val() (trainer.py, via
the progress_cb parameter) - both call .update(phase) at every observable
advance (1 effective training step, or 1 video processed in validation),
so the watchdog detects a stall regardless of where it happens.
"""
import os
import threading
import time

import torch


class ProgressState:
    def __init__(self):
        self._lock = threading.Lock()
        self.phase = "initialization"
        self.last_progress_time = time.time()

    def update(self, phase):
        with self._lock:
            self.phase = phase
            self.last_progress_time = time.time()

    def snapshot(self):
        with self._lock:
            return self.phase, self.last_progress_time


def _format_line(phase, seconds_since_progress):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        vram = f"vram_allocated={allocated_gb:.2f}GB vram_reserved={reserved_gb:.2f}GB"
    else:
        vram = "vram_allocated=n/a vram_reserved=n/a (CUDA not available)"
    return (f"[{ts}] phase={phase} | last progress: {seconds_since_progress:.0f}s ago | {vram}")


class Watchdog:
    """Daemon thread: every interval_seconds writes a line to heartbeat_path.
    start()/stop() are idempotent; stop() is blocking (join) so the last
    line is written before the caller proceeds (useful in tests)."""
    def __init__(self, progress_state: ProgressState, heartbeat_path: str, interval_seconds: float = 30.0):
        self.progress_state = progress_state
        self.heartbeat_path = heartbeat_path
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        os.makedirs(os.path.dirname(self.heartbeat_path) or ".", exist_ok=True)
        while not self._stop_event.is_set():
            phase, last_progress_time = self.progress_state.snapshot()
            line = _format_line(phase, time.time() - last_progress_time)
            with open(self.heartbeat_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._stop_event.wait(self.interval_seconds)

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="heartbeat-watchdog", daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 5)
        self._thread = None
