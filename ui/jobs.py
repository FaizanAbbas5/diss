"""Background job runner for the dashboard.

Jobs are subprocesses (the existing experiments/*.py scripts) executed by a
worker thread, one job per key (a config path), with stdout/stderr appended
to a log file the UI tails. Module-level state survives Streamlit reruns and
is shared across browser sessions for the lifetime of the server process.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    key: str
    label: str
    steps: list[list[str]]
    log_path: Path
    cwd: Path
    status: str = "running"  # running | done | failed | cancelled
    returncode: int | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    proc: subprocess.Popen | None = None
    cancel_requested: bool = False

    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def get_job(key: str) -> Job | None:
    return _JOBS.get(key)


def running_jobs() -> list[Job]:
    return [j for j in _JOBS.values() if j.status == "running"]


def start_job(key: str, label: str, steps: list[list[str]], log_path: Path, cwd: Path) -> Job:
    """Start a job unless one with the same key is already running."""
    with _LOCK:
        existing = _JOBS.get(key)
        if existing and existing.status == "running":
            return existing
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        job = Job(key=key, label=label, steps=steps, log_path=log_path, cwd=Path(cwd))
        _JOBS[key] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return job


def cancel(key: str) -> None:
    job = _JOBS.get(key)
    if job and job.status == "running":
        job.cancel_requested = True
        if job.proc and job.proc.poll() is None:
            job.proc.terminate()


def _run(job: Job) -> None:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    # Append mode so the parent's status lines and the child's output never
    # overwrite each other (both file pointers stick to end-of-file).
    with open(job.log_path, "a", encoding="utf-8", errors="replace") as log:
        for i, argv in enumerate(job.steps):
            if job.cancel_requested:
                job.status = "cancelled"
                break
            log.write(("\n" if i else "") + "$ " + " ".join(argv) + "\n")
            log.flush()
            try:
                proc = subprocess.Popen(
                    argv, stdout=log, stderr=subprocess.STDOUT,
                    cwd=str(job.cwd), env=env,
                )
            except OSError as e:
                log.write(f"failed to start: {e}\n")
                job.status = "failed"
                job.returncode = -1
                break
            job.proc = proc
            rc = proc.wait()
            job.returncode = rc
            if job.cancel_requested:
                job.status = "cancelled"
                log.write("\n[cancelled]\n")
                break
            if rc != 0:
                job.status = "failed"
                log.write(f"\n[step exited with code {rc}]\n")
                break
        else:
            job.status = "done"
    job.finished = time.time()


def tail(path: Path, max_bytes: int = 8000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
        if size > max_bytes and "\n" in data:
            data = "…" + data.split("\n", 1)[1]
        return data
    except OSError:
        return ""
