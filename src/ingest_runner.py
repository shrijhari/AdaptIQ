"""
ingest_runner.py (AdaptIQ) - run textbook ingestion in the background.

The UI calls start_ingestion([...pdf paths...]) and then polls get_status()
(or reads data/ingestion_status.json) to draw a progress bar. Nothing here
depends on Streamlit.

    start_ingestion(pdf_paths) -> Thread | None
        Launches `python src/ingest_textbook.py <pdfs>` as a child process,
        reads its output line by line and keeps the status file up to date.
        Raises IngestionBusy if a run is already in progress.
    get_status() -> dict
        The latest status (or an "idle" status if nothing has run yet).

Status file (data/ingestion_status.json):
    {
      "state":   "idle" | "running" | "done" | "failed",
      "stage":   "2/6 Classifying sections with qwen2.5:3b ...",
      "stage_number": 2,           # 0..6, matches the script's "N/6" lines
      "done":    30,               # sections classified so far (stage 2)
      "total":   309,              # sections to classify (stage 2)
      "percent": 38,               # 0..100, handy for a progress bar
      "progress": 0.38,            # same thing as a 0..1 fraction
      "files":   ["ISLP.pdf"],     # same list as "books"
      "updated_at": "...",
      "message": "last line printed by the script",
      "books":   ["ISLP.pdf", ...],
      "error":   null | "why it failed",
      "pid":     12345,
      "started_at": "...", "finished_at": "..."
    }
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATUS_PATH = DATA_DIR / "ingestion_status.json"
INGEST_SCRIPT = BASE_DIR / "src" / "ingest_textbook.py"

# "2/6 Classifying sections ..."  (stage lines start at column 0)
STAGE_RE = re.compile(r"^(\d)/6\s+(.*)")
# "   30/309 sections classified"
PROGRESS_RE = re.compile(r"^\s+(\d+)/(\d+) sections classified")

_lock = threading.Lock()
_active: threading.Thread | None = None


class IngestionBusy(RuntimeError):
    """Raised when start_ingestion is called while a run is in progress."""


# ---------------------------------------------------------------- status file

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _idle_status() -> dict:
    return {"state": "idle", "stage": "", "stage_number": 0, "done": 0,
            "total": 0, "percent": 0, "progress": 0.0, "message": "",
            "books": [], "files": [], "error": None, "pid": None,
            "started_at": None, "finished_at": None, "updated_at": None}


def _write_status(status: dict, path: Path) -> None:
    """Write atomically so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    status["progress"] = round(status.get("percent", 0) / 100, 2)
    status["files"] = list(status.get("books", []))
    status["updated_at"] = _now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def get_status(status_path: Path | str | None = None) -> dict:
    path = Path(status_path) if status_path else STATUS_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return _idle_status()


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _percent(stage_number: int, done: int, total: int) -> int:
    fraction = (done / total) if (stage_number == 2 and total) else 0.0
    return min(100, int(100 * (stage_number + fraction) / 6))


# ------------------------------------------------------------------- the run

def _monitor(proc: subprocess.Popen, status: dict, path: Path) -> None:
    """Read the child's output line by line and keep the status file fresh."""
    tail: deque[str] = deque(maxlen=8)
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            tail.append(line.strip())
            status["message"] = line.strip()

            m = STAGE_RE.match(line)
            if m:
                status["stage_number"] = int(m.group(1))
                status["stage"] = line.strip()
                status["done"] = status["total"] = 0
            else:
                p = PROGRESS_RE.match(line)
                if p:
                    status["done"], status["total"] = int(p.group(1)), int(p.group(2))

            status["percent"] = _percent(
                status["stage_number"], status["done"], status["total"])
            _write_status(status, path)
        code = proc.wait()
    except Exception as exc:  # never let the monitor die silently
        proc.kill()
        code = -1
        tail.append(f"runner error: {exc}")

    status["finished_at"] = _now()
    status["pid"] = None
    if code == 0:
        status.update(state="done", percent=100, error=None)
    else:
        status.update(state="failed",
                      error=" | ".join(tail) or f"exited with code {code}")
        status["message"] = f"Ingestion failed (exit code {code})"
    _write_status(status, path)


def start_ingestion(
    pdf_paths,
    *,
    script: Path | str | None = None,
    status_path: Path | str | None = None,
) -> threading.Thread | None:
    """Start ingestion in the background. Returns the monitor thread.

    Raises ValueError for bad input and IngestionBusy if a run is already
    going. If the child process cannot even start, the failure is recorded
    in the status file and None is returned.
    """
    global _active
    paths = [Path(p) for p in (pdf_paths or [])]
    if not paths:
        raise ValueError("Give at least one PDF path.")
    for p in paths:
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {p}")
        if not p.is_file():
            raise FileNotFoundError(f"PDF not found: {p}")

    script = Path(script) if script else INGEST_SCRIPT
    path = Path(status_path) if status_path else STATUS_PATH

    with _lock:
        previous = get_status(path)
        if _active is not None and _active.is_alive():
            raise IngestionBusy("An ingestion run is already in progress.")
        # A run started by another process (e.g. before a restart) still counts,
        # but a status stuck on "running" with a dead process is stale.
        if previous["state"] == "running" and _pid_alive(previous.get("pid")):
            raise IngestionBusy("An ingestion run is already in progress.")

        status = _idle_status()
        status.update(state="running", stage="starting", message="Starting...",
                      books=[p.name for p in paths], started_at=_now())
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(script), *map(str, paths)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(BASE_DIR),
            )
        except OSError as exc:
            status.update(state="failed", error=f"Could not start: {exc}",
                          message="Could not start ingestion",
                          finished_at=_now())
            _write_status(status, path)
            return None

        status["pid"] = proc.pid
        _write_status(status, path)
        _active = threading.Thread(
            target=_monitor, args=(proc, status, path), daemon=True)
        _active.start()
        return _active
