"""PDF upload UI and non-blocking textbook ingestion launcher."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"
DATA_DIR = ROOT / "data"
STATUS_PATH = DATA_DIR / "ingestion_status.json"
INGEST_SCRIPT = ROOT / "src" / "ingest_textbook.py"

STATUS_STEPS = [
    "Upload successful",
    "Textbook extraction in progress",
    "Knowledge-base classification in progress",
    "Chunking and embedding in progress",
    "FAISS indexing in progress",
    "Knowledge base ready",
]

def _write_status(status: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp = STATUS_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temp.replace(STATUS_PATH)

def read_ingestion_status() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"state": "idle", "message": "No textbook ingestion has been started.", "progress": 0}

def start_ingestion(pdf_paths: list[str]) -> None:
    """Launch the repository's existing ingestion script in a background thread."""
    if not pdf_paths:
        raise ValueError("Upload at least one PDF before starting ingestion.")
    if not INGEST_SCRIPT.exists():
        raise FileNotFoundError("src/ingest_textbook.py was not found.")
    current = read_ingestion_status()
    if current.get("state") == "running":
        return

    _write_status({
        "state": "running",
        "message": STATUS_STEPS[1],
        "progress": 0.10,
        "files": [Path(p).name for p in pdf_paths],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    def worker():
        try:
            command = [sys.executable, "-u", str(INGEST_SCRIPT), *[str(Path(p).resolve()) for p in pdf_paths]]
            process = subprocess.Popen(
                command, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            log_lines = []
            stage_markers = [
                ("0/6 Checking", 0.05, "Checking ingestion prerequisites"),
                ("1/6 Extracting", 0.20, STATUS_STEPS[1]),
                ("2/6 Classifying", 0.40, STATUS_STEPS[2]),
                ("3/6 Chunking", 0.65, STATUS_STEPS[3]),
                ("4/6 Embedding", 0.82, STATUS_STEPS[4]),
                ("5/6 building", 0.90, STATUS_STEPS[4]),
                ("6/6 Summary", 0.97, "Finalising knowledge base"),
            ]
            if process.stdout is not None:
                for line in process.stdout:
                    log_lines.append(line.rstrip())
                    for marker, fraction, label in stage_markers:
                        if marker.lower() in line.lower():
                            _write_status({
                                "state": "running", "message": label,
                                "progress": fraction,
                                "files": [Path(p).name for p in pdf_paths],
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                                "output": "\\n".join(log_lines[-30:]),
                            })
                            break
            return_code = process.wait()
            output = "\\n".join(log_lines)
            if return_code == 0:
                _write_status({
                    "state": "ready", "message": STATUS_STEPS[-1],
                    "progress": 1.0, "output": output[-1800:],
                    "files": [Path(p).name for p in pdf_paths],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
            else:
                _write_status({
                    "state": "failed", "message": "Textbook ingestion failed.",
                    "progress": 0.0, "error": (output or "Ingestion process failed.")[-2500:],
                    "files": [Path(p).name for p in pdf_paths],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:
            _write_status({
                "state": "failed", "message": "Could not start ingestion.",
                "progress": 0.0, "error": str(exc),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

    threading.Thread(target=worker, daemon=True).start()

def render_upload_page():
    st.header("2. Textbook upload")
    st.write("Upload one or more PDF textbooks. The diagnostic assessment remains available while ingestion runs.")
    KB_DIR.mkdir(parents=True, exist_ok=True)

    uploads = st.file_uploader(
        "Choose textbook PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Only PDF files are accepted.",
    )
    if uploads:
        if st.button("Save PDFs and start ingestion", type="primary"):
            saved_paths = []
            try:
                for uploaded in uploads:
                    if uploaded.type not in ("application/pdf", "application/x-pdf") and not uploaded.name.lower().endswith(".pdf"):
                        raise ValueError(f"{uploaded.name} is not a PDF.")
                    safe_name = Path(uploaded.name).name
                    destination = KB_DIR / safe_name
                    # Avoid overwriting an existing file accidentally.
                    if destination.exists():
                        destination = KB_DIR / f"{destination.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{destination.suffix}"
                    data = uploaded.getvalue()
                    if not data.startswith(b"%PDF"):
                        raise ValueError(f"{safe_name} does not appear to be a valid PDF file.")
                    destination.write_bytes(data)
                    saved_paths.append(str(destination))
                    if safe_name not in st.session_state.student.setdefault("uploaded_textbooks", []):
                        st.session_state.student["uploaded_textbooks"].append(safe_name)
                start_ingestion(saved_paths)
                st.success(f"Saved {len(saved_paths)} PDF file(s). Ingestion is running in the background.")
            except Exception as exc:
                st.error(f"Upload/ingestion error: {exc}")

    status = read_ingestion_status()
    state = status.get("state", "idle")
    progress = float(status.get("progress", 0.0) or 0.0)
    st.subheader("Ingestion status")
    st.progress(min(max(progress, 0.0), 1.0), text=status.get("message", "Waiting"))
    if state == "ready":
        st.success("Knowledge base ready.")
    elif state == "running":
        st.info("Ingestion is running. You can continue to the diagnostic assessment.")
        st.caption("Use the refresh button to check the latest status.")
    elif state == "failed":
        st.error(status.get("message", "Ingestion failed."))
        if status.get("error"):
            st.code(status["error"])
    else:
        st.info(status.get("message", "Upload a PDF to begin."))

    if status.get("files"):
        st.caption("Files: " + ", ".join(status["files"]))
    if status.get("output"):
        with st.expander("Ingestion log"):
            st.text(status["output"])

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Refresh ingestion status"):
            st.rerun()
    with c2:
        if st.button("Continue to diagnostic assessment"):
            st.session_state.stage = "Diagnostic assessment"
            st.session_state.student["current_stage"] = "Diagnostic assessment"
            st.rerun()
