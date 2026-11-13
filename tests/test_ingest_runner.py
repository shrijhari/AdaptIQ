import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ingest_runner
from ingest_runner import IngestionBusy, get_status, start_ingestion


def fake_script(tmp_path, body, name="fake_ingest.py"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "book.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def test_status_is_idle_when_nothing_ran(tmp_path):
    status = get_status(tmp_path / "missing.json")
    assert status["state"] == "idle"


def test_success_run_reports_stages_and_done(tmp_path, pdf):
    script = fake_script(tmp_path, """
        print("0/6 Checking Qwen...")
        print("2/6 Classifying sections...")
        print("   10/40 sections classified")
        print("   40/40 sections classified")
        print("6/6 Summary")
    """)
    status_path = tmp_path / "status.json"
    thread = start_ingestion([pdf], script=script, status_path=status_path)
    thread.join(timeout=20)

    status = get_status(status_path)
    assert status["state"] == "done"
    assert status["percent"] == 100
    assert status["stage_number"] == 6
    assert status["books"] == ["book.pdf"]
    assert status["error"] is None


def test_progress_lines_are_parsed(tmp_path, pdf):
    # Stay in stage 2 and exit non-zero so the last progress values remain.
    script = fake_script(tmp_path, """
        import sys
        print("2/6 Classifying sections...")
        print("   30/60 sections classified")
        sys.exit("stopped")
    """)
    status_path = tmp_path / "status.json"
    start_ingestion([pdf], script=script, status_path=status_path).join(timeout=20)

    status = get_status(status_path)
    assert (status["done"], status["total"]) == (30, 60)
    assert status["stage_number"] == 2


def test_failure_is_recorded_with_reason(tmp_path, pdf):
    script = fake_script(tmp_path, """
        import sys
        print("0/6 Checking Qwen...")
        sys.exit("Qwen is not running")
    """)
    status_path = tmp_path / "status.json"
    start_ingestion([pdf], script=script, status_path=status_path).join(timeout=20)

    status = get_status(status_path)
    assert status["state"] == "failed"
    assert "Qwen is not running" in status["error"]


def test_second_run_is_blocked_while_first_is_running(tmp_path, pdf):
    script = fake_script(tmp_path, """
        import time
        print("0/6 Checking Qwen...", flush=True)
        time.sleep(2)
    """)
    status_path = tmp_path / "status.json"
    thread = start_ingestion([pdf], script=script, status_path=status_path)
    with pytest.raises(IngestionBusy):
        start_ingestion([pdf], script=script, status_path=status_path)
    thread.join(timeout=20)

    # After it finishes, a new run is allowed again.
    again = start_ingestion([pdf], script=script, status_path=status_path)
    again.join(timeout=20)
    assert get_status(status_path)["state"] == "done"


def test_stale_running_status_does_not_block(tmp_path, pdf):
    status_path = tmp_path / "status.json"
    stale = ingest_runner._idle_status()
    stale.update(state="running", pid=999999)  # no such process
    ingest_runner._write_status(stale, status_path)

    script = fake_script(tmp_path, 'print("0/6 ok")')
    start_ingestion([pdf], script=script, status_path=status_path).join(timeout=20)
    assert get_status(status_path)["state"] == "done"


def test_bad_input_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        start_ingestion([], status_path=tmp_path / "s.json")
    with pytest.raises(ValueError):
        start_ingestion([tmp_path / "notes.txt"], status_path=tmp_path / "s.json")
    with pytest.raises(FileNotFoundError):
        start_ingestion([tmp_path / "missing.pdf"], status_path=tmp_path / "s.json")


def test_status_has_fields_the_upload_page_reads(tmp_path, pdf):
    script = fake_script(tmp_path, """
        print("2/6 Classifying sections...")
        print("   30/60 sections classified")
    """)
    status_path = tmp_path / "status.json"
    start_ingestion([pdf], script=script, status_path=status_path).join(timeout=20)
    status = get_status(status_path)
    assert status["progress"] == 1.0                 # finished: 0..1 fraction
    assert status["files"] == ["book.pdf"]
    assert status["updated_at"]
