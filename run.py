"""One command to run everything locally: `python run.py` (API on :8000, UI on :8501).

`python run.py --demo` forces demo mode (no API key, no LLM calls).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

COMMANDS = [
    [sys.executable, "-m", "uvicorn", "copilot.api:app", "--host", "0.0.0.0", "--port", "8000"],
    [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port", "8501",
     "--server.address", "0.0.0.0", "--server.headless", "true", "--browser.gatherUsageStats", "false"],
]


def main() -> int:
    env = dict(os.environ)
    if "--demo" in sys.argv:  # overrides DEMO_MODE in .env for this run
        env["DEMO_MODE"] = "true"
    processes = [subprocess.Popen(command, env=env) for command in COMMANDS]
    print("\nCollections Copilot  |  UI: http://localhost:8501  |  API docs: http://localhost:8000/docs\n")
    try:
        while all(p.poll() is None for p in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
    return max((p.returncode or 0) for p in processes)


if __name__ == "__main__":
    raise SystemExit(main())
