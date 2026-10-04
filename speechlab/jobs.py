import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from . import store

POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speechlab")
JOBS = {}
LOCK = threading.Lock()


def submit(fn):
    with LOCK:
        if sum(j["status"] in ("queued", "running") for j in JOBS.values()) >= 10:
            raise ValueError("The queue is full. Wait for current analyses to finish.")
        ident = store.uid()
        JOBS[ident] = {"id": ident, "status": "queued", "message": "Waiting for the model worker"}

    def run():
        def progress(message):
            with LOCK:
                JOBS[ident].update(status="running", message=message)

        try:
            progress("Starting analysis")
            result = fn(progress)
            saved = store.save("results", result)
            with LOCK:
                JOBS[ident].update(status="done", message="Analysis complete", result=saved)
        except Exception as e:
            traceback.print_exc()
            with LOCK:
                JOBS[ident].update(status="error", message=f"{type(e).__name__}: {e}")

    POOL.submit(run)
    return JOBS[ident].copy()
