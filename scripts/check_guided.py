"""Exercise the finished guided UI and real models on an isolated local test server.

SPEECHLAB_DATA=test-results/guided-data HF_HUB_OFFLINE=1 .venv/bin/uvicorn speechlab.app:app --port 8766
SPEECHLAB_TEST_URL=http://127.0.0.1:8766 .venv/bin/python scripts/check_guided.py
The fake microphone loops a corpus S crop. This is a transport/software test, not a learner evaluation.
"""

import io
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright

BASE = os.getenv("SPEECHLAB_TEST_URL", "http://127.0.0.1:8766")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-results"


def run():
    OUTPUT.mkdir(exist_ok=True)
    with httpx.Client(base_url=BASE, timeout=30) as client:
        catalog = client.get("/api/guided").json()
        first = catalog["contrasts"][0]
        source = client.get(f"/api/guided/{first['id']}/example/0/0/phone").content
        wave, sr = sf.read(io.BytesIO(source), dtype="float32")
        looped = np.tile(wave, int(np.ceil(2.0 * sr / len(wave))))[: sr * 2]
        fake = OUTPUT / "guided-fake-microphone.wav"
        sf.write(fake, np.pad(looped, (sr // 4, sr // 4)), sr, subtype="PCM_16")
    errors, page_results = [], []
    with sync_playwright() as p:
        chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        browser = p.chromium.launch(
            executable_path=chrome if Path(chrome).exists() else None,
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={fake}",
            ],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 1080}, permissions=["microphone"])
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE)
        page.wait_for_selector(".pair-card")
        assert page.locator(".pair-card").count() == len(catalog["contrasts"])
        page.screenshot(path=str(OUTPUT / "guided-home.png"), full_page=True)
        for c in catalog["contrasts"]:
            page.locator(f'[data-id="{c["id"]}"]').click()
            for side in (0, 1):
                page.locator(f'[data-word="{side}"]').click()
                page.locator(f'[data-analyze="{side}"]').click()
                page.wait_for_function(
                    "document.querySelector('#status').textContent.includes('analyzed with') || document.querySelector('#status').classList.contains('error')",
                    timeout=180000,
                )
                assert "error" not in page.locator("#status").get_attribute("class"), page.locator(
                    "#status"
                ).inner_text()
                assert page.locator(".needle").count() == 1
                page_results.append(
                    {"contrast": c["id"], "side": side, "result": page.locator("#result h3").inner_text()}
                )
            page.screenshot(path=str(OUTPUT / f"guided-{c['id']}.png"), full_page=True)
        page.locator(f'[data-id="{first["id"]}"]').click()
        page.locator("#record").click()
        page.wait_for_function("document.querySelector('#record').classList.contains('is-recording')")
        page.wait_for_function(
            "document.querySelector('#status').textContent.includes('Saved locally') || document.querySelector('#status').classList.contains('error')",
            timeout=180000,
        )
        assert "error" not in page.locator("#status").get_attribute("class"), page.locator(
            "#status"
        ).inner_text()
        assert page.locator("#replay").count() == 1
        assert page.locator(".window-dot").count() == 7
        page.locator("#replay").click()
        page.screenshot(path=str(OUTPUT / "guided-recording.png"), full_page=True)
        page.close()
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE)
        page.wait_for_selector(".pair-card")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(OUTPUT / "guided-mobile.png"), full_page=True)
        page.goto(BASE + "/evidence.html")
        page.wait_for_selector("#results table")
        assert page.locator("#results > .table-wrap tbody tr").count() == 7
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#results details").first.locator("summary").click()
        assert page.locator("#results details").first.locator("tbody tr").count() == 11
        assert not errors, errors
        browser.close()
    (OUTPUT / "guided-browser-report.json").write_text(
        json.dumps(
            {
                "ok": True,
                "page_errors": errors,
                "reference_results": page_results,
                "recording": "Looped corpus phone through Chrome fake microphone and MediaRecorder, seven windows returned",
                "mobile_width": 390,
                "tested_epoch": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(page_results, indent=2))
    print("Guided UI passed: real-model examples, audio playback, browser capture, mobile, evidence page.")


if __name__ == "__main__":
    run()
