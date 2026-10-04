"""Exercise real streaming inference through Chrome's fake microphone.

Start an isolated server on 127.0.0.1:8766 with SPEECHLAB_DATA=test-results/live-data.
Looped natural phone crops are transport fixtures, not an accuracy benchmark.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright

from research.mandarin_features import crop
from speechlab import live

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test-results"
URL = "http://127.0.0.1:8766"


def run():
    pieces = []
    for c in live.get_map("mandarin")["categories"]:
        if c["phone"] not in ["i", "a", "s", "ʂ", "ɕ"]:
            continue
        wave = crop(c["examples"][0])
        wave = wave / max(0.001, np.sqrt(np.mean(wave**2))) * 0.06
        pieces.extend([np.tile(wave, 16), np.zeros(4800)])
    fake = OUT / "live-fake-microphone.wav"
    sf.write(fake, np.concatenate(pieces), 16000, subtype="PCM_16")
    responses, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={fake}",
            ],
        )
        context = browser.new_context(viewport={"width": 1280, "height": 1100}, permissions=["microphone"])
        page = context.new_page()
        page.add_init_script(
            """window.capturedTracks = []; const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices); navigator.mediaDevices.getUserMedia = async c => { const s = await original(c); window.capturedTracks.push(...s.getTracks()); return s; };"""
        )
        page.on("pageerror", lambda error: errors.append(str(error)))

        def response(r):
            if r.url.endswith("/frame") and r.status == 200:
                responses.append(r.json())

        page.on("response", response)
        page.goto(URL + "/live.html")
        page.wait_for_function("!document.getElementById('listen').disabled")
        assert page.locator("#references .reference").count() == 9
        page.screenshot(path=OUT / "live-shared.png", full_page=True)
        page.locator("#references .reference button").first.click()
        page.wait_for_function("document.getElementById('dot-label').textContent.startsWith('Reference')")
        page.locator("#listen").click()
        page.wait_for_function("document.getElementById('listen').dataset.live === 'true'")
        page.wait_for_timeout(16000)
        active = [r for r in responses if r.get("active")]
        assert len(active) >= 30, len(active)
        assert np.ptp([r["x"] for r in active]) > 0.5
        assert np.ptp([r["y"] for r in active]) > 0.5
        page.screenshot(path=OUT / "live-streaming.png", full_page=True)
        page.locator("#listen").click()
        page.wait_for_timeout(250)
        before = len(responses)
        page.wait_for_timeout(350)
        assert len(responses) == before
        assert page.evaluate("window.capturedTracks.every(t => t.readyState === 'ended')")
        page.locator("#listen").click()
        page.wait_for_function("document.getElementById('listen').dataset.live === 'true'")
        page.locator("#references .reference button").first.click()
        page.wait_for_timeout(250)
        assert page.evaluate("window.capturedTracks.every(t => t.readyState === 'ended')")
        assert page.locator("#listen").get_attribute("data-live") == "false"
        phone = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile = phone.new_page()
        mobile.on("pageerror", lambda error: errors.append(str(error)))
        mobile.goto(URL + "/live.html")
        mobile.wait_for_function("!document.getElementById('listen').disabled")
        assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth")
        mobile.screenshot(path=OUT / "live-mobile.png", full_page=True)
        browser.close()
    assert not errors, errors
    report = {
        "active_frames": len(active),
        "server_processing_ms_median": float(np.median([r["processing_ms"] for r in active])),
        "server_processing_ms_p95": float(np.percentile([r["processing_ms"] for r in active], 95)),
        "x_range": float(np.ptp([r["x"] for r in active])),
        "y_range": float(np.ptp([r["y"] for r in active])),
        "stop_and_reference_play_release_microphone": True,
        "mobile_overflow": False,
        "javascript_errors": errors,
    }
    (OUT / "live-browser-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
