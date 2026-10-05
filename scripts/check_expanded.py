"""Check live expanded-map capture, filters, frozen trace, references, and mobile layout.

Run an isolated server on 127.0.0.1:8766. The fake microphone plays real public
word crops, including brief stops/affricates; this is a software integration check.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright

from speechlab import live

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test-results"
URL = "http://127.0.0.1:8766"


def run():
    pieces = []
    m = live.get_map("mandarin-all")
    for c in m["categories"]:
        if c["label"] not in ["b", "p", "zh", "ch", "m", "i"]:
            continue
        wave, sr = sf.read(live.reference(m["id"], c["phone"], 0), dtype="float32")
        wave *= min(0.08 / max(0.001, np.sqrt(np.mean(wave**2))), 0.85 / max(0.001, np.max(np.abs(wave))))
        pieces.extend([wave, np.zeros(6400, dtype="float32")])
    fake = OUT / "expanded-microphone.wav"
    sf.write(fake, np.concatenate(pieces), 16000, subtype="PCM_16")
    responses = []
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={fake}",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1320, "height": 1050}, permissions=["microphone"], reduced_motion="reduce"
        )
        page = context.new_page()
        page.add_init_script(
            """window.capturedTracks=[];const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=async c=>{const s=await original(c);window.capturedTracks.push(...s.getTracks());return s;};"""
        )
        page.on("pageerror", lambda error: errors.append(str(error)))

        def received(r):
            if r.url.endswith("/frame") and r.status == 200:
                responses.append(r.json())

        page.on("response", received)
        page.goto(URL + "/live.html")
        page.wait_for_function('!document.getElementById("listen").disabled')
        assert page.locator("#map-select").input_value() == "mandarin-all"
        assert page.locator("#references .reference").count() == 28
        page.screenshot(path=OUT / "expanded-map.png", full_page=True)
        page.locator("#family-select").select_option("stops")
        assert page.locator("#references .reference").count() == 6
        page.locator("#listen").click()
        page.wait_for_function('document.getElementById("listen").dataset.live==="true"')
        page.wait_for_timeout(12000)
        page.locator("#family-select").select_option("affricates")
        assert page.locator("#listen").get_attribute("data-live") == "true"
        assert page.locator("#references .reference").count() == 6
        page.wait_for_timeout(3000)
        page.locator("#listen").click()
        page.wait_for_timeout(400)
        assert page.evaluate('window.capturedTracks.every(t=>t.readyState==="ended")')
        assert "frozen" in page.locator("#status").inner_text()
        before = len(responses)
        picture = page.locator("#plot").evaluate("(c)=>c.toDataURL()")
        page.wait_for_timeout(800)
        assert len(responses) == before
        assert page.locator("#plot").evaluate("(c)=>c.toDataURL()") == picture, "Frozen trace moved"
        page.screenshot(path=OUT / "expanded-frozen.png", full_page=True)
        active = [r for r in responses if r.get("active")]
        assert len(active) > 35, len(active)
        assert np.ptp([r["x"] for r in active]) > 0.5
        assert np.ptp([r["y"] for r in active]) > 0.5
        page.locator("#references .reference button").first.click()
        page.wait_for_function('document.getElementById("dot-label").textContent.startsWith("Reference")')
        page.locator("#map-select").select_option("mandarin")
        assert page.locator("#references .reference").count() == 9
        assert page.locator("#listen").get_attribute("data-live") == "false"
        page.locator("#map-select").select_option("mandarin-all")
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.on("pageerror", lambda error: errors.append(str(error)))
        mobile.goto(URL + "/live.html")
        mobile.wait_for_function('!document.getElementById("listen").disabled')
        assert mobile.evaluate("document.documentElement.scrollWidth<=innerWidth")
        mobile.screenshot(path=OUT / "expanded-mobile.png", full_page=True)
        browser.close()
    assert not errors, errors
    result = dict(
        active_frames=len(active),
        capture_seconds=15,
        processing_ms_median=float(np.median([r["processing_ms"] for r in active])),
        processing_ms_p95=float(np.percentile([r["processing_ms"] for r in active], 95)),
        x_range=float(np.ptp([r["x"] for r in active])),
        y_range=float(np.ptp([r["y"] for r in active])),
        filter_keeps_microphone_running=True,
        frozen_trace_stays_fixed=True,
        stop_releases_microphone=True,
        original_map_available=True,
        mobile_overflow=False,
        javascript_errors=errors,
        fixture="Real public word crops through Chrome fake microphone; software check, not a learner or consonant-recognition evaluation.",
    )
    (OUT / "expanded-browser-report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
