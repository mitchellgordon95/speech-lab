"""Exercise the running app in Chrome, using generated audio (no microphone access).

Run: .venv/bin/python scripts/smoke_ui.py
Requires a server on 127.0.0.1:8765. Writes screenshots in ignored test-results/.
"""

import json
import os
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

BASE = os.getenv("SPEECHLAB_TEST_URL", "http://127.0.0.1:8765")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-results"
OUTPUT.mkdir(exist_ok=True)


def main():
    with httpx.Client(base_url=BASE, timeout=30) as client:
        client.post("/api/fixtures").raise_for_status()
        clips = [c for c in client.get("/api/clips").json() if c.get("source") == "synthetic-fixture-v1"]
        ids = [c["id"] for c in clips]
        sample = OUTPUT / "fake-microphone.wav"
        sample.write_bytes(client.get(f"/api/clips/{ids[0]}/audio").content)
    errors = []
    with sync_playwright() as p:
        chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        browser = p.chromium.launch(
            executable_path=chrome if Path(chrome).exists() else None,
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={sample}",
            ],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 1080}, permissions=["microphone"])
        context.add_init_script(f"localStorage.setItem('speechlab.clip', {json.dumps(ids[0])});")
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE)
        page.wait_for_selector(".demo-card")
        assert page.locator(".demo-card").count() == 8
        page.screenshot(path=str(OUTPUT / "overview.png"), full_page=True)

        def route(name):
            page.locator(f'#navigation a[href="#{name}"]').click()
            page.wait_for_selector("#run")

        def run(expected, timeout=180000):
            page.locator("#run").click()
            page.wait_for_function(
                "document.querySelector('#job').textContent === 'Analysis saved locally' || document.querySelector('#job').classList.contains('error')",
                timeout=timeout,
            )
            assert "error" not in page.locator("#job").get_attribute("class"), page.locator(
                "#job"
            ).inner_text()
            assert expected in page.locator("#result").inner_text()

        route("acoustics")
        run("Acoustic measurements")
        page.screenshot(path=str(OUTPUT / "acoustics.png"), full_page=True)
        # Saving an analysis must preserve synthetic fixture/session identity.
        assert page.locator("#session").input_value().startswith("fixture-day-")

        route("embeddings")
        page.locator("#compare-clips").select_option(ids[:3])
        run("Pairwise cosine distance")
        route("directions")
        page.locator("#negative-clips").select_option([c["id"] for c in clips if c["contrast"] == "A"])
        page.locator("#positive-clips").select_option([c["id"] for c in clips if c["contrast"] == "B"])
        page.locator("#axis-name").fill("Synthetic vowel A / B · smoke test")
        page.locator("#negative-label").fill("Synthetic A")
        page.locator("#positive-label").fill("Synthetic B")
        run("Held-out balanced accuracy")

        route("feedback")
        page.locator("#condition").select_option("hidden")
        run("Feedback is hidden")
        assert page.locator(".needle").count() == 0
        page.locator("#reveal").click()
        assert page.locator(".needle").count() == 1
        # An actual browser MediaRecorder path with a fake generated audio source.
        page.locator("#clip-label").fill("Browser capture · synthetic test input")
        page.locator("#record").click()
        page.wait_for_function("document.querySelector('#record').textContent.includes('Stop recording')")
        time.sleep(1.2)
        page.locator("#record").click()
        page.wait_for_function(
            "document.querySelector('.selected-clip strong').textContent.includes('Browser capture')"
        )
        assert page.locator("#speaker").input_value() == "me"
        run("Position along", timeout=180000)
        route("calibration")
        page.locator("#reference-clips").select_option(ids[:3])
        page.locator("#anchor-clips").select_option(ids[3:6])
        run("Two reference comparisons")
        route("providers")
        page.locator("#run").click()
        page.wait_for_selector("#job.error")
        assert "cloud" in page.locator("#job").inner_text().lower()
        route("articulation")
        run("Estimated articulator trajectories")
        page.screenshot(path=str(OUTPUT / "articulation.png"), full_page=True)
        route("omni")
        page.locator("#model").select_option("qwen-asr")
        page.locator("#compare-clips").select_option(ids[:2])
        run("Pairwise cosine distance")
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator('#navigation a[href="#home"]').click()
        page.screenshot(path=str(OUTPUT / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        browser.close()
    (OUTPUT / "browser-report.json").write_text(
        json.dumps({"ok": True, "page_errors": errors, "demos": 8}, indent=2)
    )
    print(
        "Browser smoke test passed: 8 routes, local analyses, cloud consent, fake microphone, mobile layout."
    )


if __name__ == "__main__":
    main()
