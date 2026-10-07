"""Record a real browser walkthrough of the local app as an MP4 demo.

Requires Playwright, a Chromium-compatible browser and ffmpeg. Set DEMO_CHROME
to an installed browser binary to avoid a Playwright browser download.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"
URL = os.environ.get("MOVIE_RAG_URL", "http://127.0.0.1:8000")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        options = {"headless": True}
        if os.environ.get("DEMO_CHROME"):
            options["executable_path"] = os.environ["DEMO_CHROME"]
        browser = playwright.chromium.launch(**options)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            device_scale_factor=1,
            record_video_dir=str(OUT),
            record_video_size={"width": 1280, "height": 800},
        )
        page = context.new_page()
        video = page.video
        started = time.monotonic()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(900)

        page.locator("#discover-input").fill("Интерстеллар")
        page.locator(".discovery-item").first.wait_for()
        page.locator('.filter-chip[data-filter="movie"]').click()
        page.wait_for_timeout(1200)
        page.locator(".discovery-item").first.click()
        page.get_by_text("Можно задавать вопрос.").wait_for(timeout=30_000)
        page.wait_for_timeout(600)

        page.locator("#question").fill("Как Купер передал Мёрф данные из чёрной дыры?")
        page.locator("#question").scroll_into_view_if_needed()
        page.wait_for_timeout(1100)
        page.locator("#ask-button").click()
        page.get_by_text("Ответ готов.").wait_for(timeout=90_000)
        answer_ready = time.monotonic() - started
        if not page.locator(".source-card").count():
            raise RuntimeError("The demo answer has no visible sources")
        page.locator("#answer-panel").scroll_into_view_if_needed()
        page.wait_for_timeout(3500)
        page.screenshot(path=str(OUT / "answer.png"), full_page=True)
        context.close()
        browser.close()
        raw = Path(video.path())

    output = OUT / "movie-rag-demo.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(f"Recording saved to {raw}; install ffmpeg to create MP4")
    subprocess.run([
        ffmpeg, "-y", "-i", str(raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "24", "-movflags", "+faststart", str(output),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw.unlink()
    print(f"Video: {output}")
    print(f"Answer ready after {answer_ready:.1f} s of recording")


if __name__ == "__main__":
    main()
