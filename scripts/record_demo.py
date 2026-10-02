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
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
            record_video_dir=str(OUT),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()
        video = page.video
        started = time.monotonic()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(1800)

        page.locator("#discover-input").fill("Интерстеллар")
        page.locator("#discover-button").click()
        page.locator(".discovery-item").first.wait_for()
        page.wait_for_timeout(1600)
        page.locator(".discovery-item").first.click()
        page.get_by_text("Материалы готовы. Задайте вопрос.").wait_for()
        page.wait_for_timeout(900)

        page.locator("#question").fill("Кто снял фильм и что известно о его создании?")
        page.locator("#question").scroll_into_view_if_needed()
        page.wait_for_timeout(1400)
        page.locator("#ask-button").click()
        loading = time.monotonic() - started
        page.locator("#answer-panel:not(.hidden)").wait_for(timeout=240_000)
        answer_ready = time.monotonic() - started
        page.locator("#answer-panel").scroll_into_view_if_needed()
        page.wait_for_timeout(8500)
        page.screenshot(path=str(OUT / "answer.png"), full_page=True)
        finished = time.monotonic() - started
        context.close()
        browser.close()
        raw = Path(video.path())

    output = OUT / "movie-rag-demo.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(f"Recording saved to {raw}; install ffmpeg to create MP4")
    # Keep a brief loading moment, then skip the model's idle inference time.
    first_end = min(loading + 2.0, answer_ready)
    second_start = max(first_end, answer_ready - 0.5)
    filters = (
        f"[0:v]trim=start=0:end={first_end:.3f},setpts=PTS-STARTPTS[v0];"
        f"[0:v]trim=start={second_start:.3f}:end={finished:.3f},"
        "setpts=PTS-STARTPTS[v1];"
        "[v0][v1]concat=n=2:v=1:a=0[v]"
    )
    subprocess.run([
        ffmpeg, "-y", "-i", str(raw), "-filter_complex", filters,
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "24", "-movflags", "+faststart", str(output),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Video: {output}")
    print(f"Model wait removed: {max(0, second_start - first_end):.1f} s")


if __name__ == "__main__":
    main()
