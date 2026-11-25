"""Run with Python and pass the path to the CLI-trained demo model."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import expect, sync_playwright


model = Path(sys.argv[1]).resolve()
with TemporaryDirectory() as root, subprocess.Popen(
    [sys.executable, "-m", "post_sentiment.web", "--port", "0", "--model", str(model)],
    cwd=root, stdout=subprocess.PIPE, text=True,
) as server:
    try:
        address = server.stdout.readline().strip().removeprefix("Open ")
        assert address.startswith("http://127.0.0.1:"), address
        with sync_playwright() as playwright:
            executable = os.environ.get("SENTIMENT_CHROMIUM")
            browser = playwright.chromium.launch(
                executable_path=executable, channel=None if executable else "chrome",
            )
            page = browser.new_page(viewport={"width": 1440, "height": 950})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(address)
            expect(page.get_by_role("heading", name="Twitter Sentiment Analysis")).to_be_visible()
            expect(page.get_by_role("heading", name="Confusion matrix")).to_be_visible()
            post = page.get_by_role("textbox", name="Tweet text")
            submit = page.get_by_role("button", name="Analyse tweet")
            expect(page.get_by_text("Fictional posts.", exact=False)).to_be_visible()
            page.get_by_role("button", name="Use post").first.click()
            expect(post).to_have_value("The rollout went smoothly and the support team was excellent.")
            expect(post).to_be_focused()
            post.fill("Great service this morning")
            submit.click()
            expect(page.locator("#scores progress")).to_have_count(3)
            expect(page.locator("#analysed-post")).to_have_text("Great service this morning")
            expect(page.get_by_role("status")).to_contain_text("Predicted sentiment:")
            assert page.locator(".hero-art").evaluate("(img) => img.complete && img.naturalWidth > 0")
            previews = os.environ.get("SENTIMENT_PREVIEWS")
            if previews:
                page.screenshot(path=Path(previews) / "sentiment-pulse-desktop.png", full_page=True)
            post.fill("🫠")
            expect(page.locator("#result")).to_be_hidden()
            submit.click()
            expect(page.get_by_role("status")).to_contain_text("training vocabulary")
            expect(page.locator("#result")).to_be_hidden()
            post.fill("   ")
            submit.click()
            expect(page.get_by_role("status")).to_have_text("Enter a post to analyse.")
            assert not errors, errors
            mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True)
            mobile.goto(address)
            mobile.keyboard.press("Tab")
            expect(mobile.get_by_role("link", name="Skip to composer")).to_be_focused()
            mobile.keyboard.press("Enter")
            expect(mobile.get_by_role("textbox", name="Tweet text")).to_be_focused()
            assert mobile.evaluate("document.documentElement.scrollWidth === innerWidth")
            if previews:
                mobile.locator(":focus").evaluate("(element) => element.blur()")
                mobile.evaluate("window.scrollTo(0, 0)")
                mobile.screenshot(path=Path(previews) / "sentiment-pulse-mobile.png", full_page=True)
    finally:
        server.terminate()
