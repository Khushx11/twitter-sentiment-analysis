"""Run with Python and pass the path to the CLI-trained demo model."""

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
            browser = playwright.chromium.launch(channel="chrome")
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(address)
            expect(page.get_by_role("heading", name="Twitter Sentiment Analysis")).to_be_visible()
            expect(page.get_by_role("heading", name="Confusion matrix")).to_be_visible()
            post = page.get_by_role("textbox", name="Tweet text")
            submit = page.get_by_role("button", name="Analyse tweet")
            post.fill("Great service this morning")
            submit.click()
            expect(page.locator("#scores progress")).to_have_count(3)
            post.fill("🫠")
            expect(page.locator("#result")).to_be_hidden()
            submit.click()
            expect(page.get_by_role("status")).to_contain_text("training vocabulary")
            expect(page.locator("#result")).to_be_hidden()
            assert not errors, errors
    finally:
        server.terminate()
