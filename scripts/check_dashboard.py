"""Run a local Chromium smoke test for the anonymized graph explorer."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromium")
    args = parser.parse_args()
    root = Path("results")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, executable_path=args.chromium)
        page = browser.new_page(viewport={"width": 1150, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((root / "flow-graph.html").resolve().as_uri())
        assert page.locator("#window option").count() == 6
        page.locator("#edges button").first.click()
        assert "Graph Δ" in page.locator("#details").inner_text()
        canvas = page.locator("#graph")
        width = canvas.bounding_box()["width"]
        height = canvas.bounding_box()["height"]
        canvas.click(position={"x":width/2, "y":height/2})
        assert "visible flows" in page.locator("#selection").inner_text()
        page.locator("#window").select_option("2")
        page.locator("#view").select_option("context")
        page.locator("#confirmed").check()
        assert "anonymized flows" in page.locator("#details").inner_text()
        page.locator("#confirmed").uncheck()
        page.locator("#view").select_option("rank")
        page.screenshot(path=root / "graph-explorer.png")
        page.set_viewport_size({"width":390, "height":844})
        assert canvas.bounding_box()["width"] < 390
        assert not errors, errors
        browser.close()
    print("Graph explorer passed interactive desktop and mobile checks.")


if __name__ == "__main__":
    main()
