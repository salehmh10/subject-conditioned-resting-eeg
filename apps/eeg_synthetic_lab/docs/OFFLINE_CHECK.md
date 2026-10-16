# Offline presentation checks

Verified on 2026-10-10 with Microsoft Edge through Playwright, after the initial internet-assisted dependency installation.

The installed Streamlit application ran on `127.0.0.1:8501`. The browser request guard allowed only loopback/file/data/blob addresses and denied external requests. All seven pages and all ten guided steps rendered successfully. The recorded external-request list is empty. There were zero page JavaScript errors. Streamlit usage telemetry is explicitly disabled in `.streamlit/config.toml`, the launch environment and command-line arguments. Application source does not import an external service client or invoke remote inference.

The standalone `exports/EEG_Synthetic_Lab_Presentation.html` was then opened via `file://` with browser offline mode enabled. All ten steps worked, including interactive embedded Plotly figures. The downloaded utility-chart HTML also rendered in offline mode. HTML data and the Plotly JavaScript bundle are inline; there are no fetch-based local data dependencies, CDN scripts or external fonts. The snapshot applies a Content Security Policy denying network connections. Reference hyperlinks are optional reading and are not automatically fetched.

An actual CSV download and an actual interactive-chart HTML download were saved to the app-owned `exports/` directory. The chart HTML embeds Plotly. Browser logs and 14 screenshots are available in `browser-checks.json` and `screenshots/`.

This verifies browser-visible offline presentation, not a machine-wide disconnection of every unrelated Windows process. The test did not disable the user's network or terminate other applications. Initial package installation may require internet; no offline-installation claim is made.
