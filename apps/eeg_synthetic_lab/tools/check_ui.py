"""Exercise real Streamlit pages, selectors, reset and guided steps."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.dont_write_bytecode = True
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.chdir(APP)

from streamlit.testing.v1 import AppTest
from lab.config import PAGES


def main():
    results = []
    app = AppTest.from_file(str(APP/'app.py'), default_timeout=45)

    def run(label):
        start = time.perf_counter()
        app.run()
        exceptions = [e.message for e in app.exception]
        results.append({'check': label, 'seconds': round(time.perf_counter()-start, 3), 'exceptions': exceptions})
        if exceptions:
            print(label, exceptions, flush=True)
        return not exceptions

    run('initial overview')
    for page in PAGES:
        app.radio(key='navigation').set_value(page)
        if not run('page: '+page):
            continue
        keys = [s.key for s in app.selectbox]
        for key in keys:
            select = app.selectbox(key=key)
            if len(select.options) < 2:
                continue
            # Select by index: formats may differ from the underlying scientific identifier.
            select.select_index(len(select.options)-1)
            run(f'selector: {page} / {select.label}')
        for kind in ('checkbox', 'toggle'):
            for control in list(getattr(app, kind)):
                key, label = control.key, control.label
                if key:
                    getattr(app, kind)(key=key).set_value(not control.value)
                    run(f'{kind}: {page} / {label}')
        for button in list(app.button):
            if str(button.key).startswith('prepare:'):
                app.button(key=button.key).click()
                run(f'HTML download creation: {page} / {button.key}')
        next(b for b in app.button if b.label == 'Reset view').click()
        run('reset: '+page)
    app.radio(key='mode').set_value('Guided Presentation')
    run('guided: 1')
    for step in range(2, 11):
        buttons = [b for b in app.button if 'Next' in b.label and not b.disabled]
        if not buttons:
            results.append({'check': f'guided: {step}', 'exceptions': ['Next button missing or disabled']})
            break
        buttons[0].click()
        run(f'guided: {step}')
    previous = next((b for b in app.button if 'Previous' in b.label and not b.disabled), None)
    if previous:
        previous.click()
        run('guided: previous')
    (APP/'docs/ui-checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    failures = [r for r in results if r.get('exceptions')]
    print(f'{len(results)} UI checks; {len(failures)} failed.', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
