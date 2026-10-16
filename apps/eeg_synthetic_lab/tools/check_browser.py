"""Headless local Edge checks; external browser network is denied for this test."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
os.environ['TEMP'] = os.environ['TMP'] = str(APP/'.tmp')
from playwright.sync_api import sync_playwright
from lab.config import PAGES, STEPS

PAGE_TITLES = {
    'Overview & models':'How close is synthetic EEG — and how useful?',
    'Signal explorer':'Look at the signal, without generating new data',
    'Fidelity explorer':'Similarity has more than one dimension',
    'Connectivity explorer':'Do the relationships between channels survive?',
    'Identity & diversity':'What remains distinctive about a participant?',
    'Classification utility':'Does synthetic EEG help eye-state prediction?',
    'Learning & methods':'See how checkpoints were selected',
}


def main():
    out = APP/'docs/screenshots'
    out.mkdir(parents=True, exist_ok=True)
    report = {'external_requests': [], 'console_errors': [], 'checks': [], 'screenshots': [], 'downloads': []}
    edge = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))/'Microsoft/Edge/Application/msedge.exe'
    with sync_playwright() as play:
        browser = play.chromium.launch(executable_path=str(edge), headless=True,
                                       args=['--disable-background-networking', '--disable-component-update', '--no-first-run'])
        context = browser.new_context(viewport={'width':1366,'height':768}, service_workers='block', accept_downloads=True)
        def guard(route):
            url = route.request.url
            parsed = urlparse(url)
            if parsed.scheme in ('file','data','blob') or parsed.hostname in ('127.0.0.1','localhost'):
                route.continue_()
            else:
                report['external_requests'].append(url)
                route.abort()
        context.route('**/*', guard)
        page = context.new_page()
        page.on('pageerror', lambda error: report['console_errors'].append(str(error)))

        def ready(title=None):
            if title:
                page.get_by_role('heading',name=title,exact=True).wait_for(timeout=45000)
            page.wait_for_function("!document.querySelector('[data-testid=stStatusWidget]') || !document.querySelector('[data-testid=stStatusWidget]').innerText.includes('Running')", timeout=45000)
            page.wait_for_timeout(350)

        def check(label):
            errors = page.locator('[data-testid="stException"]').all_text_contents()
            report['checks'].append({'label':label, 'exceptions':errors,
                'charts':page.locator('.js-plotly-plot').count(),
                'horizontal_overflow':page.evaluate('document.documentElement.scrollWidth > innerWidth + 2')})

        def capture(name, full=False):
            target=out/name
            page.screenshot(path=str(target), full_page=full)
            report['screenshots'].append(str(target.relative_to(APP)).replace('\\','/'))

        page.goto('http://127.0.0.1:8501', wait_until='networkidle', timeout=45000)
        page.get_by_role('radio', name='Overview & models', exact=True).wait_for(state='attached')
        ready()
        capture('overview-laptop-1366x768.png')
        for index, name in enumerate(PAGES):
            started=time.perf_counter()
            page.locator('[data-testid="stSidebar"]').get_by_text(name, exact=True).click()
            ready(PAGE_TITLES[name])
            check(name)
            report['checks'][-1]['browser_settle_seconds']=round(time.perf_counter()-started,3)
            capture(f'{index:02d}-laptop-1366x768.png')
            if name == 'Classification utility':
                # Download actual payloads, never rely on button presence alone.
                page.get_by_role('tab',name='Full results table',exact=True).click()
                csv_button = page.get_by_role('button', name='Download table CSV').first
                if csv_button.count():
                    with page.expect_download() as download_info:
                        csv_button.click()
                    download=download_info.value
                    path=APP/'exports'/('verified_'+download.suggested_filename)
                    download.save_as(str(path))
                    report['downloads'].append({'file':path.name,'bytes':path.stat().st_size})
                page.get_by_role('tab',name='Complete comparison',exact=True).click()
                expander = page.locator('summary:visible').filter(has_text='Download this chart').first
                if expander.count():
                    expander.click()
                else:
                    page.locator('summary:visible').filter(has_text='Download this chart and displayed data').first.click()
                page.get_by_role('button',name='Prepare interactive HTML',exact=True).first.click()
                page.get_by_role('button',name='Download interactive HTML',exact=True).first.wait_for()
                with page.expect_download() as download_info:
                    page.get_by_role('button',name='Download interactive HTML',exact=True).first.click()
                download=download_info.value
                path=APP/'exports'/('verified_'+download.suggested_filename)
                download.save_as(str(path))
                report['downloads'].append({'file':path.name,'bytes':path.stat().st_size})
        page.set_viewport_size({'width':1920,'height':1080})
        for index,name in enumerate(('Overview & models','Connectivity explorer','Classification utility')):
            page.locator('[data-testid="stSidebar"]').get_by_text(name,exact=True).click();ready(PAGE_TITLES[name])
            capture(f'{index:02d}-presentation-1920x1080.png')
            if name=='Connectivity explorer':
                capture('connectivity-full-page-1920.png',full=True)
        page.locator('[data-testid="stSidebar"]').get_by_text('Guided Presentation',exact=True).click();ready(STEPS[0])
        capture('guided-presentation-1920x1080.png')
        for step in range(10):
            check(f'guided step {step+1}')
            if step<9:
                page.get_by_role('button',name='Next',exact=False).click();ready(STEPS[step+1])
        page.get_by_role('button',name='Previous',exact=False).click();ready(STEPS[8])
        check('guided previous')
        # file:// is tested with offline mode, in addition to the deny-external route.
        context.set_offline(True)
        snapshot=APP/'exports/EEG_Synthetic_Lab_Presentation.html'
        page.goto(snapshot.as_uri(),wait_until='load')
        for step in range(10):
            page.wait_for_timeout(180)
            report['checks'].append({'label':f'offline HTML step {step+1}',
                                    'visible_sections':page.locator('section.slide.active').count(),
                                    'title':page.locator('section.slide.active h1').inner_text()})
            if step==5:
                capture('offline-html-connectivity-1920x1080.png')
            if step<9:
                page.get_by_role('button',name='Next',exact=True).click()
        downloaded_chart=APP/'exports/verified_eeg_utility_comparison.html'
        page.goto(downloaded_chart.as_uri(),wait_until='load')
        report['checks'].append({'label':'downloaded chart offline', 'charts':page.locator('.js-plotly-plot').count()})
        context.close();browser.close()
    (APP/'docs/browser-checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    failures=[c for c in report['checks'] if c.get('exceptions') or c.get('horizontal_overflow')]
    print(f"Browser checks: {len(report['checks'])}; failures: {len(failures)}; external requests: {len(report['external_requests'])}; JS errors: {len(report['console_errors'])}")
    return int(bool(failures or report['external_requests'] or report['console_errors']))


if __name__=='__main__':
    raise SystemExit(main())
