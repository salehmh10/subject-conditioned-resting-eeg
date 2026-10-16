"""Local-only launch, without installation, data preparation, or model execution."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys

APP_ROOT = Path(__file__).resolve().parents[1]


def available_port(preferred: int = 8501) -> int:
    if not 1024 <= preferred <= 65535:
        raise ValueError('Choose a port from 1024 through 65535.')
    for port in [*range(preferred, min(preferred + 30, 65536)), 0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                sock.bind(('127.0.0.1', port))
                return sock.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError('No free local port was available. Close an unused application and retry.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    if importlib.util.find_spec('streamlit') is None:
        print('Streamlit is missing. Run setup.ps1 in the application directory.')
        return 1
    if not (APP_ROOT / 'data/presentation_bundle/bundle.json').is_file():
        print('Presentation bundle missing. Run prepare_data.ps1 before launching.')
        return 1
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', STREAMLIT_BROWSER_GATHER_USAGE_STATS='false',
               STREAMLIT_SERVER_HEADLESS='true', STREAMLIT_SERVER_ADDRESS='127.0.0.1',
               STREAMLIT_SERVER_FILE_WATCHER_TYPE='none')
    for key, folder in [('TEMP', '.tmp'), ('TMP', '.tmp'), ('PIP_CACHE_DIR', '.cache/pip'),
                        ('PYTHONPYCACHEPREFIX', '.cache/pycache')]:
        dest = APP_ROOT / folder
        dest.mkdir(parents=True, exist_ok=True)
        env[key] = str(dest)
    port = available_port(args.port)
    print(f'EEG Synthetic Lab: http://127.0.0.1:{port}\nPress Ctrl+C to stop this application.', flush=True)
    try:
        return subprocess.call([sys.executable, '-m', 'streamlit', 'run', str(APP_ROOT / 'app.py'),
                                '--server.address=127.0.0.1', f'--server.port={port}',
                                '--server.headless=true', '--browser.gatherUsageStats=false'],
                               cwd=APP_ROOT, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
