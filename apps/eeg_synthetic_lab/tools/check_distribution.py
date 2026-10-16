"""Validate the shared archive and exercise its aggregate runtime in isolation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

APP=Path(__file__).resolve().parents[1]


def main():
    archive_path=APP/'exports/EEG_Synthetic_Lab.zip'
    target=(APP/'.tmp/distribution check with spaces').resolve()
    target.mkdir(parents=True,exist_ok=True)
    forbidden={'.venv','.cache','__pycache__','.git'}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name=info.filename.replace('\\','/')
            dest=(target/name).resolve()
            if not dest.is_relative_to(target) or '..' in Path(name).parts or set(Path(name).parts)&forbidden:
                raise ValueError('Unsafe ZIP path: '+name)
            if Path(name).suffix.lower() in {'.h5','.hdf5','.pt','.pth','.ckpt','.npz','.ttf','.woff','.woff2'}:
                raise ValueError('Unexpected raw/preview/model/font file: '+name)
            content=archive.read(info)
            if info.filename.endswith(('.py','.md','.json','.ps1','.html','.toml','.txt')):
                decoded=content.decode('utf-8-sig')
                if re.search(r'[A-Za-z]:[\\/](?:Users|SHARIF)[\\/]',decoded,re.I):
                    raise ValueError('Personal filesystem path in '+name)
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(content)
        members=len(archive.infolist())
    extracted=target/'eeg_synthetic_lab'
    data=json.loads((extracted/'data/presentation_bundle/bundle.json').read_text(encoding='utf-8'))
    assert data['metadata']['shared_export']
    assert len(data['connectivity_metrics'])==18
    assert all('subject_id' not in r for r in data['connectivity_metrics'])
    for section in ('connectivity','psd','alpha_effect'):
        assert not data[section].get('participants')
    code='''from streamlit.testing.v1 import AppTest
from lab.config import PAGES
app=AppTest.from_file('app.py',default_timeout=45).run()
assert not list(app.exception), [e.message for e in app.exception]
for name in PAGES:
    app.radio(key='navigation').set_value(name).run()
    assert not list(app.exception), (name,[e.message for e in app.exception])
app.radio(key='mode').set_value('Guided Presentation').run()
for index in range(10):
    assert not list(app.exception), [e.message for e in app.exception]
    if index<9:
        app.button(key='presentation_next').click().run()
print('Aggregate edition: seven pages and ten guided steps passed.')
'''
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',TEMP=str(APP/'.tmp'),TMP=str(APP/'.tmp'),STREAMLIT_BROWSER_GATHER_USAGE_STATS='false')
    result=subprocess.run([sys.executable,'-c',code],cwd=extracted,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    missing_env=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(extracted/'run.ps1')],
                               cwd=APP,env=env,capture_output=True,text=True,timeout=30)
    assert missing_env.returncode==1 and 'environment missing' in missing_env.stdout
    (APP/'docs/distribution-test-log.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    report={'members':members,'zip_bytes':archive_path.stat().st_size,'sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            'unsafe_members':0,'personal_paths':0,'participant_arrays':0,'aggregate_connectivity_rows':18,
            'isolated_runtime_exit_code':result.returncode,'runtime_stdout':result.stdout.strip(),
            'space_path_missing_environment_message':missing_env.stdout.strip()}
    (APP/'docs/distribution-checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
