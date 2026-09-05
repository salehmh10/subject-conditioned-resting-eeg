"""Read-only verification of one frozen batch; no network or raw waveform reads."""
from pathlib import Path
import argparse
import ast
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import pandas as pd
from resting_eeg.common import sha256_file,git_head
from resting_eeg.data import batches as b
from resting_eeg.data.paths import load_paths
from resting_eeg.data.mechanical_access import MechanicalHeldoutAccessGate,PURPOSES
from resting_eeg.data.access import TrainAccessGate,sample_interval
from resting_eeg.preprocessing import all_blocks as ab


def verify(batch_id='001',require_clean=False,test_report=None):
    p=b.paths(ROOT);storage=load_paths(ROOT);directory=p['metrics_root']/'batches'
    frame,plan,auth=b.validate_authorization(ROOT,batch_id,directory)
    b.verify_previous_state(ROOT,batch_id,full=True)
    ledger=b.ledger_read(directory)['batches']
    assert [e['batch_id'] for e in ledger]==[f'{i:03}' for i in range(1,int(batch_id)+1)]
    all_assigned=[s for e in ledger for s in b.verify_frozen(directory,e['batch_id'])[1]['subject_ids']]
    assert len(all_assigned)==len(set(all_assigned))
    expected_cap={'001':5*b.GIB,'002':10*b.GIB,'003':20*b.GIB}[batch_id]
    assert plan['cap_bytes']==expected_cap
    source=pd.read_csv(p['metrics_root']/'full_candidate_download_manifest.csv')
    prior=[s for e in ledger if e['batch_id']<batch_id for s in b.verify_frozen(directory,e['batch_id'])[1]['subject_ids']]
    expected_frame,expected=b.plan_membership(source,expected_cap,plan['excluded_verified_completed'],prior)
    assert expected_frame.to_dict('records')==frame.to_dict('records')
    assert expected['subject_ids']==plan['subject_ids']
    name=b.batch_name(batch_id)
    summary=b.read_json(directory/f'{name}_summary.json')
    outputs=pd.read_csv(directory/f'{name}_output_manifest.csv')
    results=pd.read_csv(directory/f'{name}_subject_results.csv').fillna('')
    runtime=b.runtime_directory(ROOT)
    history=b.read_journal(runtime/f'{name}_streaming_journal.jsonl')
    events=b.read_journal(runtime/f'{name}_download_journal.jsonl')
    assert len(results)==plan['subject_count'] and results.subject_id.tolist()==plan['subject_ids']
    assert set(results.status)=={'complete'} and len(outputs)==plan['subject_count']
    assert summary['subjects_completed']==plan['subject_count']
    assert sha256_file(runtime/f'{name}_streaming_journal.jsonl')==summary['streaming_journal_sha256']
    assert sha256_file(runtime/f'{name}_download_journal.jsonl')==summary['download_journal_sha256']
    measured=sum(r.get('received_body_bytes',0) for r in events)
    assert measured==summary['actual_measured_network_payload_bytes']
    assert set(r['subject_id'] for r in events)<=set(plan['subject_ids'])
    assert set(r['relative_path'] for r in events)<=set(frame.relative_path)
    preflight=b.read_json(directory/f'{name}_network_preflight.json')
    assert preflight['declared_bytes']==preflight['verified_existing_bytes']+preflight['valid_partial_bytes']+preflight['expected_new_network_bytes']
    # Every new subject starts after the preceding complete/cleanup decision.
    first_positions=[]
    for subject in plan['subject_ids']:
        positions=[i for i,r in enumerate(history) if r.get('subject_id')==subject]
        assert positions
        first_positions.append(min(positions))
        complete=[r for r in history if r.get('subject_id')==subject and r['status']=='complete'][-1]
        receipt=complete['result']
        checked=b.st._verify_receipt(receipt,subject,p['config_path'],p['output_root'])
        blocks=ab.expected_subject_blocks(ROOT,subject)
        assert receipt['authoritative_blocks']==blocks and len(blocks)==16
        params,param_hash=ab.verified_train_parameters(p['train_output_root']/f'{subject}_preprocessed.h5',
            b.load_config(p['config_path'])[0],b.FROZEN_SHA,blocks,p['metrics_root'])
        assert receipt['train_parameters']==params and receipt['train_parameter_source_hash']==param_hash
        row=outputs.set_index('subject_id').loc[subject]
        assert row.output_sha256==checked['output_sha256'] and row.output_bytes==checked['output_bytes']
        assert row.receipt_sha256==sha256_file(Path(checked['receipt_path']))
        raw=p['raw_root']/subject
        if complete['raw_retained']:
            assert raw.is_dir()
            if receipt['qc_pass']:
                assert complete.get('cleanup_blocker') or not complete.get('cleanup_requested')
        else:
            assert not raw.exists() and receipt['qc_pass']
            deleted=next(i for i,r in enumerate(history) if r.get('subject_id')==subject and r['status']=='raw_deleted')
            durable=next(i for i,r in enumerate(history) if r.get('subject_id')==subject and r['status']=='complete')
            verified=next(i for i,r in enumerate(history) if r.get('subject_id')==subject and r['status']=='processed_verified')
            assert verified<durable<deleted
    assert first_positions==sorted(first_positions)
    for previous,next_position in zip(plan['subject_ids'],first_positions[1:]):
        assert any(r.get('subject_id')==previous and r['status']=='complete' for r in history[:next_position])
    assert b.read_json(directory/f'{name}_canary_verification.json')['passed']
    pilot=set(pd.read_csv(p['metrics_root']/'pilot_download_manifest.csv').subject_id)
    allowed=pilot|set(all_assigned)
    heldout=MechanicalHeldoutAccessGate(p['metrics_root']/'provisional_block_split_manifest.csv',
        p['metrics_root']/'lemon_block_inventory.csv',p['metrics_root']/'pilot_mechanical_heldout_access_registry.csv',
        ROOT,p['access_config_path'],p['config_path'])
    registry=pd.read_csv(heldout.registry_path)
    assert set(registry.subject_id)<=allowed and set(registry.purpose)<=PURPOSES
    assert registry.access_allowed.all()
    for row in registry.to_dict('records'):
        block=heldout.authorize(row['subject_id'],row['block_index'],row['purpose'])
        assert block['split']==row['provisional_split']
        assert (row['native_start_sample'],row['native_stop_sample'])==sample_interval(block['start_sec'],block['end_estimate_capped_sec'],row['native_sfreq'])
    batch_registry=registry.loc[registry.subject_id.isin(plan['subject_ids'])]
    assert len(batch_registry)==6*plan['subject_count'] and not batch_registry.block_uid.duplicated().any()
    trainpath=p['metrics_root']/'streaming_train_access_registry.csv'
    train=pd.read_csv(trainpath)
    assert set(train.subject_id)==set(all_assigned) and set(train.split)=={'Train'}
    assert train.access_allowed.all() and len(train)==10*len(all_assigned) and not train.block_uid.duplicated().any()
    gate=TrainAccessGate(p['metrics_root']/'provisional_block_split_manifest.csv',p['metrics_root']/'lemon_block_inventory.csv',trainpath,ROOT)
    for row in train.to_dict('records'):
        block=gate.authorize(row['subject_id'],row['block_index'])
        assert (row['start_sample'],row['stop_sample_exclusive'])==sample_interval(block['start_sec'],block['end_estimate_capped_sec'],row['native_sfreq'])
    for path in (trainpath,heldout.registry_path):
        rows=pd.read_csv(path);new=rows.loc[rows.subject_id.isin(plan['subject_ids'])]
        head_column='code_git_head' if 'code_git_head' in new else 'git_head'
        authorizations=[b.read_json(directory/f'{b.batch_name(batch_id)}_execution_authorization.json')]
        authorizations += [b.read_json(f) for f in sorted(directory.glob(f'{b.batch_name(batch_id)}_resume_authorization_*.json'))]
        for row in new.to_dict('records'):
            matching=[a for a in authorizations if a['authorization_code_head']==row[head_column]
                      and pd.Timestamp(a['authorization_utc'])<=pd.Timestamp(row['timestamp_utc'])]
            assert matching
    committed=pd.Timestamp(subprocess.check_output(['git','show','-s','--format=%cI',auth['authorization_code_head']],cwd=ROOT,text=True).strip())
    assert committed<=pd.Timestamp(auth['authorization_utc'])
    pre=b.read_json(p['metrics_root']/'mechanical_preflight.json')
    assert sha256_file(p['config_path'])==pre['frozen_yaml_byte_sha256']
    assert sha256_file(p['metrics_root']/'pilot_waveform_access_registry.csv')==pre['train_registry_sha256']
    for name in ('all_blocks.py','signal.py'):
        tree=ast.parse((ROOT/'src/resting_eeg/preprocessing'/name).read_text())
        calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree)
               if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        assert not calls & {'compute_psd','psd','plot','wpli','savefig','fit_transform','StandardScaler'}
    for folder in ('data_raw','data_processed'):
        assert not any(f.is_file() for f in (ROOT/folder).rglob('*'))
    assert not list(storage['temp_root'].glob('*.h5')) and not list(storage['temp_root'].glob('*.tmp'))
    next_id=f'{int(batch_id)+1:03}'
    assert not list(directory.glob(f'batch_{next_id}*'))
    assert not list(runtime.glob(f'batch_{next_id}*'))
    assert (p['raw_root']/'sub-010250').is_dir()
    cleanup=b.read_json(directory/'pilot_cleanup_summary.json')
    assert len(cleanup['deleted_subjects'])==14 and cleanup['retained_subjects']==['sub-010250']
    assert all(not (p['raw_root']/s).exists() for s in cleanup['deleted_subjects'])
    assert set(b.verified_completed(ROOT))==allowed
    assert summary['normalization_fitted'] is False and summary['scientific_heldout_analysis'] is False
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    assert not any(Path(f).suffix.lower() in {'.eeg','.vhdr','.part','.h5','.hdf5','.npy','.npz'} for f in tracked)
    metadata_status=subprocess.check_output(['git','-c',f'safe.directory={storage["metadata_root"].as_posix()}',
        '-C',str(storage['metadata_root']),'status','--porcelain'],text=True)
    assert not metadata_status.strip()
    tests=None
    if test_report:
        suite=ET.parse(test_report).getroot()
        suites=[suite] if suite.tag=='testsuite' else list(suite.iter('testsuite'))
        assert all(int(s.attrib.get('failures',0))==int(s.attrib.get('errors',0))==0 for s in suites)
        tests=sum(int(s.attrib.get('tests',0)) for s in suites)
        assert tests>=267
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    if require_clean:assert not dirty
    return dict(verification='passed',batch_id=batch_id,subjects=len(outputs),blocks=16*len(outputs),
        measured_network_payload_bytes=measured,pytest_count=tests,git_head=git_head(ROOT),worktree_clean=not dirty,
        prohibited_next_batch=next_id,no_future_batch=True,prior_batch_unchanged=True,
        no_normalization=True,no_scientific_heldout_analysis=True,metadata_clone_clean=True,
        audit_limit='Application-level code, provenance, and registry verification; not OS-wide surveillance.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-id',default='001')
    parser.add_argument('--require-clean',action='store_true')
    parser.add_argument('--test-report',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.batch_id,args.require_clean,args.test_report),indent=2))
