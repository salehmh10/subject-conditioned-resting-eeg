"""Verify frozen mechanical pilot access, outputs, feasibility, and preservation.

Only fixed output verification and mask aggregation; no signal analysis or plots.
"""
from pathlib import Path
import sys
import json
import argparse
import subprocess
import ast
from collections import Counter
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import pandas as pd
import numpy as np
import h5py
from resting_eeg.common import sha256_file,write_json,canonical_hash,git_head
from resting_eeg.data.access import sample_interval
from resting_eeg.data.mechanical_access import MechanicalHeldoutAccessGate,PURPOSES,FROZEN_SHA,load_access_config
from resting_eeg.data.integrity import audit_pilot_inventory,audit_subject
from resting_eeg.preprocessing import all_blocks as ab
from resting_eeg.preprocessing.pipeline import load_config


def verify(write=False,require_clean=False,test_report=None):
    m=ROOT/'outputs/metrics/scenario2'; raw=ROOT/'data_raw/MPI-LEMON'
    pre=json.loads((m/'mechanical_preflight.json').read_text())
    config_path=ROOT/'configs/preprocessing/frozen_preprocessing_v1.yaml'
    config,digest=load_config(config_path)
    assert digest==FROZEN_SHA
    assert sha256_file(config_path)==pre['frozen_yaml_byte_sha256']
    assert sha256_file(config_path.with_suffix('.sha256'))==pre['frozen_sha_file_sha256']
    assert sha256_file(m/'pilot_waveform_access_registry.csv')==pre['train_registry_sha256']
    train=pd.read_csv(m/'pilot_waveform_access_registry.csv')
    assert len(train)==train.block_uid.nunique()==150 and set(train.split)=={'Train'}
    pilot=pd.read_csv(m/'pilot_download_manifest.csv');subjects=sorted(pilot.subject_id.unique())
    raw_inventory=audit_pilot_inventory(raw,pilot)
    current_raw={str(p.relative_to(ROOT)):sha256_file(p) for p in raw.rglob('*') if p.is_file()}
    assert current_raw==pre['raw_files'], 'Pilot raw files missing/changed or unexpected payload exists'
    for name,sha in pre['train_outputs'].items():assert sha256_file(ROOT/name)==sha
    inventory=pd.read_csv(m/'lemon_block_inventory.csv')
    full=pd.read_csv(m/'full_candidate_download_manifest.csv')
    spec,access_hash=load_access_config(ROOT/'configs/access/mechanical_heldout_v1.yaml',config_path)
    gate=MechanicalHeldoutAccessGate(m/'provisional_block_split_manifest.csv',m/'lemon_block_inventory.csv',
        m/'pilot_mechanical_heldout_access_registry.csv',ROOT,ROOT/'configs/access/mechanical_heldout_v1.yaml',config_path)
    registry=pd.read_csv(gate.registry_path)
    expected={b['block_uid'] for subject in subjects for b in ab.expected_subject_blocks(ROOT,subject) if b['split']!='Train'}
    assert len(registry)==registry.block_uid.nunique()==90 and set(registry.block_uid)==expected
    for registry_file in m.glob('*access_registry.csv'):
        assert set(pd.read_csv(registry_file).subject_id) <= set(subjects), 'Non-pilot waveform access'
    assert set(registry.subject_id)==set(subjects) and set(registry.provisional_split)=={'Validation','Test'}
    assert set(registry.purpose)<=PURPOSES and registry.access_allowed.all()
    assert set(registry.script_name)=={'resting_eeg.data.mechanical_access.MechanicalHeldoutBlockLoader'}
    assert set(registry.preprocessing_config_sha256)=={digest}
    assert set(registry.mechanical_access_config_sha256)=={access_hash}
    assert set(registry.split_manifest_sha256)=={gate.split_hash} and set(registry.block_inventory_sha256)=={gate.inventory_hash}
    for r in registry.to_dict('records'):
        b=gate.authorize(r['subject_id'],r['block_index'],r['purpose'])
        assert (r['state'],r['provisional_split'],r['start_sec'],r['end_sec'])==(b['state'],b['split'],b['start_sec'],b['end_estimate_capped_sec'])
        assert (r['native_start_sample'],r['native_stop_sample'])==sample_interval(b['start_sec'],b['end_estimate_capped_sec'],r['native_sfreq'])
        head=r['code_git_head']
        for path in ['configs/access/mechanical_heldout_v1.yaml','src/resting_eeg/data/mechanical_access.py','src/resting_eeg/preprocessing/all_blocks.py','tests/test_mechanical.py']:
            subprocess.check_output(['git','cat-file','-e',f'{head}:{path}'],cwd=ROOT)
        committed_at=pd.Timestamp(subprocess.check_output(['git','show','-s','--format=%cI',head],cwd=ROOT,text=True).strip())
        assert committed_at <= pd.Timestamp(r['timestamp_utc'])
    heads=sorted(set(registry.code_git_head))
    tree=ast.parse(Path(ab.__file__).read_text())
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls & {'fit_window_thresholds','fit_bad_channels','compute_psd','psd','plot','fit','wpli','savefig'}
    rows=[];processed=[];feasibility=[];cleanup=[];total=0
    journals=[json.loads(line) for line in (m/'streaming_all_block_status.jsonl').read_text().splitlines()]
    for subject in subjects:
        integrity=audit_subject(subject,full,raw,inventory,config['target_channels'])
        assert integrity['integrity_pass']
        blocks=ab.expected_subject_blocks(ROOT,subject)
        path=ROOT/'data_processed/MPI-LEMON-all-blocks-v1'/f'{subject}_all_blocks.h5'
        receipt=json.loads(path.with_suffix('.provenance.json').read_text())
        params,param_hash=ab.verified_train_parameters(ROOT/'data_processed/MPI-LEMON'/f'{subject}_preprocessed.h5',config,digest,blocks,m)
        assert params==receipt['train_parameters'] and param_hash==receipt['train_parameter_source_hash']
        checked=ab.verify_all_block_output(path,digest,blocks,access_hash,param_hash)
        assert checked['output_sha256']==receipt['output_sha256'] and receipt['source_hash_verified']
        completed=[r for r in journals if r.get('subject_id')==subject and r['status']=='complete']
        assert completed and completed[-1]['result']['output_sha256']==checked['output_sha256']
        assert not any(r['status']=='raw_deleted' for r in journals)
        total+=checked['output_bytes']
        processed.append(dict(subject_id=subject,output_path=str(path.relative_to(ROOT)),output_sha256=checked['output_sha256'],
            output_bytes=checked['output_bytes'],processed_verified=True,source_hash_verified=True,n_blocks=16,
            train_blocks_reused=10,heldout_blocks_processed=6,preprocessing_config_sha256=digest,
            mechanical_access_config_sha256=access_hash,train_parameter_source_hash=param_hash,processing_code_head=receipt['code_git_head']))
        with h5py.File(path,'r') as h,h5py.File(ROOT/'data_processed/MPI-LEMON'/f'{subject}_preprocessed.h5','r') as old:
            for b in blocks:
                g=h['blocks'][b['block_uid']]; mask=g['window_usable_mask'][:];reasons=g['window_qc_reason'][:].astype(str)
                if b['split']=='Train':
                    for ds in old['blocks'][b['block_uid']]:np.testing.assert_array_equal(g[ds][:],old['blocks'][b['block_uid']][ds][:])
                    for key,value in old['blocks'][b['block_uid']].attrs.items():assert g.attrs[key]==value
                counts=Counter(reason for text in reasons for reason in text.split('|') if reason!='usable')
                rows.append(dict(subject_id=subject,block_uid=b['block_uid'],block_index=b['block_index'],state=b['state'],provisional_split=b['split'],
                    candidate_windows=len(mask),usable_windows=int(mask.sum()),retained_fraction=float(mask.mean()),
                    block_eligible=bool(g.attrs['block_eligible']),finite=True,n_samples=g['eeg_uv'].shape[1],
                    rejection_reasons=json.dumps(counts,sort_keys=True)))
        frame=pd.DataFrame([r for r in rows if r['subject_id']==subject])
        result=dict(subject_id=subject,train_criterion_pass=params['summary']['qc_pass'],source_header_output_verified=True)
        reasons=[]
        if not result['train_criterion_pass']:reasons.append('frozen_Train_QC_failed')
        for split in ('Train','Validation','Test'):
            for state in ('EO','EC'):
                group=frame.loc[(frame.provisional_split==split)&(frame.state==state)]
                result[f'{state}_{split}_eligible_blocks']=int(group.block_eligible.sum())
                if split!='Train' and not group.block_eligible.all():reasons.append(f'{state}_{split}_assigned_block_ineligible')
        feasible=not reasons
        result.update(provisional_5_1_2_feasible=feasible,label='pilot_all_block_qc_feasible' if feasible else 'pilot_all_block_qc_not_feasible',reason=';'.join(reasons))
        feasibility.append(result)
        eligible=bool(params['summary']['qc_pass'])
        cleanup.append(dict(subject_id=subject,raw_directory=str(raw/subject),source_hash_verified=True,all_16_blocks_verified=True,
            output_sha256=checked['output_sha256'],provenance_verified=True,journal_complete=True,technically_eligible_for_later_cleanup=eligible,
            blocker='' if eligible else 'frozen_Train_subject_QC_failed',raw_deleted=False))
    frame=pd.DataFrame(rows)
    grouped=[]
    for (split,state),g in frame.groupby(['provisional_split','state']):
        grouped.append(dict(provisional_split=split,state=state,blocks=len(g),eligible_blocks=int(g.block_eligible.sum()),
                            candidate_windows=int(g.candidate_windows.sum()),usable_windows=int(g.usable_windows.sum())))
    qc=dict(subjects=15,blocks=240,train_subjects_pass=sum(r['train_criterion_pass'] for r in feasibility),
            provisional_5_1_2_feasible=sum(r['provisional_5_1_2_feasible'] for r in feasibility),by_split_state=grouped,
            final_cohort_locked=False,final_split_locked=False,normalization_fitted=False)
    storage=dict(subjects=15,blocks=240,pilot_bytes=total,pilot_MiB=total/1024**2,estimated_211_bytes=total*211/15,
                 estimated_211_GiB=total*211/15/1024**3,estimate_basis='measured pilot all-block mean times 211; excludes preserved Train outputs and receipts')
    summary=dict(subjects=15,train_records=150,train_registry_unchanged=True,heldout_records=90,unique_heldout_blocks=90,
        heldout_counts=registry.provisional_split.value_counts().to_dict(),train_blocks_reused=150,total_blocks=240,
        code_heads=heads,first_heldout_authorization_utc=registry.timestamp_utc.min(),authorization_commit_predates_reads=True,
        preprocessing_config_sha256=digest,mechanical_access_config_sha256=access_hash,
        no_nonpilot_raw_present=True,no_pilot_raw_deleted=True,raw_files=len(current_raw),raw_bytes=raw_inventory['raw_bytes'],
        no_heldout_parameter_fitting=True,no_scientific_heldout_computation=True,
        audit_limit='Application-level evidence: reviewed call graph, committed code, registries and provenance; not an OS-wide surveillance claim',
        technically_cleanup_eligible=sum(r['technically_eligible_for_later_cleanup'] for r in cleanup))
    tests=None
    if test_report:
        suites=list(ET.parse(test_report).getroot().iter('testsuite'))
        assert suites and all(int(s.attrib['failures'])==int(s.attrib['errors'])==0 for s in suites)
        tests=sum(int(s.attrib['tests']) for s in suites)
    if write:
        for name,data in [('pilot_all_block_qc.csv',rows),('pilot_all_block_subject_feasibility.csv',feasibility),
                          ('pilot_all_block_processed_manifest.csv',processed),('pilot_all_block_raw_cleanup_manifest.csv',cleanup)]:
            pd.DataFrame(data).to_csv(m/name,index=False)
        for name,data in [('pilot_all_block_qc_summary.json',qc),('pilot_all_block_storage_summary.json',storage),
                          ('pilot_mechanical_access_summary.json',summary)]:write_json(m/name,data)
    clean=not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    if require_clean:
        assert clean,'Final acceptance requires clean worktree'
        assert tests and tests >= 208, 'Passing JUnit report required for final acceptance'
        for name in ['README.md','docs/PROTOCOL.md','docs/DECISIONS.md','docs/MECHANICAL_HELDOUT_HANDOFF.md']:
            assert (ROOT/name).is_file()
        dry=json.loads((m/'all_block_streaming_dry_run_summary.json').read_text())
        assert dry['subjects']==211 and dry['all_disk_checks_pass'] and dry['value_reads']==dry['downloads']==dry['raw_deletions']==0
        repeated=json.loads((m/'pilot_all_block_idempotence_summary.json').read_text())
        assert repeated['subjects']==15 and repeated['all_reused'] and repeated['registry_unchanged'] and repeated['output_hashes_unchanged']
        assert len(list((ROOT/'data_processed/MPI-LEMON-all-blocks-v1').glob('*.h5')))==15
        assert not list((ROOT/'data_processed/MPI-LEMON-all-blocks-v1').glob('*.tmp'))
        tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        assert not any(Path(p).suffix.lower() in {'.eeg','.vhdr','.h5','.hdf5','.npy','.npz','.pt','.pth','.ckpt'} for p in tracked)
        for name,data in [('pilot_all_block_qc_summary.json',qc),('pilot_all_block_storage_summary.json',storage),('pilot_mechanical_access_summary.json',summary)]:
            assert json.loads((m/name).read_text())==data

    return dict(project_state='MECHANICAL_HELDOUT_PROCESSING_READY_FOR_FULL_STREAMING' if clean and require_clean else 'VERIFIED_PENDING_FINAL_ACCEPTANCE',
                git_head=git_head(ROOT),worktree_clean=clean,pytest_count=tests,qc=qc,storage=storage,access=summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true');parser.add_argument('--require-clean',action='store_true')
    parser.add_argument('--test-report',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.write,args.require_clean,args.test_report),indent=2))
