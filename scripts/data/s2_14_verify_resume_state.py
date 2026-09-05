"""Verify the manually interrupted Batch 001 without any network/value reads."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from resting_eeg.data import batches as b
from resting_eeg.data.environment import configure_environment
from resting_eeg.preprocessing import all_blocks as ab
from resting_eeg.common import sha256_file,write_json,utc_now


def verify():
    environment=configure_environment(ROOT)
    p=b.paths(ROOT);directory=p['metrics_root']/'batches';runtime=b.runtime_directory(ROOT)
    frame,plan,auth=b.validate_authorization(ROOT,'001',directory)
    protected=['batch_001_manifest.csv','batch_001_plan.json','batch_001_execution_authorization.json','batch_ledger.json']
    protected_hashes={name:sha256_file(directory/name) for name in protected}
    assert protected_hashes['batch_001_manifest.csv']=='21a5676eb5fcefbe5f776bf467d9eace8e3d63ee0ede9d102c6f5bcdf43fb865'
    assert plan['cap_bytes']==5368709120 and plan['declared_bytes']==5104883272
    assert plan['subject_ids']==['sub-010002','sub-010003','sub-010004','sub-010005','sub-010006','sub-010007','sub-010010','sub-010012']
    history=b.read_journal(runtime/'batch_001_streaming_journal.jsonl')
    prior=[r for r in history if r.get('subject_id')=='sub-010002' and r['status']=='complete'][-1]
    checked=b.st._verify_receipt(prior['result'],'sub-010002',p['config_path'],p['output_root'])
    assert not (p['raw_root']/'sub-010002').exists() and not prior['raw_retained']
    receipt=prior['result'];blocks=ab.expected_subject_blocks(ROOT,'sub-010002')
    config,digest=b.load_config(p['config_path'])
    params,param_hash=ab.verified_train_parameters(p['train_output_root']/'sub-010002_preprocessed.h5',config,digest,blocks,p['metrics_root'])
    assert params==receipt['train_parameters'] and param_hash==receipt['train_parameter_source_hash']
    assert receipt['authoritative_blocks']==blocks and len(blocks)==16
    assert receipt['fit_parameters_from']=='Train_only' and receipt['heldout_parameters_are']=='apply_only'
    assert {r['payload_sha256'] for r in receipt['source_manifest_rows']}==set(frame.loc[frame.subject_id=='sub-010002','payload_sha256'])
    row=frame.loc[(frame.subject_id=='sub-010003') & (frame.file_role=='waveform')].iloc[0]
    target=b.st.downloader.safe_destination(p['raw_root'],str(row.relative_path));part=target.with_name(target.name+'.part')
    b.st._assert_no_links(part)
    assert part.is_file() and not target.exists() and 0<part.stat().st_size<int(row.payload_size_bytes)
    report=dict(verified_utc=utc_now(),batch_id='001',frozen_artifact_sha256=protected_hashes,
        user_report='User manually stopped at approximately 600 MB C: free; manually cleared approximately 7.7 GB pip cache; C: recovered to approximately 8.8 GiB.',
        sub_010002_verified=True,canary_output_sha256=checked['output_sha256'],
        canary_receipt_sha256=sha256_file(Path(checked['receipt_path'])),canary_raw_absent=True,
        canary_train_output_sha256=params['train_output_sha256'],
        partial_subject='sub-010003',partial_path=str(part),partial_bytes=part.stat().st_size,
        partial_sha256=sha256_file(part),expected_payload_bytes=int(row.payload_size_bytes),
        expected_payload_sha256=str(row.payload_sha256),expected_annex_key=str(row.annex_key),
        range_header=f'bytes={part.stat().st_size}-',no_conflicting_final=True,
        partial_note='Resumable length and identity verified; full source SHA256 is verified only after completion.',
        environment=environment,no_batch_002=True,**b.frozen_hashes(ROOT))
    file=directory/'batch_001_interruption_resume_preflight.json'
    if file.exists():
        old=b.read_json(file)
        assert old['partial_bytes']==report['partial_bytes'] and old['partial_sha256']==report['partial_sha256']
    else:
        write_json(file,report)
        journal=b.st.downloader.DownloadJournal(runtime/'batch_001_download_journal.jsonl')
        journal.record(status='manual_interruption',subject_id='sub-010003',relative_path=str(row.relative_path),
            partial_bytes=part.stat().st_size,reason='User stopped for C: low space',received_body_bytes=0)
        b.st.append_status(runtime/'batch_001_streaming_journal.jsonl','manual-interruption-record','sub-010003',
            'user_interrupted',partial_bytes=part.stat().st_size,reason='User stopped for C: low space',
            scope='all_blocks_mechanical_provisional_v1')
    print(json.dumps(report,indent=2))


if __name__=='__main__':verify()
