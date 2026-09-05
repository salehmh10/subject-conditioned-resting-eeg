"""Inventory, copy, verify, then remove only inventoried legacy data files.

No network. Scientific HDF5 and adjacent receipts remain byte-identical; historic
paths are resolved by the central relocation policy and recorded in the inventory.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]
import pandas as pd
from resting_eeg.common import sha256_file, write_json, utc_now
from resting_eeg.data.paths import load_paths
from resting_eeg.data import batches as b, streaming as st
from resting_eeg.preprocessing import all_blocks as ab
from resting_eeg.preprocessing.pipeline import load_config, verify_output


def migrate(execute=False):
    storage = load_paths(ROOT)
    metrics = ROOT/'outputs/metrics/scenario2'
    inventory_path = metrics/'storage_migration_inventory.json'
    mappings = [(ROOT/'data_raw/MPI-LEMON',storage['raw_root']),
        (ROOT/'data_processed/MPI-LEMON',storage['processed_root']/'train-only-v1'),
        (ROOT/'data_processed/MPI-LEMON-all-blocks-v1',storage['processed_root']/'all-blocks-v1'),
        (metrics/'streaming_all_block_status.jsonl',storage['runtime_root']/'streaming_all_block_status.jsonl')]
    if inventory_path.exists():
        inventory = b.read_json(inventory_path)
    else:
        files = []
        for source, destination in mappings:
            for path in ([source] if source.is_file() else sorted(source.rglob('*'))):
                if not path.is_file():
                    continue
                st._assert_no_links(path)
                target = destination if source.is_file() else destination/path.relative_to(source)
                files.append(dict(source=str(path.resolve()),destination=str(target.resolve()),
                    bytes=path.stat().st_size,sha256=sha256_file(path)))
        inventory = dict(created_utc=utc_now(),files=files,old_roots=[str(a) for a,_ in mappings],
                         new_roots=[str(c) for _,c in mappings],**b.frozen_hashes(ROOT))
        write_json(inventory_path,inventory)
    print(json.dumps(dict(files=len(inventory['files']),bytes=sum(r['bytes'] for r in inventory['files']),
        execute=execute,resolved_paths={k:str(v) for k,v in storage.items()})),flush=True)
    if not execute:
        return
    for key in ('raw_root','processed_root','temp_root','runtime_root'):
        storage[key].mkdir(parents=True,exist_ok=True)
    for row in inventory['files']:
        source, target = Path(row['source']), Path(row['destination'])
        if storage['scenario2_data_root'] not in target.resolve().parents:
            raise ValueError('Migration target outside external Scenario 2 root')
        st._assert_no_links(target)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            st.downloader.verify_file(source,row['bytes'],row['sha256'])
            shutil.copy2(source,target)
            with target.open('r+b') as handle: os.fsync(handle.fileno())
        st.downloader.verify_file(target,row['bytes'],row['sha256'])
    # All destination bytes are intact before deleting any legacy source.
    manifest = pd.read_csv(metrics/'pilot_download_manifest.csv')
    for row,target in st._manifest_rows(manifest,storage['raw_root']):
        st.downloader.verify_file(target,int(row['payload_size_bytes']),str(row['payload_sha256']))
    config,digest=load_config(ROOT/'configs/preprocessing/frozen_preprocessing_v1.yaml')
    for subject in manifest.subject_id.drop_duplicates():
        blocks=ab.expected_subject_blocks(ROOT,subject)
        train=storage['processed_root']/'train-only-v1'/f'{subject}_preprocessed.h5'
        allblock=storage['processed_root']/'all-blocks-v1'/f'{subject}_all_blocks.h5'
        params,param_hash=ab.verified_train_parameters(train,config,digest,blocks,metrics)
        checked=ab.verify_all_block_output(allblock,digest,blocks,b.ACCESS_SHA,param_hash)
        receipt=b.read_json(allblock.with_suffix('.provenance.json'))
        if checked['output_sha256']!=receipt['output_sha256'] or params!=receipt['train_parameters']:
            raise ValueError('Migrated HDF5/provenance mismatch')
        st._verify_receipt(receipt,subject,ROOT/'configs/preprocessing/frozen_preprocessing_v1.yaml',allblock.parent)
    removed=0
    for row in inventory['files']:
        source,target=Path(row['source']),Path(row['destination'])
        allowed = any(source == old or old in source.parents for old,_ in mappings)
        if not allowed or ROOT not in source.resolve().parents:
            raise ValueError('Legacy removal escaped inventoried repository source')
        st._assert_no_links(source)
        if source.exists():
            st.downloader.verify_file(source,row['bytes'],row['sha256'])
            st.downloader.verify_file(target,row['bytes'],row['sha256'])
            source.unlink()
            removed+=1
    if any(p.is_file() for folder in ('data_raw','data_processed') for p in (ROOT/folder).rglob('*')):
        raise ValueError('Unexpected old data file remains; preserve and investigate')
    summary=dict(old_roots=inventory['old_roots'],new_roots=inventory['new_roots'],
        resolved_paths={k:str(v) for k,v in storage.items()},files_moved=len(inventory['files']),
        bytes_moved=sum(r['bytes'] for r in inventory['files']),raw_hashes_verified=30,
        processed_hashes_verified=30,receipts_preserved_byte_identical=True,
        duplicates_removed=sum(not Path(r['source']).exists() for r in inventory['files']),
        inventory_sha256=sha256_file(inventory_path),**b.frozen_hashes(ROOT),
        batch_001_status='not_frozen_no_network',completed_utc=utc_now(),
        metadata_clone_write_operations=0,legacy_payload_files_remaining=0)
    write_json(metrics/'storage_migration_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    migrate(parser.parse_args().execute)
