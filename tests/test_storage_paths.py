"""Synthetic storage isolation and relocation; never accesses dataset values."""
from pathlib import Path
import shutil
import pytest
from resting_eeg.data.paths import load_paths, validate_paths, relocated_path
from resting_eeg.data import streaming as st
from resting_eeg.data import batches as b
from test_batches import manifest


@pytest.fixture
def layout(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    data=tmp_path/'datasets/scenario2'
    values=dict(metadata_root=tmp_path/'datasets/metadata',scenario2_data_root=data,
        raw_root=data/'raw',processed_root=data/'processed',temp_root=data/'temp',runtime_root=data/'runtime')
    return repo,values


def test_roots_external_distinct(layout):
    repo,values=layout
    resolved=validate_paths(repo,values)
    for key in ('raw_root','processed_root','temp_root','runtime_root'):
        assert repo not in resolved[key].parents
        assert resolved['metadata_root'] not in resolved[key].parents


@pytest.mark.parametrize('key',['raw_root','processed_root','temp_root','runtime_root','scenario2_data_root'])
@pytest.mark.parametrize('bad',['repo','metadata','escape'])
def test_storage_overlap_refused(layout,key,bad):
    repo,values=layout
    values[key]={'repo':repo,'metadata':values['metadata_root'],'escape':repo.parent/'elsewhere'}[bad]
    if key=='scenario2_data_root' and bad=='escape':
        # Existing children would escape the newly selected base.
        pass
    with pytest.raises(ValueError):validate_paths(repo,values)


def test_readonly_metadata_cleanup_refused(layout):
    repo,v=layout
    with pytest.raises(ValueError): st.safe_subject_directory(v['metadata_root'],v['metadata_root'],'sub-001')


@pytest.mark.parametrize('subject',['..','../repo','sub-001/..','sub-001\\..','sub-001:escape'])
def test_cleanup_cannot_escape_raw(layout,subject):
    _,v=layout
    with pytest.raises(ValueError):st.safe_subject_directory(v['raw_root'],v['metadata_root'],subject)


def test_local_override_and_historical_relocation(tmp_path):
    repo=tmp_path/'repo';(repo/'configs/paths').mkdir(parents=True)
    (repo/'configs/paths/data_paths.yaml').write_text('metadata_root: ../metadata\nscenario2_data_root: ../external\nraw_root: raw\nprocessed_root: processed\ntemp_root: temp\nruntime_root: runtime\n')
    p=load_paths(repo)
    old=repo/'data_processed/MPI-LEMON-all-blocks-v1/sub-001_all_blocks.h5'
    assert relocated_path(old,repo)==p['processed_root']/'all-blocks-v1'/old.name
    assert relocated_path(repo/'data_raw/MPI-LEMON/sub-001/eeg/x.part',repo)==p['raw_root']/'sub-001/eeg/x.part'
    override=repo/'configs/paths/local_data_paths.yaml'
    override.write_text('scenario2_data_root: ../another-external\n')
    assert load_paths(repo)['raw_root']==tmp_path/'another-external/raw'


def test_storage_does_not_affect_membership_and_disk_target(layout,monkeypatch):
    _,v=layout
    frame,plan=b.plan_membership(manifest(),100);plan['batch_id']='001'
    checks=[]
    monkeypatch.setattr(st,'_free_bytes',lambda path:checks.append(path) or 100*b.GIB)
    b.network_preflight(frame,plan,v['raw_root'],v['processed_root'])
    assert checks==[v['raw_root'],v['processed_root']]
    assert b.plan_membership(manifest(),100)[1]['subject_ids']==plan['subject_ids']


def test_valid_partial_survives_relocation_and_resume(layout,monkeypatch):
    from test_download_resume import asset,PAYLOAD,install_responses,resume_response
    repo,v=layout
    old=repo/'old.eeg.part';old.write_bytes(PAYLOAD[:10])
    target=v['raw_root']/'sub-001/eeg/sub-001_task-resting_eeg.eeg'
    target.parent.mkdir(parents=True)
    part=target.with_name(target.name+'.part');shutil.move(old,part)
    calls=install_responses(monkeypatch,resume_response(10))
    st.downloader.fetch_one(asset(),target)
    assert target.read_bytes()==PAYLOAD and calls[0][0]['Range']=='bytes=10-'


def test_frozen_scientific_hashes_unchanged():
    root=Path(__file__).resolve().parents[1]
    h=b.frozen_hashes(root)
    assert h['preprocessing_config_sha256']==b.FROZEN_SHA
    assert h['mechanical_access_config_sha256']==b.ACCESS_SHA
