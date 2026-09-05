"""Offline 20-GiB membership and preservation of two completed batches."""
from pathlib import Path
import pandas as pd
import pytest
from resting_eeg.data import batches as b

ROOT=Path(__file__).resolve().parents[1]


def test_batch003_exact_cap_order_exclusions_and_two_immutable_batches(tmp_path):
    directory=ROOT/'outputs/metrics/scenario2/batches'
    source=pd.read_csv(directory.parent/'full_candidate_download_manifest.csv')
    before=b.read_json(directory/'batch_003_starting_state.json')
    assert b.cap_bytes('20')==21474836480
    first=b.read_json(directory/'batch_001_plan.json')
    second=b.read_json(directory/'batch_002_plan.json')
    pilot=[s for s in before['completed_subjects'] if s not in first['subject_ids']+second['subject_ids']]
    b.freeze_batch(tmp_path,'001',5*b.GIB,source,pilot,'synthetic',{})
    b.freeze_batch(tmp_path,'002',10*b.GIB,source,pilot+first['subject_ids'],'synthetic',{})
    old={f.name:f.read_bytes() for f in tmp_path.glob('batch_00[12]*') if f.is_file()}
    old_entries=b.ledger_read(tmp_path)['batches']
    plan=b.freeze_batch(tmp_path,'003',20*b.GIB,source,before['completed_subjects'],'synthetic',{})
    remaining=[s for s in source.subject_id.drop_duplicates() if s not in before['completed_subjects']]
    assert remaining[0]=='sub-010035'
    assert plan['subject_count']==33 and plan['asset_count']==66
    assert plan['subject_ids']==remaining[:33]
    assert len(set(plan['subject_ids']))==33
    assert plan['declared_bytes']==21299575247
    assert plan['unused_capacity_bytes']==175261233
    assert plan['first_not_included']['subject_id']==remaining[33]=='sub-010075'
    assert plan['declared_bytes']+plan['first_not_included']['declared_bytes']>plan['cap_bytes']
    assert all((tmp_path/name).read_bytes()==data for name,data in old.items())
    assert b.ledger_read(tmp_path)['batches'][:2]==old_entries
    assert not list(tmp_path.glob('batch_004*'))
    assert b.freeze_batch(tmp_path,'003',20*b.GIB,source,remaining,'new-head',{})==plan


@pytest.mark.parametrize('which',['001','002','004'])
def test_prior_two_batch_ledgers_and_future_boundary(tmp_path,monkeypatch,which):
    directory=tmp_path/'batches';directory.mkdir()
    prior=[dict(batch_id=bid,batch_status='complete') for bid in ('001','002')]
    b.write_json(directory/'batch_ledger.json',dict(batches=prior))
    b.write_json(directory/'batch_003_starting_state.json',dict(prior_ledger_entries=prior,files=[],registries={},completed_subjects=[]))
    monkeypatch.setattr(b,'paths',lambda root:dict(metrics_root=tmp_path))
    monkeypatch.setattr(b,'runtime_directory',lambda root:tmp_path/'runtime')
    assert b.verify_previous_state(tmp_path,'003',full=True)
    if which=='004':
        (directory/'batch_004_plan.json').write_text('prohibited')
    else:
        prior[int(which)-1]['batch_status']='changed'
        b.write_json(directory/'batch_ledger.json',dict(batches=prior))
    with pytest.raises(ValueError):b.verify_previous_state(tmp_path,'003',full=True)
