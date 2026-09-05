"""Offline Batch 002 membership and prior-state preservation regressions."""
from pathlib import Path
import pandas as pd
import pytest
from resting_eeg.data import batches as b

ROOT=Path(__file__).resolve().parents[1]


def test_batch002_exact_cap_contiguous_membership_and_prior_freeze(tmp_path):
    directory=ROOT/'outputs/metrics/scenario2/batches'
    source=pd.read_csv(directory.parent/'full_candidate_download_manifest.csv')
    before=b.read_json(directory/'batch_002_starting_state.json')
    batch1=b.read_json(directory/'batch_001_plan.json')
    selected,plan=b.plan_membership(source,b.cap_bytes('10'),before['completed_subjects'],batch1['subject_ids'])
    assert plan['cap_bytes']==10737418240
    assert plan['subject_ids']==['sub-'+s for s in ('010016','010017','010019','010020','010021','010022','010024','010026','010027','010028','010029','010030','010031','010032','010033','010034')]
    assert plan['declared_bytes']==10141417744
    assert plan['unused_capacity_bytes']==596000496
    assert plan['first_not_included']['subject_id']=='sub-010035'
    assert plan['declared_bytes']+plan['first_not_included']['declared_bytes']>plan['cap_bytes']
    assert len(selected)==32 and not selected.relative_path.duplicated().any()
    assert not set(plan['subject_ids']) & set(before['completed_subjects'])
    # Freeze in an isolated synthetic ledger using authoritative metadata only.
    pilot=[s for s in before['completed_subjects'] if s not in batch1['subject_ids']]
    b.freeze_batch(tmp_path,'001',5*b.GIB,source,pilot,'synthetic',{})
    old={f.name:f.read_bytes() for f in tmp_path.glob('batch_001*') if f.is_file()}
    old_entry=b.ledger_read(tmp_path)['batches'][0]
    frozen=b.freeze_batch(tmp_path,'002',10*b.GIB,source,before['completed_subjects'],'synthetic',{})
    assert frozen['subject_ids']==plan['subject_ids']
    assert all((tmp_path/n).read_bytes()==data for n,data in old.items())
    assert b.ledger_read(tmp_path)['batches'][0]==old_entry
    assert not list(tmp_path.glob('batch_003*'))


@pytest.mark.parametrize('mutation',['artifact','ledger','registry','future'])
def test_previous_state_rejects_tampering(tmp_path,monkeypatch,mutation):
    directory=tmp_path/'batches'; directory.mkdir()
    artifact=tmp_path/'prior.json'; artifact.write_text('unchanged')
    registry=tmp_path/'registry.csv'; pd.DataFrame([dict(subject_id='sub-prior',block=1)]).to_csv(registry,index=False)
    rows=pd.read_csv(registry).to_dict('records')
    entry=dict(batch_id='001',batch_status='complete')
    b.write_json(directory/'batch_ledger.json',dict(batches=[entry]))
    b.write_json(directory/'batch_002_starting_state.json',dict(completed_subjects=['sub-prior'],batch_001_ledger_entry=entry,
        files=[dict(path=str(artifact),bytes=9,sha256=b.sha256_file(artifact))],
        registries={'registry.csv':dict(count=1,sha256=b.canonical_hash(rows))}))
    monkeypatch.setattr(b,'paths',lambda root:dict(metrics_root=tmp_path))
    monkeypatch.setattr(b,'runtime_directory',lambda root:tmp_path/'runtime')
    assert b.verify_previous_state(tmp_path,'002',full=True)
    if mutation=='artifact': artifact.write_text('changed')
    if mutation=='ledger': b.write_json(directory/'batch_ledger.json',dict(batches=[dict(entry,batch_status='executing')]))
    if mutation=='registry': registry.write_text('subject_id,block\nsub-prior,2\n')
    if mutation=='future': (directory/'batch_003_manifest.csv').write_text('forbidden')
    with pytest.raises(ValueError): b.verify_previous_state(tmp_path,'002',full=True)
