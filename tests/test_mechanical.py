"""Synthetic-only authorization, apply-only, storage and streaming regressions."""
from pathlib import Path
import json
import ast
import h5py
import mne
import numpy as np
import pandas as pd
import pytest
import yaml
from resting_eeg.common import canonical_hash, sha256_file
from resting_eeg.data.access import WaveformAccessError, PhysicalBlock
from resting_eeg.data import mechanical_access as access
from resting_eeg.preprocessing import all_blocks as ab, pipeline, signal
from test_preprocessing import synthetic_repo

ROOT = Path(__file__).resolve().parents[1]


def spec_for(repo, config_path):
    spec = yaml.safe_load((ROOT/'configs/access/mechanical_heldout_v1.yaml').read_text())
    m = repo/'outputs/metrics/scenario2'
    spec['split_manifest_sha256'] = sha256_file(m/'provisional_block_split_manifest.csv')
    spec['block_inventory_sha256'] = sha256_file(m/'lemon_block_inventory.csv')
    path = repo/'access.yaml'
    path.write_text(yaml.safe_dump(spec))
    path.with_suffix('.sha256').write_text(canonical_hash(spec))
    return path


@pytest.fixture
def gated(tmp_path):
    config_path = ROOT/'configs/preprocessing/frozen_preprocessing_v1.yaml'
    m = tmp_path/'outputs/metrics/scenario2';m.mkdir(parents=True)
    rows = [dict(subject_id='sub-001',block_index=i,acquisition_segment_id=1,state='EO',
        start_sec=i*10.,end_estimate_capped_sec=(i+1)*10.,is_study_block=i!=4,
        is_interruption_fragment=i==4,is_complete=i!=5,split=role,split_status='PROVISIONAL_METADATA_ONLY')
        for i,role in enumerate(['Train','Validation','Test','Validation','Test','Validation'])]
    for file in ('provisional_block_split_manifest.csv','lemon_block_inventory.csv'):
        pd.DataFrame(rows).to_csv(m/file,index=False)
    path = spec_for(tmp_path,config_path)
    return access.MechanicalHeldoutAccessGate(m/'provisional_block_split_manifest.csv',m/'lemon_block_inventory.csv',
            tmp_path/'registry.csv',ROOT,path,config_path)


@pytest.mark.parametrize('index,split',[(1,'Validation'),(2,'Test')])
def test_mechanical_permitted(gated,index,split):
    assert gated.authorize('sub-001',index,'fixed_waveform_qc')['split']==split


@pytest.mark.parametrize('purpose', ['alpha','psd','connectivity','classifier','plotting','waveform_visualization',
    'iaf','wpli','generator','tstr','augmentation','hyperparameter_selection','threshold_fitting','normalization_fitting','unknown',''])
def test_purpose_denied(gated,purpose):
    with pytest.raises(WaveformAccessError):gated.authorize('sub-001',1,purpose)
    assert not gated.registry_path.exists()


@pytest.mark.parametrize('index',[0,4,5,999])
def test_train_fragment_incomplete_unknown_denied(gated,index):
    with pytest.raises(WaveformAccessError):gated.authorize('sub-001',index)


@pytest.mark.parametrize('target',['access','inventory','split'])
def test_hash_change_denied(gated,target):
    path = {'access':gated.access_path,'inventory':gated.inventory_path,'split':gated.split_path}[target]
    path.write_text(path.read_text().replace('apply_only','fit') if target=='access' else path.read_text()+'# changed')
    with pytest.raises((WaveformAccessError,ValueError)):gated.authorize('sub-001',1)


def test_authorization_durable_before_io(gated,tmp_path,monkeypatch):
    class Raw:
        info={'sfreq':500};n_times=30000;ch_names=[f'c{i}' for i in range(62)]
        def set_montage(self,*a,**kw):pass
        def get_channel_types(self):return ['eeg']*61+['eog']
        def get_data(self,**kw):
            rows=pd.read_csv(gated.registry_path)
            assert len(rows)==1 and rows.iloc[0].provisional_split=='Validation'
            assert rows.iloc[0].purpose=='frozen_preprocessing'
            assert (kw['start'],kw['stop'])==(5000,10000)
            return np.zeros((62,5000))
        def close(self):pass
    monkeypatch.setattr(mne.io,'read_raw_brainvision',lambda *a,**kw:Raw())
    header=tmp_path/'sub-001/eeg/sub-001_task-resting_eeg.vhdr';header.parent.mkdir(parents=True);header.write_text('stub')
    integrity=dict(subject_id='sub-001',integrity_pass=True,verified_header_path=str(header.resolve()),header_sha256=sha256_file(header))
    with access.MechanicalHeldoutBlockLoader('sub-001',header,gated,integrity,tmp_path/'lock') as loader:
        with pytest.raises(WaveformAccessError):loader.read(1,'alpha')
        assert not gated.registry_path.exists()
        assert not hasattr(loader,'raw')
        assert loader.read(1,'frozen_preprocessing').values_v.shape==(62,5000)


@pytest.mark.parametrize('split',['Validation','Test'])
@pytest.mark.parametrize('fit',['threshold','bad_channel'])
def test_heldout_cannot_fit(split,fit):
    c=yaml.safe_load((ROOT/'configs/preprocessing/frozen_preprocessing_v1.yaml').read_text())
    frame=pd.DataFrame([dict(provisional_split=split)])
    with pytest.raises(ValueError,match='Only Train'):
        if fit=='threshold':signal.fit_window_thresholds(frame,c)
        else:signal.fit_bad_channels(frame,frame,[],[],c)


@pytest.fixture
def combined(tmp_path,monkeypatch):
    config=yaml.safe_load((ROOT/'configs/preprocessing/frozen_preprocessing_v1.yaml').read_text())
    args=synthetic_repo(tmp_path,config)
    m=args['metrics_root']; repo=args['repo_root']
    blocks=pd.read_csv(m/'provisional_block_split_manifest.csv')
    rows=[]
    for i in range(16):
        row=blocks.iloc[0].to_dict()
        row.update(block_index=i+1,state=['EO','EC'][i%2],start_sec=i*12.5,end_estimate_capped_sec=(i+1)*12.5,
                   split='Train' if i<10 else 'Validation' if i<12 else 'Test')
        rows.append(row)
    for name in ['provisional_block_split_manifest.csv','lemon_block_inventory.csv']:
        pd.DataFrame(rows).to_csv(m/name,index=False)
    header=args['raw_root']/'sub-001/eeg/sub-001_task-resting_eeg.vhdr'
    audit=lambda subject,*a:dict(subject_id=subject,integrity_pass=True,verified_header_path=str(header.resolve()),header_sha256=sha256_file(header))
    monkeypatch.setattr(pipeline,'audit_subject',audit);monkeypatch.setattr(ab,'audit_subject',audit)
    monkeypatch.setattr(pipeline,'git_head',lambda root:'a'*40)
    import resting_eeg.data.access as ta
    monkeypatch.setattr(ta,'git_head',lambda root:'a'*40)
    monkeypatch.setattr(access,'git_head',lambda root:'b'*40)
    monkeypatch.setattr(ab,'require_committed_clean',lambda root:'b'*40)
    old=pipeline.process_subject('sub-001',**args)
    old_hash=sha256_file(old['output_path']);registry_hash=sha256_file(m/'pilot_waveform_access_registry.csv')
    # Any raw Train reread or held-out fitting is an immediate regression failure.
    monkeypatch.setattr(ab,'process_subject',lambda *a,**kw:pytest.fail('raw Train reread'))
    monkeypatch.setattr(signal,'fit_window_thresholds',lambda *a,**kw:pytest.fail('held-out threshold fit'))
    monkeypatch.setattr(signal,'fit_bad_channels',lambda *a,**kw:pytest.fail('held-out bad-channel fit'))
    mechanical=dict(args,output_root=repo/'combined',train_output_root=args['output_root'],access_config_path=spec_for(repo,args['config_path']))
    result=ab.process_all_blocks('sub-001',**mechanical)
    assert sha256_file(old['output_path'])==old_hash
    assert sha256_file(m/'pilot_waveform_access_registry.csv')==registry_hash
    return mechanical,result,old


def test_all_block_copy_apply_and_idempotence(combined):
    args,result,old=combined
    assert len(result['expected_blocks'])==16
    before=sha256_file(args['metrics_root']/'pilot_mechanical_heldout_access_registry.csv')
    again=ab.process_all_blocks('sub-001',**args)
    assert again['idempotent_reuse'] and again['output_sha256']==result['output_sha256']
    assert sha256_file(args['metrics_root']/'pilot_mechanical_heldout_access_registry.csv')==before
    rows=pd.read_csv(args['metrics_root']/'pilot_mechanical_heldout_access_registry.csv')
    assert len(rows)==6 and set(rows.provisional_split)=={'Validation','Test'}
    with h5py.File(result['output_path'],'r') as h,h5py.File(old['output_path'],'r') as t:
        assert json.loads(h.attrs['qc_thresholds'])==json.loads(t.attrs['qc_thresholds'])
        for uid in t['blocks']:
            for key in t['blocks'][uid]:
                np.testing.assert_array_equal(h['blocks'][uid][key][:],t['blocks'][uid][key][:])
        for uid,g in h['blocks'].items():
            starts=g['window_start_samples'][:]
            assert (starts+512<=g['eeg_uv'].shape[1]).all()
            assert (np.diff(starts)==256).all()
            assert g.attrs['provisional_split']==next(b['split'] for b in result['authoritative_blocks'] if b['block_uid']==uid)


@pytest.mark.parametrize('damage',['missing','split','config','shape','parameters'])
def test_output_corruption_detected(combined,damage):
    args,r,_=combined
    with h5py.File(r['output_path'],'r+') as h:
        uid=r['expected_blocks'][-1];g=h['blocks'][uid]
        if damage=='missing':del h['blocks'][uid]
        if damage=='split':g.attrs['provisional_split']='Train'
        if damage=='config':g.attrs['preprocessing_config_sha256']='wrong'
        if damage=='shape':g['window_start_samples'][0]=100000
        if damage=='parameters':h.attrs['train_parameter_source_hash']='wrong'
    with pytest.raises(ValueError):ab.process_all_blocks('sub-001',**args)


def test_apply_parameters_unchanged(combined,monkeypatch):
    args,r,_=combined
    params=r['train_parameters'];before=canonical_hash(params)
    config,_=pipeline.load_config(args['config_path'])
    rng=np.random.default_rng(2)
    physical=PhysicalBlock(rng.normal(0,1e-5,(62,10000)),params['channel_names'],params['channel_types'],500,0,10000,
                           dict(split='Test',block_uid='test'))
    original=ab.apply_window_qc
    def check(features,thresholds,eligible):
        assert thresholds==params['summary']['thresholds']
        return original(features,thresholds,eligible)
    monkeypatch.setattr(ab,'apply_window_qc',check)
    ab.apply_heldout_block(physical,params,config)
    assert canonical_hash(params)==before


def test_allblock_streaming_local_existing(combined,tmp_path,monkeypatch):
    from resting_eeg.data import streaming
    args,r,_=combined
    metadata=tmp_path/'metadata';folder=metadata/'sub-001/eeg';folder.mkdir(parents=True)
    for suffix in streaming.REQUIRED_SIDECARS:(folder/f'sub-001_task-resting_{suffix}').write_text('stub')
    monkeypatch.setattr(streaming.downloader,'copy_sidecars',lambda *a:None)
    monkeypatch.setattr(streaming.downloader,'fetch_one',lambda *a,**kw:pytest.fail('network forbidden'))
    frame=pd.read_csv(args['metrics_root']/'full_candidate_download_manifest.csv')
    # Synthetic manifest gets URL fields required by downloader validation.
    frame['source_url']='https://nemar.s3.us-east-2.amazonaws.com/synthetic-test-only'
    kw=dict(args,metadata_root=metadata,processing_scope='all-blocks-mechanical')
    plan=streaming.stream_subjects(frame,**kw)
    assert plan[0]['scope']==ab.SCOPE
    result=streaming.stream_subjects(frame,**kw,execute=True,local_existing_only=True)
    assert result[0]['status']=='complete'
    assert (args['raw_root']/'sub-001').exists()


def test_no_scientific_or_fit_calls_in_mechanical_path():
    tree=ast.parse(Path(ab.__file__).read_text())
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls & {'fit_window_thresholds','fit_bad_channels','compute_psd','psd','plot','fit','wpli','savefig'}


def test_frozen_config_unchanged():
    assert pipeline.load_config(ROOT/'configs/preprocessing/frozen_preprocessing_v1.yaml')[1]==access.FROZEN_SHA


def test_cli_allblock_and_fit_denial():
    from scripts.data.s2_07_stream_preprocess_subjects import build_parser
    parser=build_parser()
    assert parser.parse_args(['--processing-scope','all-blocks-mechanical','--dry-run']).processing_scope=='all-blocks-mechanical'
    with pytest.raises(SystemExit):parser.parse_args(['--fit-parameters-from','Test'])
