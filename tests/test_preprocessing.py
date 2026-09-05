from pathlib import Path
import json

import h5py
import mne
import numpy as np
import pandas as pd
import pytest
import yaml

from resting_eeg.common import canonical_hash
from resting_eeg.data.access import TrainAccessGate, WaveformAccessError, sample_interval, TrainBlockLoader
from resting_eeg.preprocessing.signal import (
    apply_window_qc, expected_samples, fit_bad_channels, fit_window_thresholds,
    mad_scale, reference_and_select, sanity_metrics, synthetic_validation,
    window_features, window_starts,
)
from resting_eeg.preprocessing import pipeline


@pytest.fixture
def config():
    return yaml.safe_load(Path('configs/preprocessing/frozen_preprocessing_v1.yaml').read_text())


def make_gate(tmp_path):
    rows = [dict(subject_id='sub-001', block_index=i, acquisition_segment_id=1, state='EO',
                 start_sec=float(i*60), end_estimate_capped_sec=float(i*60+60), is_study_block=i != 4,
                 is_interruption_fragment=i == 4, is_complete=i != 5, split=role,
                 split_status='PROVISIONAL_METADATA_ONLY') for i, role in enumerate(['Train','Validation','Test','Train','Train','Train'])]
    frame = pd.DataFrame(rows)
    frame.to_csv(tmp_path/'split.csv', index=False)
    frame.to_csv(tmp_path/'blocks.csv', index=False)
    return TrainAccessGate(tmp_path/'split.csv',tmp_path/'blocks.csv',tmp_path/'registry.csv',Path.cwd())


def test_train_allowed_before_io(tmp_path):
    assert make_gate(tmp_path).authorize('sub-001',0)['split'] == 'Train'


@pytest.mark.parametrize('index', [1,2,4,5,999])
def test_forbidden_access_fails_explicitly(tmp_path, index):
    with pytest.raises(WaveformAccessError):
        make_gate(tmp_path).authorize('sub-001',index)
    assert not (tmp_path/'registry.csv').exists()


def test_changed_manifest_fails_closed(tmp_path):
    gate = make_gate(tmp_path)
    with (tmp_path/'split.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(WaveformAccessError, match='changed'):
        gate.authorize('sub-001',0)


def test_inventory_disagreement_fails(tmp_path):
    make_gate(tmp_path)
    frame = pd.read_csv(tmp_path/'blocks.csv');frame.loc[0,'start_sec']=1
    frame.to_csv(tmp_path/'blocks.csv',index=False)
    gate=TrainAccessGate(tmp_path/'split.csv',tmp_path/'blocks.csv',tmp_path/'registry.csv',Path.cwd())
    with pytest.raises(WaveformAccessError,match='disagreement'):
        gate.authorize('sub-001',0)


@pytest.mark.parametrize('rate', [500,1000,2500])
def test_exact_sample_conversion(rate):
    assert sample_interval('1.002', '61.002', rate) == (int(1.002*rate+.5),int(61.002*rate+.5))
    assert expected_samples(60*rate,rate) == 7680
    assert expected_samples(60*rate+1,rate) == 7681


def test_fractional_boundary_includes_only_sample_times_inside():
    assert sample_interval(.0001, .0021, 1000) == (1,3)


def test_gate_logs_before_read_and_prevents_heldout_calls(tmp_path, monkeypatch):
    gate=make_gate(tmp_path)
    class FakeRaw:
        info={'sfreq':500};n_times=200000;ch_names=[f'c{i}' for i in range(62)]
        def set_montage(self,*a,**kw): pass
        def get_channel_types(self):return ['eeg']*61+['eog']
        def get_data(self,**kw):
            assert (tmp_path/'registry.csv').exists()
            assert kw['start']==0 and kw['stop']==30000
            return np.zeros((62,30000))
        def close(self):pass
    monkeypatch.setattr(mne.io,'read_raw_brainvision',lambda *a,**kw: FakeRaw())
    from resting_eeg.common import sha256_file
    header=tmp_path/'sub-001/eeg/sub-001_task-resting_eeg.vhdr'
    header.parent.mkdir(parents=True);header.write_text('synthetic stub')
    integrity=dict(integrity_pass=True,subject_id='sub-001',verified_header_path=str(header.resolve()),header_sha256=sha256_file(header))
    loader=TrainBlockLoader('sub-001',header,gate,integrity,tmp_path/'active.lock')
    with loader:
        with pytest.raises(WaveformAccessError):loader.read(1,'test')
        block=loader.read(0,'test')
        assert block.unit=='V'
        with pytest.raises(RuntimeError):
            with TrainBlockLoader('sub-001',header,gate,integrity,tmp_path/'active.lock'):pass
    registry=pd.read_csv(tmp_path/'registry.csv')
    assert len(registry)==1 and registry.iloc[0]['split']=='Train'
    with pytest.raises(WaveformAccessError,match='verified subject'):
        TrainBlockLoader('sub-001',tmp_path/'other.vhdr',gate,integrity,tmp_path/'other.lock')


def test_integrity_failure_prevents_loader(tmp_path):
    with pytest.raises(WaveformAccessError,match='integrity'):
        TrainBlockLoader('sub-001','x',None,{'integrity_pass':False},tmp_path/'lock')


def test_filter_antialias_transient_support_and_determinism(config):
    results=synthetic_validation(config)
    assert results.pass_validation.all()
    assert results.deterministic.all()
    assert results.fir_length.tolist()==[3301,6601,16501]
    assert (results.combined_half_support_sec<=config['edge']['guard_sec']).all()


@pytest.mark.parametrize('n', [0,511,512,513,768,7000])
def test_windows_never_cross_blocks(n, config):
    starts=window_starts(n,config)
    assert np.all(starts+512<=n)
    assert np.all(np.diff(starts)==256)
    if n==512:assert starts.tolist()==[0]


def test_bad_channel_two_metrics_and_catastrophic(config):
    names=[f'C{i}' for i in range(61)]
    metrics=pd.DataFrame(dict(channel=names,scale_uv=[10]*60+[1e5],
                              derivative_scale_uv=[2]*61,median_abs_correlation=[.5]*61))
    sanity=pd.DataFrame(dict(channel=names,finite_fraction=[1.]*61,robust_scale=[10.]*61))
    bads,eligible,_=fit_bad_channels(metrics,sanity,names,[],config)
    assert not bads # amplitude alone cannot designate a bad channel
    metrics.loc[60,'median_abs_correlation']=.01
    bads,eligible,decisions=fit_bad_channels(metrics,sanity,names,[],config)
    assert bads==['C60'] and eligible
    sanity.loc[1,'finite_fraction']=.5
    bads,_,_=fit_bad_channels(metrics,sanity,names,[],config)
    assert bads==['C1','C60']


def test_threshold_fit_is_state_pooled_and_state_permutation_invariant(config):
    names=config['qc']['features']
    frame=pd.DataFrame({name:np.arange(1.,21.)*50 for name in names})
    frame['finite'],frame['flat']=True,False
    frame['state']=['EO']*10+['EC']*10
    thresholds=fit_window_thresholds(frame,config)
    swapped=frame.copy();swapped['state']=list(reversed(swapped.state))
    assert fit_window_thresholds(swapped,config)==thresholds
    assert thresholds[names[0]]==min(1000,max(150,np.median(frame[names[0]])+8*mad_scale(frame[names[0]])))


def test_car_full_61_then_deterministic_target_order_and_eog(config):
    montage=mne.channels.make_standard_montage('standard_1005')
    names=config['target_channels']+[n for n in montage.ch_names if n not in config['target_channels']][:42]+['VEOG']
    x=np.zeros((62,1000));x[60]=61e-6;x[61]=123e-6
    target,eog=reference_and_select(x,names,['eeg']*61+['eog'],[],config)
    np.testing.assert_allclose(target,-1.)
    np.testing.assert_allclose(eog,123.)


def test_units_sanity_and_spike_qc(config):
    x=np.array([[1e-6,2e-6,3e-6,4e-6]])
    metrics=sanity_metrics(x,['Fp1'])[0]
    assert metrics['minimum']==1 and metrics['maximum']==4 and metrics['unit']=='uV'
    features=pd.DataFrame([dict(finite=True,flat=False,**{f:0. for f in config['qc']['features']})])
    features.loc[0,'veog_ptp_uv']=2000
    mask,reasons=apply_window_qc(features,dict(zip(config['qc']['features'],config['qc']['threshold_caps'])))
    assert not mask[0] and reasons==['veog_ptp_uv']


def synthetic_repo(tmp_path, config):
    repo=tmp_path/'repo';raw=repo/'data_raw/MPI-LEMON';metrics=repo/'outputs/metrics/scenario2'
    subject='sub-001';folder=raw/subject/'eeg';folder.mkdir(parents=True);metrics.mkdir(parents=True)
    montage=mne.channels.make_standard_montage('standard_1005')
    names=config['target_channels']+[n for n in montage.ch_names if n not in config['target_channels']][:42]+['VEOG']
    base=folder/f'{subject}_task-resting_eeg'
    rng=np.random.default_rng(36)
    common=rng.normal(size=(100000,1))*20
    values=(common+rng.normal(size=(100000,62))*10).astype('<f4')
    values.tofile(base.with_suffix('.eeg'))
    base.with_suffix('.vmrk').write_text('Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile='+base.name+'.eeg\n[Marker Infos]\n',encoding='utf-8')
    header='Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nCodepage=UTF-8\nDataFile='+base.name+'.eeg\nMarkerFile='+base.name+'.vmrk\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=62\nSamplingInterval=2000\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\n'
    header+='\n'.join(f'Ch{i+1}={n},,1,µV' for i,n in enumerate(names))
    base.with_suffix('.vhdr').write_text(header,encoding='utf-8')
    from resting_eeg.common import sha256_file
    manifest=pd.DataFrame([dict(subject_id=subject,relative_path=str(p.relative_to(raw)).replace('\\','/'),payload_sha256=sha256_file(p),payload_size_bytes=p.stat().st_size) for p in [base.with_suffix('.eeg'),base.with_suffix('.vhdr')]])
    manifest.to_csv(metrics/'full_candidate_download_manifest.csv',index=False)
    blocks=pd.DataFrame([dict(subject_id=subject,block_index=i+1,acquisition_segment_id=1,state=['EO','EC'][i%2],start_sec=i*20.,end_estimate_capped_sec=(i+1)*20.,is_study_block=True,is_interruption_fragment=False,is_complete=True,split='Train',split_status='PROVISIONAL_METADATA_ONLY') for i in range(10)])
    blocks.to_csv(metrics/'lemon_block_inventory.csv',index=False);blocks.to_csv(metrics/'provisional_block_split_manifest.csv',index=False)
    config_path=repo/'config.yaml';config_path.write_text(yaml.safe_dump(config))
    config_path.with_suffix('.sha256').write_text(canonical_hash(config))
    return dict(repo_root=repo,raw_root=raw,output_root=repo/'processed',config_path=config_path,metrics_root=metrics)


def test_hdf5_provenance_idempotence_and_scientific_rebuild(tmp_path, config, monkeypatch):
    args=synthetic_repo(tmp_path,config)
    from resting_eeg.common import sha256_file
    header=args['raw_root']/'sub-001/eeg/sub-001_task-resting_eeg.vhdr'
    monkeypatch.setattr(pipeline,'audit_subject',lambda subject,*a:dict(subject_id=subject,integrity_pass=True,verified_header_path=str(header.resolve()),header_sha256=sha256_file(header)))
    monkeypatch.setattr(pipeline,'git_head',lambda root:'a'*40)
    import resting_eeg.data.access as access
    monkeypatch.setattr(access,'git_head',lambda root:'a'*40)
    first=pipeline.process_subject('sub-001',**args)
    second=pipeline.process_subject('sub-001',**args)
    assert first['output_sha256']==second['output_sha256'] and second['idempotent_reuse']
    assert len(pd.read_csv(args['metrics_root']/'pilot_waveform_access_registry.csv'))==10
    args['output_root']=args['repo_root']/'rebuild'
    third=pipeline.process_subject('sub-001',**args)
    with h5py.File(first['output_path']) as a,h5py.File(third['output_path']) as b:
        for uid in a['blocks']:
            for dataset in a['blocks'][uid]:
                np.testing.assert_array_equal(a['blocks'][uid][dataset][:],b['blocks'][uid][dataset][:])
    Path(third['output_path']).with_suffix('.provenance.json').unlink()
    recovered=pipeline.process_subject('sub-001',**args)
    assert recovered['processed_verified'] and list(args['output_root'].glob('*.orphan-*'))
    with h5py.File(first['output_path'],'r+') as handle:
        handle['blocks'][first['expected_blocks'][0]].attrs['channel_names']='[]'
    with pytest.raises(ValueError,match='channel order'):
        pipeline.verify_output(first['output_path'],first['config_sha256'],first['expected_blocks'])


def test_frozen_config_digest_refuses_mutation(tmp_path, config):
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(config));p.with_suffix('.sha256').write_text('a'*64)
    with pytest.raises(ValueError,match='digest'):
        pipeline.load_config(p)


def test_native_flat_block_cannot_be_hidden_by_car(config):
    rows=pd.DataFrame([dict(block_uid='b',channel='Fp1',finite_fraction=1.,robust_scale=0.,flat_fraction=1.)])
    assert pipeline.native_block_failure(rows,'b',[],config)
    assert not pipeline.native_block_failure(rows,'b',['Fp1'],config) # explicitly repaired EEG
    rows['channel']='VEOG';rows['finite_fraction']=.9
    assert pipeline.native_block_failure(rows,'b',[],config)


def test_flat_veog_is_not_a_valid_artifact_proxy(config):
    rng=np.random.default_rng(41)
    features=window_features(rng.normal(size=(19,512)),np.zeros(512),config)
    assert features.iloc[0].flat


def test_empty_reference_diagnostics_remain_rejected(config):
    row=dict(finite=False,flat=True,qc_reference_failed=True,**{name:0. for name in config['qc']['features']})
    mask,reasons=apply_window_qc(pd.DataFrame([row]),dict(zip(config['qc']['features'],config['qc']['threshold_caps'])))
    assert not mask[0] and 'no_valid_Train_QC_reference' in reasons[0]
