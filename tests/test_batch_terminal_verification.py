"""Do not conflate retained download failures with successful processing/QC."""
import pandas as pd
import pytest
from scripts.data.s2_12_verify_streaming_batch import terminal_records


def records():
    rows=pd.DataFrame([
        dict(subject_id='sub-ok',status='complete',completed=True,raw_retained=False,raw_deleted=True,failure_reason=''),
        dict(subject_id='sub-failed',status='download_interrupted',completed=False,raw_retained=True,raw_deleted=False,failure_reason='DNS unavailable')])
    history=[dict(subject_id='sub-ok',status='complete'),
        dict(subject_id='sub-failed',status='download_interrupted',raw_preserved=True,error='DNS unavailable')]
    return rows,history


def test_all_terminal_does_not_mean_all_technically_complete():
    rows,history=records()
    result=terminal_records(rows,history,['sub-ok','sub-failed'])
    assert len(result)==2 and sum(rows.completed)==1
    assert result['sub-failed']['status']=='download_interrupted'


@pytest.mark.parametrize('mutation',['missing_terminal','false_success','deleted_raw','changed_reason','unverified_processing','wrong_order'])
def test_terminal_verification_rejects_inconsistent_evidence(mutation):
    rows,history=records();ids=['sub-ok','sub-failed']
    if mutation=='missing_terminal':history.pop()
    if mutation=='false_success':rows.loc[1,'completed']=True
    if mutation=='deleted_raw':rows.loc[1,'raw_deleted']=True
    if mutation=='changed_reason':rows.loc[1,'failure_reason']='different'
    if mutation=='unverified_processing':
        rows.loc[1,'status']='preprocessing_failed';history[-1]['status']='preprocessing_failed'
    if mutation=='wrong_order':ids.reverse()
    with pytest.raises(AssertionError):terminal_records(rows,history,ids)
