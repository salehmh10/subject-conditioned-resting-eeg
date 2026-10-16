r"""s2_75A-v2: freeze the explicitly revised 11-condition utility training contract.

Unlike s2_72 / s2_75A-v1 (8976 fixed updates, no Validation selection),
this POST-PILOT revision uses up to 120 *actual* training-set epochs per
condition, and s2_29's validation-BA early-stopping rule (patience 6, min 8).

This change was decided AFTER observing one s2_75B-v1 Real-Only validation
result. It is not prospectively frozen relative to all available development
outcomes; preserve the original protocols and clearly label this v2 revision.
NO EEG signals are read and no classifier is trained by this file.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path

BASE_AMEND_REL = Path('outputs/protocols/scenario2/utility/b0_amended_11_condition_v1/manifests/s2_75a_amendment_v1.json')
BASE_AMEND_SHA256 = 'cdef4c6bc90f63c69429aba8d7da232f9b73fe6e2ef88fe5635044eae11ed2b7'
BASE_SOURCE_REL = Path('scripts/data/s2_75a_freeze_b0_utility_amendment.py')
ORIG_PROTO_REL = Path('outputs/protocols/scenario2/utility/b1_vs_b2_utility_protocol_v1/metrics/s2_72_b1_vs_b2_downstream_utility_protocol_v1.json')
ORIG_PROTO_SHA = '5b98ff09d977006b96ea80e43d4f13f33742745b734d5dfa94bb1eaacb682a22'
OUT_REL = Path('outputs/protocols/scenario2/utility/b0_amended_11_condition_120epoch_es_v2/manifests/s2_75a_v2_120epoch_es_amendment.json')
STATUS = 'S2_75A_V2_120EP_EARLYSTOP_FROZEN'
VERSION = 's2_75a_v2_120epoch_earlystopping'
MAX_EPOCHS = 120
MIN_EPOCHS = 8
PATIENCE = 6
BATCH_SIZE = 128


class ContractError(RuntimeError): pass

def need(ok,why):
    if not ok: raise ContractError(why)

def canonical(o):
    return json.dumps(o,sort_keys=True,ensure_ascii=True,separators=(',', ':'),allow_nan=False).encode()

def load_json(path):
    def unique(items):
        result={}
        for k,v in items:
            need(k not in result,'Duplicate JSON key: '+k)
            result[k]=v
        return result
    def no_constant(value):
        raise ContractError('Invalid JSON constant: '+value)
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f,object_pairs_hook=unique,parse_constant=no_constant)

def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()

def base_context(repo):
    rel=BASE_AMEND_REL
    basepath=(repo/rel).resolve()
    need(basepath.is_relative_to(repo) and basepath.is_file() and not basepath.is_symlink(),
         'Missing frozen parent amendment')
    need(sha_file(basepath)==BASE_AMEND_SHA256,
         's2_75A-v1 amendment SHA mismatch: do not edit it')
    base=load_json(basepath)
    need(base['status']=='S2_75A_11_CONDITION_AMENDMENT_FROZEN' and
         len(base['utility_conditions'])==11 and
         base['training']['EEGNet_fixed_optimizer_updates']==8976,
         'Original amendment is not the frozen 11-condition 8976-update baseline')
    source=(repo/BASE_SOURCE_REL).resolve()
    need(source.is_relative_to(repo) and source.is_file() and not source.is_symlink() and
         sha_file(source)==base['amendment_implementation_sha256'],
         'Original s2_75A source changed; refusing to prepare v2')
    spec=importlib.util.spec_from_file_location('_s75a_v1_verified',source)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    mod.freeze(repo,verify_only=True)
    need(tuple(c['id'] for c in base['utility_conditions'])==mod.ORIGINAL_IDS+mod.ADDED_IDS,
         'Original 11 conditions do not match frozen v1 source')
    return base


def make_amendment(base, implementation_sha):
    b=copy.deepcopy(base)
    b['schema']=VERSION
    b['status']=STATUS
    b['amendment_implementation_sha256']=implementation_sha
    b['parent_s2_75a_v1_amendment']={
        'path':BASE_AMEND_REL.as_posix(),'sha256':BASE_AMEND_SHA256}
    b['timing']=(
        'Revised AFTER examining s2_75B-v1 REAL_ONLY EEGNet seed=20262301 '
        'Validation performance and historical s2_29 best_epoch=24. '
        'All other results and original frozen protocols remain separately archived. '
        'This is a post-pilot revised main utility protocol, NOT the original '
        'prospectively frozen s2_72 method.')
    b['training']={
        'classifier_architecture_and_nonbudget_hyperparameters':'Unchanged original s2_29 EEGNetV4Baseline, Adam lr=0.001 wd=0, dropout=0.5, batch_size=128; max-norm after updates',
        'EEGNet_max_epochs':MAX_EPOCHS,
        'EEGNet_min_epochs':MIN_EPOCHS,
        'EEGNet_early_stopping_patience':PATIENCE,
        'EEGNet_epoch_definition':'One complete shuffled pass through that condition\'s own training rows (ceil(N/128) optimizer steps, including final partial batch).',
        'EEGNet_validation_frequency':'After every completed epoch; Real Validation, never Final Test.',
        'EEGNet_best_checkpoint_rule':'Highest Real-Validation subject-macro balanced accuracy; if tied within 1e-12, lower sample-weighted Validation cross-entropy; all other ties retain earlier best.',
        'EEGNet_early_stop_rule':'Stop if epoch>=8 and six consecutive epochs do not improve the best checkpoint under the same ordered criterion.',
        'EEGNet_min_ba_delta':1e-12,
        'EEGNet_fixed_optimizer_updates':False,
        'EEGNet_validation_checkpoint_selection':True,
        'EEGNet_epoch_update_counts':{'47835':374, '71592':560,'95670':748},
        'EEGNet_training_budget_caveat':'Equal maximum number of full passes, but NOT equal number of optimizer updates across dataset sizes; doubled data can receive ~2x more updates.',
        'EEGNet_seeds':[20262301,20262302,20262303],
        'no_additional_hyperparameter_tuning':True,
        'training_synthetic_only_and_or_real_train':True,
        'classical_classifier_recipes_unchanged':True,
    }
    # Preserve the original classifier protocol verbatim as historic provenance,
    # but make the live v2 EEGNet policy unambiguous rather than retaining the
    # conflicting old s2_72 fixed-update/no-selection instructions.
    b['historical_s2_72_classifier_families']=copy.deepcopy(base['classifier_families'])
    b['classifier_families']['EEGNETV4']['checkpoint_rule']=(
        'Select the highest Real-Validation subject-macro BA checkpoint; '
        'tie within 1e-12: lower Validation cross-entropy; leave earlier best '
        'unchanged if still tied. Early stop at patience 6 after min epoch 8.')
    b['classifier_families']['EEGNETV4']['optimization_budget_rule']=(
        'Train at most 120 actual full passes through the condition-specific '
        'training pool; validate after every epoch; updates per epoch differ '
        'with condition sample count. No longer the fixed-8976-update protocol.')
    b['training_policy_revision_from_s2_72']=True
    b['original_s2_72_primary_contrast_pairs_retained_but_training_policy_changed']=True
    b['validation_used_for_EEGNet_checkpoint_selection']=True
    b['final_test_remains_sealed']=True
    b['validation_is_development_and_not_independent_evidence']=True
    return b


def freeze(repo,verify_only=False):
    repo=Path(repo).resolve()
    need((repo/'scripts/data').is_dir(),'Not the repository root')
    base=base_context(repo)
    own=Path(__file__).resolve()
    need(own==(repo/'scripts/data'/own.name).resolve(),
         'Amendment v2 source must be inside scripts/data of repo')
    amendment=make_amendment(base,sha_file(own))
    out=repo/OUT_REL
    if out.exists():
        need(not out.is_symlink() and canonical(load_json(out))==canonical(amendment),
             'Refusing to overwrite inconsistent frozen s2_75A-v2 amendment')
        print('S2_75A_V2_REUSED_IDENTICAL_FROZEN_AMENDMENT',flush=True)
    else:
        need(not verify_only, 'No frozen v2 amendment: run without --verify-only first')
        out.parent.mkdir(parents=True,exist_ok=True)
        tmp=out.with_name('.'+out.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with tmp.open('xb') as f:
                f.write(canonical(amendment)+b'\n');f.flush();os.fsync(f.fileno())
            os.replace(tmp,out)
        finally:
            if tmp.exists():tmp.unlink()
        print('S2_75A_V2_NEW_AMENDMENT_FROZEN',flush=True)
    print(STATUS,flush=True)
    print('Conditions: 11 (8 original + 3 B0); EEGNet: max epochs=120, min=8, patience=6',flush=True)
    print('Selection: best Real Validation subject-macro BA; tie: lower Validation CE',flush=True)
    print('NOTE: post-pilot main protocol revision; s2_72 and s2_75A-v1 are NOT rewritten',flush=True)
    print('No model trained. Validation EEG accessed: NO; Final Test accessed: NO',flush=True)
    print('AMENDMENT:',out,flush=True)
    print('SHA256:',sha_file(out),flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    try:
        freeze(a.repo_root,a.verify_only)
        return 0
    except Exception as e:
        print('S2_75A_V2_FAILED:',type(e).__name__+': '+str(e),file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
