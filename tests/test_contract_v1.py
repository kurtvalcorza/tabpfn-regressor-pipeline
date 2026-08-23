from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[1]
def load(path:str): return json.loads((ROOT/path).read_text(encoding='utf-8'))

def test_contract_identity_and_model_limits():
    c=load('contract/contract-v1.json')
    assert c['contractVersion']==1 and c['taskType']=='tabular_regression'
    assert c['datasetFingerprintAlgorithm']=='sha256-path-content-v1'
    assert c['modelVersions']['v2']['max_features']==500
    assert c['modelVersions']['v3']['max_samples']==1000000

def test_all_json_schemas_are_valid_draft_2020_12():
    for path in (ROOT/'contract/schemas').glob('*.schema.json'):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding='utf-8')))

def test_validation_success_and_failure_envelopes():
    v=Draft202012Validator(load('contract/schemas/validation-result.schema.json'))
    meta={'contractVersion':1,'taskType':'tabular_regression','datasetFingerprint':'a'*64,'datasetFingerprintAlgorithm':'sha256-path-content-v1','resolvedModel':{'version':'v3'}}
    v.validate({'successful':True,'contractVersion':1,'checks':[],'metadata':meta})
    v.validate({'successful':False,'contractVersion':1,'code':'INVALID_PARAMETER','metadata':{**meta,'datasetFingerprint':None}})

def test_training_success_requires_portable_artifact_and_reload_check():
    v=Draft202012Validator(load('contract/schemas/training-result.schema.json'))
    meta={'contractVersion':1,'taskType':'tabular_regression','datasetFingerprint':'b'*64,'datasetFingerprintAlgorithm':'sha256-path-content-v1','resolvedModel':{'version':'v2'}}
    good={'successful':True,'contractVersion':1,'metadata':meta,'metrics':{},'provenance':{},'artifacts':{'modelArtifact':{'path':'fine-tuning/r/artifacts/model.tabpfn_fit'},'foundationCheckpoint':{'path':'fine-tuning/r/artifacts/model.ckpt'},'manifest':{'path':'fine-tuning/r/artifacts/artifact_manifest.json'},'reloadCheck':{'passed':True}}}
    v.validate(good)
    bad=json.loads(json.dumps(good)); del bad['artifacts']['reloadCheck']; assert list(v.iter_errors(bad))

def test_worker_parameter_schema_rejects_out_of_range_and_unknown():
    v=Draft202012Validator(load('contract/schemas/worker-parameters.schema.json'))
    assert not list(v.iter_errors({'preprocessing':{'validation_split':0.2},'hyperparameters':{'epochs':30}}))
    assert list(v.iter_errors({'preprocessing':{'validation_split':0.9},'hyperparameters':{}}))
    assert list(v.iter_errors({'preprocessing':{'surprise':1},'hyperparameters':{}}))

def test_component_candidate_snapshots_are_canonical():
    subprocess.run([sys.executable,'scripts/check_component_contracts.py'],cwd=ROOT,check=True)
