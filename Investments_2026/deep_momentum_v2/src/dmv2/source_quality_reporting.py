"""Read a pinned repair handoff without loading market values or estimators."""
from pathlib import Path
import hashlib
import json

from .universe import DataContractError


def read_source_quality_repair(root, *, maximum_files=600, maximum_bytes=2*1024**3):
    root = Path(root).resolve()
    config = root/'configs/source_quality_repair_20261016.json'
    if not config.is_file() or config.stat().st_size > 65536:
        raise DataContractError('Missing bounded source-quality repair configuration')
    policy = json.loads(config.read_text())
    pin = policy['handoff']

    def verify(item):
        path = root/item['path']
        if (Path(item['path']).is_absolute() or not path.resolve().is_relative_to(root)
                or any(part.is_symlink() for part in [path,*path.parents])
                or not path.is_file() or path.stat().st_size != item['size_bytes']):
            raise DataContractError('Repair artifact path or size changed')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024),b''):
                digest.update(chunk)
        if digest.hexdigest() != item['sha256'].removeprefix('sha256:'):
            raise DataContractError('Repair artifact SHA256 changed')
        return path

    if pin['size_bytes'] > 256*1024:
        raise DataContractError('Repair handoff exceeds metadata budget')
    handoff = json.loads(verify(pin).read_text())
    files = handoff['artifacts']
    if (len(files) > maximum_files or sum(item['size_bytes'] for item in files) > maximum_bytes
            or len({item['path'] for item in files}) != len(files)):
        raise DataContractError('Repair handoff exceeds its integrity-check budget')
    if (handoff['status'] != 'REPAIRED_INPUTS_READY_MODELS_STALE'
            or handoff['financial_2024_plus_analysis'] is not False
            or handoff['new_model_fits'] != 0 or handoff['new_backtests'] != 0
            or handoff['existing_runs_modified'] is not False):
        raise DataContractError('Repair handoff changed its scope or completion claim')
    for item in files:
        verify(item)
    return {'status':handoff['status'],'summary':handoff['summary'],
            'dependent_fits':handoff['dependent_fits'],
            'verified_files':len(files),'verified_bytes':sum(item['size_bytes'] for item in files),
            'financial_values_loaded':False,'models_loaded_or_fitted':False}
