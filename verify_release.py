"""Verify the current RGB4 checkpoint registry before loading model weights."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = ['drone', 'bird', 'airplane', 'helicopter']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify(root=ROOT, full=False, require_models=False):
    root = Path(root).resolve()
    data = json.loads((root / 'models/registry.json').read_text(encoding='utf-8'))
    if data['classes'] != CLASSES:
        raise ValueError('Expected the current four-class RGB4 registry')
    for name, model in data['models'].items():
        weight = model['weights']['pt']
        path = (root / weight['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Model path escapes the repository')
        if not path.is_file():
            if require_models:
                raise FileNotFoundError('Missing current checkpoint: ' + weight['path'])
            continue
        if path.stat().st_size != weight['bytes'] or sha(path) != weight['sha256']:
            raise ValueError('Checkpoint fingerprint mismatch: ' + name)
    if full:
        from tools.prepare_current_dataset import validate
        validate(root / 'dataset')
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--full', action='store_true', help='Also verify every dataset image and label')
    args = parser.parse_args()
    result = verify(args.root, args.full, require_models=True)
    print('VERIFIED:', result['release'], 'current model checksums' + (' and full dataset' if args.full else ''))
