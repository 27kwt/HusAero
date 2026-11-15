"""Verify the ready-to-use RGB4 dataset; never download, copy or train."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLASSES = ['drone', 'bird', 'airplane', 'helicopter']
EXPECTED = {'train': 13954, 'val': 3423, 'test': 2159}

def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def validate(root):
    root = Path(root).resolve()
    manifest = root / 'provenance/manifest.jsonl'
    summary = json.loads((root / 'provenance/summary.json').read_text(encoding='utf-8'))
    if fingerprint(manifest) != summary['manifest_sha256']:
        raise ValueError('Manifest fingerprint differs from the dataset summary')
    counts = Counter()
    expected_files = {'images': set(), 'labels': set()}
    groups, image_splits = {}, {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        split = row['split']
        if split not in EXPECTED:
            raise ValueError('Unknown dataset split')
        for kind, key in [('images', 'image'), ('labels', 'label')]:
            relative = row[key]
            path = (root / relative).resolve()
            if not path.is_relative_to(root / kind / split) or relative in expected_files[kind]:
                raise ValueError('Invalid or repeated dataset path: ' + relative)
            if not path.is_file() or fingerprint(path) != row[key + '_sha256']:
                raise ValueError('Missing or changed file: ' + relative)
            expected_files[kind].add(relative)
        if Path(row['image']).stem != Path(row['label']).stem:
            raise ValueError('Image/label filenames do not match')
        for annotation in (root / row['label']).read_text().splitlines():
            parts = annotation.split()
            if len(parts) != 5 or int(parts[0]) not in range(4):
                raise ValueError('Expected four-class YOLO labels: ' + row['label'])
            if not all(math.isfinite(float(x)) and 0 <= float(x) <= 1 for x in parts[1:]):
                raise ValueError('Invalid normalized box coordinates: ' + row['label'])
        group = (row['source'], row['group'])
        digest = row['image_sha256']
        if groups.get(group, split) != split or image_splits.get(digest, split) != split:
            raise ValueError('A recorded group or duplicate image crosses partitions')
        groups[group], image_splits[digest] = split, split
        counts[split] += 1
    for kind in expected_files:
        actual = {p.relative_to(root).as_posix() for split in EXPECTED
                  for p in (root / kind / split).rglob('*') if p.is_file()}
        if actual != expected_files[kind]:
            raise ValueError('Missing or additional files in ' + kind)
    if dict(counts) != EXPECTED or summary['classes'] != CLASSES:
        raise ValueError('Dataset size or class order changed')
    return dict(status='PASS', images=dict(counts), labels=dict(counts), classes=CLASSES,
                data=str(root / 'data.yaml'), training_started=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copy-images', action='store_true',
                        help='Compatibility option: images are already in place; no copy is needed')
    parser.add_argument('--data-root', type=Path, default=REPO / 'dataset')
    args = parser.parse_args()
    print(json.dumps(validate(args.data_root), indent=2))

if __name__ == '__main__':
    main()
