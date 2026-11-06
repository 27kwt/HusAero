"""Verify the downloaded release before loading its bundled model weights."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify(root=ROOT,full=False,require_models=False):
    root=Path(root).resolve();manifest=root/'release_manifest.json'
    if not manifest.is_file():raise FileNotFoundError('Use the complete model release ZIP, including release_manifest.json')
    data=json.loads(manifest.read_text(encoding='utf-8'))
    if require_models:
        missing=[name for name in ('HusAero-8n.pt','HusAero-11n.pt') if not (root/'models'/name).is_file()]
        if missing:
            raise FileNotFoundError('Trained weights are missing: '+', '.join(missing)+'. Download the Complete ZIP from GitHub Releases and extract it. The source-code ZIP does not include weights.')
    for row in data['files']:
        if row['path'].startswith('dataset/') and not full:continue
        path=(root/row['path']).resolve()
        if root not in path.parents:raise ValueError('Manifest path escapes the release')
        if not path.is_file() or sha(path)!=row['sha256']:raise ValueError('Missing or changed release file: '+row['path'])
    return data
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--full',action='store_true');a=p.parse_args();m=verify(a.root,a.full);print('VERIFIED:',m['release'],'all packaged files' if a.full else 'packaged files excluding the full dataset')
