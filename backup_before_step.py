"""Copy selected essential files into an exclusive Riyadh-timestamped folder.

Every copied file is SHA-256 verified. Reuse verified immutable dataset snapshots
by reference; pass their verification reports with --reference. Never delete or
overwrite an existing backup. A .partial marker means a failed/incomplete backup.
"""
import argparse,hashlib,json,re,shutil
from datetime import datetime,timezone,timedelta
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def snapshot(paths,destination,label,references=()):
    stamp=datetime.now(timezone(timedelta(hours=3))).strftime('%Y%m%d_%H%M%S_%f')
    label=re.sub(r'[^A-Za-z0-9_-]+','_',label)
    dest=Path(destination).resolve()/f'backup_{stamp}_{label}'
    dest.mkdir(parents=True,exist_ok=False);marker=dest/'BACKUP_INCOMPLETE.json';marker.write_text('{}')
    records=[]
    for i,path in enumerate(paths):
        src=Path(path).resolve()
        if not src.exists():raise FileNotFoundError(src)
        if src==dest or src in dest.parents:raise ValueError('Backup cannot contain itself')
        files=sorted(src.rglob('*')) if src.is_dir() else [src]
        for f in files:
            if not f.is_file():continue
            if any(part in {'.venv','__pycache__','.git'} for part in f.relative_to(src).parts if src.is_dir()):continue
            rel=Path(f'{i:02d}_{src.name}')/(f.relative_to(src) if src.is_dir() else Path(f.name))
            target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
            expected=sha(f);shutil.copy2(f,target);actual=sha(target)
            if actual!=expected:raise RuntimeError('Backup mismatch: '+str(f))
            records.append(dict(source=str(f),backup=rel.as_posix(),bytes=target.stat().st_size,sha256=actual))
    refs=[]
    for path in references:
        p=Path(path).resolve();r=json.loads(p.read_text())
        if r.get('status')!='VERIFIED':raise ValueError('Unverified reference '+str(p))
        if not Path(r['archive']).is_file():raise FileNotFoundError(r['archive'])
        refs.append(dict(verification_report=str(p),verification_sha256=sha(p),archive=r['archive']))
    report=dict(status='VERIFIED',created_at=datetime.now(timezone(timedelta(hours=3))).isoformat(),label=label,
                copied_files=records,immutable_dataset_references=refs,source_modified=False)
    (dest/'backup_manifest.json').write_text(json.dumps(report,indent=2));marker.unlink();return dest

def main():
    p=argparse.ArgumentParser();p.add_argument('--path',type=Path,action='append',required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--reference',type=Path,action='append',default=[]);a=p.parse_args()
    print(snapshot(a.path,a.destination,a.label,a.reference))
if __name__=='__main__':main()
