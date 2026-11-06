"""Freeze a representative, training-only RGB calibration array for both models."""
import argparse,hashlib,json,random
from pathlib import Path
import cv2
import numpy as np
import yaml

def label_path(image):
    parts=list(Path(image).parts)
    positions=[i for i,part in enumerate(parts) if part=='images']
    if not positions:raise ValueError('Training image path must contain an images directory: '+str(image))
    parts[positions[-1]]='labels'
    return Path(*parts).with_suffix('.txt')

def recorded_path(image,base):
    return Path(image).resolve().relative_to(Path(base).resolve()).as_posix()

def resolve_recorded_image(row,base,allowed):
    relative=Path(row['path'])
    if relative.is_absolute():raise ValueError('Selection must use dataset-relative paths; regenerate the calibration manifest')
    path=(Path(base)/relative).resolve()
    if path not in allowed or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('Selection is not an unchanged training image')
    return path

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=1024);p.add_argument('--selection',type=Path,help='Optional recorded ordered training image inventory');a=p.parse_args()
    cfg=yaml.safe_load(a.data.read_text());base=Path(cfg.get('path') or '.')
    base=(base if base.is_absolute() else a.data.resolve().parent/base).resolve();train=Path(cfg['train']);train=(train if train.is_absolute() else base/train).resolve()
    if train.is_dir():paths=sorted(train.glob('*.jpg'))
    else:paths=[(p if p.is_absolute() else train.parent/p).resolve() for s in train.read_text().splitlines() if s.strip() for p in [Path(s)]]
    if len(paths)<a.count or a.count<1024:raise ValueError('Require at least 1024 training images')
    groups={i:[] for i in range(len(cfg['names']))}
    for path in paths:
        lp=label_path(path)
        for c in {int(l.split()[0]) for l in lp.read_text().splitlines() if l.strip()}:groups[c].append(path)
    rng=random.Random(42);chosen=set()
    for c,pool in groups.items():
        if not pool:raise ValueError('Missing training class '+str(c))
        chosen.update(rng.sample(pool,min(a.count//len(groups),len(pool))))
    remaining=[p for p in paths if p not in chosen];chosen.update(rng.sample(remaining,a.count-len(chosen)))
    selected=sorted(chosen,key=str)
    if a.selection:
        recorded=json.loads(a.selection.read_text());selected=[];allowed=set(paths)
        for row in recorded['images']:
            selected.append(resolve_recorded_image(row,base,allowed))
        if len(selected)!=a.count or len(set(selected))!=a.count:raise ValueError('Calibration selection must have the requested number of unique training images')
    a.out.mkdir(parents=True,exist_ok=False)
    array=np.lib.format.open_memmap(a.out/'calibration.npy',mode='w+',dtype=np.uint8,shape=(a.count,640,640,3));records=[]
    for i,path in enumerate(selected):
        frame=cv2.imread(str(path))
        if frame is None:raise RuntimeError(str(path))
        h,w=frame.shape[:2];gain=640/max(h,w);nw,nh=round(w*gain),round(h*gain);left,top=(640-nw)//2,(640-nh)//2
        canvas=np.full((640,640,3),114,np.uint8);canvas[top:top+nh,left:left+nw]=cv2.resize(frame,(nw,nh))
        array[i]=cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)
        records.append(dict(path=recorded_path(path,base),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),split='train'))
    array.flush();del array
    (a.out/'manifest.json').write_text(json.dumps(dict(seed=42,count=a.count,source_data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
         scope='Training only; identical calibration inputs for YOLOv8n and YOLO11n',images=records),indent=2));print(a.out/'calibration.npy')
if __name__=='__main__':main()
