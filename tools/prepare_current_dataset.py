"""Prepare the unified-drone YOLO directory without downloading or changing source images."""
import argparse,hashlib,json,os,shutil
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copy-images',action='store_true',help='Copy images instead of sharing read-only image files through hard links')
    args=parser.parse_args()
    root=REPO/'dataset/combined_rgb4'
    rows=[json.loads(line) for line in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    counts={s:0 for s in ['train','val','test']}
    for row in rows:
        source=(root/row['canonical_image']).resolve();target=root/row['image'];label=root/row['label']
        assert source.is_relative_to((REPO/'dataset/combined_rgb5_v2/images').resolve())
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row['image_sha256'],source
        assert hashlib.sha256(label.read_bytes()).hexdigest()==row['label_sha256'],label
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest()==row['image_sha256'],target
        else:
            if args.copy_images:shutil.copy2(source,target)
            else:
                try:os.link(source,target)
                except OSError:shutil.copy2(source,target)
        for line in label.read_text().splitlines():
            parts=line.split();assert len(parts)==5 and int(parts[0]) in range(4)
            assert all(0<=float(v)<=1 for v in parts[1:])
        counts[row['split']]+=1
    assert counts=={'train':13954,'val':3423,'test':2159}
    print(json.dumps(dict(status='PASS',images=counts,classes=['drone','bird','airplane','helicopter'],data=str(root/'data.yaml'),training_started=False),indent=2))
if __name__=='__main__':main()
