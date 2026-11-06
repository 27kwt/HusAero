"""Extract reviewed-source RGB frames into an exclusive YOLO dataset directory.

Input: sequence/visible.mp4 plus source visible.json (exist and gt_rect xywh).
No annotation is drawn onto training images. Timestamp groups must be disjoint.
"""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np,yaml

def main():
 p=argparse.ArgumentParser();p.add_argument('--video-root',type=Path,required=True);p.add_argument('--groups',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--train-stride',type=int,default=50);p.add_argument('--val-stride',type=int,default=100);p.add_argument('--include-source-absence',action='store_true',help='Only after visual review: target absence alone does not establish an empty five-class scene');a=p.parse_args()
 groups=json.loads(a.groups.read_text());sets={k:set(groups[k]) for k in ['train','val','reserved']}
 if sets['train']&sets['val'] or sets['train']&sets['reserved'] or sets['val']&sets['reserved']:raise ValueError('Timestamp groups overlap')
 if min(a.train_stride,a.val_stride)<1:raise ValueError('Stride must be positive')
 a.out.mkdir(parents=True,exist_ok=False);records=[];seen=set();rejected=[]
 for split in ['train','val']:
  for kind in ['images','labels']:(a.out/kind/split).mkdir(parents=True)
  for folder in sorted(p for p in a.video_root.iterdir() if p.is_dir() and p.name[:15] in sets[split]):
   video=folder/'visible.mp4';annotation=folder/'visible.json';truth=json.loads(annotation.read_text());cap=cv2.VideoCapture(str(video));assert cap.isOpened();count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert count==len(truth['exist'])==len(truth['gt_rect']);stride=a.train_stride if split=='train' else a.val_stride
   for i in range(count):
    ok,frame=cap.read();assert ok
    if i%stride or not truth['exist'][i] and not a.include_source_absence:continue
    h,w=frame.shape[:2];digest=hashlib.sha256(frame.tobytes()).hexdigest()
    if digest in seen:continue
    label='';correction=None
    if truth['exist'][i]:
     x,y,bw,bh=map(float,truth['gt_rect'][i]);raw=[x,y,x+bw,y+bh];box=np.array(raw);box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h);x1,y1,x2,y2=box
     if x2-x1<3 or y2-y1<3:rejected.append(dict(sequence=folder.name,frame=i+1,reason='Invalid or sub-three-pixel source box'));continue
     if not np.array_equal(box,raw):correction=dict(original_xyxy=raw,clipped_xyxy=box.tolist())
     label=f'0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}\n'
    seen.add(digest);name=folder.name+'_'+str(i+1).zfill(5);image=a.out/'images'/split/(name+'.jpg');assert cv2.imwrite(str(image),frame,[cv2.IMWRITE_JPEG_QUALITY,95]);(a.out/'labels'/split/(name+'.txt')).write_text(label);records.append(dict(split=split,sequence=folder.name,frame=i+1,pixel_sha256=digest,image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),source_annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),label_correction=correction,output='images/'+split+'/'+image.name))
   cap.release()
 (a.out/'data.yaml').write_text(yaml.safe_dump(dict(train='images/train',val='images/val',names=['drone','bird','airplane','helicopter','kite']),sort_keys=False));(a.out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));(a.out/'review_required.json').write_text(json.dumps(dict(status='REQUIRES_SEMANTIC_REVIEW',images=len(records),rejected=rejected,groups=groups,limitations='Decoded-pixel uniqueness and timestamp grouping are not full near-duplicate/session certification. Source annotations are not newly verified ground truth. Empty labels from source target absence require review for other airborne classes. This video-only dataset labels drone; combine reviewed bird/aircraft examples before five-class fine-tuning.'),indent=2));print('Extracted to',a.out,'Review labels before training.')
if __name__=='__main__':main()
