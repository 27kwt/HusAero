"""Compare SORT and Ultralytics ByteTrack on exactly the same cached detections.

Uses visible.mp4/visible.json only. Single-target localization diagnostics, not a
general multi-object tracking benchmark. No ground truth is used by a tracker.
"""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np

def load_sort():
    p=Path(__file__).parent/'third_party/sort/sort_core.py'
    spec=importlib.util.spec_from_file_location('sort_core',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.Sort

def overlap(boxes,gt):
    if not len(boxes):return np.empty(0)
    inter=np.maximum(0,np.minimum(boxes[:,2:4],gt[2:])-np.maximum(boxes[:,:2],gt[:2])).prod(axis=1)
    areas=np.maximum(0,boxes[:,2:4]-boxes[:,:2]).prod(axis=1);ga=np.maximum(0,gt[2:]-gt[:2]).prod()
    return inter/(areas+ga-inter+1e-9)

def replay(cache,gt,fps,output):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.engine.results import Boxes
    sort=load_sort()(max_age=30,min_hits=3,iou_threshold=0.3)
    byte=BYTETracker(SimpleNamespace(track_buffer=30,track_high_thresh=.25,track_low_thresh=.1,
                     new_track_thresh=.25,match_thresh=.8,fuse_score=True))
    stats={name:dict(frames=0,present=0,localized_iou50=0,iou_sum=0.,absent_frames=0,
           absent_frames_with_tracks=0,matched_id_changes=0,tracked_boxes=0,tracker_seconds=0.) for name in ['SORT','ByteTrack']}
    previous={name:None for name in stats};counts=0
    with cache.open() as f,(output/'tracks.jsonl').open('w') as saved:
        for line in f:
            r=json.loads(line);i=r['frame'];d=np.asarray(r['detections'],dtype=np.float32).reshape(-1,6)
            t=time.perf_counter();s=sort.update(d[d[:,4]>=.25,:5]);sort_time=time.perf_counter()-t
            t=time.perf_counter();b=byte.update(Boxes(d,tuple(r['shape'])));byte_time=time.perf_counter()-t
            b=np.asarray(b,dtype=np.float32).reshape(-1,8)
            for name,tracks,seconds in [('SORT',s,sort_time),('ByteTrack',b,byte_time)]:
                st=stats[name];st['frames']+=1;st['tracked_boxes']+=len(tracks);st['tracker_seconds']+=seconds
                present=bool(gt['exist'][i]);box=np.asarray(gt['gt_rect'][i],dtype=float)
                if present:
                    if box.shape!=(4,) or box[2]<=0 or box[3]<=0:raise ValueError('Invalid present GT box')
                    st['present']+=1;box[2:]+=box[:2];ious=overlap(tracks[:,:4],box)
                    best=int(np.argmax(ious)) if len(ious) else None;best_iou=float(ious[best]) if best is not None else 0.
                    st['iou_sum']+=best_iou
                    if best_iou>=.5:
                        st['localized_iou50']+=1;identity=int(tracks[best,4])
                        if previous[name] is not None and previous[name]!=identity:st['matched_id_changes']+=1
                        previous[name]=identity
                else:
                    st['absent_frames']+=1;st['absent_frames_with_tracks']+=int(len(tracks)>0)
                saved.write(json.dumps(dict(frame=i,tracker=name,tracks=tracks.tolist()))+'\n')
            counts+=1
    for name,s in stats.items():
        s['target_recall_iou50']=s['localized_iou50']/max(s['present'],1)
        s['mean_best_iou_present']=s['iou_sum']/max(s['present'],1)
        s['mean_tracker_ms']=s['tracker_seconds']/max(s['frames'],1)*1000
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--videos',type=Path,required=True)
    p.add_argument('--sequences',nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='0')
    p.add_argument('--max-frames',type=int,default=0);a=p.parse_args()
    from ultralytics import YOLO
    import ultralytics,torch
    if len(set(a.sequences))!=len(a.sequences):raise ValueError('Repeated sequence')
    a.out.mkdir(parents=True,exist_ok=False);model=YOLO(str(a.weights))
    if model.names[0]!='drone':raise ValueError('Expected drone class at ID 0')
    all_results={}
    for name in a.sequences:
        folder=(a.videos/name).resolve()
        if a.videos.resolve() not in folder.parents:raise ValueError('Sequence outside video root')
        video=folder/'visible.mp4';ann=folder/'visible.json';gt=json.loads(ann.read_text())
        input_sha=hashlib.sha256(video.read_bytes()).hexdigest()
        ann_sha=hashlib.sha256(ann.read_bytes()).hexdigest()
        if len(gt['exist'])!=len(gt['gt_rect']):raise ValueError('GT lengths differ')
        dest=a.out/name;dest.mkdir();cache=dest/'detections.jsonl';cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():raise RuntimeError('Cannot read '+str(video))
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        if n!=len(gt['exist']):raise ValueError(f'Frame/annotation mismatch: {n}/{len(gt["exist"])}')
        limit=min(n,a.max_frames) if a.max_frames else n
        tick=time.perf_counter();read=0
        with cache.open('w') as f:
            for i in range(limit):
                ok,frame=cap.read()
                if not ok:raise RuntimeError(f'Video ended early at {i}/{limit}')
                pred=model.predict(frame,imgsz=640,conf=.1,iou=.45,max_det=300,classes=[0],device=a.device,
                                  quantize=32,augment=False,verbose=False)[0]
                f.write(json.dumps(dict(frame=i,shape=list(frame.shape[:2]),detections=pred.boxes.data.cpu().numpy().tolist()))+'\n');read+=1
                if (i+1)%200==0:print(name,'detections',i+1,'/',limit,flush=True)
        cap.release();elapsed=time.perf_counter()-tick
        if hashlib.sha256(video.read_bytes()).hexdigest()!=input_sha or hashlib.sha256(ann.read_bytes()).hexdigest()!=ann_sha:
            raise RuntimeError('Evaluation inputs changed during inference')
        metrics=replay(cache,gt,fps,dest)
        result=dict(sequence=name,frames=read,sequence_frames=n,complete_sequence=read==n,fps=fps,
           detector_wall_seconds=elapsed,metrics=metrics,video=str(video),video_sha256=input_sha,annotation_sha256=ann_sha,
           cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
        (dest/'metrics.json').write_text(json.dumps(result,indent=2));all_results[name]=result;print(json.dumps(result,indent=2),flush=True)
    report=dict(status='DRONE_TRACKER_DIAGNOSTIC',weights=str(a.weights.resolve()),
      weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),ultralytics=ultralytics.__version__,torch=torch.__version__,
      detector=dict(imgsz=640,confidence_floor=.1,nms_iou=.45,max_det=300,classes=[0],half=False),
      sort=dict(max_age=30,min_hits=3,iou_threshold=.3,confidence=.25),
      bytetrack=dict(track_buffer=30,track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,match_thresh=.8,fuse_score=True),
      results=all_results,limitations=['Detection-training/video overlap remains unverified; these are not held-out benchmark claims.',
       'Single annotated target per clip. Mean best IoU uses best overlapping output track during scoring only.',
       'Matched ID changes count consecutive successfully localized target identities, including gaps; not standard MOT IDSW.',
       'No formal IDF1 or HOTA is computed. Other visible targets may be unannotated.',
       'The supplied checkpoint is evaluated on drone class 0 only; other classes are filtered before tracking.',
       'Tracker update timings are CPU wall times; detector time includes decode and serialization, not Pi FPS.'])
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
