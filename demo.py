"""Run the downloaded five-class models on images or video without retraining."""
import argparse,json
from datetime import datetime
from pathlib import Path
from verify_release import ROOT,verify
NAMES=['drone','bird','airplane','helicopter','kite']
MODELS=['YOLOKW11n','YOLOKW8n']
def tracking_config(activation,source):
    import yaml
    config=yaml.safe_load(Path(source).read_text())
    config['track_high_thresh']=activation;config['new_track_thresh']=activation
    config['track_low_thresh']=min(float(config['track_low_thresh']),activation/2)
    return config

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS+['both'],default='YOLOKW8n')
    p.add_argument('--source',default=str(ROOT/'samples/images'));p.add_argument('--device',default='cpu')
    p.add_argument('--conf',type=float,default=.25,help='Detection confidence; with --track this sets high/new-track activation while lower scores are retained for association');p.add_argument('--track',action='store_true',help='Ultralytics ByteTrack on drone class only; video input required')
    p.add_argument('--out',type=Path);a=p.parse_args()
    if not 0<a.conf<1:raise ValueError('Confidence must be between 0 and 1')
    if a.track and Path(a.source).suffix.lower() not in {'.mp4','.avi','.mov','.mkv'}:raise ValueError('Tracking requires a video file')
    manifest=verify(require_models=True);from ultralytics import YOLO
    out=(a.out or ROOT/'runs'/('demo_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve();out.mkdir(parents=True,exist_ok=False)
    summary=[]
    detector_confidence=a.conf;tracker_path=None
    if a.track:
        import yaml
        config=tracking_config(a.conf,ROOT/'configs/bytetrack.yaml')
        detector_confidence=config['track_low_thresh'];tracker_path=out/'bytetrack.yaml'
        tracker_path.write_text(yaml.safe_dump(config,sort_keys=False))
    for name in (MODELS if a.model=='both' else [a.model]):
        model=YOLO(str(ROOT/'models'/(name+'.pt')))
        if list(model.names.values())!=NAMES:raise ValueError('Unexpected model classes')
        kwargs=dict(source=a.source,imgsz=640,rect=False,conf=detector_confidence,iou=.45,max_det=300,device=a.device,save=True,save_txt=True,save_conf=True,project=str(out),name=name,exist_ok=False,stream=True,verbose=False,quantize=32)
        results=model.track(tracker=str(tracker_path),classes=[0],**kwargs) if a.track else model.predict(**kwargs)
        frames=boxes=0
        with (out/(name+'_predictions.jsonl')).open('w',encoding='utf-8') as log:
            for result in results:
                frames+=1;rows=result.boxes.data.cpu().tolist();boxes+=len(rows)
                log.write(json.dumps(dict(input=Path(result.path).name,shape=list(result.orig_shape),boxes=rows,names=NAMES,tracking=a.track))+'\n')
        if frames==0:raise RuntimeError('Input produced no readable images or video frames: '+str(a.source))
        summary.append(dict(model=name,processed_frames=frames,predicted_boxes=boxes))
    report=dict(release=manifest['release'],device=a.device,confidence=a.conf,detector_confidence=detector_confidence,tracking_activation_confidence=a.conf if a.track else None,results=summary,
        note='Demo confidence is not system accuracy. PC tracking here uses Ultralytics ByteTrack; the current Pi follow results use FlowAssistedTracker with Lucas-Kanade optical flow.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print('Saved to:',out)
if __name__=='__main__':main()
