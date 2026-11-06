"""Run RGB Hailo detection and optional single-target follow, without annotations.

All five-class detections are logged. Detection and follow videos are separate
so a selected target cannot be mistaken for the complete detector output.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from deployment.hailo_adapter import prepare
from deployment.husaero_hailo import decode,NAMES
from flow_assisted_tracker import FlowAssistedTracker
from postprocess import suppress_nested_drone_boxes

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def validate_video_metadata(fps,count):
 if not np.isfinite(fps) or fps<=0 or count<=0:
  raise ValueError('Video needs a positive FPS and frame count. Re-encode the recorded clip before inference; unknown-length/live streams are not supported here.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--hef',type=Path,required=True);p.add_argument('--video-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--save-videos',action='store_true');p.add_argument('--display-confidence',type=float,default=.25);p.add_argument('--activation',type=float,default=.1);p.add_argument('--keep-threshold',type=float,default=.05);p.add_argument('--max-gap-seconds',type=float,default=.5);p.add_argument('--tracking-config',type=Path);a=p.parse_args()
 if a.tracking_config:
  cfg=json.loads(a.tracking_config.read_text());assert set(cfg)=={'activation','keep_threshold','max_gap_seconds'}
  for key,value in cfg.items():setattr(a,key,float(value))
 if not 0<a.keep_threshold<=a.activation<1 or not 0<a.max_gap_seconds<=2:raise ValueError('Invalid tracker thresholds or gap duration')
 from hailo_platform import HEF,VDevice,ConfigureParams,HailoStreamInterface,InputVStreamParams,OutputVStreamParams,InferVStreams,FormatType
 videos=sorted(a.video_root.glob('*/visible.mp4'))
 if not videos:raise ValueError('Expected video-root/sequence/visible.mp4')
 a.out.mkdir(parents=True,exist_ok=False);hef=HEF(str(a.hef));ins=hef.get_input_vstream_infos();outs=hef.get_output_vstream_infos();assert len(ins)==len(outs)==1 and tuple(ins[0].shape)==(640,640,3);reports=[]
 with VDevice() as device:
  group=device.configure(hef,ConfigureParams.create_from_hef(hef=hef,interface=HailoStreamInterface.PCIe))[0];inp=InputVStreamParams.make(group,quantized=True,format_type=FormatType.UINT8);out=OutputVStreamParams.make(group,quantized=False,format_type=FormatType.FLOAT32)
  with InferVStreams(group,inp,out) as pipe:
   pipe.set_nms_score_threshold(.01);pipe.set_nms_iou_threshold(.45);pipe.set_nms_max_proposals_per_class(300)
   with group.activate(group.create_params()):
    # Warm the configured network before timing the complete video pipeline.
    pipe.infer({ins[0].name:np.zeros((1,640,640,3),np.uint8)})
    for video in videos:
     cap=cv2.VideoCapture(str(video));assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));dest=a.out/video.parent.name;dest.mkdir();tracker=FlowAssistedTracker(fps,activation=a.activation,keep_threshold=a.keep_threshold,max_gap_seconds=a.max_gap_seconds);writers={};durations=dict(read=0.,detector=0.,tracker=0.,draw_write_log=0.);states={};start=time.perf_counter()
     validate_video_metadata(fps,count)
     with (dest/'detections.jsonl').open('w') as detlog,(dest/'frame_states.jsonl').open('w') as statefile,(dest/'tracks.csv').open('w',newline='') as trackfile:
      csvout=csv.writer(trackfile);csvout.writerow(['frame','time_seconds','track_id','x1','y1','x2','y2','state'])
      try:
       if a.save_videos:
        for mode in ['detections','follow']:
         writers[mode]=cv2.VideoWriter(str(dest/(mode+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));assert writers[mode].isOpened()
       for i in range(count):
        t=time.perf_counter();ok,frame=cap.read();assert ok;durations['read']+=time.perf_counter()-t
        t=time.perf_counter();batch,transform=prepare(frame);raw=pipe.infer({ins[0].name:batch});pred=decode(raw[outs[0].name],frame.shape,transform,floor=.01);pred=pred[np.argsort(-pred[:,4],kind='stable')][:300];durations['detector']+=time.perf_counter()-t
        t=time.perf_counter();tracks=tracker.update(frame,suppress_nested_drone_boxes(pred));durations['tracker']+=time.perf_counter()-t
        t=time.perf_counter();state=tracker.state;states[state]=states.get(state,0)+1
        detlog.write(json.dumps(dict(frame=i+1,detections=pred.tolist()))+'\n');statefile.write(json.dumps(dict(frame=i+1,state=state,tracks=tracks.tolist()))+'\n')
        for x1,y1,x2,y2,identity in tracks:csvout.writerow([i+1,i/fps,int(identity),x1,y1,x2,y2,state])
        if a.save_videos:
         display=frame.copy()
         for x1,y1,x2,y2,conf,c in pred[pred[:,4]>=a.display_confidence]:
          xy=tuple(map(int,[x1,y1,x2,y2]));cv2.rectangle(display,xy[:2],xy[2:],(30,210,255),2);cv2.putText(display,f'{NAMES[int(c)]} {conf:.2f}',(xy[0],max(75,xy[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,210,255),2)
         cv2.rectangle(display,(0,0),(w,55),(15,20,25),-1);cv2.putText(display,f'ALL DETECTIONS | confidence >= {a.display_confidence:.2f} | RGB',(10,33),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2);writers['detections'].write(display)
         display=frame.copy()
         for x1,y1,x2,y2,identity in tracks:
          xy=tuple(map(int,[x1,y1,x2,y2]));color=(40,220,60) if state=='DETECTION_CONFIRMED' else (20,210,255);cv2.rectangle(display,xy[:2],xy[2:],color,2);cv2.putText(display,f'drone ID {int(identity)}',(xy[0],max(75,xy[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
         cv2.rectangle(display,(0,0),(w,55),(15,20,25),-1);cv2.putText(display,'SINGLE TARGET FOLLOW | '+state,(10,33),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2);writers['follow'].write(display)
        durations['draw_write_log']+=time.perf_counter()-t
        if (i+1)%250==0:print(video.parent.name,i+1,'/',count,flush=True)
      finally:
       cap.release()
       for writer in writers.values():writer.release()
     wall=time.perf_counter()-start
     report=dict(sequence=video.parent.name,frames=count,source_fps=fps,source_video_sha256=sha(video),hef_sha256=sha(a.hef),runtime_sha256=sha(Path(__file__)),tracker_sha256=sha(Path(__file__).resolve().parent.parent/'flow_assisted_tracker.py'),states=states,wall_seconds=wall,end_to_end_fps=count/wall,component_seconds=durations,save_videos=a.save_videos,display_confidence=a.display_confidence,tracker=dict(activation=a.activation,keep_threshold=a.keep_threshold,max_gap_seconds=a.max_gap_seconds),nms_iou=.45,confidence_floor=.01,names=NAMES,scope='Actual sequential Pi/Hailo RGB decode + inference + tracker + logging'+(' + two full video renders/encodes' if a.save_videos else '')+'. No annotations read. Not a live-camera latency measurement.')
     reports.append(report);(dest/'runtime.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 result=dict(sequences=reports,frames=sum(r['frames'] for r in reports),wall_seconds=sum(r['wall_seconds'] for r in reports));result['end_to_end_fps']=result['frames']/result['wall_seconds'];(a.out/'runtime.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
