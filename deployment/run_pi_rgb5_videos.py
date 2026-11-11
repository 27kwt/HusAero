"""Actual RGB5/Hailo inference for two recorded RGB clips, with per-frame evidence."""
import hashlib,importlib.util,json,platform,subprocess,time,urllib.request,zipfile
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    settings=json.loads((ROOT/'video_inputs.json').read_text())
    spec=importlib.util.spec_from_file_location('husaero_demo',ROOT/'husaero_rgb5_live.py')
    demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    info=subprocess.run(['hailortcli','fw-control','identify'],capture_output=True,text=True,timeout=20).stdout
    assert 'HAILO8' in info
    results=dict(hostname=platform.node(),device_info=info,backend='Hailo-8',runs=[],
                 scope='Real RGB inference with basic centre association, not ByteTrack or optical flow. Unannotated source footage: counts are operation evidence, not accuracy or recall. Preserve source text/watermarks. Video playback uses source FPS, not measured processing throughput.')
    for model in ['HusAero-8n','HusAero-11n']:
        path=ROOT/model/(model+'_RGB5.hef');manifest=json.loads((ROOT/model/'deployment.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['hef_sha256']
        with demo.detector('hailo',path) as predict:
            for item in settings:
                source=ROOT/item['file'];assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
                cap=cv2.VideoCapture(str(source));assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);assert fps>0
                tracker=demo.Tracker();target=out/(model+'_'+item['id']+'.mp4');writer=None;times=[]
                counters=dict(frames=0,detected_frames=0,tracked_frames=0,subtype_outputs={k:0 for k in demo.NAMES[:2]})
                with target.with_suffix('.jsonl').open('w') as log:
                    for index in range(200):
                        ok,frame=cap.read();assert ok
                        tic=time.perf_counter();detections=predict(frame);times.append(time.perf_counter()-tic)
                        selected,state=tracker.update(detections,frame.shape)
                        original=frame.copy();display=demo.draw_tracking(frame.copy(),detections,selected,tracker,state,'')
                        # Place experiment titles outside the source frame; retain original source text.
                        raw=cv2.resize(original,(display.shape[1],display.shape[0]))
                        display[:30]=raw[:30]
                        header=np.zeros((44,display.shape[1],3),np.uint8)
                        footer=np.zeros((30,display.shape[1],3),np.uint8)
                        cv2.putText(header,model+' RGB5 | Raspberry Pi + Hailo-8 | '+state,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,245,250),1,cv2.LINE_AA)
                        cv2.putText(footer,item['id']+' | RGB source frame '+str(index+1)+' | Recorded footage',(10,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,245,250),1,cv2.LINE_AA)
                        display=np.vstack([header,display,footer])
                        if writer is None:
                            writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(display.shape[1],display.shape[0]));assert writer.isOpened()
                        writer.write(display)
                        counters['frames']+=1;counters['detected_frames']+=int(bool(detections));counters['tracked_frames']+=int(selected is not None)
                        for _,_,cls in detections:counters['subtype_outputs'][demo.NAMES[cls]]+=1
                        log.write(json.dumps(dict(frame=index+1,state=state,track_id=tracker.track_id,detections=detections))+'\n')
                    writer.release();cap.release()
                counters.update(model=model,clip=item['id'],source_video_sha256=item['sha256'],source_fps=fps,
                                output_file=target.name,output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                hef_sha256=manifest['hef_sha256'],inference_ms_mean=float(np.mean(times)*1000))
                results['runs'].append(counters)
                print(model+' / '+item['id']+': '+str(counters['frames'])+' frames saved',flush=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    archive=ROOT/'Pi_RGB5_Videos.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
        for p in out.iterdir():z.write(p,p.name)
    callback=json.loads((ROOT/'callback.json').read_text())
    req=urllib.request.Request(callback['url'],data=archive.read_bytes(),method='POST',headers={'Content-Type':'application/zip'})
    with urllib.request.urlopen(req,timeout=120) as r:print('Actual videos returned to PC: '+str(r.status),flush=True)
if __name__=='__main__':main()
