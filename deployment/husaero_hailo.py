"""Evaluate or demonstrate a five-class HusAero HEF on Hailo-8."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np
if __package__:
    from .hailo_adapter import prepare
    from .detection_scores import Scores,read_labels
else:
    from hailo_adapter import prepare
    from detection_scores import Scores,read_labels
NAMES=['drone','bird','airplane','helicopter','kite']
def decode(raw,shape,transform,floor=.001):
    if len(raw)!=1 or len(raw[0])!=len(NAMES):raise ValueError('Expected five-class Hailo NMS output')
    h,w=shape[:2];sx,sy,left,top=transform;rows=[]
    for c,boxes in enumerate(raw[0]):
        for y1,x1,y2,x2,score in np.asarray(boxes,dtype=np.float32).reshape(-1,5):
            if not np.isfinite([y1,x1,y2,x2,score]).all() or score<floor:continue
            box=np.array([(x1*640-left)/sx,(y1*640-top)/sy,(x2*640-left)/sx,(y2*640-top)/sy]);box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h)
            if box[2]>box[0] and box[3]>box[1]:rows.append([*box,float(score),c])
    return np.asarray(rows,dtype=np.float32).reshape(-1,6)
def main():
    p=argparse.ArgumentParser();p.add_argument('--hef',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--save-images',action='store_true');p.add_argument('--split',choices=['val','test'],default='test');a=p.parse_args()
    from hailo_platform import HEF,VDevice,ConfigureParams,HailoStreamInterface,InputVStreamParams,OutputVStreamParams,InferVStreams,FormatType
    files=sorted((a.data/'images'/a.split).glob('*.jpg'))
    if not files:raise ValueError('Expected dataset/images/test/*.jpg')
    a.out.mkdir(parents=True,exist_ok=False);scores=Scores(NAMES);hef=HEF(str(a.hef));ins=hef.get_input_vstream_infos();outs=hef.get_output_vstream_infos();inputs=[];infer_seconds=0;t=time.perf_counter()
    if len(ins)!=1 or tuple(ins[0].shape)!=(640,640,3) or len(outs)!=1:raise ValueError('HEF interface mismatch')
    with VDevice() as device:
        group=device.configure(hef,ConfigureParams.create_from_hef(hef=hef,interface=HailoStreamInterface.PCIe))[0]
        inp=InputVStreamParams.make(group,quantized=True,format_type=FormatType.UINT8);out=OutputVStreamParams.make(group,quantized=False,format_type=FormatType.FLOAT32)
        with InferVStreams(group,inp,out) as pipe:
            pipe.set_nms_score_threshold(.001);pipe.set_nms_iou_threshold(.45);pipe.set_nms_max_proposals_per_class(300)
            with group.activate(group.create_params()),(a.out/'predictions.jsonl').open('w') as log:
                for i,path in enumerate(files,1):
                    frame=cv2.imread(str(path));h,w=frame.shape[:2];label=a.data/'labels'/a.split/(path.stem+'.txt');gt=read_labels(label,w,h)
                    tick=time.perf_counter();batch,transform=prepare(frame);raw=pipe.infer({ins[0].name:batch});pred=decode(raw[outs[0].name],frame.shape,transform);pred=pred[pred[:,4].argsort()[::-1]][:300];infer_seconds+=time.perf_counter()-tick;scores.add(gt,pred)
                    inputs.append(dict(image=path.name,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()));log.write(json.dumps(dict(image=path.name,predictions=pred.tolist()))+'\n')
                    if a.save_images:
                        for x1,y1,x2,y2,conf,c in pred:
                            if conf<.25:continue
                            x1,y1,x2,y2=map(int,[x1,y1,x2,y2]);cv2.rectangle(frame,(x1,y1),(x2,y2),(30,210,255),2);cv2.putText(frame,f'{NAMES[int(c)]} {conf:.2f}',(x1,max(18,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,210,255),1)
                        cv2.imwrite(str(a.out/path.name),frame)
                    if i%50==0:print(i,'/',len(files),flush=True)
    report=scores.result();report.update(hef_sha256=hashlib.sha256(a.hef.read_bytes()).hexdigest(),wall_seconds=time.perf_counter()-t,detector_preprocess_infer_decode_seconds=infer_seconds,input_size=640,confidence_floor=.001,nms_iou=.45,max_detections=300,split=a.split,scope='Five-class HusAero '+a.split+' evaluation; session independence unverified')
    (a.out/'metrics.json').write_text(json.dumps(report,indent=2));(a.out/'test_input_hashes.json').write_text(json.dumps(inputs,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
