"""PyTorch evaluation using the exact shared NumPy scoring used for Hailo."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from deployment.husaero_hailo import NAMES
from deployment.detection_scores import Scores,read_labels

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='0');a=p.parse_args()
    from ultralytics import YOLO
    a.out.mkdir(parents=True,exist_ok=False);model=YOLO(str(a.weights));names=list(model.names.values());scores=Scores(names);inputs=[];t=time.perf_counter()
    if names not in [NAMES,NAMES[:5]]:raise ValueError('Unexpected class mapping')
    source=a.data/'test.txt' if (a.data/'test.txt').is_file() else a.data/'images/test'
    expected=len(source.read_text().splitlines()) if source.is_file() else len(list(source.glob('*.jpg')))
    results=model.predict(source=str(source),stream=True,batch=16,imgsz=640,rect=False,
              conf=.001,iou=.45,max_det=300,device=a.device,quantize=32,augment=False,verbose=False)
    with (a.out/'predictions.jsonl').open('w') as saved:
        for i,result in enumerate(results,1):
            path=Path(result.path);h,w=result.orig_shape;label=a.data/'labels/test'/(path.stem+'.txt')
            gt=read_labels(label,w,h);pred=result.boxes.data.cpu().numpy().reshape(-1,6);scores.add(gt,pred)
            saved.write(json.dumps(dict(image=path.name,shape=[h,w],predictions=pred.tolist()))+'\n')
            inputs.append(dict(image=path.name,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
            if i%500==0:print(i,'/',expected,flush=True)
    if scores.images!=expected:raise RuntimeError('Evaluation image count differs from frozen manifest')
    result=scores.result();result.update(weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),wall_seconds=time.perf_counter()-t,
         input_size=640,confidence_floor=.001,nms_iou=.45,max_detections=300,rect=False,
         scope='Shared NumPy metric for PC vs Hailo; official Ultralytics metrics reported separately')
    (a.out/'metrics.json').write_text(json.dumps(result,indent=2));(a.out/'test_input_hashes.json').write_text(json.dumps(inputs,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
