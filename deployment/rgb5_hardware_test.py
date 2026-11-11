"""Evaluate a fixed RGB5 sample using real Hailo-8 inference. No PyTorch required."""
import argparse, hashlib, json, platform, subprocess, time, urllib.request
from pathlib import Path
import cv2
import numpy as np

NAMES=['fixed_wing_drone','multirotor_drone','bird','airplane','helicopter']

def iou(a,b):
    if len(a)==0 or len(b)==0:return np.zeros((len(a),len(b)))
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter,1e-12)

def score(records):
    thresholds=np.linspace(.5,.95,10); conf=[]; predcls=[]; truthcls=[]; correct=[]
    for r in records:
        gt=np.array(r['ground_truth'],dtype=float).reshape(-1,5)
        pred=np.array(r['predictions'],dtype=float).reshape(-1,6)
        truthcls.extend(gt[:,0].tolist());conf.extend(pred[:,1].tolist());predcls.extend(pred[:,0].tolist())
        ok=np.zeros((len(pred),10),dtype=bool)
        overlap=iou(gt[:,1:],pred[:,2:]);same=gt[:,0,None]==pred[None,:,0]
        for k,th in enumerate(thresholds):
            gi,pi=np.where((overlap>=th)&same)
            if len(gi):
                matches=np.stack([gi,pi,overlap[gi,pi]],axis=1)
                matches=matches[np.argsort(-matches[:,2])]
                matches=matches[np.unique(matches[:,1],return_index=True)[1]]
                matches=matches[np.argsort(-matches[:,2])]
                matches=matches[np.unique(matches[:,0],return_index=True)[1]]
                ok[matches[:,1].astype(int),k]=True
        correct.extend(ok.tolist())
    conf=np.array(conf);predcls=np.array(predcls);truthcls=np.array(truthcls)
    correct=np.array(correct,dtype=bool).reshape(-1,10)
    x=np.linspace(0,1,1000); curves={key:[] for key in ['PR','P','R','F1']};per=[]
    for cls,name in enumerate(NAMES):
        ids=np.where(predcls==cls)[0];ids=ids[np.argsort(-conf[ids],kind='stable')]
        count=int((truthcls==cls).sum()); tp=correct[ids].cumsum(0);fp=(~correct[ids]).cumsum(0)
        rec=tp/max(count,1);precision=tp/np.maximum(tp+fp,1e-12)
        aps=[]
        for k in range(10):
            mr=np.concatenate(([0],rec[:,k],[1]));mp=np.concatenate(([1],precision[:,k],[0]))
            mp=np.maximum.accumulate(mp[::-1])[::-1]
            # COCO-style 101 recall points, trapezoidal integration, same as Ultralytics AP utility.
            integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
            aps.append(float(integrate(np.interp(np.linspace(0,1,101),mr,mp),np.linspace(0,1,101))) if len(ids) else 0.)
            if k==0:curves['PR'].append(np.interp(x,mr,mp).tolist() if len(ids) else np.zeros(1000).tolist())
        if len(ids):
            pc=np.interp(-x,-conf[ids],precision[:,0],left=1)
            rc=np.interp(-x,-conf[ids],rec[:,0],left=0)
        else:pc=np.zeros(1000);rc=np.zeros(1000)
        curves['P'].append(pc.tolist());curves['R'].append(rc.tolist());curves['F1'].append((2*pc*rc/np.maximum(pc+rc,1e-12)).tolist())
        keep=conf[ids]>=.25;ntp=int(correct[ids][keep,0].sum());nfp=int(keep.sum())-ntp
        per.append(dict(name=name,ground_truth=count,predictions_at_025=int(keep.sum()),TP=ntp,FP=nfp,FN=count-ntp,
                        precision=ntp/max(ntp+nfp,1),recall=ntp/max(count,1),AP50=aps[0],AP50_95=float(np.mean(aps))))
    return dict(precision=float(np.mean([r['precision'] for r in per])),recall=float(np.mean([r['recall'] for r in per])),
                mAP50=float(np.mean([r['AP50'] for r in per])),mAP50_95=float(np.mean([r['AP50_95'] for r in per])),
                per_class=per,curves=dict(x=x.tolist(),**curves))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();root=args.root
    sample=json.loads((root/'sample.json').read_text()); records=sample['images']
    from hailo_platform import HEF,VDevice,HailoStreamInterface,ConfigureParams,InputVStreamParams,OutputVStreamParams,InferVStreams,FormatType
    try:device_info=subprocess.run(['hailortcli','fw-control','identify'],capture_output=True,text=True,timeout=20).stdout
    except Exception as e:device_info=str(e)
    result=dict(id=sample['id'],timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                platform=platform.platform(),hostname=platform.node(),device_info=device_info,
                sample_sha256=hashlib.sha256((root/'sample.json').read_bytes()).hexdigest(),
                sample_images=len(records),source_counts=sample['source_counts'],models=[],
                protocol='Real Hailo-8, 640x640 RGB, compiled NMS confidence 0.001 / IoU 0.45, maximum 300 proposals/class; P/R at confidence 0.25. AP uses IoU 0.50:0.95, confidence ranking and 101 recall points. Separately labelled from native Ultralytics full-test metrics. Stratified sample is not population weighted. No test loss is calculated by the compiled HEF.')
    for name in ['HusAero-8n','HusAero-11n']:
        folder=root/name;manifest=json.loads((folder/'deployment.json').read_text());path=folder/(name+'_RGB5.hef')
        assert manifest['names']==NAMES and hashlib.sha256(path.read_bytes()).hexdigest()==manifest['hef_sha256']
        hef=HEF(str(path));input_info=hef.get_input_vstream_infos()[0];output_info=hef.get_output_vstream_infos()
        assert tuple(input_info.shape)==(640,640,3) and len(output_info)==1
        rows=[];times=[];start=time.perf_counter()
        print('Testing '+name+' on '+str(len(records))+' test images',flush=True)
        with VDevice() as device:
            group=device.configure(hef,ConfigureParams.create_from_hef(hef=hef,interface=HailoStreamInterface.PCIe))[0]
            inp=InputVStreamParams.make(group,quantized=True,format_type=FormatType.UINT8)
            out=OutputVStreamParams.make(group,quantized=False,format_type=FormatType.FLOAT32)
            with InferVStreams(group,inp,out) as pipeline:
                with group.activate(group.create_params()):
                    first=cv2.cvtColor(cv2.imread(str(root/records[0]['image'])),cv2.COLOR_BGR2RGB)[None]
                    for _ in range(3):pipeline.infer({input_info.name:first})
                    for index,row in enumerate(records):
                        image=root/row['image'];assert hashlib.sha256(image.read_bytes()).hexdigest()==row['image_sha256']
                        frame=cv2.imread(str(image));assert frame is not None and frame.shape==(640,640,3)
                        data=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)[None])
                        tic=time.perf_counter();raw=pipeline.infer({input_info.name:data})[output_info[0].name];times.append(time.perf_counter()-tic)
                        classes=raw[0];assert len(classes)==5
                        preds=[]
                        for cls,boxes in enumerate(classes):
                            a=np.asarray(boxes,dtype=float)
                            if a.size==0:continue
                            assert a.ndim==2 and a.shape[1]==5,a.shape
                            for y1,x1,y2,x2,confidence in a:
                                if confidence<.001 or not np.isfinite([y1,x1,y2,x2,confidence]).all():continue
                                box=np.clip([x1*640,y1*640,x2*640,y2*640],0,640).tolist()
                                if box[2]>box[0] and box[3]>box[1]:preds.append([cls,float(confidence),*box])
                        rows.append(dict(image=row['image'],source=row['source'],split=row['split'],ground_truth=row['ground_truth'],predictions=preds))
                        if (index+1)%50==0:print(name+': '+str(index+1)+'/'+str(len(records)),flush=True)
        # Partition metrics stay separate: training scores are not independent test performance.
        inference_wall=time.perf_counter()-start
        split_metrics={s:dict(images=sum(r['split']==s for r in rows),**score([r for r in rows if r['split']==s])) for s in sorted({r['split'] for r in rows})}
        metrics=split_metrics['test']
        elapsed=time.perf_counter()-start
        output=dict(model=name,hef_sha256=manifest['hef_sha256'],checkpoint_sha256=manifest['checkpoint_sha256'],
                    images=len(rows),metrics=metrics,inference_ms_mean=float(np.mean(times)*1000),
                    inference_ms_median=float(np.median(times)*1000),wall_seconds=elapsed,
                    file_throughput_fps=len(rows)/inference_wall,wall_inference_seconds=inference_wall,
                    split_metrics=split_metrics,predictions=rows)
        result['models'].append(output)
        (root/(name+'_result.json')).write_text(json.dumps(output))
        print(name+' mAP50='+str(metrics['mAP50'])+' mAP50-95='+str(metrics['mAP50_95']),flush=True)
    target=root/'Pi_test_results.json';target.write_text(json.dumps(result));print('Saved '+str(target),flush=True)
    callback=root/'callback.json'
    if callback.exists():
        config=json.loads(callback.read_text());request=urllib.request.Request(config['url'],data=target.read_bytes(),method='POST',headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=90) as response:print('Results returned to PC: '+str(response.status),flush=True)

if __name__=='__main__':main()
