"""Evaluate the packaged mini test set as a smoke test, not a new benchmark."""
import argparse,json
from pathlib import Path
from verify_release import ROOT,verify
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['YOLOKW11n','YOLOKW8n'],default='YOLOKW11n');p.add_argument('--device',default='cpu');p.add_argument('--data',type=Path,default=ROOT/'samples/data.yaml');a=p.parse_args();verify(require_models=True)
    from ultralytics import YOLO
    import yaml
    from datetime import datetime
    out=ROOT/'runs'/('sample_eval_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    source=a.data.resolve();cfg=yaml.safe_load(source.read_text());base=Path(cfg.get('path') or '.')
    cfg['path']=str((base if base.is_absolute() else source.parent/base).resolve());config=out/'resolved_data.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    model=YOLO(str(ROOT/'models'/(a.model+'.pt')))
    processed={}
    def record_processed(validator):processed['images']=int(validator.seen)
    model.add_callback('on_val_end',record_processed)
    m=model.val(data=str(config),split='test',imgsz=640,batch=4,device=a.device,workers=0,rect=False,conf=.001,iou=.45,max_det=300,quantize=32,plots=True,save_json=False,project=str(out),name=a.model)
    if processed.get('images',0)<=0:raise RuntimeError('Validation processed no images; inspect the test split in data.yaml')
    report=dict(model=a.model,images=processed['images'],scope='Current five-class test evaluation. Packaged samples are a smoke subset of the Complete test split, not an independent benchmark. Historical six-class scores are separate.',metrics=m.results_dict)
    (out/'sample_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
