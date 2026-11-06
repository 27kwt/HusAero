"""Continue learning from release weights in a new run, without overwriting them."""
import argparse,json
from datetime import datetime
from pathlib import Path
from verify_release import ROOT,verify
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['YOLOKW8n','YOLOKW11n'],default='YOLOKW11n')
    p.add_argument('--data',type=Path,default=ROOT/'dataset/data.yaml');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--device',default='cpu');p.add_argument('--batch',type=int,default=16);p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--smoke-test',action='store_true',help='One tiny plumbing test, not a valid training experiment')
    a=p.parse_args();verify(require_models=True)
    import yaml,torch
    from ultralytics import YOLO
    source=ROOT/'samples/data.yaml' if a.smoke_test else a.data.resolve()
    if not source.is_file():raise FileNotFoundError('Training dataset missing. Use the Complete release or pass --data /path/data.yaml')
    cfg=yaml.safe_load(source.read_text());base=Path(cfg.get('path') or '.')
    cfg['path']=str((base if base.is_absolute() else source.parent/base).resolve())
    out=ROOT/'runs'/('finetune_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    config=out/'resolved_data.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    if a.smoke_test:torch.set_num_threads(4)
    model=YOLO(str(ROOT/'models'/(a.model+'.pt')))
    model.train(data=str(config),epochs=1 if a.smoke_test else a.epochs,imgsz=64 if a.smoke_test else a.imgsz,
        batch=4 if a.smoke_test else a.batch,device=a.device,workers=0,seed=42,optimizer='SGD',lr0=.001,
        project=str(out),name=a.model,exist_ok=False,plots=True,save=True,patience=0,amp=False if a.smoke_test else True)
    (out/'run_scope.json').write_text(json.dumps(dict(mode='SMOKE_TEST_NOT_BENCHMARK' if a.smoke_test else 'ADDITIONAL_FINE_TUNING',
        explanation='Starts from release weights with a new optimizer. Not an exact resume of the original training run.'),indent=2))
    print('New fine-tuned model saved under:',out)
if __name__=='__main__':main()
