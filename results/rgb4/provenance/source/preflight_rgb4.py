"""Validate RGB4 data and prepare independent FOUR-class initializations. NO TRAINING."""
import argparse,copy,hashlib,json,os,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path

NATIVE=['fixed_wing_drone','multirotor_drone','bird','airplane','helicopter']
CURRENT=['drone','bird','airplane','helicopter']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--models-only',action='store_true',help='Development smoke check only; skips the full dataset checksum scan')
    args=parser.parse_args();repo=args.repo.resolve()
    os.environ['YOLO_AUTOINSTALL']='false'
    import torch,yaml,ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    torch.set_num_threads(4)
    torch.manual_seed(42)
    data=repo/'dataset/combined_rgb4/data.yaml'
    definition=yaml.safe_load(data.read_text(encoding='utf-8'))
    assert definition['names']==CURRENT,'Wrong dataset class order'
    if not args.models_only:
        subprocess.run([sys.executable,str(repo/'tools/prepare_current_dataset.py')],check=True)
    folder=repo/'results'/('RGB4_Preflight_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(status='preparing',training_started=False,epochs_completed=0,
        dataset=str(data),classes=CURRENT,full_dataset_check=not args.models_only,
        ultralytics=ultralytics.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        models=[],planned_training=dict(initial_epochs=100,maximum_epochs=200,patience=30,imgsz=640,batch=32,amp=True,resume=False),
        scope='Independent initialized four-output models, not trained checkpoints or accuracy results. Final classification outputs are newly initialized; compatible tensors are retained. No optimizer, old epoch or old best score is carried forward.')
    for name in ['HusAero-8n','HusAero-11n']:
        source=repo/'models/rgb5_v2'/name/(name+'_RGB5.pt')
        hef=source.with_suffix('.hef');original_sha=sha(source)
        backup=folder/'original_backups'/name;backup.mkdir(parents=True)
        for path in [source,hef,source.parent/'deployment.json']:
            if path.exists():shutil.copy2(path,backup/path.name)
        history=repo/'results/rgb5_v2'/name/'Training_reference'
        if history.exists():shutil.copytree(history,backup/'Training_reference')
        print('Checking '+name+' -> four classes',flush=True)
        old=YOLO(str(source));assert list(old.names.values())==NATIVE
        old.model.float().cpu()
        cfg=copy.deepcopy(old.model.yaml)
        if cfg.get('scales') and not cfg.get('scale'):cfg['scale']='n'
        target=DetectionModel(cfg=cfg,ch=3,nc=4,verbose=False).float().cpu()
        assert len(old.model.model)==len(target.model)
        head_prefix='model.'+str(len(target.model)-1)+'.'
        previous=old.model.state_dict();fresh=target.state_dict()
        transferred={key:value for key,value in previous.items() if key in fresh and value.shape==fresh[key].shape}
        missing=[key for key in fresh if key not in transferred]
        # Any incompatible tensor outside the classification branches is unexpected.
        unexpected=[key for key in missing if not (key.startswith(head_prefix+'cv3.') or key.startswith(head_prefix+'one2one_cv3.'))]
        assert not unexpected,('Unexpected backbone/box-head mismatch',unexpected)
        assert missing,'Expected fresh classification tensors for changed class count'
        target.load_state_dict(transferred,strict=False)
        target.names=dict(enumerate(CURRENT));target.yaml['nc']=4
        assert target.model[-1].nc==4 and target.model[-1].no==target.model[-1].reg_max*4+4
        assert torch.equal(old.model.stride,target.stride)
        for key in transferred:assert torch.equal(target.state_dict()[key],previous[key]),key
        target.eval()
        with torch.inference_mode():prediction=target(torch.zeros(1,3,640,640))[0]
        assert prediction.ndim==3 and prediction.shape[1]==8 and torch.isfinite(prediction).all(),prediction.shape
        initialized=folder/(name+'_RGB4_initialized.pt')
        prepared=copy.copy(old);prepared.model=target
        prepared.ckpt={'epoch':-1,'best_fitness':None,'optimizer':None,'updates':0,'train_results':None,
                       'train_args':{'task':'detect','data':str(data),'imgsz':640,'model':str(initialized),'resume':False},
                       'preflight_only':True,'source_sha256':original_sha,'training_started':False}
        prepared.save(initialized)
        loaded=YOLO(str(initialized));assert list(loaded.names.values())==CURRENT and loaded.model.model[-1].nc==4
        assert loaded.ckpt['epoch']==-1 and loaded.ckpt['optimizer'] is None
        assert sha(source)==original_sha==sha(backup/source.name),'Original checkpoint changed'
        if hef.exists():assert sha(hef)==sha(backup/hef.name)
        report['models'].append(dict(model=name,source_sha256=original_sha,initialized=str(initialized),
            initialized_sha256=sha(initialized),transferred_tensors=len(transferred),fresh_tensors=missing,
            class_count=4,forward_shape=list(prediction.shape),original_preserved=True,training_started=False))
        print(name+': PASS | nc=4 | original preserved | NO TRAINING',flush=True)
        del loaded,prepared,target,old
    report['status']='PASS'
    (folder/'preflight_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
    print('PREFLIGHT PASS — NO TRAINING STARTED',flush=True)
    print('Output: '+str(folder),flush=True)

if __name__=='__main__':main()
