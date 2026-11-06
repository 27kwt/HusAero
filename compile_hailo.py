"""Hailo-8 DFC 3.x parser/compiler for five-class YOLOv8/11 detection ONNX.

Run in Linux x86_64 with the licensed Hailo Dataflow Compiler installed. Outputs
are exclusive. Calibration arrays must be RGB uint8 NHWC, from training only.
Use --parse-only to validate graph compatibility without quantization/compile.
"""
import argparse,hashlib,json,re,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--calibration',type=Path);p.add_argument('--parse-only',action='store_true')
    p.add_argument('--optimization-level',type=int,choices=[0,1,2],default=0)
    p.add_argument('--class-names',nargs='+',default=['drone','bird','airplane','helicopter','kite'])
    p.add_argument('--compiler-effort',type=int,choices=[0,1],default=1);a=p.parse_args();nc=len(a.class_names)
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',a.name):raise ValueError('Invalid network name')
    import onnx
    from hailo_sdk_client import ClientRunner
    graph=onnx.load(str(a.onnx));nodes={n.name for n in graph.graph.node}
    pattern=re.compile(r'^/model\.(\d+)/cv([23])\.([012])/cv\2\.\3\.2/Conv$')
    matched=[(pattern.match(n),n) for n in nodes if pattern.match(n)]
    if len(matched)!=6:raise ValueError(f'Expected six YOLO head convolutions, found {len(matched)}')
    endpoints=[n for m,n in sorted(matched,key=lambda t:(int(t[0].group(3)),int(t[0].group(2))))]
    a.out.mkdir(parents=True,exist_ok=False);t=time.time()
    runner=ClientRunner(hw_arch='hailo8');runner.translate_onnx_model(str(a.onnx),a.name,end_node_names=endpoints)
    runner.save_har(str(a.out/'parsed.har'))
    hn=runner.get_hn();hn=json.loads(hn) if isinstance(hn,str) else hn
    (a.out/'network.json').write_text(json.dumps(hn,indent=2))
    layers=runner.get_hn_model().get_output_layers()
    outputs=[layer.inputs[0] for layer in layers]
    if len(outputs)!=6:raise ValueError('Expected six parsed output heads')
    report=dict(stage='PARSED',network=a.name,hardware='hailo8',endpoints=endpoints,parsed_outputs=outputs,
       onnx_sha256=hashlib.sha256(a.onnx.read_bytes()).hexdigest(),elapsed_seconds=time.time()-t)
    (a.out/'compile_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    mapping={}
    for endpoint in endpoints:
        hits=[(n,layer) for n,layer in hn['layers'].items() if endpoint in layer.get('original_names',[]) and layer['type']=='conv']
        if len(hits)!=1:raise ValueError('Ambiguous head mapping: '+endpoint)
        mapping[endpoint]=hits[0]
    decoders=[];class_layers=[]
    for scale,stride in enumerate([8,16,32]):
        reg,cls=endpoints[2*scale:2*scale+2];rn,rl=mapping[reg];cn,cl=mapping[cls]
        if rl['output_shapes'][0][1:]!=[640//stride,640//stride,64]:raise ValueError('Unexpected regression output shape')
        if cl['output_shapes'][0][1:]!=[640//stride,640//stride,nc]:raise ValueError('Unexpected class output shape')
        decoders.append(dict(name=a.name+f'/bbox_decoder{scale}',stride=stride,reg_layer=rn,cls_layer=cn));class_layers.append(cn)
    nms=dict(nms_scores_th=.001,nms_iou_th=.45,image_dims=[640,640],max_proposals_per_class=300,
             classes=nc,regression_length=16,background_removal=False,bbox_decoders=decoders)
    config=a.out.resolve()/'nms_config.json';config.write_text(json.dumps(nms,indent=2))
    report['head_mapping_verified']=True
    (a.out/'compile_report.json').write_text(json.dumps(report,indent=2))
    if a.parse_only:return
    if a.calibration is None:raise ValueError('Calibration dataset required for quantization')
    calib=np.load(a.calibration,mmap_mode='r')
    if calib.dtype!=np.uint8 or calib.ndim!=4 or tuple(calib.shape[1:])!=(640,640,3) or len(calib)<1024:
        raise ValueError('Expected at least 1024 RGB uint8 calibration images shaped [N,640,640,3]')
    script=['normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])']
    script += [f'change_output_activation({name.rsplit("/",1)[-1]}, sigmoid)' for name in class_layers]
    script += [f'performance_param(compiler_optimization_level={a.compiler_effort})',f'model_optimization_flavor(optimization_level={a.optimization_level})',f'model_optimization_config(calibration, calibset_size={len(calib)})',
               f'nms_postprocess("{config}", meta_arch=yolov8, engine=cpu)']
    (a.out/'model.alls').write_text('\n'.join(script)+'\n');runner.load_model_script('\n'.join(script))
    runner.optimize(calib);runner.save_har(str(a.out/'optimized.har'))
    report.update(stage='OPTIMIZED',optimization_level=a.optimization_level,calibration_shape=list(calib.shape),calibration_sha256=hashlib.sha256(a.calibration.read_bytes()).hexdigest())
    (a.out/'compile_report.json').write_text(json.dumps(report,indent=2))
    hef=runner.compile();target=a.out/'model.hef';target.write_bytes(hef)
    report.update(stage='COMPILED',hef_sha256=hashlib.sha256(hef).hexdigest(),elapsed_seconds=time.time()-t,
                  names=a.class_names,compiler_effort=a.compiler_effort,input='RGB uint8 NHWC 640x640, letterbox 114')
    (a.out/'compile_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
