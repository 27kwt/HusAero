"""Export a copied checkpoint into an exclusive folder; never overwrite active weights."""
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    from ultralytics import YOLO
    import onnx
    a.out.mkdir(parents=True,exist_ok=False);copy=a.out/'model.pt';shutil.copy2(a.weights,copy)
    model=YOLO(str(copy));artifact=Path(model.export(format='onnx',opset=11,imgsz=640,batch=1,dynamic=False,simplify=False,nms=False,device='cpu'))
    graph=onnx.load(str(artifact));onnx.checker.check_model(graph)
    result=dict(source=str(a.weights.resolve()),checkpoint_sha256=hashlib.sha256(copy.read_bytes()).hexdigest(),
      onnx=str(artifact.resolve()),onnx_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),names=model.names,
      input_shape=[1,3,640,640],opset=11,onnx_checker='PASS',nms=False)
    (a.out/'export.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
