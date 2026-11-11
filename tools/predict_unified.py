"""Conservative drone-focused output adapter for existing native RGB5 checkpoints.

This remaps predictions, not trained weights. Four-class retraining is not performed.
"""
import argparse,json
from pathlib import Path

NATIVE=['fixed_wing_drone','multirotor_drone','bird','airplane','helicopter']
OUTPUT=['drone','bird','airplane','helicopter']
MAPPING={0:0,1:0,2:1,3:2,4:3}

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1e-9)

def adapt(rows,min_score=.5,margin=.15,show_all=False):
    """Keep the strongest overlapping hypothesis; abstain on low/conflicting scores.

    Confidence thresholds are conservative display rules, not calibrated certainty.
    A high model score can still be wrong on unfamiliar scenes.
    """
    result=[]
    for row in sorted(rows,key=lambda r:r['score'],reverse=True):
        if any(iou(row['box'],r['box'])>=.45 for r in result):continue
        mapped=MAPPING[row['native_class_id']]
        alternatives=[r for r in rows if MAPPING[r['native_class_id']]!=mapped and iou(row['box'],r['box'])>=.45]
        conflict=bool(alternatives) and row['score']-max(r['score'] for r in alternatives)<margin
        uncertain=row['score']<min_score or conflict
        # Default deployment focus is drone, not asserting other aircraft types.
        display=OUTPUT[mapped] if not uncertain and (show_all or mapped==0) else 'unknown_airborne_object'
        result.append(dict(box=row['box'],score=row['score'],mapped_class_id=mapped,mapped_class=OUTPUT[mapped],
                           display_label=display,abstained=display=='unknown_airborne_object',
                           reason='low_score' if row['score']<min_score else ('conflicting_predictions' if conflict else ('non_drone_type_unverified' if not show_all and mapped!=0 else 'model_prediction')),
                           native_class_id=row['native_class_id'],native_class=NATIVE[row['native_class_id']]))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--source',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--min-score',type=float,default=.5);parser.add_argument('--margin',type=float,default=.15)
    parser.add_argument('--show-all-labels',action='store_true',help='Show confident mapped background labels as well; these are model predictions, not verified identity')
    args=parser.parse_args()
    if not 0<=args.min_score<=1 or not 0<=args.margin<=1:parser.error('Score and margin must be between 0 and 1')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model=YOLO(str(args.model))
    if list(model.names.values())!=NATIVE:raise ValueError('Expected an existing native RGB5 checkpoint. Do not overwrite its names.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8') as stream:
        for index,r in enumerate(model.predict(args.source,stream=True,device=args.device,imgsz=640,rect=False,conf=.1,iou=.7,verbose=False)):
            raw=[dict(box=box,score=float(score),native_class_id=int(cls)) for box,score,cls in zip(r.boxes.xyxy.tolist(),r.boxes.conf.tolist(),r.boxes.cls.tolist())]
            record=dict(frame_or_image=index+1,source=r.path,predictions=adapt(raw,args.min_score,args.margin,args.show_all_labels),native_predictions=raw,
                        scope='Existing five-output checkpoint with a four-category output adapter. No retraining. Unknown is a display abstention, not a learned fifth category or proof of safety/identity.')
            stream.write(json.dumps(record)+'\n')
    print('Saved remapped predictions: '+str(args.output))
if __name__=='__main__':main()
