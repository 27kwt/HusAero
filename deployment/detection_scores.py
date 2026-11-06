"""Shared NumPy detector scoring for PC/Hailo comparison at identical settings.

IoU matching and 101-point trapezoid AP are explicit; this is not the official
COCO evaluator. Preserve this implementation/version alongside reported scores.
"""
import numpy as np

THRESHOLDS=np.linspace(.5,.95,10)

def match_boxes(gt,pred):
    correct=np.zeros((len(pred),10),bool)
    if not len(gt) or not len(pred):return correct
    wh=np.maximum(0,np.minimum(gt[:,None,2:],pred[None,:,2:4])-np.maximum(gt[:,None,:2],pred[None,:,:2]))
    inter=wh.prod(axis=2);ag=np.maximum(0,gt[:,2:]-gt[:,:2]).prod(axis=1);ap=np.maximum(0,pred[:,2:4]-pred[:,:2]).prod(axis=1)
    iou=inter/(ag[:,None]+ap-inter+1e-7)
    for k,threshold in enumerate(THRESHOLDS):
        gi,pi=np.where(iou>=threshold)
        if not len(gi):continue
        m=np.column_stack((gi,pi,iou[gi,pi]));m=m[np.argsort(-m[:,2])]
        m=m[np.unique(m[:,1],return_index=True)[1]];m=m[np.argsort(-m[:,2])];m=m[np.unique(m[:,0],return_index=True)[1]]
        correct[m[:,1].astype(int),k]=True
    return correct

class Scores:
    def __init__(self,names):
        self.names=names;self.parts={c:[] for c in range(len(names))};self.gt={c:0 for c in self.parts};self.images=0
        self.fixed={c:dict(tp=0,fp=0) for c in self.parts}
    def add(self,labels,predictions):
        # labels: class,x1,y1,x2,y2; predictions: x1,y1,x2,y2,score,class
        self.images+=1
        for c in self.parts:
            gt=labels[labels[:,0]==c,1:5];p=predictions[predictions[:,5]==c,:5];self.gt[c]+=len(gt)
            self.parts[c].append((match_boxes(gt,p),p[:,4]))
            # Match again after filtering: a discarded low-confidence box must
            # not consume the GT match used to measure fixed-threshold P/R.
            selected=p[p[:,4]>=.25]
            true=int(match_boxes(gt,selected)[:,0].sum())
            self.fixed[c]['tp']+=true;self.fixed[c]['fp']+=len(selected)-true
    def result(self):
        result=[]
        for c in self.parts:
            n=self.gt[c];correct=np.concatenate([p[0] for p in self.parts[c]]) if self.parts[c] else np.zeros((0,10),bool)
            scores=np.concatenate([p[1] for p in self.parts[c]]) if self.parts[c] else np.empty(0)
            order=np.argsort(-scores,kind='stable');scores=scores[order];correct=correct[order]
            aps=[]
            if len(scores) and n:
                tp=np.cumsum(correct,axis=0);fp=np.cumsum(~correct,axis=0);recall=tp/n;precision=tp/np.maximum(tp+fp,1)
                for j in range(10):
                    r=np.r_[0,recall[:,j],1];p=np.r_[1,precision[:,j],0];p=np.maximum.accumulate(p[::-1])[::-1]
                    aps.append(float(np.trapezoid(np.interp(np.linspace(0,1,101),r,p),np.linspace(0,1,101))))
            else:aps=[0.]*10
            true=self.fixed[c]['tp'];false=self.fixed[c]['fp']
            result.append(dict(class_id=c,name=self.names[c],ground_truth=n,predictions=len(scores),
                  precision_at_025=true/max(true+false,1),recall_at_025=true/max(n,1),TP_at_025=true,FP_at_025=false,FN_at_025=n-true,
                  AP50=aps[0] if n else None,AP50_95=float(np.mean(aps)) if n else None))
        valid=[r for r in result if r['ground_truth']]
        return dict(images=self.images,per_class=result,mAP50=float(np.mean([r['AP50'] for r in valid])) if valid else None,
             mAP50_95=float(np.mean([r['AP50_95'] for r in valid])) if valid else None,
             implementation='class-aware IoU matching + 101-point trapezoid AP; threshold-first P/R at 0.25',
             ap_matching_version='iou-greedy-v1',fixed_confidence_matching_version='threshold-first-v2',
             matching_iou_thresholds=THRESHOLDS.tolist())

def read_labels(path,width,height):
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip():continue
        c,x,y,w,h=map(float,line.split());rows.append([c,(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])
    return np.asarray(rows,np.float32).reshape(-1,5)
