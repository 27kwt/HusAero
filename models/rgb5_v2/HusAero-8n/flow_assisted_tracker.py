"""Experimental single-drone tracker assisted by short-lived Lucas-Kanade flow.

No annotations, target coordinates or motor commands enter this runtime. States
distinguish fresh detections from optical-flow or brief motion propagation.
"""
import cv2,numpy as np

class FlowAssistedTracker:
    def __init__(self,fps,activation=.1,keep_threshold=.05,max_gap_seconds=.5):
        self.activation=activation;self.keep_threshold=keep_threshold;self.max_gap=max(1,round(fps*max_gap_seconds));self.identity=0;self.box=None;self.gray=None;self.points=None;self.pending=None;self.pending_hits=0;self.missed=0;self.velocity=np.zeros(2);self.state='SEARCHING';self.no_flow=0
    @staticmethod
    def _iou(a,b):
        wh=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2]));inter=wh.prod(axis=1);return inter/(np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[:,2:]-b[:,:2]).prod(axis=1)-inter+1e-9)
    def _features(self,gray,box):
        h,w=gray.shape;x1,y1,x2,y2=np.round(box).astype(int);x1,x2=np.clip([x1,x2],0,w);y1,y2=np.clip([y1,y2],0,h)
        if x2-x1<4 or y2-y1<4:return None
        p=cv2.goodFeaturesToTrack(gray[y1:y2,x1:x2],maxCorners=40,qualityLevel=.01,minDistance=2,blockSize=3)
        if p is not None:p+=np.array([x1,y1],np.float32).reshape(1,1,2)
        return p
    def _flow(self,gray):
        if self.points is None or len(self.points)<3:return None
        h,w=gray.shape;cx,cy=(self.box[:2]+self.box[2:])/2;bw,bh=self.box[2:]-self.box[:2];rx,ry=max(48,2*bw),max(48,2*bh);x1,x2=np.clip(np.round([cx-rx,cx+rx]).astype(int),0,w);y1,y2=np.clip(np.round([cy-ry,cy+ry]).astype(int),0,h)
        if x2-x1<24 or y2-y1<24:return None
        origin=np.array([x1,y1],np.float32).reshape(1,1,2);p0=self.points-origin;old=self.gray[y1:y2,x1:x2];new=gray[y1:y2,x1:x2];kwargs=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.01))
        p1,s1,_=cv2.calcOpticalFlowPyrLK(old,new,p0,None,**kwargs)
        if p1 is None:return None
        back,s2,_=cv2.calcOpticalFlowPyrLK(new,old,p1,None,**kwargs)
        if back is None:return None
        good=(s1.ravel()>0)&(s2.ravel()>0)&(np.linalg.norm((back-p0).reshape(-1,2),axis=1)<1.5)
        if good.sum()<3:return None
        changes=(p1-p0).reshape(-1,2)[good];motion=np.median(changes,axis=0);disp=np.linalg.norm(changes-motion,axis=1);changes=changes[disp<=max(3,.1*np.hypot(bw,bh))]
        if len(changes)<3:return None
        motion=np.median(changes,axis=0)
        if np.linalg.norm(motion)>max(30,2.5*np.hypot(bw,bh)):return None
        return motion
    def update(self,frame,detections):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);p=np.asarray(detections,np.float32).reshape(-1,6);p=p[np.isfinite(p).all(axis=1)&(p[:,5]==0)]
        if self.box is None:
            p=p[p[:,4]>=self.activation]
            if not len(p):self.pending=None;self.pending_hits=0;self.gray=gray;self.state='SEARCHING';return np.empty((0,5),np.float32)
            candidate=p[np.argmax(p[:,4]),:4]
            self.pending_hits=self.pending_hits+1 if self.pending is not None and self._iou(self.pending,candidate[None])[0]>=.2 else 1;self.pending=candidate.copy()
            if self.pending_hits<2:self.gray=gray;self.state='CONFIRMING';return np.empty((0,5),np.float32)
            self.identity+=1;self.box=candidate.copy();self.velocity=np.zeros(2);self.missed=0;self.no_flow=0;self.state='DETECTION_CONFIRMED'
        else:
            motion=self._flow(gray);self.no_flow=0 if motion is not None else self.no_flow+1;delta=motion if motion is not None else self.velocity;pred=self.box+np.tile(delta,2);old_center=(self.box[:2]+self.box[2:])/2
            candidates=p[p[:,4]>=self.keep_threshold];matched=None;resize_match=False
            if len(candidates):
                boxes=candidates[:,:4];area=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(axis=1);old_area=max(1,np.maximum(0,pred[2:]-pred[:2]).prod());ratio=area/old_area;distance=np.linalg.norm((boxes[:,:2]+boxes[:,2:])/2-(pred[:2]+pred[2:])/2,axis=1)/max(16,np.linalg.norm(pred[2:]-pred[:2]));overlap=self._iou(pred,boxes)
                intersection=np.maximum(0,np.minimum(boxes[:,2:],pred[2:])-np.maximum(boxes[:,:2],pred[:2])).prod(axis=1)
                contained=(intersection/np.maximum(np.minimum(area,old_area),1)>=.9)&(distance<=.4)&(ratio>=.15)&(ratio<=6)
                ok=((ratio>=.4)&(ratio<=2.5)&(distance<=1.5)&(overlap>=.1))|contained
                if ok.any():indices=np.flatnonzero(ok);j=indices[np.argmax(candidates[indices,4]+.5*overlap[indices]-.05*np.abs(np.log(ratio[indices])))];matched=boxes[j];resize_match=bool(contained[j] and (ratio[j]<.4 or ratio[j]>2.5))
            if matched is not None:self.box=matched.copy() if resize_match else .65*matched+.35*pred;self.missed=0;self.state='DETECTION_CONFIRMED'
            else:
                self.missed+=1
                if self.missed>self.max_gap or self.no_flow>2:
                    self.box=None;self.points=None;self.pending=None;self.pending_hits=0;self.gray=gray;self.state='LOST';return np.empty((0,5),np.float32)
                self.box=pred;self.state='FLOW_ASSISTED' if motion is not None else 'MOTION_PREDICTED'
            self.velocity=.5*self.velocity+.5*((self.box[:2]+self.box[2:])/2-old_center)
        h,w=gray.shape;self.box[[0,2]]=np.clip(self.box[[0,2]],0,w-1);self.box[[1,3]]=np.clip(self.box[[1,3]],0,h-1);self.points=self._features(gray,self.box);self.gray=gray
        return np.array([[*self.box,self.identity]],np.float32)
