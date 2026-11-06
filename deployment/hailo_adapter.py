"""BGR-to-RGB letterbox preprocessing shared by the five-class Hailo runtime."""
import cv2
import numpy as np

def prepare(frame):
    if frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8:raise ValueError('Expected BGR uint8 image')
    h,w=frame.shape[:2];gain=640/max(h,w);nw,nh=max(1,round(w*gain)),max(1,round(h*gain));left,top=(640-nw)//2,(640-nh)//2
    image=np.full((640,640,3),114,np.uint8);image[top:top+nh,left:left+nw]=cv2.resize(frame,(nw,nh))
    return np.ascontiguousarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)[None]),(nw/w,nh/h,left,top)

