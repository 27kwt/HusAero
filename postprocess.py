"""Optional duplicate-box experiment, separate from the frozen benchmark.

Suppresses a large drone box only when a similarly confident, much smaller drone
box is almost completely inside it. Cannot repair every inaccurate large box.
"""
import numpy as np
def suppress_nested_drone_boxes(predictions,containment=.9,area_ratio=3.,confidence_ratio=.95):
    p=np.asarray(predictions).reshape(-1,6)
    if len(p)<2:return p.copy()
    area=np.maximum(0,p[:,2:4]-p[:,:2]).prod(axis=1);keep=np.ones(len(p),dtype=bool)
    for outer in np.where(p[:,5]==0)[0]:
        if area[outer]<=0:continue
        for inner in np.where((p[:,5]==0)&(area*area_ratio<area[outer])&(p[:,4]>=p[outer,4]*confidence_ratio))[0]:
            if area[inner]<=0:continue
            inter=np.maximum(0,np.minimum(p[outer,2:4],p[inner,2:4])-np.maximum(p[outer,:2],p[inner,:2])).prod()
            if inter/area[inner]>=containment:keep[outer]=False;break
    return p[keep].copy()
