import unittest
import cv2,numpy as np
from flow_assisted_tracker import FlowAssistedTracker
class FlowTrackerTests(unittest.TestCase):
    def image(self,shift=0):
        a=np.zeros((120,160,3),np.uint8);cv2.rectangle(a,(50+shift,40),(75+shift,65),(255,255,255),-1);cv2.line(a,(50+shift,40),(75+shift,65),(50,50,50),2);return a
    def test_never_invents_initial_target(self):
        t=FlowAssistedTracker(20);self.assertEqual(len(t.update(self.image(),np.empty((0,6)))),0)
    def test_confirm_then_short_flow_and_timeout(self):
        t=FlowAssistedTracker(20,max_gap_seconds=.15);p=np.array([[49,39,77,67,.9,0]],np.float32)
        self.assertEqual(len(t.update(self.image(),p)),0);self.assertEqual(len(t.update(self.image(),p)),1)
        self.assertEqual(len(t.update(self.image(1),np.empty((0,6)))),1)
        for _ in range(4):last=t.update(self.image(1),np.empty((0,6)))
        self.assertEqual(len(last),0)
    def test_large_outlier_does_not_resize_confirmed_target(self):
        t=FlowAssistedTracker(20);p=np.array([[49,39,77,67,.9,0]],np.float32);t.update(self.image(),p);t.update(self.image(),p)
        r=t.update(self.image(),[[0,0,159,119,.99,0]]);self.assertEqual(len(r),1);self.assertLess(r[0,2]-r[0,0],40)
if __name__=='__main__':unittest.main()
