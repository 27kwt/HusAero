"""Exercise the new class count and coordinate recovery without Hailo hardware."""
import unittest
import numpy as np
from deployment.yolokw_hailo import decode
from deployment.hailo_adapter import prepare

class FiveClassDecoderTests(unittest.TestCase):
    def test_letterbox_coordinates_roundtrip(self):
        _,transform=prepare(np.zeros((480,1280,3),dtype=np.uint8))
        sx,sy,left,top=transform
        x1,y1,x2,y2=100,80,420,300
        raw=[[np.empty((0,5),np.float32) for _ in range(5)]]
        raw[0][4]=np.array([[(y1*sy+top)/640,(x1*sx+left)/640,(y2*sy+top)/640,(x2*sx+left)/640,.9]])
        found=decode(raw,(480,1280,3),transform)
        np.testing.assert_allclose(found[0,:4],[x1,y1,x2,y2],atol=.0001)
        self.assertEqual(int(found[0,5]),4)

    def test_reject_six_class_hef(self):
        with self.assertRaises(ValueError):decode([[[] for _ in range(6)]],(640,640,3),(1,1,0,0))

    def test_nonfinite_and_low_scores_removed(self):
        raw=[[[] for _ in range(5)]];raw[0][0]=[[0,0,.5,.5,float('nan')],[0,0,.5,.5,.0001]]
        self.assertEqual(decode(raw,(640,640,3),(1,1,0,0)).shape,(0,6))

if __name__=='__main__':unittest.main()
