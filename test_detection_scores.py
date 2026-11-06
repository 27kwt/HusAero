import unittest
import numpy as np
from deployment.detection_scores import Scores

class ScoreTests(unittest.TestCase):
    def test_perfect_detection_and_duplicate(self):
        s=Scores(['drone']);s.add(np.array([[0,10,10,20,20]],float),np.array([[10,10,20,20,.9,0],[10,10,20,20,.8,0]],float));r=s.result()['per_class'][0]
        self.assertEqual(r['TP_at_025'],1);self.assertEqual(r['FP_at_025'],1);self.assertEqual(r['recall_at_025'],1)
        self.assertGreater(r['AP50'],.99)
    def test_wrong_class_is_not_a_true_positive(self):
        s=Scores(['drone','bird']);s.add(np.array([[0,10,10,20,20]],float),np.array([[10,10,20,20,.9,1]],float));r=s.result()
        self.assertEqual(r['per_class'][0]['FN_at_025'],1);self.assertEqual(r['per_class'][1]['FP_at_025'],1);self.assertEqual(r['mAP50'],0)
    def test_low_confidence_box_cannot_steal_fixed_threshold_match(self):
        s=Scores(['drone'])
        s.add(np.array([[0,0,0,10,10]],float),np.array([[0,0,9,10,.9,0],[0,0,10,10,.1,0]],float))
        result=s.result();r=result['per_class'][0]
        self.assertEqual((r['TP_at_025'],r['FP_at_025'],r['FN_at_025']),(1,0,0))
        self.assertEqual((r['precision_at_025'],r['recall_at_025']),(1,1))
        # Preserve historical AP matching; only fixed-threshold P/R is corrected.
        self.assertAlmostEqual(r['AP50'],.4975)
        self.assertEqual(result['ap_matching_version'],'iou-greedy-v1')
        self.assertEqual(result['fixed_confidence_matching_version'],'threshold-first-v2')
if __name__=='__main__':unittest.main()
