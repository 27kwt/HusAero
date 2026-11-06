"""Regression checks for portable calibration and low-score track association."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from create_calibration import label_path,recorded_path,resolve_recorded_image
from demo import tracking_config

class CalibrationPathTests(unittest.TestCase):
    def test_nested_images_parent_maps_the_dataset_labels(self):
        image=Path('images')/'project'/'dataset'/'images'/'train'/'a.jpg'
        self.assertEqual(label_path(image),Path('images')/'project'/'dataset'/'labels'/'train'/'a.txt')

    def test_selection_replays_from_relocated_dataset(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';new=root/'new'
            path=Path('images/train/a.jpg')
            for base in [old,new]:
                (base/path).parent.mkdir(parents=True);(base/path).write_bytes(b'unchanged image fixture')
            row={'path':recorded_path(old/path,old),'sha256':hashlib.sha256((old/path).read_bytes()).hexdigest()}
            self.assertEqual(resolve_recorded_image(row,new,{(new/path).resolve()}),(new/path).resolve())
            (new/path).write_bytes(b'changed')
            with self.assertRaises(ValueError):resolve_recorded_image(row,new,{(new/path).resolve()})

class ByteTrackInputTests(unittest.TestCase):
    def test_low_confidence_detection_keeps_confirmed_identity(self):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.engine.results import Boxes
        config=tracking_config(.25,Path(__file__).parent/'configs/bytetrack.yaml')
        tracker=BYTETracker(SimpleNamespace(**config))
        first=tracker.update(Boxes(np.array([[10,10,30,30,.9,0]],np.float32),(64,64)))
        self.assertEqual(len(first),1)
        low=np.array([[10,10,30,30,.15,0]],np.float32)
        low=low[low[:,4]>=config['track_low_thresh']]
        second=tracker.update(Boxes(low,(64,64)))
        self.assertEqual(len(second),1)
        self.assertEqual(int(first[0,4]),int(second[0,4]))

if __name__=='__main__':unittest.main()
