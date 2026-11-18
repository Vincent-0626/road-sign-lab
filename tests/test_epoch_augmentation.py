import unittest

import pandas as pd
import torch

from src.config import PROJECT_ROOT
from src.dataset import GTSRBDataset


class EpochAugmentationTests(unittest.TestCase):
    def test_epoch_changes_augmentation_but_is_reproducible(self):
        sample = PROJECT_ROOT / 'samples' / 'sample_stop.ppm'
        frame = pd.DataFrame([{'path': str(sample), 'class_id': 14}])
        first = GTSRBDataset(frame, train=True, seed=42, quiet=True)
        second = GTSRBDataset(frame, train=True, seed=42, quiet=True)
        first.set_epoch(0)
        a, _ = first[0]
        first.set_epoch(1)
        b, _ = first[0]
        second.set_epoch(1)
        c, _ = second[0]
        self.assertFalse(torch.equal(a, b))
        self.assertTrue(torch.equal(b, c))
        self.assertTrue(first._epoch.is_shared())

    def test_negative_epoch_rejected(self):
        sample = PROJECT_ROOT / 'samples' / 'sample_stop.ppm'
        dataset = GTSRBDataset(pd.DataFrame([{'path': str(sample), 'class_id': 14}]), quiet=True)
        with self.assertRaises(ValueError):
            dataset.set_epoch(-1)


if __name__ == '__main__':
    unittest.main()
