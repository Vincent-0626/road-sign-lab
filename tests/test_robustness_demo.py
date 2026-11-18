"""Tests of added perturbations and integration with the upstream checkpoint."""
import unittest
import numpy as np

from src.config import CHECKPOINT_DIR, PROJECT_ROOT
from src.predict import load_for_inference
from src.robustness_demo import ZH_CLASSES, predict_array, transform_image
from src.transforms import load_image


class RobustnessDemoTests(unittest.TestCase):
    def test_perturbation_is_deterministic_and_keeps_input(self):
        source = np.full((30, 30, 3), 200, dtype=np.uint8)
        output = transform_image(source, brightness=0.5, blur=3, occlusion=0.3)
        self.assertTrue(np.all(source == 200))
        self.assertEqual(output.shape, source.shape)
        np.testing.assert_array_equal(output, transform_image(source, 0.5, 3, 0.3))
        self.assertTrue(np.any(output != source))

    def test_invalid_parameters_are_rejected(self):
        source = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            transform_image(source, blur=4)
        with self.assertRaises(ValueError):
            transform_image(source, occlusion=0.9)

    def test_sample_predictions_with_upstream_weights(self):
        model, cfg, _, norm, device = load_for_inference(CHECKPOINT_DIR / 'best.pt', device='cpu')
        for name, expected in [('sample_stop.ppm', 14), ('sample_no_entry.ppm', 17), ('sample_speed_30.ppm', 1)]:
            with self.subTest(name=name):
                image = load_image(PROJECT_ROOT / 'samples' / name)
                result = predict_array(model, cfg, norm, device, image)
                self.assertEqual(len(result), 3)
                self.assertEqual(result[0][0], expected)
                self.assertGreaterEqual(result[0][1], result[-1][1])
                self.assertLess(result[0][0], len(ZH_CLASSES))


if __name__ == '__main__':
    unittest.main()
