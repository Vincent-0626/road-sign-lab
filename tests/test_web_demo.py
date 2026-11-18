import base64
import io
import unittest
from pathlib import Path

from PIL import Image

from src.config import CHECKPOINT_DIR, PROJECT_ROOT
from src.web_demo import Service


class WebDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = Service(CHECKPOINT_DIR / 'best.pt')

    def test_sample_and_perturbation(self):
        with Image.open(PROJECT_ROOT / 'samples/sample_stop.ppm') as image:
            out = io.BytesIO()
            image.save(out, format='PNG')
        data = 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode('ascii')
        result = self.service.analyze({'image': data, 'brightness': 0.7, 'blur': 3, 'occlusion': 0.3})
        self.assertEqual(result['original'][0]['class_id'], 14)
        self.assertEqual(len(result['modified']), 3)
        self.assertTrue(result['modified_image'].startswith('data:image/png;base64,'))

    def test_rejects_invalid_image(self):
        with self.assertRaises(ValueError):
            self.service.analyze({'image': 'not-an-image'})


if __name__ == '__main__':
    unittest.main()
