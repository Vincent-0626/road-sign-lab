"""Local-only web UI for classifying an already cropped traffic sign."""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import CHECKPOINT_DIR, PROJECT_ROOT
from .predict import load_for_inference
from .robustness_demo import ZH_CLASSES, predict_array, transform_image
from .utils import configure_threads

PAGE = Path(__file__).with_name('web_demo.html')
MAX_BODY = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


class Service:
    def __init__(self, checkpoint):
        configure_threads()
        self.model, self.cfg, _, self.norm, self.device = load_for_inference(checkpoint)
        self.lock = threading.Lock()

    def analyze(self, payload):
        data_url = payload.get('image', '')
        if not isinstance(data_url, str) or not data_url.startswith('data:image/') or ',' not in data_url:
            raise ValueError('请选择图片文件')
        header, data = data_url.split(',', 1)
        if ';base64' not in header:
            raise ValueError('图片格式错误')
        try:
            raw = base64.b64decode(data, validate=True)
            with Image.open(io.BytesIO(raw)) as loaded:
                image = ImageOps.exif_transpose(loaded).convert('RGB')
                image.thumbnail((1600, 1600))
                pixels = np.asarray(image).copy()
        except (binascii.Error, OSError, ValueError, UnidentifiedImageError) as error:
            raise ValueError('图片无法读取') from error
        brightness = float(payload.get('brightness', 1.0))
        blur = int(payload.get('blur', 0))
        occlusion = float(payload.get('occlusion', 0.0))
        modified = transform_image(pixels, brightness, blur, occlusion)
        start = time.perf_counter()
        with self.lock:
            original = predict_array(self.model, self.cfg, self.norm, self.device, pixels)
            altered = predict_array(self.model, self.cfg, self.norm, self.device, modified)
        elapsed_ms = (time.perf_counter() - start) * 1000
        output = io.BytesIO()
        Image.fromarray(modified).save(output, format='PNG')
        rows = lambda results: [{'class_id': i, 'name': ZH_CLASSES[i], 'score': round(score, 6)} for i, score in results]
        return {'original': rows(original), 'modified': rows(altered),
                'modified_image': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii'),
                'elapsed_ms': round(elapsed_ms, 1)}


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def _response(self, status, data, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                self._response(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/api/sample':
                output = io.BytesIO()
                with Image.open(PROJECT_ROOT / 'samples/sample_stop.ppm') as image:
                    image.convert('RGB').save(output, format='PNG')
                self._response(200, output.getvalue(), 'image/png')
            else:
                self._response(404, b'not found', 'text/plain')

        def do_POST(self):
            if self.path != '/api/analyze':
                self._response(404, b'not found', 'text/plain')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 1 or length > MAX_BODY:
                    raise ValueError('图片太大，请使用小于 7 MB 的图片')
                data = json.loads(self.rfile.read(length))
                result = service.analyze(data)
                self._response(200, json.dumps(result, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')
            except (ValueError, TypeError, KeyError, Image.DecompressionBombError) as error:
                self._response(400, json.dumps({'error': str(error)}, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

    return Handler


def main():
    parser = argparse.ArgumentParser(description='GTSRB traffic-sign crop demo (localhost only)')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_DIR / 'best.pt')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    service = Service(args.checkpoint)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(service))
    print(f'已启动：http://127.0.0.1:{server.server_port} · 设备 {service.device}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
