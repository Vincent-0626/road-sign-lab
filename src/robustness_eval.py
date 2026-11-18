"""Evaluate fixed image perturbations on the same held-out test set.

The perturbations are applied to the raw sign crop before ROI cropping and
the checkpoint's normal preprocessing. No parameters are tuned on test data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from .config import CLASS_NAMES, DATA_DIR
from .predict import load_for_inference
from .robustness_demo import transform_image
from .transforms import load_image, preprocess_deterministic, to_tensor
from .utils import configure_threads

CONDITIONS = {
    'clean': {'brightness': 1.0, 'blur': 0, 'occlusion': 0.0},
    'dark': {'brightness': 0.45, 'blur': 0, 'occlusion': 0.0},
    'blur': {'brightness': 1.0, 'blur': 5, 'occlusion': 0.0},
    'occlusion': {'brightness': 1.0, 'blur': 0, 'occlusion': 0.3},
}


class PerturbedDataset(Dataset):
    def __init__(self, frame, raw_images, cfg, norm, condition):
        self.frame, self.raw, self.cfg, self.norm, self.condition = frame, raw_images, cfg, norm, condition

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        raw = transform_image(self.raw[i], **self.condition)
        roi = (int(row.x1), int(row.y1), int(row.x2), int(row.y2))
        image = preprocess_deterministic(raw, roi, self.cfg.img_size, self.cfg.use_clahe)
        return to_tensor(image, self.norm['mean'], self.norm['std']), int(row.class_id)


def evaluate(checkpoint: Path, data_dir: Path, output: Path, batch_size=256):
    configure_threads()
    model, cfg, _, norm, device = load_for_inference(checkpoint)
    frame = pd.read_csv(data_dir / 'manifests' / 'test.csv').reset_index(drop=True)
    if len(frame) != 12630:
        raise ValueError(f'expected 12,630 official test images, got {len(frame)}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        images = list(pool.map(load_image, frame['path'].tolist()))
    report = {'checkpoint': str(checkpoint), 'split': 'official_test', 'n_images': len(frame),
              'conditions': CONDITIONS, 'metrics': {}}
    for name, condition in CONDITIONS.items():
        dataset = PerturbedDataset(frame, images, cfg, norm, condition)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        predictions, targets = [], []
        with torch.inference_mode():
            for image, labels in loader:
                predictions.extend(model(image.to(device)).argmax(dim=1).cpu().tolist())
                targets.extend(labels.tolist())
        errors = Counter((int(t), int(p)) for t, p in zip(targets, predictions) if t != p)
        metrics = {'accuracy': float(accuracy_score(targets, predictions)),
                   'macro_f1': float(f1_score(targets, predictions, labels=list(range(43)), average='macro', zero_division=0)),
                   'most_confused': [{'true_id': a, 'true': CLASS_NAMES[a], 'pred_id': b,
                                      'pred': CLASS_NAMES[b], 'count': n}
                                     for (a, b), n in errors.most_common(10)]}
        report['metrics'][name] = metrics
        print(f'{name}: accuracy={metrics["accuracy"]:.4f} macro_f1={metrics["macro_f1"]:.4f}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'结果保存到 {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data-dir', default=DATA_DIR, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.output, args.batch_size)


if __name__ == '__main__':
    main()
