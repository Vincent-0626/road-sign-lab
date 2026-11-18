"""Reconstruct track-aware GTSRB manifests from a public parquet mirror.

The mirror preserves original images, labels, train/test origin, physical-sign
track IDs and ROI coordinates. Every mirror file is checked against its hash.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = 'Yuvrajg2107/pranavx-gtsrb-reliability'
ENDPOINT = 'https://hf-mirror.com'


def download(root: Path):
    url = f'{ENDPOINT}/api/datasets/{REPO}/tree/main?recursive=true'
    with urllib.request.urlopen(url, timeout=45) as response:
        files = json.load(response)
    files = [f for f in files if f.get('path', '').endswith('.parquet') and f.get('lfs')]
    if not files:
        raise RuntimeError('镜像中没有找到 parquet 数据文件')
    root.mkdir(parents=True, exist_ok=True)

    def get(item):
        path = item['path']
        if not re.fullmatch(r'data/[a-z_]+/[a-z_]+-\d+\.parquet', path):
            raise ValueError(f'异常数据路径: {path}')
        target = root / Path(path).name
        digest = item['lfs']['oid']
        if target.exists() and target.stat().st_size == item['size']:
            if hashlib.sha256(target.read_bytes()).hexdigest() == digest:
                return f'已校验 {target.name}'
        temp = target.with_suffix('.part')
        for attempt in range(3):
            try:
                request = urllib.request.Request(f'{ENDPOINT}/datasets/{REPO}/resolve/main/{path}',
                                                 headers={'User-Agent': 'cv-traffic-demo/1.0'})
                sha = hashlib.sha256()
                with urllib.request.urlopen(request, timeout=120) as response, open(temp, 'wb') as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                        sha.update(chunk)
                if temp.stat().st_size != item['size'] or sha.hexdigest() != digest:
                    raise ValueError(f'校验失败: {path}')
                os.replace(temp, target)
                return f'下载并校验 {target.name} ({item["size"] / 1e6:.1f} MB)'
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    print(f'镜像数据文件：{len(files)}；并发下载 8 个', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(get, item) for item in files]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)
    (root / 'mirror_manifest.json').write_text(json.dumps(files, indent=2), encoding='utf-8')


def prepare(data_dir: Path, mirror_dir: Path):
    import io
    import pandas as pd
    import pyarrow.parquet as pq
    from PIL import Image
    from .config import CLASS_NAMES, NUM_CLASSES
    from .prepare import compute_norm_stats, track_aware_split
    from .utils import save_json

    image_root = data_dir / 'mirror_images'
    image_root.mkdir(parents=True, exist_ok=True)
    rows = {'train': [], 'test': []}
    paths_seen = set()
    for parquet in sorted(mirror_dir.glob('*.parquet')):
        source = pq.ParquetFile(parquet)
        for batch in source.iter_batches(batch_size=512):
            for item in batch.to_pylist():
                original = str(item['original_path']).replace('\\', '/')
                name = Path(original).name
                label = int(item['class_id'])
                if not 0 <= label < NUM_CLASSES or not re.fullmatch(r'\d+(?:_\d+)?\.ppm', name):
                    raise ValueError(f'非法图片或类别: {original}')
                train = item['source_split'] == 'official_train'
                if not train and item['source_split'] != 'official_test':
                    raise ValueError('未知来源划分')
                dest = image_root / ('train' if train else 'test') / f'{label:05d}' / name
                if str(dest) in paths_seen:
                    raise ValueError(f'重复原图路径: {original}')
                paths_seen.add(str(dest))
                image = item['image']
                blob = image.get('bytes') if isinstance(image, dict) else None
                if not blob:
                    raise ValueError(f'图片字节缺失: {original}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
                with Image.open(io.BytesIO(blob)) as im:
                    width, height = im.size
                roi = [int(item[key]) for key in ('roi_x1', 'roi_y1', 'roi_x2', 'roi_y2')]
                if not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
                    raise ValueError(f'标注框超出图片边界: {original}')
                row = {'path': str(dest.resolve()), 'filename': name, 'class_id': label,
                       'width': width, 'height': height,
                       'x1': roi[0], 'y1': roi[1], 'x2': roi[2], 'y2': roi[3]}
                if train:
                    match = re.fullmatch(r'(\d+)_(\d+)\.ppm', name)
                    if not match:
                        raise ValueError(f'训练图片缺少 track 编号: {name}')
                    track, frame = map(int, match.groups())
                    row.update(track_id=f'{label}_{track}', track_num=track, frame_num=frame)
                rows['train' if train else 'test'].append(row)
        print(f'已转换 {parquet.name}：训练侧 {len(rows["train"])}，测试侧 {len(rows["test"])}', flush=True)
    if len(rows['train']) != 39209 or len(rows['test']) != 12630:
        raise ValueError(f'数据数量异常 train={len(rows["train"])}, test={len(rows["test"])}')
    train_all, test = pd.DataFrame(rows['train']), pd.DataFrame(rows['test'])
    train, val, summary = track_aware_split(train_all, val_frac=0.2, seed=42)
    manifest = data_dir / 'manifests'
    manifest.mkdir(parents=True, exist_ok=True)
    train.to_csv(manifest / 'train.csv', index=False)
    val.to_csv(manifest / 'val.csv', index=False)
    test.to_csv(manifest / 'test.csv', index=False)
    save_json(CLASS_NAMES, manifest / 'class_names.json')
    save_json(compute_norm_stats(train, img_size=32, use_clahe=True), manifest / 'norm_stats.json')
    save_json({'source': f'{ENDPOINT}/datasets/{REPO}', 'roi': 'original GTSRB annotation',
               'train': len(train), 'val': len(val), 'test': len(test), 'track_overlap': 0,
               'per_class': summary}, manifest / 'mirror_protocol.json')
    print(f'划分完成：训练 {len(train)}、验证 {len(val)}、测试 {len(test)}；训练/验证 track 重合 0', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['download', 'prepare'])
    parser.add_argument('--data-dir', type=Path, default=Path('/root/cv-traffic-sign/data'))
    args = parser.parse_args()
    mirror = args.data_dir / 'mirror_parquet'
    if args.phase == 'download':
        download(mirror)
    else:
        prepare(args.data_dir, mirror)


if __name__ == '__main__':
    main()
