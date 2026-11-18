"""Interactive experiments for traffic-sign crops; uses upstream model and preprocessing."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .transforms import preprocess_deterministic, to_tensor

ZH_CLASSES = [
    '限速 20', '限速 30', '限速 50', '限速 60', '限速 70', '限速 80',
    '解除限速 80', '限速 100', '限速 120', '禁止超车', '禁止货车超车',
    '前方交叉路口优先通行', '优先道路', '减速让行', '停车让行', '禁止车辆通行',
    '禁止 3.5 吨以上车辆通行', '禁止驶入', '其他危险', '左侧急弯', '右侧急弯',
    '连续弯路', '路面不平', '路面湿滑', '右侧车道变窄', '道路施工', '交通信号灯',
    '注意行人', '注意儿童', '注意自行车', '注意冰雪', '注意野生动物',
    '解除限速和禁止超车', '前方右转', '前方左转', '只准直行', '直行或右转',
    '直行或左转', '靠右行驶', '靠左行驶', '环形交叉路口', '解除禁止超车',
    '解除货车禁止超车',
]
assert len(ZH_CLASSES) == 43


def transform_image(image: np.ndarray, brightness: float = 1.0, blur: int = 0,
                    occlusion: float = 0.0) -> np.ndarray:
    """Apply reproducible perturbations to RGB pixels, before model preprocessing."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('需要 RGB 图片')
    if not 0.2 <= brightness <= 1.8 or not 0 <= occlusion <= 0.5 or blur not in (0, 3, 5, 7):
        raise ValueError('干扰参数超出范围')
    out = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    if blur:
        out = cv2.GaussianBlur(out, (blur, blur), 0)
    if occlusion:
        height, width = out.shape[:2]
        size = max(1, round(width * occlusion))
        out[height // 3: min(height, height // 3 + size), :size] = 80
    return out


def predict_array(model, cfg, norm, device, image: np.ndarray, topk: int = 3):
    processed = preprocess_deterministic(image, None, cfg.img_size, cfg.use_clahe)
    tensor = to_tensor(processed, norm['mean'], norm['std']).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(probs, min(topk, probs.numel()))
    return [(int(i), float(p)) for i, p in zip(indices.cpu(), values.cpu())]
