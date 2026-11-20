"""Random shadow augmentation: darken a soft-edged random region of the image, labels unchanged."""
import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


def add_random_shadow(img, rng, k_min=0.35, k_max=0.7):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    kind = rng.integers(3)
    if kind == 0:      # elongated band (shadow of a plant row / pole)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, np.pi)
        L, T = rng.uniform(0.5, 1.5) * max(h, w), rng.uniform(0.04, 0.2) * min(h, w)
        d = np.array([np.cos(ang), np.sin(ang)]); n = np.array([-d[1], d[0]])
        pts = np.array([[cx, cy] + s1 * d * L / 2 + s2 * n * T / 2
                        for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    elif kind == 1:    # random quadrilateral
        pts = np.stack([rng.uniform(-0.2, 1.2, 4) * w, rng.uniform(-0.2, 1.2, 4) * h], 1)
    else:              # cloud shadow covering one side of the image
        y0 = rng.uniform(0.2, 0.8) * h
        slope = rng.uniform(-0.5, 0.5)
        pts = np.array([[0, y0 - slope * w / 2], [w, y0 + slope * w / 2], [w, h], [0, h]])
        if rng.random() < 0.5:
            pts[:, 1] = h - pts[:, 1]
    cv2.fillPoly(m, [pts.astype(np.int32)], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), max(3, int(0.02 * min(h, w))))
    gain = 1.0 - (1.0 - rng.uniform(k_min, k_max)) * m
    out = np.clip(img.astype(np.float32) * gain[..., None], 0, 255)
    return out.astype(img.dtype)


@TRANSFORMS.register_module()
class RandomShadow(BaseTransform):
    def __init__(self, prob=0.5, k_min=0.35, k_max=0.7):
        self.prob, self.k_min, self.k_max = prob, k_min, k_max

    def transform(self, results):
        if np.random.rand() < self.prob:
            results['img'] = add_random_shadow(results['img'], np.random.default_rng(), self.k_min, self.k_max)
        return results
