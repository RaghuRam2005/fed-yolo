"""
Image + box augmentations for YOLOv8 training: letterbox resize, HSV color
jitter, horizontal flip, a combined rotate/translate/scale/shear affine, and
4-image mosaic. No OpenCV dependency -- built on PIL + numpy.

Labels are passed around here as pixel-space `(cls, x1, y1, x2, y2)` arrays
(`Nx5` float32); the dataset layer converts to/from normalized xywh at its
boundary. Hyperparameter defaults mirror the values previously hardcoded in
the deleted (ultralytics-based) `obj_yolo/utils.py`.

**Documented simplification vs. Ultralytics:** mosaic here always tiles the
4 images into fixed quadrants of a `2*imgsz` canvas (no random mosaic
center), then relies on the subsequent random-affine step for the actual
scale/position augmentation. Ultralytics instead randomizes the mosaic
center directly. Both approaches combine 4 images' objects/context into one
training sample; this one is simpler to reason about and verify. `mixup` is
not implemented (noted as follow-up in the project plan).
"""
import math
import random

import numpy as np
from PIL import Image

DEFAULT_HYP: dict[str, float] = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
}


def letterbox(
    img: Image.Image, new_shape: int = 640, color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[Image.Image, float, tuple[float, float]]:
    """Aspect-ratio-preserving resize + center pad to a `new_shape` x `new_shape` square."""
    w, h = img.size
    r = min(new_shape / h, new_shape / w)
    new_unpad = (max(1, int(round(w * r))), max(1, int(round(h * r))))
    dw, dh = (new_shape - new_unpad[0]) / 2, (new_shape - new_unpad[1]) / 2

    resized = img.resize(new_unpad, Image.BILINEAR)
    canvas = Image.new("RGB", (new_shape, new_shape), color)
    canvas.paste(resized, (int(round(dw - 0.1)), int(round(dh - 0.1))))
    return canvas, r, (dw, dh)


# ---------------------------------------------------------------------------
# Vectorized RGB<->HSV (numpy, float [0, 1] channels) -- avoids an OpenCV
# dependency. Standard formulas (equivalent to colorsys / matplotlib.colors).
# ---------------------------------------------------------------------------
def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(-1)
    minc = rgb.min(-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1), 0.0)

    delta_safe = np.where(delta == 0, 1, delta)
    rc = (maxc - r) / delta_safe
    gc = (maxc - g) / delta_safe
    bc = (maxc - b) / delta_safe

    h = np.zeros_like(maxc)
    h = np.where(maxc == r, bc - gc, h)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(delta == 0, 0.0, h)
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.astype(int) % 6

    conds = [i == k for k in range(6)]
    r = np.select(conds, [v, q, p, p, t, v])
    g = np.select(conds, [t, v, v, q, p, p])
    b = np.select(conds, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def augment_hsv(img: np.ndarray, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4) -> np.ndarray:
    """Random HSV jitter (uint8 HxWx3 in, uint8 HxWx3 out)."""
    if not (hgain or sgain or vgain):
        return img
    gains = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hsv = _rgb_to_hsv(img.astype(np.float32) / 255.0)
    hsv[..., 0] = (hsv[..., 0] * gains[0]) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 1)
    rgb = _hsv_to_rgb(hsv)
    return (rgb * 255.0).clip(0, 255).astype(np.uint8)


def fliplr(img: np.ndarray, labels_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal flip; `labels_xyxy` is `(cls, x1, y1, x2, y2)` in pixel coords."""
    w = img.shape[1]
    img = np.ascontiguousarray(img[:, ::-1, :])
    if labels_xyxy.shape[0]:
        labels_xyxy = labels_xyxy.copy()
        x1, x2 = labels_xyxy[:, 1].copy(), labels_xyxy[:, 3].copy()
        labels_xyxy[:, 1] = w - x2
        labels_xyxy[:, 3] = w - x1
    return img, labels_xyxy


def random_affine(
    img: Image.Image,
    labels_xyxy: np.ndarray,
    out_size: int,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    min_box_px: float = 2.0,
) -> tuple[Image.Image, np.ndarray]:
    """Combined rotate + uniform-scale + shear + translate, applied to both the
    image (via PIL's inverse-mapped AFFINE transform) and box corners (via the
    forward matrix, then re-deriving the axis-aligned enclosing box)."""
    w, h = img.size

    C = np.eye(3)
    C[0, 2], C[1, 2] = -w / 2, -h / 2

    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    cos_a, sin_a = math.cos(math.radians(a)), math.sin(math.radians(a))
    R = np.eye(3)
    R[0, 0], R[0, 1] = s * cos_a, -s * sin_a
    R[1, 0], R[1, 1] = s * sin_a, s * cos_a

    Sh = np.eye(3)
    Sh[0, 1] = math.tan(math.radians(random.uniform(-shear, shear)))
    Sh[1, 0] = math.tan(math.radians(random.uniform(-shear, shear)))

    T = np.eye(3)
    T[0, 2] = (0.5 + random.uniform(-translate, translate)) * out_size
    T[1, 2] = (0.5 + random.uniform(-translate, translate)) * out_size

    M = T @ Sh @ R @ C

    m_inv = np.linalg.inv(M)
    out_img = img.transform(
        (out_size, out_size),
        Image.AFFINE,
        data=tuple(m_inv[:2, :].flatten()),
        resample=Image.BILINEAR,
        fillcolor=(114, 114, 114),
    )

    if labels_xyxy.shape[0] == 0:
        return out_img, labels_xyxy

    cls = labels_xyxy[:, 0]
    x1, y1, x2, y2 = labels_xyxy[:, 1], labels_xyxy[:, 2], labels_xyxy[:, 3], labels_xyxy[:, 4]
    corners = np.stack(
        [
            np.stack([x1, y1, np.ones_like(x1)], -1),
            np.stack([x2, y1, np.ones_like(x1)], -1),
            np.stack([x2, y2, np.ones_like(x1)], -1),
            np.stack([x1, y2, np.ones_like(x1)], -1),
        ],
        axis=1,
    )  # (n, 4 corners, 3)
    new_corners = corners @ M.T  # (n, 4, 3)
    xs, ys = new_corners[..., 0], new_corners[..., 1]

    new_x1, new_x2 = xs.min(1), xs.max(1)
    new_y1, new_y2 = ys.min(1), ys.max(1)
    new_x1, new_x2 = np.clip(new_x1, 0, out_size), np.clip(new_x2, 0, out_size)
    new_y1, new_y2 = np.clip(new_y1, 0, out_size), np.clip(new_y2, 0, out_size)

    keep = (new_x2 - new_x1 > min_box_px) & (new_y2 - new_y1 > min_box_px)
    new_labels = np.stack([cls, new_x1, new_y1, new_x2, new_y2], axis=1)[keep].astype(np.float32)
    return out_img, new_labels


def mosaic4(
    samples: list[tuple[Image.Image, np.ndarray]], imgsz: int
) -> tuple[Image.Image, np.ndarray]:
    """
    Combine 4 `(PIL image, normalized-xywh labels Nx5)` samples into one
    `2*imgsz` square canvas, tiled into fixed quadrants (see module
    docstring for how this differs from Ultralytics' random-center mosaic).
    Returns `(canvas, labels_xyxy)` with labels in the canvas's pixel space.
    """
    s = imgsz
    canvas = Image.new("RGB", (2 * s, 2 * s), (114, 114, 114))
    offsets = [(0, 0), (s, 0), (0, s), (s, s)]
    all_labels = []

    for (img, labels), (ox, oy) in zip(samples, offsets):
        tile, ratio, pad = letterbox(img, s)
        canvas.paste(tile, (ox, oy))

        if labels.shape[0]:
            w0, h0 = img.size
            xc, yc = labels[:, 1] * w0, labels[:, 2] * h0
            bw, bh = labels[:, 3] * w0, labels[:, 4] * h0
            x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
            x1 = x1 * ratio + pad[0] + ox
            x2 = x2 * ratio + pad[0] + ox
            y1 = y1 * ratio + pad[1] + oy
            y2 = y2 * ratio + pad[1] + oy
            all_labels.append(np.stack([labels[:, 0], x1, y1, x2, y2], axis=1))

    labels_xyxy = (
        np.concatenate(all_labels, axis=0).astype(np.float32) if all_labels else np.zeros((0, 5), dtype=np.float32)
    )
    return canvas, labels_xyxy
