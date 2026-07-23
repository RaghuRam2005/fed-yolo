"""
PyTorch `Dataset` over the YOLO-format `images/`+`labels/` layout produced by
`obj_yolo.dataset.PrepareData` (per-image `.txt` label files: one row per box,
`cls xc yc w h`, all normalized to [0, 1]).
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from obj_yolo.data.augment import DEFAULT_HYP, augment_hsv, fliplr, letterbox, mosaic4, random_affine

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _norm_xywh_to_pixel_xyxy(
    labels: np.ndarray, w0: int, h0: int, ratio: float, pad: tuple[float, float]
) -> np.ndarray:
    """Map a single image's normalized-xywh labels into letterboxed pixel-xyxy space."""
    if labels.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)
    cls = labels[:, 0]
    xc, yc = labels[:, 1] * w0, labels[:, 2] * h0
    bw, bh = labels[:, 3] * w0, labels[:, 4] * h0
    x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
    x1, x2 = x1 * ratio + pad[0], x2 * ratio + pad[0]
    y1, y2 = y1 * ratio + pad[1], y2 * ratio + pad[1]
    return np.stack([cls, x1, y1, x2, y2], axis=1).astype(np.float32)


def _pixel_xyxy_to_norm_xywh(labels_xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    """Map pixel-xyxy labels back to normalized xywh, clipped to [0, 1]."""
    if labels_xyxy.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)
    cls = labels_xyxy[:, 0]
    x1, y1, x2, y2 = labels_xyxy[:, 1], labels_xyxy[:, 2], labels_xyxy[:, 3], labels_xyxy[:, 4]
    xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    out = np.stack([cls, xc, yc, bw, bh], axis=1).astype(np.float32)
    out[:, 1:] = np.clip(out[:, 1:], 0.0, 1.0)
    return out


class YoloDataset(Dataset):
    """
    Args:
        images_dir, labels_dir: e.g. `<client_dir>/images/train`, `<client_dir>/labels/train`.
        imgsz: square training resolution.
        augment: apply mosaic/HSV/flip/affine augmentation (train split); if
            False, only letterbox-resize (val split).
        hyp: augmentation hyperparameters; defaults to `obj_yolo.data.augment.DEFAULT_HYP`.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        imgsz: int = 640,
        augment: bool = True,
        hyp: dict | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = hyp or DEFAULT_HYP

        self.img_files = sorted(p for p in self.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not self.img_files:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.img_files)

    def _load_labels(self, img_path: Path) -> np.ndarray:
        label_path = self.labels_dir / (img_path.stem + ".txt")
        if not label_path.exists() or label_path.stat().st_size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        rows = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
        return rows

    def _load(self, index: int) -> tuple[Image.Image, np.ndarray]:
        img_path = self.img_files[index]
        img = Image.open(img_path).convert("RGB")
        return img, self._load_labels(img_path)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hyp = self.hyp

        if self.augment and random.random() < hyp["mosaic"]:
            samples = [self._load(index)] + [self._load(random.randint(0, len(self) - 1)) for _ in range(3)]
            canvas, labels_xyxy = mosaic4(samples, self.imgsz)
            img, labels_xyxy = random_affine(
                canvas,
                labels_xyxy,
                self.imgsz,
                degrees=hyp["degrees"],
                translate=hyp["translate"],
                scale=hyp["scale"],
                shear=hyp["shear"],
            )
        else:
            img, labels = self._load(index)
            w0, h0 = img.size
            img, ratio, pad = letterbox(img, self.imgsz)
            labels_xyxy = _norm_xywh_to_pixel_xyxy(labels, w0, h0, ratio, pad)
            if self.augment:
                img, labels_xyxy = random_affine(
                    img,
                    labels_xyxy,
                    self.imgsz,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                )

        img_np = np.array(img, dtype=np.uint8)
        if self.augment:
            img_np = augment_hsv(img_np, hyp["hsv_h"], hyp["hsv_s"], hyp["hsv_v"])
            if random.random() < hyp["fliplr"]:
                img_np, labels_xyxy = fliplr(img_np, labels_xyxy)

        labels_norm = _pixel_xyxy_to_norm_xywh(labels_xyxy, self.imgsz, self.imgsz)
        # Drop degenerate (zero-area) boxes produced by clipping.
        keep = (labels_norm[:, 3] > 0) & (labels_norm[:, 4] > 0)
        labels_norm = labels_norm[keep]

        img_t = torch.from_numpy(img_np.transpose(2, 0, 1).copy()).float() / 255.0
        cls_t = torch.from_numpy(labels_norm[:, 0:1]).float()
        box_t = torch.from_numpy(labels_norm[:, 1:5]).float()
        return img_t, cls_t, box_t


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack images; flatten per-image (cls, box) pairs into (batch_idx, cls, x, y, w, h)
    rows, as expected by `obj_yolo.loss.loss.DetectionLoss`."""
    imgs, clss, boxes = zip(*batch)
    imgs = torch.stack(imgs, 0)

    batch_idx, cls_all, box_all = [], [], []
    for i, (c, b) in enumerate(zip(clss, boxes)):
        n = c.shape[0]
        if n:
            batch_idx.append(torch.full((n,), i, dtype=torch.float32))
            cls_all.append(c.squeeze(-1))
            box_all.append(b)

    batch_idx = torch.cat(batch_idx) if batch_idx else torch.zeros(0)
    cls_all = torch.cat(cls_all) if cls_all else torch.zeros(0)
    box_all = torch.cat(box_all) if box_all else torch.zeros(0, 4)

    return {"img": imgs, "batch_idx": batch_idx, "cls": cls_all, "bboxes": box_all}
