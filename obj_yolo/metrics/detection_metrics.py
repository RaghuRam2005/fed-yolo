"""
Detection metrics: precision, recall, F1, AP50, AP50:95 per class -- the
standard Ultralytics/COCO-style pipeline (greedy IoU-threshold matching +
101-point interpolated AP), reimplemented here so evaluation doesn't depend
on `torchmetrics`/`pycocotools` (the latter is a native-extension build that
was finicky to install on this machine).
"""
import numpy as np
import torch
from torchvision.ops import box_iou

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # COCO convention: 0.50, 0.55, ..., 0.95


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_cls: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
    iou_thresholds: np.ndarray = IOU_THRESHOLDS,
) -> np.ndarray:
    """
    For each prediction, which IoU thresholds is it a "correct" detection at
    (matched to a same-class gt box, greedily by highest IoU, each gt used
    at most once)?

    Returns:
        `(n_pred, n_thresholds)` bool array.
    """
    n_pred = pred_boxes.shape[0]
    correct = np.zeros((n_pred, len(iou_thresholds)), dtype=bool)
    if n_pred == 0 or gt_boxes.shape[0] == 0:
        return correct

    iou = box_iou(gt_boxes, pred_boxes).cpu().numpy()  # (n_gt, n_pred)
    same_class = gt_cls.cpu().numpy()[:, None] == pred_cls.cpu().numpy()[None, :]
    iou = iou * same_class

    for t, thr in enumerate(iou_thresholds):
        matches = np.argwhere(iou >= thr)  # (k, 2): [gt_idx, pred_idx]
        if matches.shape[0] == 0:
            continue
        if matches.shape[0] > 1:
            match_ious = iou[matches[:, 0], matches[:, 1]]
            matches = matches[match_ious.argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # unique pred, best iou
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # unique gt, best iou
        correct[matches[:, 1], t] = True
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated average precision (COCO convention): precision
    envelope made monotonically non-increasing, then integrated over recall."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))  # trapezoidal rule


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    iou_thresholds: np.ndarray = IOU_THRESHOLDS,
    eps: float = 1e-16,
) -> dict:
    """
    Args:
        tp: `(n_pred, n_thresholds)` bool "correct" matrix (see `match_predictions`).
        conf: `(n_pred,)` confidence scores.
        pred_cls: `(n_pred,)` predicted class ids.
        target_cls: `(n_gt,)` ground-truth class ids across the whole
            dataset (only used to know how many gt boxes each class had).
        iou_thresholds: the IoU thresholds `tp`'s columns correspond to
            (must match what was passed to `match_predictions`) -- only used
            to label `map_per_iou`'s keys.

    Returns:
        dict with dataset-mean `precision`/`recall`/`f1`/`map50`/`map`,
        `map_per_iou: {"0.50": ..., ..., "0.95": ...}` (mean AP *across
        classes* at each individual IoU threshold), and
        `per_class: {cls_id: {p, r, f1, ap50, ap}}`. Classes with zero
        ground-truth instances in the dataset are excluded (standard COCO
        convention).
    """
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]

    unique_classes, n_gt_per_class = np.unique(target_cls, return_counts=True)
    n_classes = unique_classes.shape[0]
    n_thresholds = tp.shape[1]

    ap = np.zeros((n_classes, n_thresholds))
    p_at_best_f1 = np.zeros(n_classes)
    r_at_best_f1 = np.zeros(n_classes)

    for ci, cls_id in enumerate(unique_classes):
        mask = pred_cls == cls_id
        n_l, n_p = n_gt_per_class[ci], mask.sum()
        if n_p == 0 or n_l == 0:
            continue

        fpc = (~tp[mask]).cumsum(0)
        tpc = tp[mask].cumsum(0)
        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc + eps)

        for t in range(n_thresholds):
            ap[ci, t] = compute_ap(recall[:, t], precision[:, t])

        # Report P/R at the confidence operating point maximizing F1 (IoU=0.5 column).
        f1 = 2 * precision[:, 0] * recall[:, 0] / (precision[:, 0] + recall[:, 0] + eps)
        best = int(f1.argmax())
        p_at_best_f1[ci] = precision[best, 0]
        r_at_best_f1[ci] = recall[best, 0]

    ap50, ap_mean = ap[:, 0], ap.mean(1)
    f1_per_class = 2 * p_at_best_f1 * r_at_best_f1 / (p_at_best_f1 + r_at_best_f1 + eps)

    per_class = {
        int(cls_id): {
            "p": float(p_at_best_f1[ci]),
            "r": float(r_at_best_f1[ci]),
            "f1": float(f1_per_class[ci]),
            "ap50": float(ap50[ci]),
            "ap": float(ap_mean[ci]),
        }
        for ci, cls_id in enumerate(unique_classes)
    }

    map_per_threshold = ap.mean(0) if n_classes else np.zeros(n_thresholds)
    map_per_iou = {f"{t:.2f}": float(v) for t, v in zip(iou_thresholds, map_per_threshold)}

    return {
        "precision": float(p_at_best_f1.mean()) if n_classes else 0.0,
        "recall": float(r_at_best_f1.mean()) if n_classes else 0.0,
        "f1": float(f1_per_class.mean()) if n_classes else 0.0,
        "map50": float(ap50.mean()) if n_classes else 0.0,
        "map": float(ap_mean.mean()) if n_classes else 0.0,
        "map_per_iou": map_per_iou,
        "per_class": per_class,
    }


class DetMetrics:
    """Accumulates per-image detections/gt across a dataset, then computes
    dataset-wide precision/recall/F1/mAP via `ap_per_class`."""

    def __init__(self, nc: int, iou_thresholds: np.ndarray = IOU_THRESHOLDS) -> None:
        self.nc = nc
        self.iou_thresholds = iou_thresholds
        self._tp: list[np.ndarray] = []
        self._conf: list[np.ndarray] = []
        self._pred_cls: list[np.ndarray] = []
        self._target_cls: list[np.ndarray] = []

    def update(self, detections: torch.Tensor, gt_boxes: torch.Tensor, gt_cls: torch.Tensor) -> None:
        """
        Args:
            detections: `(n, 6)` `[x1, y1, x2, y2, conf, cls]` for one image
                (e.g. `obj_yolo.val.postprocess`'s per-image output).
            gt_boxes: `(m, 4)` xyxy for the same image.
            gt_cls: `(m,)`.
        """
        if gt_cls.shape[0]:
            self._target_cls.append(gt_cls.cpu().numpy())
        if detections.shape[0] == 0:
            return
        pred_boxes, conf, pred_cls = detections[:, :4], detections[:, 4], detections[:, 5]
        correct = match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls, self.iou_thresholds)
        self._tp.append(correct)
        self._conf.append(conf.cpu().numpy())
        self._pred_cls.append(pred_cls.cpu().numpy())

    def compute(self) -> dict:
        if not self._tp:
            empty_per_iou = {f"{t:.2f}": 0.0 for t in self.iou_thresholds}
            return {
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "map50": 0.0, "map": 0.0,
                "map_per_iou": empty_per_iou, "per_class": {},
            }
        tp = np.concatenate(self._tp, 0)
        conf = np.concatenate(self._conf, 0)
        pred_cls = np.concatenate(self._pred_cls, 0)
        target_cls = np.concatenate(self._target_cls, 0) if self._target_cls else np.zeros(0)
        return ap_per_class(tp, conf, pred_cls, target_cls, self.iou_thresholds)
