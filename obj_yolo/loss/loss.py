"""
YOLOv8 detection loss: Task-Aligned Assigner (see `obj_yolo.loss.tal`) for
label assignment, then classification BCE (soft targets, over *all* anchors)
+ CIoU box loss + DFL regression loss (both restricted to assigned
positives, weighted by each anchor's alignment-normalized target score).

Verified against Ultralytics' `ultralytics/utils/loss.py::v8DetectionLoss`
and `BboxLoss`/`DFLoss`. Loss-term gains (box=7.5, cls=0.5, dfl=1.5) are
Ultralytics' published defaults (`ultralytics/cfg/default.yaml`).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from obj_yolo.loss.bbox_utils import bbox2dist, bbox_iou, dist2bbox, make_anchors
from obj_yolo.loss.tal import TaskAlignedAssigner


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    return torch.cat((xy - wh, xy + wh), -1)


class DFLoss(nn.Module):
    """Cross-entropy over the two nearest integer bins of a continuous DFL
    target (Generalized Focal Loss, arXiv:2006.04388, eq. 6)."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """CIoU box-regression loss + (if reg_max > 1) DFL loss, both restricted to
    `fg_mask` positives and weighted by each positive's target-score sum."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, ciou=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss is not None:
            # bbox2dist's own "reg_max" clamp bound is (bins - 1), i.e. the
            # highest valid DFL bin index -- not the bin count.
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class DetectionLoss:
    """
    Callable: `loss(preds, batch) -> (total_loss, detached_per_term_loss)`.

    `preds` is the *raw* (training-mode) `Detect` output: a list of 3
    per-level tensors, each `(batch, no, H, W)` with `no = nc + 4*reg_max`.
    `batch` is a dict with flat `"batch_idx"`, `"cls"`, `"bboxes"` (normalized
    xywh) tensors, as produced by `obj_yolo.data.yolo_dataset`'s collate_fn.
    """

    def __init__(
        self,
        nc: int,
        stride: torch.Tensor,
        reg_max: int = 16,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.nc = nc
        self.no = nc + reg_max * 4
        self.reg_max = reg_max
        self.device = device or torch.device("cpu")
        self.stride = stride.to(self.device)
        self.box_gain, self.cls_gain, self.dfl_gain = box_gain, cls_gain, dfl_gain

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(reg_max)
        self.proj = torch.arange(reg_max, dtype=torch.float, device=self.device)

    def _preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Flat (batch_idx, cls, x, y, w, h) rows -> padded (bs, max_boxes, 5) [cls, xyxy_px]."""
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        img_idx = targets[:, 0]
        _, counts = img_idx.unique(return_counts=True)
        out = torch.zeros(batch_size, int(counts.max()), 5, device=self.device)
        for j in range(batch_size):
            matches = img_idx == j
            n = int(matches.sum())
            if n:
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def _bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Predicted per-side bin logits -> expected ltrb distance -> xyxy box (grid units)."""
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: list[torch.Tensor], batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.zeros(3, device=self.device)  # [box, cls, dfl]
        feats = preds

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self._preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self._bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes = target_bboxes / stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.box_gain
        loss[1] *= self.cls_gain
        loss[2] *= self.dfl_gain

        # Sum-then-scale-by-batch_size is Ultralytics' own convention (losses
        # are normalized per-target, not per-image, so this keeps gradient
        # magnitude comparable across batch sizes).
        return loss.sum() * batch_size, loss.detach()
