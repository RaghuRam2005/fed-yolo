"""
Bounding-box utilities shared by the model's inference-time decoding
(`obj_yolo.model.head.Detect`) and the training loss (`obj_yolo.loss.loss`).

Verified against Ultralytics' `ultralytics/utils/tal.py` /
`ultralytics/utils/metrics.py` (`bbox_iou`) and Zheng et al., "Distance-IoU
Loss" (arXiv:1911.08287) for the CIoU formula.
"""
import math
from typing import Optional

import torch


def make_anchors(
    feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate anchor-free grid center points + a matching per-anchor stride tensor.

    Args:
        feats: list of per-level feature maps (or shape-only tensors), used
            only for their (H, W) and device/dtype.
        strides: 1D tensor of per-level strides, e.g. [8., 16., 32.].
        grid_cell_offset: offset added to integer grid coords (0.5 = cell center).
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[-2:]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1) -> torch.Tensor:
    """Decode ltrb distances (from DFL) + anchor points -> xyxy or xywh boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Encode xyxy boxes -> ltrb distances from anchor points, clamped to DFL's bin range."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    return torch.cat((lt, rb), -1).clamp_(0, reg_max - 0.01)


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    ciou: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """IoU (optionally CIoU) between two sets of boxes, broadcastable on the last dim.

    CIoU = IoU - (center_dist^2 / diag_dist^2) - alpha * v, where `v` penalizes
    aspect-ratio inconsistency (arXiv:1911.08287, eq. 10-11).
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if not ciou:
        return iou

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw**2 + ch**2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    v = (4 / math.pi**2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)
