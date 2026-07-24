"""
Anchor-free, decoupled YOLOv8 detection head.

Verified against Ultralytics' `ultralytics/nn/modules/head.py` (`Detect`,
`DFL`) and Li et al., "Generalized Focal Loss" (arXiv:2006.04388) for the
DFL expectation trick.
"""
from typing import Optional

import torch
import torch.nn as nn

from obj_yolo.model.blocks import Conv
from obj_yolo.loss.bbox_utils import make_anchors, dist2bbox


class DFL(nn.Module):
    """
    Distribution Focal Loss decoder: turns a discrete distribution over
    `c1` bins (per side, per anchor) into a continuous expected distance,
    via a fixed (non-trainable) 1x1 conv whose weight is `arange(c1)` --
    i.e. `softmax(logits) @ [0, 1, ..., c1-1]`.
    """

    def __init__(self, c1: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # batch, 4*c1, num_anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    """
    Decoupled anchor-free detection head: per feature level, an independent
    box-regression branch (predicts a discrete ltrb distance distribution,
    `4 * reg_max` channels) and classification branch (`nc` channels),
    each a pair of 3x3 `Conv` followed by a plain 1x1 `Conv2d`.

    In train mode returns the raw per-level `(box, cls)` concatenated maps
    (what `DetectionLoss` expects). In eval mode additionally decodes them
    into `(x, y, w, h, cls_probs...)` boxes in input-image pixel coordinates.
    """

    def __init__(self, nc: int, ch: tuple[int, ...]) -> None:
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        # Registered as a non-persistent buffer: `model.to(device)` now moves
        # it like every other tensor (a plain attribute never would have),
        # while `persistent=False` keeps it out of state_dict()/checkpoints/
        # FL aggregation payloads, exactly like before -- it's architecture-
        # derived (same nc+scale always gives the same stride), never learned.
        # Populated for real by YOLOv8._init_strides(); nn.Module.__setattr__
        # routes that later plain `self.stride = ...` reassignment back into
        # this same registered buffer rather than shadowing it.
        self.register_buffer("stride", torch.zeros(self.nl), persistent=False)

        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max)

        self._anchors: torch.Tensor = torch.empty(0)
        self._strides: torch.Tensor = torch.empty(0)
        self._cached_shape: Optional[torch.Size] = None

    def forward(self, feats: list[torch.Tensor]):
        for i in range(self.nl):
            feats[i] = torch.cat((self.cv2[i](feats[i]), self.cv3[i](feats[i])), 1)
        if self.training:
            return feats

        shape = feats[0].shape
        if self._cached_shape != shape:
            anchors, strides = make_anchors(feats, self.stride, 0.5)
            self._anchors, self._strides = anchors.transpose(0, 1), strides.transpose(0, 1)
            self._cached_shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in feats], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self._anchors.unsqueeze(0), xywh=True, dim=1) * self._strides
        return torch.cat((dbox, cls.sigmoid()), 1)
