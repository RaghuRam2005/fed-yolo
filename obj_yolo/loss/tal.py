"""
Task-Aligned Assigner: dynamically assigns ground-truth boxes to anchor
points based on a joint classification+localization "alignment metric",
rather than a fixed IoU/center-distance rule.

Verified against Ultralytics' `ultralytics/utils/tal.py::TaskAlignedAssigner`
and Feng et al., "TOOD: Task-aligned One-stage Object Detection"
(arXiv:2108.07755), which introduced this assignment scheme (their eq. 9 for
the alignment metric `t = s^alpha * u^beta`, s = classification score,
u = IoU).
"""
from typing import Optional

import torch
import torch.nn as nn

from obj_yolo.loss.bbox_utils import bbox_iou


def select_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """For each (batch, gt), mask of anchor centers strictly inside that gt's xyxy box."""
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(
    mask_pos: torch.Tensor, overlaps: torch.Tensor, n_max_boxes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """If an anchor was matched to >1 gt, keep only the highest-IoU gt for that anchor."""
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
        max_overlaps_idx = overlaps.argmax(1)
        is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
        fg_mask = mask_pos.sum(-2)
    target_gt_idx = mask_pos.argmax(-2)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    """
    Args (ultralytics YOLOv8 defaults, set by `obj_yolo.loss.loss.DetectionLoss`):
        topk: candidates considered per gt before the in-box/uniqueness filters.
        num_classes: number of classes.
        alpha, beta: exponents in the alignment metric `t = score^alpha * iou^beta`.
    """

    def __init__(self, topk: int = 10, num_classes: int = 80, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9) -> None:
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pd_scores: (bs, num_anchors, nc) sigmoid class probabilities.
            pd_bboxes: (bs, num_anchors, 4) predicted xyxy boxes (pixel coords).
            anc_points: (num_anchors, 2) anchor center points (pixel coords).
            gt_labels: (bs, max_num_obj, 1)
            gt_bboxes: (bs, max_num_obj, 4) xyxy (pixel coords)
            mask_gt: (bs, max_num_obj, 1) 1 for real gt, 0 for padding.

        Returns:
            target_labels, target_bboxes, target_scores (soft, alignment-normalized),
            fg_mask (bool, positives), target_gt_idx.
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self._get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize soft targets by each gt's best alignment metric, scaled by its best IoU
        # (TOOD eq. 14-ish): keeps the target "confidence" tied to how well-aligned the match is.
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def _get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self._get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self._select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def _get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long, device=pd_scores.device)
        ind[0] = torch.arange(end=self.bs, device=pd_scores.device).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, ciou=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def _select_topk_candidates(self, metrics: torch.Tensor, topk_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # An anchor picked more than once for the same gt (degenerate topk ties) doesn't count.
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]

        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        target_labels.clamp_(0)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores
