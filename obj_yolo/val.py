"""
Evaluation entrypoint: decode YOLOv8 predictions (DFL -> ltrb -> xyxy), NMS,
and compute precision/recall/F1/mAP50/mAP50-95 via the custom
`obj_yolo.metrics.detection_metrics` engine (no torchmetrics/pycocotools
dependency).

    uv run python -m obj_yolo.val --data dataset/client_0/data.yaml --weights runs/train/weights/best.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from obj_yolo.config_validation import validate_data_yaml, validate_imgsz
from obj_yolo.data.yolo_dataset import YoloDataset, collate_fn
from obj_yolo.metrics.detection_metrics import DetMetrics
from obj_yolo.model.yolov8 import YOLOv8


@torch.no_grad()
def postprocess(
    preds: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45, max_det: int = 300
) -> list[torch.Tensor]:
    """
    Args:
        preds: `(bs, 4+nc, num_anchors)` decoded `[x, y, w, h, cls_probs...]`
            (eval-mode `Detect` output).

    Returns:
        Per-image list of `(n, 6)` tensors: `[x1, y1, x2, y2, conf, cls]`.
    """
    preds = preds.transpose(1, 2)  # (bs, num_anchors, 4+nc)
    outputs = []
    for pred in preds:
        boxes_xywh = pred[:, :4]
        scores, cls_idx = pred[:, 4:].max(1)
        keep = scores > conf_thres
        boxes_xywh, scores, cls_idx = boxes_xywh[keep], scores[keep], cls_idx[keep]

        x, y, w, h = boxes_xywh.unbind(1)
        boxes_xyxy = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=1)

        keep_idx = torchvision.ops.batched_nms(boxes_xyxy, scores, cls_idx, iou_thres)[:max_det]
        det = torch.cat([boxes_xyxy[keep_idx], scores[keep_idx, None], cls_idx[keep_idx, None].float()], dim=1)
        outputs.append(det)
    return outputs


@torch.no_grad()
def evaluate(
    model: YOLOv8,
    loader: DataLoader,
    device: torch.device,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
) -> dict:
    """Run `model` over `loader` and compute precision/recall/F1/mAP50/mAP50-95
    (dataset means + per-class breakdown) via `DetMetrics`."""
    was_training = model.training
    model.eval()
    metric = DetMetrics(nc=model.nc)

    for batch in loader:
        imgs = batch["img"].to(device)
        preds = model(imgs)  # eval-mode Detect: decoded (bs, 4+nc, num_anchors)
        dets = postprocess(preds, conf_thres=conf_thres, iou_thres=iou_thres)

        batch_idx = batch["batch_idx"]
        imgsz = imgs.shape[-1]
        for i in range(imgs.shape[0]):
            m = batch_idx == i
            boxes_n, cls_n = batch["bboxes"][m], batch["cls"][m]
            if boxes_n.shape[0]:
                x, y, w, h = (boxes_n * imgsz).unbind(1)
                gt_xyxy = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=1)
            else:
                gt_xyxy = torch.zeros(0, 4)
            metric.update(dets[i].cpu(), gt_xyxy.cpu(), cls_n.long().cpu())

    result = metric.compute()
    model.train(was_training)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 checkpoint (mAP50, mAP50-95).")
    p.add_argument("--data", required=True, help="path to data.yaml (train/val/nc/names)")
    p.add_argument("--weights", required=True, help="path to a state_dict checkpoint (.pt)")
    p.add_argument("--scale", default="n", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        validate_imgsz(args.imgsz)
        data_cfg = validate_data_yaml(args.data)
        if "val" not in data_cfg:
            raise ValueError(f"{args.data} is missing required key 'val'")
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    nc = int(data_cfg["nc"])
    val_dir = Path(data_cfg["val"])
    labels_dir = Path(str(val_dir).replace("images", "labels"))

    model = YOLOv8(nc=nc, scale=args.scale).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    val_ds = YoloDataset(val_dir, labels_dir, imgsz=args.imgsz, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn
    )

    metrics = evaluate(model, val_loader, device)
    print(
        f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
        f"mAP50={metrics['map50']:.4f}  mAP50-95={metrics['map']:.4f}"
    )


if __name__ == "__main__":
    main()
