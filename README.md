# YOLOv8 Object Detection (from scratch, PyTorch)

A from-scratch PyTorch implementation of YOLOv8 — model, loss, dataset
pipeline, and training script — with no dependency on the `ultralytics`
package. Trained/evaluated on KITTI and BDD100K.

> **Status:** this is a centralized (single-machine) training pipeline.
> The project previously used `flwr` for federated learning across simulated
> clients; that integration was removed during the rewrite to `ultralytics`/
> `flwr`-free PyTorch and is planned as **future work** once the model itself
> is verified correct.

## Overview

- **Model:** YOLOv8 (CSPDarknet backbone + PAN-FPN neck + anchor-free
  decoupled head), implemented directly in PyTorch under `obj_yolo/model/`.
  Architecture is parameterized by the standard n/s/m/l/x scale table, same
  as `yolo_config/*_yolo8.yaml`.
- **Loss:** `obj_yolo/loss/` — Task-Aligned Assigner for label assignment,
  classification BCE (soft targets over all anchors), CIoU box loss and DFL
  (Distribution Focal Loss) regression loss over assigned positives.
- **Data:** KITTI and BDD100K, prepared into YOLO-format `images/`+`labels/`
  folders by `obj_yolo/dataset.py`, loaded via a PyTorch `Dataset` in
  `obj_yolo/data/` with letterbox resize, HSV/flip/affine augmentation and
  4-image mosaic.

## Datasets

### KITTI Dataset

- **Images:** Download the left color images from the KITTI 2D object
  detection benchmark.
    - [KITTI Dataset Download](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
- **Note:** The official KITTI dataset contains only images.
- **Labels:** Original labels (not YOLO format) are on the KITTI website;
  YOLO-formatted labels are available pre-converted on
  [Kaggle](https://www.kaggle.com/datasets).

### BDD100K Dataset

- Detection labels + `images/100k/train` from the
  [BDD100K dataset](https://bdd-data.berkeley.edu/).

Prepare either dataset with `obj_yolo/dataset.py`'s `PrepareData` class. For
centralized (non-federated) training, use `clientCount=1` to get a single
`images/{train,val}` + `labels/{train,val}` + `data.yaml` tree:

```python
from pathlib import Path
from obj_yolo.dataset import PrepareData

PrepareData(
    dataName="kitti",  # or "bdd100k"
    baseDataPath=Path("./kitti_dataset"),
    finalDataPath=Path("./dataset/client_0"),
    clientCount=1,
).start()
```

## Getting Started

1. **Install dependencies:**

    ```bash
    uv sync
    ```

    This installs the CPU build of PyTorch by default. `--device` in
    `train.py`/`val.py` auto-selects `cuda` when available and falls back to
    `cpu` otherwise, but the *installed wheel* also has to be CUDA-enabled —
    on a CUDA-capable machine, install a matching build instead, e.g.:

    ```bash
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    ```

    (pick the `cuXXX` tag matching your installed CUDA driver — see
    [pytorch.org/get-started](https://pytorch.org/get-started/locally/)).

2. **Prepare the dataset** (see above), then **train:**

    ```bash
    uv run python -m obj_yolo.train --data dataset/client_0/data.yaml --scale n --epochs 100 --imgsz 640 --batch 16
    ```

3. **Evaluate:**

    ```bash
    uv run python -m obj_yolo.val --data dataset/client_0/data.yaml --weights runs/train/weights/best.pt
    ```

## References

1. [Ultralytics `ultralytics` source (architecture ground truth)](https://github.com/ultralytics/ultralytics)
2. Reis et al., *"Real-Time Flying Object Detection with YOLOv8"*, [arXiv:2305.09972](https://arxiv.org/abs/2305.09972)
3. Feng et al., *"TOOD: Task-aligned One-stage Object Detection"*, [arXiv:2108.07755](https://arxiv.org/abs/2108.07755)
4. Li et al., *"Generalized Focal Loss"*, [arXiv:2006.04388](https://arxiv.org/abs/2006.04388)
5. Zheng et al., *"Distance-IoU Loss"*, [arXiv:1911.08287](https://arxiv.org/abs/1911.08287)
6. [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/)
7. [BDD100K Dataset](https://bdd-data.berkeley.edu/)
8. [Federated Learning Wikipedia](https://en.wikipedia.org/wiki/Federated_learning) (future work reference)

---

Contributions and issues are welcome!
