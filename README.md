# Object Detection using YOLO and Federated Learning

This project showcases object detection using the Ultralytics YOLO model, trained in a simulated federated learning environment across the KITTI and BDD100K datasets.

## Overview

- **YOLO Model:** Real-time, state-of-the-art object detection (YOLOv8 / YOLO11).
- **Federated Learning:** Distributed training across multiple simulated clients, built on Flower's message-based ServerApp/ClientApp API.
- **Datasets:** KITTI and BDD100K for 2D object detection.
- **Algorithm selection:** Pick the federated algorithm via a single config key — no code changes or branch switching required.

## Datasets

### KITTI Dataset

- **Images:** Download the left color images from the KITTI 2D object detection benchmark.
    - [KITTI Dataset Download](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
- **Note:** The official KITTI dataset contains only images.
- **Labels:** Original labels (not YOLO format) are on the KITTI website; YOLO-formatted labels are available pre-converted on [Kaggle](https://www.kaggle.com/datasets).

### BDD100K Dataset

- Detection labels + `images/100k/train` from the [BDD100K dataset](https://bdd-data.berkeley.edu/).
- Used by the **FedTag** algorithm below, which partitions clients by each image's weather attribute (clear/overcast/snowy/rainy/cloudy/foggy).

Prepare either dataset with `obj_yolo/dataset.py`'s `PrepareData` class:

```python
from pathlib import Path
from obj_yolo.dataset import PrepareData

PrepareData(
    dataName="bdd100k",  # or "kitti"
    baseDataPath=Path("./bdd100k_dataset"),
    finalDataPath=Path("./dataset/clients"),
    clientCount=5,
).start()
```

## Algorithms

Set `strategy-name` in `pyproject.toml` (or via `flwr run . --run-config "strategy-name='fedadam'"`) to pick one:

- **fedavg** (default): standard federated averaging.
- **fedadam**: adaptive server-side optimization ([Reddi et al., 2020](https://arxiv.org/abs/2003.00295)); tunable via the `fedadam-*` config keys.
- **fedtag**: weather-tag-conditioned sparse training (FedWeg) — each BDD100K client keeps a persistent weather tag, applies an L1 sparsity penalty on BatchNorm gamma factors, and the strategy adjusts each tag's penalty strength every round based on mAP convergence. Use with `dataset-name = "bdd100k"`; tunable via the `fedtag-*` config keys.

## Getting Started

1. **Run the Simulation using:**

    ```bash
    flwr run .
    ```

## References

1. [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
2. [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/)
3. [BDD100K Dataset](https://bdd-data.berkeley.edu/)
4. [Federated Learning Wikipedia](https://en.wikipedia.org/wiki/Federated_learning)
5. [Adaptive Federated Optimization (FedAdam)](https://arxiv.org/abs/2003.00295)
6. [Efficient and Secure Object Detection with Sparse Federated Training](https://doi.org/10.1109/TITS.2024.3389212)
7. [Efficient CNNs through Network Slimming](https://doi.org/10.48550/arXiv.1708.06519)

---

Contributions and issues are welcome!
