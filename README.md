# Object Detection using YOLO and Federated Learning

This project showcases object detection using the Ultralytics YOLO model, trained in a simulated federated learning environment with the KITTI dataset.

## Overview

- **YOLO Model:** Real-time, state-of-the-art object detection.
- **Federated Learning:** Distributed training across multiple simulated clients.
- **Dataset:** KITTI dataset for 2D object detection.
- **Sparse Training:** Implements FedWeg algorithm with client-side sparsity and inverse sparsity aggregation.

## Datasets

### KITTI Dataset

- **Images:** Download the left color images from the KITTI 2D object detection benchmark.
    - [KITTI Dataset Download](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
- **Note:** The official KITTI dataset contains only images.

### Labels

- **Original Labels:** Available on the KITTI website (not in YOLO format).
- **YOLO-formatted Labels:** Find pre-converted labels on [Kaggle](https://www.kaggle.com/datasets).

## Algorithms

- **FedWeg:** Introduces sparse training on clients and aggregates results using inverse sparsity, improving efficiency and security in federated learning scenarios.

## Getting Started

To set up and run the project:

1. **Clone the repository**
    ```bash
    git clone https://github.com/RaghuRam2005/obj_yolo.git
    cd obj_yolo
    ```

2. **Install Python package manager [uv](https://docs.astral.sh/uv/getting-started/installation/)**
    - Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. **Download KITTI images and YOLO-formatted labels**
    - Get images from the [KITTI site](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
    - Download YOLO labels from [Kaggle](https://www.kaggle.com/datasets) or convert original labels using provided scripts.

4. **Set up folders**
    - Extract images and labels into the same base directory.

5. **Run the simulation**
    ```bash
    uv run simulation.py
    ```

## References

1. [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
2. [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/)
3. [Federated Learning Wikipedia](https://en.wikipedia.org/wiki/Federated_learning)
4. [Efficient and Secure Object Detection with Sparse Federated Training](https://doi.org/10.1109/TITS.2024.3389212)
5. [Efficient CNNs through Network Slimming](https://doi.org/10.48550/arXiv.1708.06519)

---

Contributions and issues are welcome!
