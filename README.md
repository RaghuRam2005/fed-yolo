# Object Detection using YOLO and Federated Learning

This project demonstrates object detection using the YOLO model (by Ultralytics) trained in a simulated federated learning environment with the KITTI dataset.

## Overview

- **YOLO Model:** State-of-the-art, real-time object detection.
- **Federated Learning:** Training is distributed across multiple simulated clients.
- **Dataset:** KITTI dataset for 2D object detection.

## Datasets

### KITTI Dataset

- **Images:** Download the left color images from the KITTI 2D object detection benchmark.
    - [KITTI Dataset Download](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
- **Note:** The official KITTI dataset only contains images.

### Labels

- **Original Labels:** Provided on the KITTI website, but not in YOLO format.
- **YOLO-formatted Labels:** Available on Kaggle for easier integration.
    - [KITTI YOLO Labels on Kaggle](https://www.kaggle.com/datasets)
    - Alternatively, you can convert the original labels to YOLO format using available scripts.

## Getting Started

Follow these steps to set up and run the project:

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/obj_yolo.git
    cd obj_yolo
    ```

2. **Install the Python package manager [uv](https://docs.astral.sh/uv/getting-started/installation/)**
    - Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. **Download the KITTI images and YOLO-formatted labels**
    - Download KITTI images from the [official site](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
    - Download YOLO-formatted labels from [Kaggle](https://www.kaggle.com/datasets) or convert the original labels using available scripts.

4. **Set up environment variables and folders**
    - Copy `.env.example` to `.env` in the `obj_yolo` folder.
    - Create any required folders as specified in `.env.example`.
    - Update the paths in `.env` to match your local setup.

5. **Prepare the dataset (run once after downloading the data)**
    ```bash
    uv run ./obj_yolo/data_prep.py
    ```

6. **Start the federated learning server**
    ```bash
    uv run ./obj_yolo/server.py
    ```

7. **Run one or more clients (in separate terminals)**
    ```bash
    uv run ./obj_yolo/client.py
    ```

### Notes

- on a GPU machine with cuda, we install cuda based pytorch using:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

- then you can run the code using python instead of `uv`.


## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/)
- [Federated Learning Overview](https://en.wikipedia.org/wiki/Federated_learning)

---

Feel free to contribute or open issues for improvements!
