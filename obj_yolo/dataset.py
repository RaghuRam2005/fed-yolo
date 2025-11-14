# dataset.py
""" data preparation for clients """
import os
import yaml
import shutil
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

# KITTI dataset constants
KITTI_NC = 8
KITTI_CLASSES = [
    "Car", "Pedestrian", "Van", "Cyclist",
    "Truck", "Misc", "Tram", "Person_sitting"
]

def _read_image_class(label_path: Path) -> set[int]:
    if not label_path.exists():
        return set()
    cls_ids = set()
    with open(label_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                cls_ids.add(int(line.split()[0]))
            except Exception:
                pass
    return cls_ids

def _split_train_and_val(items, train_count, val_count, rng):
    items = list(items)
    rng.shuffle(items)
    return items[:train_count], items[train_count : train_count + val_count]

def _partition_iid_equal(image_files, num_clients, rng):
    files = list(image_files)
    rng.shuffle(files)

    total = len(files)
    base = total // num_clients
    leftover = total % num_clients

    client_data = {i: [] for i in range(num_clients)}

    ptr = 0
    for i in range(num_clients):
        count = base + (1 if i < leftover else 0)
        client_data[i] = files[ptr:ptr + count]
        ptr += count

    return client_data

def _partition_iid_unequal(image_files, num_clients, rng):
    files = list(image_files)
    rng.shuffle(files)

    total = len(files)
    cut_points = sorted(rng.sample(range(1, total), num_clients - 1))

    client_data = {}
    prev = 0
    for i, cp in enumerate(cut_points + [total]):
        client_data[i] = files[prev:cp]
        prev = cp

    return client_data

def prepare_kitti_data(
    base_data_path: Path,
    output_path: Path,
    num_clients: int = 5,
    train_ratio: float = 0.8,
    partition_strategy: str = "iid-equal",
    seed: int = 42,
):

    rng = random.Random(seed)
    np.random.seed(seed)

    prepared_data_path = output_path
    prepared_data_path.mkdir(parents=True, exist_ok=True)

    base_img_path = base_data_path / "training" / "image_2"
    base_label_path = base_data_path / "labels"

    image_files = sorted([f for f in os.listdir(base_img_path) if f.endswith((".png", ".jpg"))])

    # class maps
    img_to_classes = {}
    class_to_imgs = defaultdict(list)
    for img in image_files:
        lbl = base_label_path / (img.rsplit(".", 1)[0] + ".txt")
        cls_set = _read_image_class(lbl)
        img_to_classes[img] = cls_set
        for c in cls_set:
            class_to_imgs[c].append(img)
         
    if partition_strategy == "iid-equal":
        client_to_all_imgs = _partition_iid_equal(image_files, num_clients, rng)

    elif partition_strategy == "iid-unequal":
        client_to_all_imgs = _partition_iid_unequal(image_files, num_clients, rng)

    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")

    for client, img_list in client_to_all_imgs.items():
        img_list = list(dict.fromkeys(img_list))

        client_dir = prepared_data_path / f"client_{client}"
        img_train = client_dir / "images/train"
        img_val = client_dir / "images/val"
        lbl_train = client_dir / "labels/train"
        lbl_val = client_dir / "labels/val"

        for p in [img_train, img_val, lbl_train, lbl_val]:
            p.mkdir(parents=True, exist_ok=True)

        total_images = len(img_list)
        if total_images == 1:
            t_count, v_count = 1, 0
        else:
            t_count = int(total_images * train_ratio)
            v_count = total_images - t_count
            if v_count == 0:
                t_count -= 1
                v_count = 1

        train_files, val_files = _split_train_and_val(img_list, t_count, v_count, rng)

        for f in train_files:
            shutil.copy(base_img_path / f, img_train / f)
            shutil.copy(base_label_path / (f.rsplit(".", 1)[0] + ".txt"), lbl_train / (f.rsplit(".", 1)[0] + ".txt"))

        for f in val_files:
            shutil.copy(base_img_path / f, img_val / f)
            shutil.copy(base_label_path / (f.rsplit(".", 1)[0] + ".txt"), lbl_val / (f.rsplit(".", 1)[0] + ".txt"))

        yaml.dump(
            {
                "train": str(img_train),
                "val": str(img_val),
                "nc": KITTI_NC,
                "classes": KITTI_CLASSES,
            },
            open(client_dir / "data.yaml", "w"),
        )

    print(f"Prepared {num_clients} clients at {output_path} using {partition_strategy}")


if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent
    BASE_DATA_PATH = CURRENT_DIR.parent / "dataset"
    OUTPUT_PATH = CURRENT_DIR.parent / "dataset" / "clients"

    prepare_kitti_data(
        base_data_path=BASE_DATA_PATH,
        output_path=OUTPUT_PATH,
        num_clients=3,
        partition_strategy="iid-equal",
        seed=42,
    )
