# dataset.py
"""
Data preparation for federated learning clients (KITTI)
Includes NEW strategies:
 - iid-equal   (equal images per client)
 - iid-unequal (different images per client)
"""

import os
import yaml
import math
import shutil
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple

# KITTI dataset constants
KITTI_NC = 8
KITTI_CLASSES = [
    "Car", "Pedestrian", "Van", "Cyclist",
    "Truck", "Misc", "Tram", "Person_sitting"
]


# ========================================================
# READ IMAGE CLASS LABELS
# ========================================================
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
            except:
                pass
    return cls_ids


# ========================================================
# TRAIN / VAL SPLIT
# ========================================================
def _split_train_and_val(items, train_count, val_count, rng):
    items = list(items)
    rng.shuffle(items)
    return items[:train_count], items[train_count : train_count + val_count]


# ========================================================
# IID — EQUAL SPLIT (Zero overlap)
# ========================================================
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


# ========================================================
# IID — UNEQUAL SPLIT (Zero overlap)
# ========================================================
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


# ========================================================
# EXISTING STRATEGIES
# ========================================================
def _partition_dirichlet(image_files, class_to_imgs, num_clients, dirichlet_alpha):
    client_to_all_imgs = {i: [] for i in range(num_clients)}
    unassigned = set(image_files)

    for c, imgs in class_to_imgs.items():
        n = len(imgs)
        if n == 0:
            continue

        proportions = np.random.dirichlet([dirichlet_alpha] * num_clients)
        quotas = [int(round(p * n)) for p in proportions]

        while sum(quotas) > n:
            quotas[np.argmax(quotas)] -= 1
        while sum(quotas) < n:
            quotas[np.argmin(quotas)] += 1

        idx = 0
        for client, q in enumerate(quotas):
            for _ in range(q):
                if idx < n and imgs[idx] in unassigned:
                    client_to_all_imgs[client].append(imgs[idx])
                    unassigned.remove(imgs[idx])
                idx += 1

    # leftover unlabeled images
    for img in unassigned:
        min_client = min(client_to_all_imgs, key=lambda x: len(client_to_all_imgs[x]))
        client_to_all_imgs[min_client].append(img)

    return client_to_all_imgs


def _partition_shards(image_list, img_to_classes, num_clients, num_shards_per_client, rng):
    def get_primary(img):
        return min(img_to_classes[img]) if img_to_classes[img] else float('inf')

    sorted_images = sorted(image_list, key=get_primary)
    total_shards = num_clients * num_shards_per_client
    shard_size = math.ceil(len(sorted_images) / total_shards)

    shards = [
        sorted_images[i * shard_size : (i + 1) * shard_size]
        for i in range(total_shards)
    ]

    rng.shuffle(shards)
    client_to_all_imgs = {i: [] for i in range(num_clients)}

    for client in range(num_clients):
        selected = shards[client * num_shards_per_client:(client + 1) * num_shards_per_client]
        for s in selected:
            client_to_all_imgs[client].extend(s)

    return client_to_all_imgs


def _partition_quantity_imbalance(image_list, num_clients, imbalance_beta, rng):
    rng.shuffle(image_list)

    proportions = np.random.dirichlet([imbalance_beta] * num_clients)
    counts = [int(round(p * len(image_list))) for p in proportions]

    while sum(counts) > len(image_list):
        counts[np.argmax(counts)] -= 1
    while sum(counts) < len(image_list):
        counts[np.argmin(counts)] += 1

    client_to_all_imgs = {i: [] for i in range(num_clients)}
    ptr = 0
    for i in range(num_clients):
        client_to_all_imgs[i] = image_list[ptr:ptr + counts[i]]
        ptr += counts[i]

    return client_to_all_imgs


# ========================================================
# MAIN FUNCTION (your same structure)
# ========================================================
def prepare_kitti_data(
    base_data_path: Path,
    output_path: Path,
    num_clients: int = 5,
    train_ratio: float = 0.8,
    partition_strategy: str = "dirichlet",
    dirichlet_alpha: float = 0.3,
    num_shards_per_client: int = 2,
    imbalance_beta: float = 0.5,
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

    # =====================================================
    # CHOOSE STRATEGY
    # =====================================================
    if partition_strategy == "dirichlet":
        client_to_all_imgs = _partition_dirichlet(image_files, class_to_imgs, num_clients, dirichlet_alpha)

    elif partition_strategy == "shards":
        client_to_all_imgs = _partition_shards(image_files, img_to_classes, num_clients, num_shards_per_client, rng)

    elif partition_strategy == "quantity-imbalance":
        client_to_all_imgs = _partition_quantity_imbalance(image_files, num_clients, imbalance_beta, rng)

    # NEW STRATEGIES:
    elif partition_strategy == "iid-equal":
        client_to_all_imgs = _partition_iid_equal(image_files, num_clients, rng)

    elif partition_strategy == "iid-unequal":
        client_to_all_imgs = _partition_iid_unequal(image_files, num_clients, rng)

    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")

    # =====================================================
    # OUTPUT CLIENT FOLDERS (your same logic)
    # =====================================================
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

    print(f"✔ Prepared {num_clients} clients at {output_path} using {partition_strategy}")


# ========================================================
# RUN
# ========================================================
if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent
    BASE_DATA_PATH = CURRENT_DIR.parent / "dataset"
    OUTPUT_PATH = CURRENT_DIR.parent / "dataset" / "clients"

    # Change strategy here:
    prepare_kitti_data(
        base_data_path=BASE_DATA_PATH,
        output_path=OUTPUT_PATH / "iid-equal",     # or "iid-unequal"
        num_clients=3,
        partition_strategy="iid-equal",            # or "iid-unequal"
        seed=42,
    )
