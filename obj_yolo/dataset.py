# dataset.py
"""
Data preparation for federated learning clients (KITTI)
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


def _read_image_class(label_path: Path) -> set[int]:
    """Extract class ids from a label file."""
    if not label_path.exists():
        return set()
    cls_ids = set()
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls_id = int(parts[0])
                cls_ids.add(cls_id)
            except Exception:
                continue
    return cls_ids


def _split_train_and_val(
    items: List[str], train_count: int, val_count: int, rng: random.Random
) -> Tuple[List[str], List[str]]:
    """Shuffle and split a list of items."""
    items = list(items)
    rng.shuffle(items)
    return items[:train_count], items[train_count : train_count + val_count]


def _partition_dirichlet(
    image_files: List[str],
    class_to_imgs: Dict[int, List[str]],
    num_clients: int,
    dirichlet_alpha: float,
) -> Dict[int, List[str]]:
    """
    Partition data using a Dirichlet distribution based on class labels.
    Images with no labels will be distributed to clients with the fewest images.
    """
    client_to_all_imgs = {i: [] for i in range(num_clients)}
    unassigned = set(image_files)

    for c, imgs in class_to_imgs.items():
        n = len(imgs)
        if n == 0:
            continue

        # Get client proportions for this class
        proportions = np.random.dirichlet([dirichlet_alpha] * num_clients)
        quotas = [int(round(p * n)) for p in proportions]

        # Normalize rounding errors
        while sum(quotas) > n:
            quotas[np.argmax(quotas)] -= 1
        while sum(quotas) < n:
            quotas[np.argmin(quotas)] += 1

        # Assign images to clients
        idx = 0
        for client, q in enumerate(quotas):
            for _ in range(q):
                if idx < n and imgs[idx] in unassigned:
                    client_to_all_imgs[client].append(imgs[idx])
                    unassigned.remove(imgs[idx])
                idx += 1

    # Distribute leftover images (those with no labels)
    for img in unassigned:
        smallest_client = min(
            client_to_all_imgs, key=lambda x: len(client_to_all_imgs[x])
        )
        client_to_all_imgs[smallest_client].append(img)

    return client_to_all_imgs


def _partition_shards(
    image_list: List[str],
    img_to_classes: Dict[str, Set[int]],
    num_clients: int,
    num_shards_per_client: int,
    rng: random.Random,
) -> Dict[int, List[str]]:
    """
    Partition data by sorting by class and distributing shards.
    This creates a non-IID class distribution.
    """
    def get_primary_class(img):
        classes = img_to_classes.get(img)
        if not classes:
            return float('inf')
        return min(classes)

    sorted_images = sorted(image_list, key=get_primary_class)

    # Create shards
    total_shards = num_clients * num_shards_per_client
    shard_size = math.ceil(len(sorted_images) / total_shards)
    shards = [
        sorted_images[i * shard_size : (i + 1) * shard_size]
        for i in range(total_shards)
    ]

    # Shuffle and assign shards to clients
    rng.shuffle(shards)
    client_to_all_imgs = {i: [] for i in range(num_clients)}

    for client_id in range(num_clients):
        client_shards = shards[
            client_id * num_shards_per_client : (client_id + 1) * num_shards_per_client
        ]
        for shard in client_shards:
            client_to_all_imgs[client_id].extend(shard)

    return client_to_all_imgs


def _partition_quantity_imbalance(
    image_list: List[str],
    num_clients: int,
    imbalance_beta: float,
    rng: random.Random,
) -> Dict[int, List[str]]:
    """
    Partition data by shuffling (IID) but assigning imbalanced quantities
    to each client based on a Dirichlet distribution.
    """
    # Shuffle data for IID distribution
    rng.shuffle(image_list)

    # Get imbalanced proportions
    proportions = np.random.dirichlet([imbalance_beta] * num_clients)
    counts = [int(round(p * len(image_list))) for p in proportions]

    # Normalize rounding errors
    while sum(counts) > len(image_list):
        counts[np.argmax(counts)] -= 1
    while sum(counts) < len(image_list):
        counts[np.argmin(counts)] += 1

    # Assign IID data in imbalanced quantities
    client_to_all_imgs = {i: [] for i in range(num_clients)}
    ptr = 0
    for i in range(num_clients):
        client_to_all_imgs[i] = image_list[ptr : ptr + counts[i]]
        ptr += counts[i]

    return client_to_all_imgs


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
    """
    Main function to prepare and partition KITTI data for clients.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Base paths
    prepared_data_path = output_path
    prepared_data_path.mkdir(parents=True, exist_ok=True)

    base_img_path = base_data_path / "training" / "image_2"
    base_label_path = base_data_path / "labels"

    if not base_img_path.exists() or not base_label_path.exists():
        raise FileNotFoundError(
            f"KITTI dataset folders not found at {base_data_path}"
        )

    # Collect images
    image_files = sorted(
        [f for f in os.listdir(base_img_path) if f.endswith((".png", ".jpg"))]
    )

    # Build class mapping
    img_to_classes = {}
    class_to_imgs = defaultdict(list)
    for img in image_files:
        lbl_file = base_label_path / (img.rsplit(".", 1)[0] + ".txt")
        cls_set = _read_image_class(lbl_file)
        img_to_classes[img] = cls_set
        for c in cls_set:
            class_to_imgs[c].append(img)

    # Partition Methods
    if partition_strategy == "dirichlet":
        client_to_all_imgs = _partition_dirichlet(
            image_files, class_to_imgs, num_clients, dirichlet_alpha
        )
    elif partition_strategy == "shards":
        client_to_all_imgs = _partition_shards(
            image_files, img_to_classes, num_clients, num_shards_per_client, rng
        )
    elif partition_strategy == "quantity-imbalance":
        client_to_all_imgs = _partition_quantity_imbalance(
            image_files, num_clients, imbalance_beta, rng
        )
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")

    # Write client folders
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
        if total_images < 2:
            # Need at least 2 images for a train/val split
            t_count = total_images
            v_count = 0
        else:
            t_count = int(total_images * train_ratio)
            v_count = total_images - t_count
            
            # Ensure val set is not empty (if we have >1 image)
            if v_count == 0:
                t_count -= 1
                v_count = 1
        
        train_files, val_files = _split_train_and_val(
            img_list, t_count, v_count, rng
        )

        # Copy files to client directories
        for f in train_files:
            shutil.copy(base_img_path / f, img_train / f)
            shutil.copy(
                base_label_path / (f.rsplit(".", 1)[0] + ".txt"),
                lbl_train / (f.rsplit(".", 1)[0] + ".txt"),
            )

        for f in val_files:
            shutil.copy(base_img_path / f, img_val / f)
            shutil.copy(
                base_label_path / (f.rsplit(".", 1)[0] + ".txt"),
                lbl_val / (f.rsplit(".", 1)[0] + ".txt"),
            )

        # Write config YAML
        yaml.dump(
            {
                "train": str(img_train),
                "val": str(img_val),
                "nc": KITTI_NC,
                "classes": KITTI_CLASSES,
            },
            open(client_dir / "data.yaml", "w"),
        )

    print(
        f"Prepared {num_clients} clients at {output_path} "
        f"using strategy = {partition_strategy}"
    )


if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent
    BASE_DATA_PATH = CURRENT_DIR.parent / "dataset"
    OUTPUT_PATH = CURRENT_DIR.parent / "dataset" / "clients"

    print("--- Preparing data with 'dirichlet' strategy ---")
    prepare_kitti_data(
        base_data_path=BASE_DATA_PATH,
        output_path=OUTPUT_PATH / "dirichlet",
        num_clients=5,
        partition_strategy="dirichlet",
        dirichlet_alpha=0.3,
        seed=42,
    )

    #print("\n--- Preparing data with 'shards' strategy ---")
    #prepare_kitti_data(
    #    base_data_path=BASE_DATA_PATH,
    #    output_path=OUTPUT_PATH / "shards",
    #    num_clients=5,
    #    partition_strategy="shards",
    #    num_shards_per_client=2,
    #    seed=42,
    #)

    #print("\n--- Preparing data with 'quantity-imbalance' strategy ---")
    #prepare_kitti_data(
    #    base_data_path=BASE_DATA_PATH,
    #    output_path=OUTPUT_PATH / "quantity-imbalance",
    #    num_clients=5,
    #    partition_strategy="quantity-imbalance",
    #    imbalance_beta=0.5,
    #    seed=42,
    #)