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


def _split_train_and_val(img_list, t_count, v_count, rng):
    """Split images into training and validation sets."""
    shuffled = img_list.copy()
    rng.shuffle(shuffled)
    return shuffled[:t_count], shuffled[t_count:t_count + v_count]


def _partition_equally(img_files: dict[str, list], num_clients: int):
    """Partition images equally across clients."""
    client_files = {f"{client}": [] for client in range(num_clients)}
    
    for class_id, images in img_files.items():
        img_count = len(images) // num_clients
        completed = 0
        for client in range(num_clients):
            client_images = images[completed:completed + img_count]
            client_files[f"{client}"].extend(client_images)
            completed += img_count
    
    return client_files


def _partition_unequally(img_files: dict[str, list], num_clients: int, 
                         minority_threshold: int = 100, seed: int = 42):
    """
    Partition images with class imbalance for federated learning.
    
    Args:
        img_files: Dictionary mapping class_id -> list of image files
        num_clients: Number of federated learning clients
        minority_threshold: Classes with fewer instances distributed to all clients
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping client_id to list of image files
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize client data structures
    client_data = {f"{client}": [] for client in range(num_clients)}
    
    # Identify minority and majority classes
    minority_classes = []
    majority_classes = []
    
    for class_id, images in img_files.items():
        if len(images) < minority_threshold:
            minority_classes.append(class_id)
        else:
            majority_classes.append(class_id)
    
    print(f"Minority classes (< {minority_threshold} samples): {[KITTI_CLASSES[c] for c in minority_classes]}")
    print(f"Majority classes: {[KITTI_CLASSES[c] for c in majority_classes]}")
    
    # Distribute minority classes to ALL clients
    for class_id in minority_classes:
        images = img_files[class_id].copy()
        random.shuffle(images)
        
        # Split minority class roughly equally among all clients
        splits = np.array_split(images, num_clients)
        for client_idx in range(num_clients):
            client_data[f"{client_idx}"].extend(splits[client_idx].tolist())
    
    # Distribute majority classes with imbalance
    for client_idx in range(num_clients):
        # Randomly select 2-4 classes to be dominant for this client
        num_dominant = random.randint(2, min(4, len(majority_classes)))
        dominant_classes = random.sample(majority_classes, num_dominant)
        
        print(f"Client {client_idx} dominant classes: {[KITTI_CLASSES[c] for c in dominant_classes]}")
        
        for class_id in majority_classes:
            images = img_files[class_id].copy()
            random.shuffle(images)
            
            if class_id in dominant_classes:
                # This client gets a larger share (40-60%)
                proportion = random.uniform(0.4, 0.6)
            else:
                # This client gets a smaller share (5-15%)
                proportion = random.uniform(0.05, 0.15)
            
            # Calculate number of samples for this client
            num_samples = int(len(images) * proportion / num_clients)
            num_samples = max(1, min(num_samples, len(images)))
            
            # Assign samples to this client
            client_data[f"{client_idx}"].extend(images[:num_samples])
    
    # Print statistics
    print("\n=== Client Statistics ===")
    for client_id, img_list in client_data.items():
        print(f"Client {client_id}: {len(img_list)} total images")
    
    return client_data


def prepare_kitti_data(
    base_data_path: Path,
    output_path: Path,
    num_clients: int = 5,
    train_ratio: float = 0.8,
    equal_classes: bool = True,
    minority_threshold: int = 100,
    seed: int = 42,
):
    """
    Prepare KITTI dataset for federated learning.
    
    Args:
        base_data_path: Path to KITTI dataset
        output_path: Path to save prepared client data
        num_clients: Number of clients to create
        train_ratio: Ratio of training to validation split
        equal_classes: If True, distribute classes equally; if False, create imbalance
        minority_threshold: Threshold for minority class detection (only used when equal_classes=False)
        seed: Random seed for reproducibility
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    
    prepared_data_path = output_path
    prepared_data_path.mkdir(parents=True, exist_ok=True)
    
    base_img_path = base_data_path / "training" / "image_2"
    base_label_path = base_data_path / "labels"
    
    image_files = sorted([f for f in os.listdir(base_img_path) if f.endswith((".png", ".jpg"))])
    
    # Build class to image mapping
    class_img_map = defaultdict(list)
    for img in image_files:
        lbl = base_label_path / (img.rsplit(".", 1)[0] + ".txt")
        cls = _read_image_class(lbl)
        if not cls:
            continue
        for c in cls:
            class_img_map[c].append(img)
    
    print("\n=== Class Distribution in Dataset ===")
    for class_id, images in sorted(class_img_map.items()):
        print(f"{KITTI_CLASSES[class_id]}: {len(images)} images")
    
    # Partition data based on strategy
    if equal_classes:
        print(f"\npartitioning for {num_clients} clients (equally)")
        client_files = _partition_equally(img_files=class_img_map, num_clients=num_clients)
    else:
        print(f"\npartitioning with class imbalance for {num_clients} clients")
        client_files = _partition_unequally(
            img_files=class_img_map, 
            num_clients=num_clients,
            minority_threshold=minority_threshold,
            seed=seed
        )
    
    if not client_files:
        raise ValueError("Partitioning failed: no client files were generated.")
    
    # Create client directories and copy files
    for client_id, img_list in client_files.items():
        # Remove duplicates while preserving order
        img_list = list(dict.fromkeys(img_list))
        
        client_dir = prepared_data_path / f"client_{client_id}"
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
        
        # Copy training files
        for f in train_files:
            shutil.copy(base_img_path / f, img_train / f)
            label_file = f.rsplit(".", 1)[0] + ".txt"
            shutil.copy(base_label_path / label_file, lbl_train / label_file)
        
        # Copy validation files
        for f in val_files:
            shutil.copy(base_img_path / f, img_val / f)
            label_file = f.rsplit(".", 1)[0] + ".txt"
            shutil.copy(base_label_path / label_file, lbl_val / label_file)
        
        # Create data.yaml for each client
        yaml.dump(
            {
                "train": str(img_train),
                "val": str(img_val),
                "nc": KITTI_NC,
                "names": KITTI_CLASSES,
            },
            open(client_dir / "data.yaml", "w"),
        )
        
        print(f"Client {client_id}: {t_count} train, {v_count} val images")
    
    partition_strategy = "equal" if equal_classes else "unequal"
    print(f"\n Prepared {num_clients} clients at {output_path} using {partition_strategy} partitioning")


# Example usage
if __name__ == "__main__":
    prepare_kitti_data(
        base_data_path=Path("./kitti_dataset"),
        output_path=Path("./federated_data"),
        num_clients=5,
        train_ratio=0.8,
        equal_classes=False,  # Set to True for equal distribution, False for imbalance
        minority_threshold=100,
        seed=42
    )