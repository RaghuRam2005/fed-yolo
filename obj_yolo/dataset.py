# dataset.py
"""data preparation file"""
import os
import shutil
import yaml
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool, cpu_count
from PIL import Image

def load_data_kitti(
        base_path:str,
        client_base_path:str,
        client_id:int,
        train_images: List[str],
        val_images: List[str]) -> Path:
    """
    Creates data yaml and copies data from base data, using pre-partitioned
    lists of image files. (for KITTI dataset preparation)

    Args:
        base_path (str): base path that contains images and labals
        client_base_path (str): base path for clients data
        client_id (int): id of the client
        train_images (List[str]): List of image filenames for the training set.
        val_images (List[str]): List of image filenames for the validation set.

    Returns:
        Path: returns yaml path for prepared data.
    """
    if not Path(base_path).exists():
        raise FileNotFoundError("base path does not exist")
    
    base_i_path = Path(base_path) / "image_2"
    base_l_path = Path(base_path) / "labels"

    if (not base_i_path.exists()) or (not base_l_path.exists()):
        raise FileNotFoundError("base path does not contain images or labels")

    if not Path(client_base_path).exists():
        os.makedirs(client_base_path, exist_ok=True)

    client_path = Path(client_base_path) / f"client_{client_id}"
    client_i_path = client_path / "images"
    client_l_path = client_path / "labels"

    # If data is already prepared, just return the path
    yaml_path = client_path / "data.yaml"
    if yaml_path.exists() and os.listdir(client_i_path / "train"):
        return yaml_path
    
    # Create train/val directories
    (client_i_path / "train").mkdir(parents=True, exist_ok=True)
    (client_l_path / "train").mkdir(parents=True, exist_ok=True)
    (client_i_path / "val").mkdir(parents=True, exist_ok=True)
    (client_l_path / "val").mkdir(parents=True, exist_ok=True)

    # Copy data for the TRAINING set from the provided list
    for image in train_images:
        shutil.copy(base_i_path / image, client_i_path / "train" / image)
        label_file = image.replace(".png", ".txt")
        shutil.copy(base_l_path / label_file, client_l_path / "train" / label_file)
    
    # Copy data for the VALIDATION set from the provided list
    for image in val_images:
        shutil.copy(base_i_path / image, client_i_path / "val" / image)
        label_file = image.replace(".png", ".txt")
        shutil.copy(base_l_path / label_file, client_l_path / "val" / label_file)

    content = {
        "train" : str(client_i_path / "train"),
        "val" : str(client_i_path / "val"),
        "nc" : 8,
        "classes" : ["Car", "Pedestrian", "Van", "Cyclist", "Truck", "Misc", "Tram", "Person_sitting"]
    }

    with open(yaml_path, "w") as f:
        yaml.dump(content, f, sort_keys=True)
    return yaml_path

# YOLO class mapping for BDD100K
YOLO_CLASSES = ["bus", "light", "sign", "person", "bike", "truck", "motor", "car", "train", "rider"]
CLASS2ID = {name: i for i, name in enumerate(YOLO_CLASSES)}


# ----------------------------
# 1. MULTI-FILE VERSION
# ----------------------------
def _process_label_file(file_path: Path) -> Tuple[Dict, Dict]:
    weather_dict = {}
    scene_dict = {}

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logging.warning(f"Skipping {file_path.name} due to error: {e}")
        return weather_dict, scene_dict

    img_name = data.get("name")
    attributes = data.get("attributes", {})
    weather = attributes.get("weather", "unknown")
    scene = attributes.get("scene", "unknown")

    for obj in data.get("labels", []):
        if "box2d" not in obj:
            continue

        box = obj["box2d"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        category = obj.get("category", "unknown")

        entry = {"img_name": img_name, "category": category, "bbox": bbox}

        weather_dict.setdefault(weather, []).append(entry)
        scene_dict.setdefault(scene, []).append(entry)

    return weather_dict, scene_dict


def _merge_dicts(dicts: List[Dict]) -> Dict:
    merged = {}
    for d in dicts:
        for k, v in d.items():
            merged.setdefault(k, []).extend(v)
    return merged


def create_tag_dicts(label_path: str) -> None:
    base_path = Path(label_path)
    if not base_path.exists():
        logging.error("Base label path does not exist")
        return

    if (base_path / "weather_dict.json").exists() and (base_path / "scene_dict.json").exists():
        logging.info("Data indexing already completed")
        return

    label_files = [base_path / f for f in os.listdir(base_path) if f.endswith(".json")]
    if not label_files:
        logging.warning("No JSON files found")
        return

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(_process_label_file, label_files)

    weather_parts, scene_parts = zip(*results)
    weather_dict = _merge_dicts(weather_parts)
    scene_dict = _merge_dicts(scene_parts)

    with open(base_path / "weather_dict.json", "w") as wf:
        json.dump(weather_dict, wf, indent=4)
    with open(base_path / "scene_dict.json", "w") as sf:
        json.dump(scene_dict, sf, indent=4)

    logging.info("Weather and scene dicts created successfully (multi-file).")


# ----------------------------
# 2. SINGLE-FILE VERSION
# ----------------------------
def _process_entries(entries: List[Dict]) -> Tuple[Dict, Dict]:
    weather_dict: Dict[str, Any] = {}
    scene_dict: Dict[str, Any] = {}

    for entry in entries:
        img_name = entry.get("name")
        attributes = entry.get("attributes", {})
        weather = attributes.get("weather", "unknown")
        scene = attributes.get("scene", "unknown")

        for obj in entry.get("labels", []):
            if "box2d" not in obj:
                continue

            box = obj["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            category = obj.get("category", "unknown")

            item = {"img_name": img_name, "category": category, "bbox": bbox}
            weather_dict.setdefault(weather, []).append(item)
            scene_dict.setdefault(scene, []).append(item)

    return weather_dict, scene_dict


def create_tag_dicts_from_single(label_file: str) -> None:
    file_path = Path(label_file)
    if not file_path.exists():
        logging.error("Label file does not exist")
        return

    if (file_path.parent / "weather_dict.json").exists() and (file_path.parent / "scene_dict.json").exists():
        logging.info("Data indexing already completed")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    if "root" not in data or not isinstance(data["root"], list):
        logging.error("Invalid JSON format: expected top-level 'root'")
        return

    entries = data["root"]
    num_workers = cpu_count()
    chunk_size = max(1, len(entries) // num_workers)
    chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]

    with Pool(processes=num_workers) as pool:
        results = pool.map(_process_entries, chunks)

    weather_parts, scene_parts = zip(*results)
    weather_dict = _merge_dicts(weather_parts)
    scene_dict = _merge_dicts(scene_parts)

    with open(file_path.parent / "weather_dict.json", "w") as wf:
        json.dump(weather_dict, wf, indent=4)
    with open(file_path.parent / "scene_dict.json", "w") as sf:
        json.dump(scene_dict, sf, indent=4)

    logging.info("Weather and scene dicts created successfully (single-file).")


# ----------------------------
# 3. PREPARE YOLO DATASET
# ----------------------------
def _convert_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_center, y_center, w, h


def prepare_yolo_dataset(dict_path: str, img_dir: str, filter_key: str, filter_val: str,
                         client_id: int, data_count: int, outputdir: str,
                         train_ratio: float = 0.8) -> str:
    dict_file = Path(dict_path)
    img_dir = Path(img_dir)
    outputdir = Path(outputdir) / f"client_{client_id}"

    with open(dict_file, "r") as f:
        data_dict = json.load(f)

    if filter_val not in data_dict:
        raise ValueError(f"{filter_key} value '{filter_val}' not found in {dict_file}")

    entries = data_dict[filter_val][:data_count]
    random.shuffle(entries)

    split_idx = int(len(entries) * train_ratio)
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]
    subsets = {"train": train_entries, "val": val_entries}

    for subset, subset_entries in subsets.items():
        img_out = Path(outputdir) / "images" / subset
        lbl_out = Path(outputdir) / "labels" / subset
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for item in subset_entries:
            img_name = item["img_name"]
            bbox = item["bbox"]
            category = item.get("category", "unknown")

            if category not in CLASS2ID:
                continue  # skip unknown categories

            src_img = img_dir / img_name
            dst_img = img_out / img_name
            if not src_img.exists():
                continue

            shutil.copy(src_img, dst_img)

            with Image.open(src_img) as im:
                img_w, img_h = im.size

            class_id = CLASS2ID[category]
            x_center, y_center, w, h = _convert_to_yolo(bbox, img_w, img_h)

            lbl_path = lbl_out / (Path(img_name).stem + ".txt")
            with open(lbl_path, "a") as lf:
                lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    yaml_path = Path(outputdir) / "dataset.yaml"
    yaml_dict = {
        "train": str(Path(outputdir) / "images" / "train"),
        "val": str(Path(outputdir) / "images" / "val"),
        "nc": len(YOLO_CLASSES),
        "names": YOLO_CLASSES
    }
    with open(yaml_path, "w") as yf:
        yaml.dump(yaml_dict, yf)

    logging.info(f"YOLO dataset prepared at {outputdir}")
    return str(yaml_path)
