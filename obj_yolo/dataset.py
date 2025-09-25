# dataset.py
"""data preparation file"""
import os
import shutil
import yaml
import json
import random
from pathlib import Path
from typing import List, Dict
from PIL import Image
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_data_kitti(
        base_path:str,
        client_base_path:str,
        client_id:str,
        train_images: List[str],
        val_images: List[str]) -> Path:
    """
    Creates data yaml and copies data from base data, using pre-partitioned
    lists of image files. (for KITTI dataset preparation)

    Args:
        base_path (str): base path that contains images and labals
        client_base_path (str): base path for clients data
        client_id (str): id of the client
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

# YOLO class names
YOLO_CLASSES = ["bus", "light", "sign", "person", "bike", "truck", "motor", "car", "train", "rider"]
CLASS2ID = {name: i for i, name in enumerate(YOLO_CLASSES)}

def _get_or_create_attribute_index(base_label_path: Path, index_file_path: Path) -> Dict:
    """
    Loads the attribute index from a file, or creates it if it doesn't exist.
    The index maps attributes (like 'weather' or 'scene') and their values
    to a list of corresponding label filenames. This provides a significant speedup.
    """
    if index_file_path.exists():
        print("✅ Loading existing metadata index.")
        with open(index_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("⏳ Creating new metadata index. This might take a moment...")
    index = {"weather": {}, "scene": {}}
    for label_file in base_label_path.glob("*.json"):
        with open(label_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                attrs = data.get("attributes", {})
                weather = attrs.get("weather")
                scene = attrs.get("scene")
                if weather:
                    index["weather"].setdefault(weather, []).append(label_file.name)
                if scene:
                    index["scene"].setdefault(scene, []).append(label_file.name)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not decode JSON from {label_file.name}")
                continue
    
    with open(index_file_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"✅ Metadata index created and saved to {index_file_path}")
    return index

def _process_single_file(img_file, lbl_file, split_label_dir, split_img_dir):
    """
    Convert a single image + label pair into YOLO format.
    """
    try:
        with Image.open(img_file) as im:
            w, h = im.size

        yolo_txt = split_label_dir / f"{img_file.stem}.txt"
        with open(lbl_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(yolo_txt, "w", encoding="utf-8") as out:
            for obj in data.get("labels", []):
                category = obj.get("category")
                if category not in CLASS2ID: continue
                class_id = CLASS2ID[category]
                if "box2d" not in obj: continue

                x1, y1, x2, y2 = obj["box2d"]["x1"], obj["box2d"]["y1"], obj["box2d"]["x2"], obj["box2d"]["y2"]
                x_center, y_center = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                width, height = (x2 - x1) / w, (y2 - y1) / h
                out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        shutil.copy(img_file, split_img_dir / img_file.name)
    except Exception as e:
        print(f"❌ Error processing {img_file}: {e}")

def _prepare_yolo_dataset(base_img_path, base_label_path, client_id, client_data_path,
                          filter_key, filter_value, count):
    """
    Internal function to prepare YOLO dataset. Uses a pre-computed index for speed.
    """
    output_dir = Path(client_data_path) / f"client_{client_id}"
    yaml_path = output_dir / "data.yaml"

    base_img_path, base_label_path = Path(base_img_path), Path(base_label_path)
    
    # --- OPTIMIZATION: Use metadata index ---
    index_file = base_label_path.parent / "metadata_index.json"
    metadata_index = _get_or_create_attribute_index(base_label_path, index_file)
    
    try:
        label_filenames = metadata_index[filter_key][filter_value]
    except KeyError:
        raise ValueError(f"No matches found for {filter_key}={filter_value} in the index.")
    
    matched = [(base_img_path / f"{Path(name).stem}.jpg", base_label_path / name)
               for name in label_filenames if (base_img_path / f"{Path(name).stem}.jpg").exists()]
    
    if not matched:
        raise ValueError(f"No image files found for the matches of {filter_key}={filter_value}")

    if output_dir.exists(): shutil.rmtree(output_dir)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    random.shuffle(matched)
    matched = matched[:count]
    train_files, val_files = train_test_split(matched, test_size=0.2, random_state=42)

    def _process_split(pairs, split):
        split_label_dir, split_img_dir = output_dir / "labels" / split, output_dir / "images" / split
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(_process_single_file, img, lbl, split_label_dir, split_img_dir) for img, lbl in pairs]
            for f in as_completed(futures):
                try:
                    _ = f.result()
                except Exception as e:
                    print(f"A worker process failed: {e}")

    _process_split(train_files, "train")
    _process_split(val_files, "val")

    yaml_dict = {"train": str((output_dir / "images" / "train").resolve()),
                 "val": str((output_dir / "images" / "val").resolve()),
                 "nc": len(YOLO_CLASSES), "names": YOLO_CLASSES}
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f)
    return yaml_path

def weather_data_process(base_img_path, base_label_path, client_id, client_data_path, weather_tag, count):
    return _prepare_yolo_dataset(base_img_path, base_label_path, client_id, client_data_path, "weather", weather_tag, count)

def scene_data_process(base_img_path, base_label_path, client_id, client_data_path, scene_tag, count):
    return _prepare_yolo_dataset(base_img_path, base_label_path, client_id, client_data_path, "scene", scene_tag, count)
