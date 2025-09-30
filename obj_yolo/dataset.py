# dataset.py
"""data preparation file"""
import os
import json
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count

def kitti_client_data(
        base_path:str,
        client_base_path:str,
        client_id:str,
        train_images: List[str],
        val_images: List[str]) -> Path:
    """
    Creates data yaml and copies data from base data, using pre-partitioned
    lists of image files.

    Dataset: KITTI 2D Image dataset

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


# YOLO class mapping for BDD100K
YOLO_CLASSES = ["bus", "light", "sign", "person", "bike", "truck", "motor", "car", "train", "rider"]
CLASS2ID = {name: i for i, name in enumerate(YOLO_CLASSES)}


def bbox_to_yolo(bbox, img_w, img_h, category):
    """
    Convert COCO bbox [x1,y1,w,h] → YOLO [cls, x_center, y_center, w, h]
    """
    cls_id = CLASS2ID.get(category)
    if cls_id is None:
        return None
    x1, y1, w, h = bbox
    return (
        cls_id,
        (x1 + w / 2) / img_w,
        (y1 + h / 2) / img_h,
        w / img_w,
        h / img_h,
    )


def process_item(item):
    img_name = item["name"]
    img_w, img_h = 1280, 720 

    yolo_boxes = [
        bbox_to_yolo(
            [
                obj["box2d"]["x1"],
                obj["box2d"]["y1"],
                obj["box2d"]["x2"] - obj["box2d"]["x1"],
                obj["box2d"]["y2"] - obj["box2d"]["y1"]
            ],
            img_w,
            img_h,
            obj["category"]
        )
        for obj in item.get("labels", [])
        if "box2d" in obj and obj["category"] in CLASS2ID
    ]

    yolo_boxes = [box for box in yolo_boxes if box]

    weather = item["attributes"].get("weather", "unknown")
    scene = item["attributes"].get("scene", "unknown")

    return weather, scene, (img_name, yolo_boxes)


def build_dicts(label_file: str):
    with open(label_file, "r") as f:
        data = json.load(f)

    weather_dict, scene_dict = {}, {}

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_item, data)

    for weather, scene, entry in results:
        weather_dict.setdefault(weather, []).append(entry)
        scene_dict.setdefault(scene, []).append(entry)

    return weather_dict, scene_dict


def bdd_client_data(
    base_data_path: str,
    client_data_path: str,
    client_id: int,
    train_images: List[Tuple[str, List]],
    val_images: List[Tuple[str, List]],
) -> Path:
    client_dir = os.path.join(client_data_path, "bdd_clients", f"client_{client_id}")
    img_train_dir = os.path.join(client_dir, "images", "train")
    img_val_dir = os.path.join(client_dir, "images", "val")
    lbl_train_dir = os.path.join(client_dir, "labels", "train")
    lbl_val_dir = os.path.join(client_dir, "labels", "val")

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)
    os.makedirs(lbl_val_dir, exist_ok=True)

    def write_data(image_list, img_dir, lbl_dir):
        for img_name, boxes in image_list:
            src_img = os.path.join(base_data_path, "images", "100k", "train", img_name) 
            dst_img = os.path.join(img_dir, os.path.basename(img_name))
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)

            label_path = os.path.join(lbl_dir, os.path.splitext(os.path.basename(img_name))[0] + ".txt")
            with open(label_path, "w") as f:
                for box in boxes:
                    f.write(" ".join(map(str, box)) + "\n")

    # write train and val
    write_data(train_images, img_train_dir, lbl_train_dir)
    write_data(val_images, img_val_dir, lbl_val_dir)

    # create dataset.yaml
    yaml_path = Path(client_dir) / "dataset.yaml"
    content = {
        "train": str(img_train_dir),
        "val": str(img_val_dir),
        "nc": len(YOLO_CLASSES),
        "names": YOLO_CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, sort_keys=False)
    
    return yaml_path