# dataset.py
"""data preparation file"""
import os
import shutil
import yaml
from pathlib import Path
from typing import List

def load_data(
        base_path:str,
        client_base_path:str,
        client_id:str,
        train_images: List[str],
        val_images: List[str]) -> Path:
    """
    Creates data yaml and copies data from base data, using pre-partitioned
    lists of image files.

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
