# dataset.py
"""data preparation file"""
import os
import shutil
import random
import yaml
from pathlib import Path

def load_data(
        base_path:str, 
        client_base_path:str, 
        client_id:str,
        client_data_count:int) -> Path:
    """
    creates data yaml and copies data from base data, to crete data
    prepared for yaml data.

    Args:
        base_path (str): base path that contains images and labals
        client_base_path (str): base path for clients data
        client_id (str): id of the client
        client_data_count (int): number of instances of data in each client

    Returns:
        Path: returns yaml path for prepared data.
    """
    if not Path(base_path).exists():
        raise FileNotFoundError("base path does not exist")
    
    base_i_path = Path(base_path) / "image_2" # can change this based on the data you downloaded
    base_l_path = Path(base_path) / "labels"

    if (not base_i_path.exists()) or (not base_l_path.exists()):
        raise FileNotFoundError("base path does not contain images or labels")

    if not Path(client_base_path).exists():
        os.mkdir(client_base_path)

    client_i_path = Path(client_base_path) / f"client_{client_id}" / "images"
    client_l_path = Path(client_base_path) / f"client_{client_id}" / "labels"

    if client_i_path.exists() and os.listdir(client_i_path) != None:
        return Path(client_base_path) / f"client_{client_id}" / "data.yaml"
    
    image_list = os.listdir(base_i_path)
    random.shuffle(image_list)

    for image in image_list[:client_data_count]:
        shutil.copy(base_i_path / image, client_i_path / "train" / image)
        label_file = image.replace(".png", ".txt") # Assign the new filename
        shutil.copy(base_l_path / label_file, client_l_path / "train" / label_file)
    
    for image in image_list[client_data_count:client_data_count+50]:
        shutil.copy(base_i_path / image, client_i_path / "val" / image)
        label_file = image.replace(".png", ".txt") # Assign the new filename
        shutil.copy(base_l_path / label_file, client_l_path / "val" / label_file)

    content = {
        "train" : str(client_i_path / "train"),
        "val" : str(client_i_path / "val"),
        "nc" : 8,
        "classes" : ["Car", "Pedestrian", "Van", "Cyclist", "Truck", "Misc", "Tram", "Person_sitting"]
    }

    yaml_path = Path(client_base_path) / f"client_{client_id}" / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, sort_keys=True)
    return yaml_path
