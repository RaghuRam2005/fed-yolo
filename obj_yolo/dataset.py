# dataset.py
""" data preparation for clients """
import os
import yaml
import shutil
import random
from pathlib import Path

KITTI_NC=8
KITTI_CLASSES=["Car", "Pedestrian", "Van", "Cyclist", "Truck", "Misc", "Tram", "Person_sitting"]

def prepare_kitti_data(num_clients:int=3, train_data_count:int=1000, test_data_count:int=200) -> None:
    BASE_LIB_PATH = os.path.abspath(os.path.dirname(__file__))
    BASE_DIR_PATH = os.path.dirname(BASE_LIB_PATH)
    BASE_DATA_PATH = Path(BASE_DIR_PATH) / "dataset"

    prepared_data_path = BASE_DATA_PATH / "clients"
    if not prepared_data_path.exists():
        prepared_data_path.mkdir(parents=True, exist_ok=True)

    base_img_path = BASE_DATA_PATH / "training" / "image_2"
    base_label_path = BASE_DATA_PATH / "labels"
    if not base_img_path.exists() or not base_label_path.exists():
        raise Exception("Base data path does not exist")

    image_files = os.listdir(base_img_path)
    random.shuffle(image_files)
    completed_images = 0

    for client in range(num_clients):
        client_data_path = prepared_data_path / f"client_{client}"
        client_yaml_path = client_data_path / "data.yaml"
        if client_yaml_path.exists():
            continue
        
        client_img_path = client_data_path / "images"
        client_label_path = client_data_path / "labels"

        (client_img_path / "train").mkdir(parents=True, exist_ok=True)
        (client_label_path / "train").mkdir(parents=True, exist_ok=True)
        (client_img_path / "val").mkdir(parents=True, exist_ok=True)
        (client_label_path / "val").mkdir(parents=True, exist_ok=True)

        train_files = image_files[completed_images:completed_images+train_data_count]
        completed_images += train_data_count
        val_files = image_files[completed_images:completed_images+test_data_count]
        completed_images += test_data_count

        for file in train_files:
            shutil.copy(base_img_path / file, client_img_path / "train" / file)
            label = file.replace(".png", ".txt")
            shutil.copy(base_label_path / label, client_label_path / "train" / label)

        for file in val_files:
            shutil.copy(base_img_path / file, client_img_path / "val" / file)
            label = file.replace(".png", ".txt")
            shutil.copy(base_label_path / label, client_label_path / "val" / label)
        
        content = {
                "train" : str(client_img_path / "train" ),
                "val" : str(client_img_path / "val" ),
                "nc" : KITTI_NC,
                "classes" : KITTI_CLASSES
        }
        
        with open(client_yaml_path, "w") as f:
            yaml.dump(content, f, sort_keys=True)

    print("All clients data preparation complete")

if __name__ == "__main__":
    prepare_kitti_data(
        num_clients=5,
        train_data_count=2000,
        test_data_count=500,
    )
