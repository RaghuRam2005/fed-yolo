import os
from pathlib import Path
import shutil
import random
import yaml

# import configuration
from config import (
    BASE_DATA_PATH,

    CLIENT_DATA_PATH,
    CLIENT_DATA_COUNT,
    CLIENTS_COUNT,

    GLOBAL_DATA_PATH,
    GLOBAL_DATA_COUNT,

    NC,
    CLASSES,

    DATA_SPLIT
)


# Ensure the base data path exists
if not BASE_DATA_PATH or not os.path.exists(BASE_DATA_PATH):
    raise Exception(
        "BASE_DATA_PATH not found. Please set it in your .env file or environment."
    )

def write_yolo_yaml(dataset_path: Path):
    """
    Writes a YOLOv8 compatible YAML file for a given dataset path.

    Args:
        dataset_path (Path): The root path of the specific dataset
                             (e.g., ./yolo_datasets/global_dataset).
    """
    yaml_content = { 
        "train": f"{dataset_path}\\images\\train",  # path to training images
        "val": f"{dataset_path}\\images\\val",      # path to validation images
        "nc": NC,
        "names": CLASSES,
    }
    
    yaml_file_path = dataset_path / "data.yaml"
    with open(yaml_file_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Generated YAML file at: {yaml_file_path}")


def prepare_data(base_data_path: str) -> None:
    """
    Prepares and splits data into training and validation sets for global and
    client-specific datasets, and generates corresponding YOLO YAML files.
    """
    base_path = Path(base_data_path)
    img_dir = base_path / "image_2"
    label_dir = base_path / "labels"

    if not img_dir.exists() or not label_dir.exists():
        raise Exception(f"Source image or label directory does not exist in {base_path}")
    
    # Get all image files and shuffle them for random distribution
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    random.shuffle(image_files)

    # Split files into training and validation sets
    split_index = int(len(image_files) * DATA_SPLIT)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    print(f"Total files: {len(image_files)}. Training files: {len(train_files)}, Validation files: {len(val_files)}")

    # --- Process Global Dataset ---
    print("\n--- Preparing Global Dataset ---")
    global_train_files = train_files[:GLOBAL_DATA_COUNT]
    
    # Calculate validation count based on the training split ratio
    global_val_count = int(GLOBAL_DATA_COUNT * ((1 - DATA_SPLIT) / DATA_SPLIT))
    global_val_files = val_files[:global_val_count]
    
    # Create directories
    (GLOBAL_DATA_PATH / "images" / "train").mkdir(parents=True, exist_ok=True)
    (GLOBAL_DATA_PATH / "images" / "val").mkdir(parents=True, exist_ok=True)
    (GLOBAL_DATA_PATH / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (GLOBAL_DATA_PATH / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Copy training files
    for file in global_train_files:
        shutil.copy(img_dir / file, GLOBAL_DATA_PATH / "images/train" / file)
        shutil.copy(label_dir / file.replace('.png', '.txt'), GLOBAL_DATA_PATH / "labels/train" / file.replace('.png', '.txt'))

    # Copy validation files
    for file in global_val_files:
        shutil.copy(img_dir / file, GLOBAL_DATA_PATH / "images/val" / file)
        shutil.copy(label_dir / file.replace('.png', '.txt'), GLOBAL_DATA_PATH / "labels/val" / file.replace('.png', '.txt'))

    print(f"Prepared global dataset with {len(global_train_files)} train and {len(global_val_files)} val samples.")
    write_yolo_yaml(GLOBAL_DATA_PATH)
    
    # Keep track of used files
    processed_train_count = len(global_train_files)
    processed_val_count = len(global_val_files)

    # --- Process Client Datasets ---
    print("\n--- Preparing Client Datasets ---")
    for client_id in range(CLIENTS_COUNT):
        client_path = CLIENT_DATA_PATH / f"client_{client_id}"
        
        # Get slices for this client
        client_train_files = train_files[processed_train_count : processed_train_count + CLIENT_DATA_COUNT]
        
        client_val_count = int(CLIENT_DATA_COUNT * ((1 - DATA_SPLIT) / DATA_SPLIT))
        client_val_files = val_files[processed_val_count : processed_val_count + client_val_count]

        if not client_train_files:
            print(f"Warning: No more training files available for client {client_id}.")
            continue

        # Create directories
        (client_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (client_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (client_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (client_path / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # Copy training files
        for file in client_train_files:
            shutil.copy(img_dir / file, client_path / "images/train" / file)
            shutil.copy(label_dir / file.replace('.png', '.txt'), client_path / "labels/train" / file.replace('.png', '.txt'))

        # Copy validation files
        for file in client_val_files:
            shutil.copy(img_dir / file, client_path / "images/val" / file)
            shutil.copy(label_dir / file.replace('.png', '.txt'), client_path / "labels/val" / file.replace('.png', '.txt'))
        
        print(f"Prepared data for client {client_id} with {len(client_train_files)} train and {len(client_val_files)} val samples.")
        write_yolo_yaml(client_path)

        # Update processed counts
        processed_train_count += len(client_train_files)
        processed_val_count += len(client_val_files)

if __name__ == "__main__":
    prepare_data(BASE_DATA_PATH)
    print("\nData preparation complete.")

