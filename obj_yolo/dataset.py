# dataset.py
"""data preparation file"""
import os
import shutil
import json
import yaml
import random
from pathlib import Path
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count

KITTI_NC=8
KITTI_CLASSES=["Car", "Pedestrian", "Van", "Cyclist", "Truck", "Misc", "Tram", "Person_sitting"]

BDD_NC=10
BDD_CLASSES=["bus", "light", "sign", "person", "bike", "truck", "motor", "car", "train", "rider"]
BDD_ID={name:i for i, name in enumerate(BDD_CLASSES)}

class KittiData:
    def __init__(
            self,
            base_data_path:str,
            prep_data_path:str,
            exist_ok:bool=False
        ) -> None:
        self.base_data_path = base_data_path
        self.prep_data_path = prep_data_path
        self.exist_ok = exist_ok
        if not Path(base_data_path).exists():
            raise Exception(f"Base Data not found at {self.base_data_path}")
        if Path(prep_data_path).exists() and not self.exist_ok:
            [shutil.rmtree(p) if p.is_dir() else p.unlink() for p in Path(prep_data_path).iterdir()]
        if not Path(prep_data_path).exists():
            Path(prep_data_path).mkdir(parents=True)

    def prepare_global_data(self, data_count:int) -> str:
        pass
    
    def prepare_client_data(self, client_id:int, train_data_count:int, val_data_count:int) -> str:
        base_img_path = Path(self.base_data_path) / "image_2"
        base_label_path = Path(self.base_data_path) / "labels"

        if not base_img_path.exists() or not base_label_path.exists():
            raise Exception(f"Base Data not found at {self.base_data_path}")
        
        client_data_path = Path(self.prep_data_path) / f"client_{client_id}"
        client_img_path = client_data_path / "images"
        client_label_path = client_data_path / "labels"
        client_yaml_path = client_data_path / "data.yaml"

        if client_yaml_path.exists():
            return client_yaml_path
        
        (client_img_path / "train").mkdir(parents=True, exist_ok=self.exist_ok)
        (client_label_path / "train").mkdir(parents=True, exist_ok=self.exist_ok)
        (client_img_path / "val").mkdir(parents=True, exist_ok=self.exist_ok)
        (client_label_path / "val").mkdir(parents=True, exist_ok=self.exist_ok)

        image_list = os.listdir(base_img_path)
        random.shuffle(image_list)
        train_file_list = image_list[:train_data_count]
        val_file_list = image_list[train_data_count:train_data_count+val_data_count]

        for file in train_file_list:
            shutil.copy(base_img_path / file, client_img_path / "train" / file)
            label = file.replace(".png", ".txt")
            shutil.copy(base_label_path / label, client_label_path / "train" / label)

        for file in val_file_list:
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

        return str(client_yaml_path)

class BddData:
    def __init__(
            self,
            base_data_path:str,
            prep_data_path:str,
            exist_ok:bool=False
        ) -> None:
        self.base_data_path = base_data_path
        self.prep_data_path = prep_data_path
        self.exist_ok = exist_ok
        if not Path(base_data_path).exists():
            raise Exception(f"Base Data not found at {self.base_data_path}")
        if Path(prep_data_path).exists() and not self.exist_ok:
            [shutil.rmtree(p) if p.is_dir() else p.unlink() for p in Path(prep_data_path).iterdir()]
        if not Path(prep_data_path).exists():
            Path(prep_data_path).mkdir(parents=True)
    
    def bbox_to_yolo(self, bbox, img_w, img_h, category):
        cls_id = BDD_ID.get(category)
        if cls_id is None:
            return None
        x1, y1, w, h = bbox
        return (
                cls_id,
                (x1 + w/2) / img_w,
                (y1 + h/2) / img_h,
                w / img_w,
                h / img_h
        )

    def process_labels(self, item):
        img_name = item["name"]
        img_w, img_h = 1280, 720

        yolo_boxes = [
                self.bbox_to_yolo(
                    [
                        obj["box2d"]["x1"],
                        obj["box2d"]["y1"],
                        obj["box2d"]["x2"] - obj["box2d"]["x1"],
                        obj["box2d"]["y2"] - obj["box2d"]["y1"],
                    ],
                    img_w,
                    img_h,
                    obj["category"]
                )
                for obj in item.get("labels", [])
                if "box2d" in obj and obj["category"] in BDD_ID
        ]

        yolo_boxes = [box for box in yolo_boxes if box]

        weather = item["attributes"].get("weather", "unknown")
        scene = item["attributes"].get("scene", "unknown")

        return weather, scene, (img_name, yolo_boxes)


    def create_tag_dicts(self) -> Tuple[Dict]:
        label_file_path = Path(self.base_data_path) / "labels" / "label_train.json"
        if not label_file_path.exists():
            raise Exception(f"Label file not found at {label_file_path}")

        with open(label_file_path, "r") as f:
            label_data = json.load(f)

        weather_dict, scene_dict = {}, {}

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self.process_labels, label_data)

        for weather, scene, entry in results:
            weather_dict.setdefault(weather, []).append(entry)
            scene_dict.setdefault(scene, []).append(entry)

        return weather_dict, scene_dict
    
    def prepare_client_data(self, client_id:int, train_img_list:List[str], val_img_list:List[str]) -> str:
        base_img_path = Path(self.base_data_path) / "100k" / "train"
        if not Path(base_img_path).exists():
            raise Exception(f"base images not found at {base_img_path}")

        client_data_path = Path(self.prep_data_path) / f"client_{client_id}"

        client_img_path = client_data_path / "images"
        client_label_path = client_data_path / "labels"

        (client_img_path / "train").mkdir(parents=True, exist_ok=self.exist_ok)
        (client_img_path / "val").mkdir(parents=True, exist_ok=self.exist_ok)
        (client_label_path / "train").mkdir(parents=True, exist_ok=self.exist_ok)
        (client_label_path / "val").mkdir(parents=True, exist_ok=self.exist_ok)

        def write_data(image_list:List[str], img_dir:Path, label_dir:Path):
            for img_name, boxes in image_list:
                src_img = base_img_path / img_name
                dst_img = img_dir / img_name
                if src_img.exists():
                    shutil.copy(src_img, dst_img)
                
                label_name = str(img_name).replace(".jpg", ".txt")
                label_path = label_dir / label_name

                with open(label_path, "w") as f:
                    for box in boxes:
                        f.write(" ".join(map(str, box)) + "\n")
        
        write_data(train_img_list, client_img_path / "train", client_label_path / "train")
        write_data(val_img_list, client_img_path / "val", client_label_path / "val")

        # create dataset yaml
        
        client_yaml_path = client_data_path / "data.yaml"

        content = {
                "train":str(client_img_path / "train"),
                "val" : str(client_img_path / "val"),
                "nc" : BDD_NC,
                "names": BDD_CLASSES,
        }

        with open(client_yaml_path, "w") as f:
            yaml.dump(content, f, sort_keys=False)

        return client_yaml_path
