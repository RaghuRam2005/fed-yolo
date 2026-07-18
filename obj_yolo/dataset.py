# dataset.py
""" data preparation for clients """
import os
import json
import yaml
import shutil
import random
from pathlib import Path
from typing import Optional

CLASSES = {
    "bdd100k": {
        "bicycle": 0,
        "bus": 1,
        "car": 2,
        "motorcycle": 3,
        "other person": 4,
        "other vehicle": 5,
        "pedestrian": 6,
        "rider": 7,
        "traffic light": 8,
        "traffic sign": 9,
        "trailer": 10,
        "train": 11,
        "truck": 12,
    },
    "kitti": {
        "Car": 0,
        "Pedestrain": 1,
        "Van": 2,
        "Cyclist": 3,
        "Truck": 4,
        "Misc": 5,
        "Tram": 6,
        "Person_sitting": 8,
    },
}

# BDD100K images are always 1280x720
BDD100K_IMG_W: int = 1280
BDD100K_IMG_H: int = 720

# BDD100K weather attribute values used to tag clients for FedTag
BDD_WEATHER_TAGS = ["clear", "overcast", "snowy", "rainy", "cloudy", "foggy"]

class PrepareData:
    """
    Unified dataset preparation for KITTI and BDD100K federated learning.

    Usage
    -----
    >>> prep = PrepareData(
    ...     dataName="kitti2D",
    ...     baseDataPath=Path("./kitti_dataset"),
    ...     finalDataPath=Path("./dataset/clients"),
    ...     clientCount=5,
    ... )
    >>> prep.start()
    """

    def __init__(
        self,
        dataName: str,
        baseDataPath: Path,
        finalDataPath: Path,
        clientCount: int,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        """
        Args:
            dataName      : "kitti" or "bdd100k"
            baseDataPath  : Root of the raw downloaded dataset
            finalDataPath : Where per-client folders will be written
            clientCount   : Number of federated clients to create
            train_ratio   : Fraction of each client's images used for training
            seed          : RNG seed for reproducibility
        """
        if dataName not in CLASSES:
            raise ValueError(
                f"Unknown dataset '{dataName}'. Expected one of {list(CLASSES.keys())}"
            )
        self.dataName = dataName
        self.baseDataPath = Path(baseDataPath)
        self.finalDataPath = Path(finalDataPath)
        self.clientCount = clientCount
        self.train_ratio = train_ratio
        self.seed = seed
        self.rng = random.Random(seed)

        self._class_map: dict = CLASSES[dataName]
        self._id_to_name: dict = {v: k for k, v in self._class_map.items()}

    def start(self) -> None:
        """Dispatch to the correct dataset preparation routine."""
        self.finalDataPath.mkdir(parents=True, exist_ok=True)

        if self.dataName == "kitti":
            self._prepare_kitti_data()
        elif self.dataName == "bdd100k":
            self._prepare_bdd100k_data()
        else:
            raise ValueError(f"No preparation routine for dataset '{self.dataName}'")

    @staticmethod
    def _convert_yolo_label(
        box: dict, img_w: int, img_h: int
    ) -> tuple[float, float, float, float]:
        """Convert an absolute bounding box to YOLO normalised format."""
        x_center = (box["x1"] + box["x2"]) / 2 / img_w
        y_center = (box["y1"] + box["y2"]) / 2 / img_h
        width    = (box["x2"] - box["x1"]) / img_w
        height   = (box["y2"] - box["y1"]) / img_h
        return x_center, y_center, width, height

    def _make_client_dirs(self, client_dir: Path) -> tuple[Path, Path, Path, Path]:
        """Create and return the four standard subdirectories for a client."""
        img_train = client_dir / "images" / "train"
        img_val   = client_dir / "images" / "val"
        lbl_train = client_dir / "labels" / "train"
        lbl_val   = client_dir / "labels" / "val"
        for p in (img_train, img_val, lbl_train, lbl_val):
            p.mkdir(parents=True, exist_ok=True)
        return img_train, img_val, lbl_train, lbl_val

    def _write_yaml(self, client_dir: Path, img_train: Path, img_val: Path) -> None:
        """Write the data.yaml file for a client."""
        yaml.dump(
            {
                "train": str(img_train),
                "val":   str(img_val),
                "nc":    len(self._class_map),
                "names": self._id_to_name,
            },
            open(client_dir / "data.yaml", "w"),
            default_flow_style=False,
            sort_keys=True,
        )

    def _calc_train_val_counts(self, total: int) -> tuple[int, int]:
        """Compute safe (t_count, v_count) ensuring at least 1 val image."""
        if total == 1:
            return 1, 0
        t = int(total * self.train_ratio)
        v = total - t
        if v == 0:
            t -= 1
            v = 1
        return t, v

    def _random_split_into_clients(self, all_images: list[str]) -> dict[str, list[str]]:
        """Randomly shuffle and split images equally across clients."""
        images = all_images.copy()
        self.rng.shuffle(images)

        client_files: dict[str, list[str]] = {str(c): [] for c in range(self.clientCount)}
        chunk = len(images) // self.clientCount

        for idx in range(self.clientCount):
            start = idx * chunk
            # Last client absorbs any remainder images
            end = start + chunk if idx < self.clientCount - 1 else len(images)
            client_files[str(idx)] = images[start:end]

        return client_files

    def _assign_tags_to_clients(self) -> dict[str, str]:
        """Randomly assign one BDD100K weather tag to each client (tags may repeat)."""
        return {str(cid): self.rng.choice(BDD_WEATHER_TAGS) for cid in range(self.clientCount)}

    def _slice_images_by_tag(
        self,
        tag_images: dict[str, list[str]],
        client_tag_map: dict[str, str],
    ) -> dict[str, list[str]]:
        """Split each tag's image pool equally among the clients assigned that tag."""
        tag_to_clients: dict[str, list[str]] = {}
        for cid, tag in client_tag_map.items():
            tag_to_clients.setdefault(tag, []).append(cid)

        client_files: dict[str, list[str]] = {cid: [] for cid in client_tag_map}
        for tag, clients in tag_to_clients.items():
            pool = tag_images.get(tag, []).copy()
            self.rng.shuffle(pool)
            if not pool:
                print(f"  WARNING: tag '{tag}' has no images — {clients} will be empty")
                continue
            chunk = len(pool) // len(clients)
            for idx, cid in enumerate(clients):
                start = idx * chunk
                end = start + chunk if idx < len(clients) - 1 else len(pool)
                client_files[cid] = pool[start:end]

        return client_files

    def _write_client_tags(self, client_tag_map: dict[str, str]) -> None:
        """Persist the client_id -> weather_tag mapping for later use by FedTag."""
        with open(self.finalDataPath / "client_tags.json", "w") as f:
            json.dump(client_tag_map, f, indent=2)

    @staticmethod
    def load_client_tags(finalDataPath: Path) -> dict[str, str]:
        """Load the client_id -> weather_tag mapping written by _write_client_tags."""
        tag_path = Path(finalDataPath) / "client_tags.json"
        if not tag_path.exists():
            raise FileNotFoundError(
                f"client_tags.json not found at {tag_path}. Run PrepareData for bdd100k first."
            )
        with open(tag_path, "r") as f:
            return json.load(f)

    def _write_client_files(
        self,
        client_files: dict[str, list[str]],
        src_img_path: Path,
        src_lbl_path: Path,
    ) -> None:
        """
        Copy images and labels into per-client directories and write data.yaml.

        Args:
            client_files : Mapping client_id -> list of image filenames
            src_img_path : Directory containing source images
            src_lbl_path : Directory containing source YOLO .txt label files
        """
        for client_id, img_list in client_files.items():
            client_dir = self.finalDataPath / f"client_{client_id}"
            img_train, img_val, lbl_train, lbl_val = self._make_client_dirs(client_dir)

            t_count, v_count = self._calc_train_val_counts(len(img_list))

            # img_list is already shuffled from _random_split_into_clients
            train_files = img_list[:t_count]
            val_files   = img_list[t_count:t_count + v_count]

            def _copy(files: list[str], dst_img: Path, dst_lbl: Path) -> None:
                for fname in files:
                    shutil.copy(src_img_path / fname, dst_img / fname)
                    lbl_name = Path(fname).stem + ".txt"
                    lbl_src  = src_lbl_path / lbl_name
                    if lbl_src.exists():
                        shutil.copy(lbl_src, dst_lbl / lbl_name)
                    else:
                        # Create an empty label file to keep YOLO happy
                        (dst_lbl / lbl_name).touch()

            _copy(train_files, img_train, lbl_train)
            _copy(val_files,   img_val,   lbl_val)

            self._write_yaml(client_dir, img_train, img_val)
            print(f"  Client {client_id}: {t_count} train, {v_count} val images")

    def _prepare_kitti_data(self) -> None:
        """
        Prepare KITTI dataset (Kaggle YOLO-format version).

        Expected raw layout
        -------------------
        baseDataPath/
            training/
                image_02/   <- PNG/JPG images
            labels/         <- YOLO .txt files (same stem as images)
        """
        src_img_path = self.baseDataPath / "training" / "image_2"
        src_lbl_path = self.baseDataPath / "labels"

        if not src_img_path.exists():
            raise FileNotFoundError(f"KITTI image directory not found: {src_img_path}")
        if not src_lbl_path.exists():
            raise FileNotFoundError(f"KITTI label directory not found: {src_lbl_path}")

        all_images = sorted(
            f for f in os.listdir(src_img_path) if f.endswith((".png", ".jpg"))
        )
        if not all_images:
            raise ValueError(f"No images found in {src_img_path}")

        print(f"Found {len(all_images)} KITTI images — splitting across {self.clientCount} clients …")

        client_files = self._random_split_into_clients(all_images)

        print("\n=== Writing client data ===")
        self._write_client_files(client_files, src_img_path, src_lbl_path)

        print(
            f"\nKITTI preparation complete — {self.clientCount} clients "
            f"written to {self.finalDataPath}"
        )

    def _prepare_bdd100k_data(self) -> None:
        """
        Prepare BDD100K dataset for federated learning.

        Expected raw layout
        -------------------
        baseDataPath/
            bdd100k/
                images/
                    100k/
                        train/  <- JPEG images
                labels/
                    bdd100k_labels_images_train.json
        """
        src_img_path = self.baseDataPath / "bdd100k" / "images" / "100k" / "train"
        json_path    = (
            self.baseDataPath / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"
        )

        if not src_img_path.exists():
            raise FileNotFoundError(f"BDD100K image directory not found: {src_img_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"BDD100K label JSON not found: {json_path}")

        print("Loading BDD100K labels JSON (this may take a moment) …")
        with open(json_path, "r") as f:
            annotations: list[dict] = json.load(f)

        staging_lbl_path = self.finalDataPath / "_staging_labels"
        staging_lbl_path.mkdir(parents=True, exist_ok=True)

        valid_images: list[str] = []
        image_weather: dict[str, str] = {}
        skipped = 0

        for entry in annotations:
            img_name: str = entry.get("name", "")
            if not img_name:
                continue

            if not (src_img_path / img_name).exists():
                skipped += 1
                continue

            weather = str(entry.get("attributes", {}).get("weather", "")).lower().strip()
            if weather not in BDD_WEATHER_TAGS:
                weather = "clear"
            image_weather[img_name] = weather

            labels: list[dict] = entry.get("labels", []) or []
            yolo_lines: list[str] = []

            for lbl in labels:
                category: str         = lbl.get("category", "")
                box2d: Optional[dict] = lbl.get("box2d")

                if box2d is None:
                    continue

                cls_id: Optional[int] = self._class_map.get(category)
                if cls_id is None:
                    continue

                x_c, y_c, w, h = self._convert_yolo_label(box2d, BDD100K_IMG_W, BDD100K_IMG_H)

                # Clamp to [0, 1] to guard against annotation noise
                x_c = max(0.0, min(1.0, x_c))
                y_c = max(0.0, min(1.0, y_c))
                w   = max(0.0, min(1.0, w))
                h   = max(0.0, min(1.0, h))

                yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            stem = Path(img_name).stem
            with open(staging_lbl_path / (stem + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines))

            valid_images.append(img_name)

        if skipped:
            print(f"  Skipped {skipped} entries with no matching image on disk.")

        if not valid_images:
            raise ValueError("No valid labelled images found in BDD100K JSON.")

        print(
            f"Found {len(valid_images)} BDD100K images"
            f"splitting across {self.clientCount} clients by weather tag"
        )

        tag_images: dict[str, list[str]] = {}
        for img_name in valid_images:
            tag_images.setdefault(image_weather[img_name], []).append(img_name)

        client_tag_map = self._assign_tags_to_clients()
        print(f"  Client -> weather tag assignment: {client_tag_map}")
        client_files = self._slice_images_by_tag(tag_images, client_tag_map)
        self._write_client_tags(client_tag_map)

        print("\n=== Writing client data ===")
        self._write_client_files(client_files, src_img_path, staging_lbl_path)

        # Clean up staging labels
        shutil.rmtree(staging_lbl_path, ignore_errors=True)

        print(
            f"\nBDD100K preparation complete, {self.clientCount} clients "
        )


if __name__ == "__main__":
    PrepareData(
        dataName="kitti",
        baseDataPath=Path("./kitti_dataset"),
        finalDataPath=Path("./dataset/clients"),
        clientCount=5,
        train_ratio=0.8,
        seed=42,
    ).start()