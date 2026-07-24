"""
Config/argument validation for the training CLIs (`obj_yolo/train.py`,
`obj_yolo/val.py`, `obj_yolo/train_federated.py`). Every function here
raises `ValueError`/`FileNotFoundError` with a message meant to be shown
directly to a user, not a traceback -- callers should catch those exceptions
at the top of `main()` and exit cleanly (see `obj_yolo.train.main` for the
pattern).
"""
from pathlib import Path
from typing import Optional

import yaml


def _dir_has_files(path: Path) -> bool:
    """`path` exists, is a directory, and contains at least one entry.
    Never raises -- a path that exists but isn't a directory (e.g. `train:`
    accidentally pointing at a file) is just reported as "no files", not a
    crash from `.iterdir()`."""
    try:
        return path.exists() and any(path.iterdir())
    except OSError:
        return False


def validate_imgsz(imgsz: int) -> None:
    """The model's neck upsamples/downsamples by factors of 2 three times
    (stride 8/16/32); a non-multiple-of-32 imgsz produces mismatched concat
    shapes deep in the network instead of a clear error at startup."""
    if imgsz <= 0 or imgsz % 32 != 0:
        raise ValueError(f"--imgsz must be a positive multiple of 32 (got {imgsz})")


def validate_positive(value: int, name: str) -> None:
    """Generic `--<name>` must be a positive integer check (e.g. --epochs, --rounds)."""
    if value <= 0:
        raise ValueError(f"{name} must be positive (got {value})")


def validate_val_pair(val_images: Optional[str], val_labels: Optional[str]) -> None:
    """--val-images/--val-labels must be given together or not at all --
    otherwise centralized evaluation silently never runs with no warning."""
    if bool(val_images) != bool(val_labels):
        raise ValueError(
            f"--val-images and --val-labels must both be set or both omitted "
            f"(got val_images={val_images!r}, val_labels={val_labels!r})"
        )


def validate_data_yaml(path: str | Path) -> dict:
    """Load and sanity-check a data.yaml (train/val/nc/names), returning the parsed dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data.yaml not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    for key in ("train", "nc"):
        if key not in cfg:
            raise ValueError(f"{path} is missing required key '{key}'")

    train_dir = Path(cfg["train"])
    if not _dir_has_files(train_dir):
        raise FileNotFoundError(f"{path}'s train dir is missing, empty, or not a directory: {train_dir}")

    labels_dir = Path(str(train_dir).replace("images", "labels"))
    if not _dir_has_files(labels_dir):
        raise FileNotFoundError(f"{path}'s train labels dir is missing, empty, or not a directory: {labels_dir}")

    if int(cfg["nc"]) <= 0:
        raise ValueError(f"{path}'s 'nc' must be positive (got {cfg['nc']})")

    return cfg


def validate_client_root(data_root: str | Path, num_clients: int) -> None:
    """Check every `<data_root>/client_<i>/{images,labels}/train` exists and is non-empty."""
    data_root = Path(data_root)
    validate_positive(num_clients, "--num-clients")

    missing = []
    for i in range(num_clients):
        img_dir = data_root / f"client_{i}" / "images" / "train"
        lbl_dir = data_root / f"client_{i}" / "labels" / "train"
        if not _dir_has_files(img_dir):
            missing.append(str(img_dir))
        if not _dir_has_files(lbl_dir):
            missing.append(str(lbl_dir))
    if missing:
        raise FileNotFoundError(
            f"missing, empty, or non-directory client data paths under {data_root}: {missing}. "
            f"Run obj_yolo.dataset.PrepareData with clientCount={num_clients} first."
        )
