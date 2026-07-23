"""
Federated (simulated) YOLOv8 training entrypoint -- basic FedAvg over a
sequential, single-process client simulation (see `obj_yolo/federated/`).

Expects the `<data-root>/client_<i>/{images,labels}/{train,val}` layout that
`obj_yolo.dataset.PrepareData` already produces (call it with
`clientCount=N` to generate it):

    uv run python -m obj_yolo.train_federated \\
        --data-root dataset/clients --num-clients 5 \\
        --rounds 20 --local-epochs 2 --scale n --imgsz 640 --batch 8 \\
        --val-images dataset/clients/client_0/images/val \\
        --val-labels dataset/clients/client_0/labels/val
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from obj_yolo.data.yolo_dataset import YoloDataset, collate_fn
from obj_yolo.federated.client import FedClient
from obj_yolo.federated.simulation import FedAvgSimulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulated federated YOLOv8 training (basic FedAvg).")
    p.add_argument("--data-root", required=True, help="root containing client_<i>/ folders")
    p.add_argument("--num-clients", type=int, required=True)
    p.add_argument("--nc", type=int, required=True, help="number of classes")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--fraction-fit", type=float, default=1.0)
    p.add_argument("--scale", default="n", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--val-images", default=None, help="optional central held-out val images dir")
    p.add_argument("--val-labels", default=None, help="optional central held-out val labels dir")
    p.add_argument("--val-batch", type=int, default=8)
    p.add_argument("--project", default="runs/federated")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data_root = Path(args.data_root)

    clients = []
    for i in range(args.num_clients):
        client_dir = data_root / f"client_{i}"
        clients.append(
            FedClient(
                client_id=f"client_{i}",
                images_dir=client_dir / "images" / "train",
                labels_dir=client_dir / "labels" / "train",
                nc=args.nc,
                scale=args.scale,
                imgsz=args.imgsz,
                device=device,
            )
        )

    val_loader = None
    if args.val_images and args.val_labels:
        val_ds = YoloDataset(args.val_images, args.val_labels, imgsz=args.imgsz, augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.val_batch, shuffle=False, collate_fn=collate_fn)

    sim = FedAvgSimulation(clients, nc=args.nc, scale=args.scale, imgsz=args.imgsz, device=device, seed=args.seed)
    result = sim.run(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr0=args.lr0,
        batch=args.batch,
        fraction_fit=args.fraction_fit,
        val_loader=val_loader,
        out_dir=args.project,
    )
    print(f"Federated training complete. best mAP50-95={result['best_map']:.4f}")


if __name__ == "__main__":
    main()
