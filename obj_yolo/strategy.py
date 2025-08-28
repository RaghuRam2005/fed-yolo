import torch
import logging
from collections import OrderedDict
from typing import List, Tuple, Dict, Any
from pathlib import Path
from ultralytics.engine.model import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FedAvg:
    @staticmethod
    def aggregate_fit(new_parameters: List[OrderedDict], client_data_counts: List[int]) -> OrderedDict:
        assert len(new_parameters) == len(client_data_counts), "mismatched clients vs counts"
        if not new_parameters:
            raise ValueError("no client parameters provided")

        total_data_count = int(sum(int(x) for x in client_data_counts))
        if total_data_count == 0:
            raise ValueError("total_data_count is zero; cannot aggregate")

        # initialize aggregated with zeros of first client
        aggregated = OrderedDict({k: torch.zeros_like(v) for k, v in new_parameters[0].items()})

        for client_sd, n in zip(new_parameters, client_data_counts):
            w = float(n) / float(total_data_count)
            for k, v in client_sd.items():
                # skip num_batches_tracked counters
                if k.endswith("num_batches_tracked"):
                    continue
                if aggregated[k].dtype.is_floating_point and v.dtype.is_floating_point:
                    aggregated[k] += v * w
                else:
                    # For non-floating params (rare), take the first client's value if not yet set
                    if aggregated[k].numel() == 0 or torch.count_nonzero(aggregated[k]).item() == 0:
                        aggregated[k] = v.clone()

        return aggregated

    @staticmethod
    def client_train(model: Model,
                     client_id: int,
                     epochs: int,
                     data_path: Path) -> OrderedDict:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        logging.info(f"Started local model training (client {client_id})")
        model.train(
            data=str(data_path),
            epochs=epochs,
            batch=8,
            save=False,
            project="fed_yolo",
            name=f"client_{client_id}_train",
            exist_ok=True,
            device=-1,
            cos_lr=True,
            warmup_epochs=3,
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        logging.info("Local training complete.")
        return model.state_dict()


class FedWeg:
    """
    FedWeg: L1 on BN gamma during client training, client sends sparse BN-gamma updates (mask + masked values)
    while other params are sent as dense deltas. Server aggregates per-element using only clients that contributed.
    """

    @staticmethod
    def _wrap_loss_with_bn_l1(model: Model, lam: float = 1e-5) -> bool:
        """
        Try to wrap model's loss/criterion to add lam * sum(abs(BN_gamma)).
        Returns True if wrapping succeeded, False otherwise.
        This approach tries a few common attribute paths for ultralytics models.
        """
        def bn_l1_term() -> torch.Tensor:
            device = None
            total = None
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d) and getattr(m, "weight", None) is not None:
                    if device is None:
                        device = m.weight.device
                    if total is None:
                        total = torch.sum(torch.abs(m.weight))
                    else:
                        total = total + torch.sum(torch.abs(m.weight))
            if total is None:
                # no BN gamma found; return zero tensor on some device
                return torch.tensor(0.0, device=device if device is not None else torch.device("cpu"))
            return total

        # Try a few places where ultralytics stores the loss function
        candidates = []
        if hasattr(model, "criterion"):
            candidates.append(("model.criterion", model, "criterion"))
        # sometimes the internal yolo model is at model.model
        if hasattr(model, "model") and hasattr(model.model, "criterion"):
            candidates.append(("model.model.criterion", model.model, "criterion"))
        # some versions name loss function differently
        if hasattr(model, "loss_fn"):
            candidates.append(("model.loss_fn", model, "loss_fn"))
        if hasattr(model, "model") and hasattr(model.model, "loss_fn"):
            candidates.append(("model.model.loss_fn", model.model, "loss_fn"))

        wrapped_one = False
        for name, owner, attr in candidates:
            orig = getattr(owner, attr, None)
            if orig is None or not callable(orig):
                continue

            def make_wrapped(orig_fn):
                def wrapped(*args, **kwargs):
                    result = orig_fn(*args, **kwargs)
                    # YOLO loss often returns either loss tensor or (loss, components...)
                    if isinstance(result, tuple):
                        loss = result[0]
                        rest = result[1:]
                    else:
                        loss = result
                        rest = ()
                    # ensure loss is tensor
                    if not torch.is_tensor(loss):
                        loss = torch.as_tensor(loss, device=next(model.parameters()).device, dtype=torch.float32)
                    loss = loss + lam * bn_l1_term()
                    if rest:
                        return (loss, *rest)
                    return loss
                return wrapped

            setattr(owner, attr, make_wrapped(orig))
            logging.info(f"Wrapped loss at {name} to include BN-L1")
            wrapped_one = True
            break

        if not wrapped_one:
            logging.warning("Failed to wrap loss to add BN L1. You may need a custom Trainer or different ultralytics version.")
        return wrapped_one

    @staticmethod
    def _make_sparse_update(client_state: OrderedDict, server_state: OrderedDict, mask_threshold: float
                            ) -> Tuple[OrderedDict, Dict[str, torch.Tensor]]:
        """
        Computes the delta = client_state - server_state elementwise.
        For BN gamma params (heuristic: param name endswith '.weight' and 1D tensor) apply mask threshold:
            mask = abs(gamma_delta) >= mask_threshold  (or we can threshold on gamma itself)
            sparse_delta = delta * mask.float()
        Returns:
            sparse_update: OrderedDict of deltas (for BN deltas masked; other deltas full)
            masks: dict param_name->mask (bool tensor) for BN params (others not included)
        Note: both sparse_update and masks contain tensors on the same device/dtype as server_state.
        """
        sparse_update = OrderedDict()
        masks: Dict[str, torch.Tensor] = {}

        for k, server_val in server_state.items():
            client_val = client_state.get(k, None)
            if client_val is None:
                # client missing key: treat as no update (zeros)
                sparse_update[k] = torch.zeros_like(server_val)
                continue

            # compute delta
            if client_val.dtype != server_val.dtype or client_val.device != server_val.device:
                client_val = client_val.to(device=server_val.device, dtype=server_val.dtype)

            delta = client_val - server_val

            # Heuristic to detect BN gamma: 1D parameter named "*.weight" and likely small length
            is_bn_gamma = (k.endswith(".weight") and delta.dim() == 1)
            if is_bn_gamma:
                # build mask based on absolute gamma values after client training (not delta)
                # For simplicity we threshold delta's magnitude (could be gamma's magnitude instead)
                mask = delta.abs() >= mask_threshold
                masks[k] = mask.clone().to(dtype=torch.bool)
                if mask.any():
                    # keep masked values (masked zeros elsewhere)
                    masked_delta = delta * mask.to(dtype=delta.dtype)
                    sparse_update[k] = masked_delta
                else:
                    # all zeros
                    sparse_update[k] = torch.zeros_like(delta)
            else:
                # send dense delta for other params
                sparse_update[k] = delta

        return sparse_update, masks

    @staticmethod
    def client_train(model: Model,
                     server_state: OrderedDict,
                     client_id: int,
                     epochs: int,
                     data_path: Path,
                     mask_threshold: float = 1e-3,
                     lam_bn_l1: float = 1e-5) -> Tuple[OrderedDict, Dict[str, torch.Tensor]]:
        """
        Trains the client model (local copy), applies BN-L1 via wrapped loss if possible,
        and returns a sparse_update (deltas) and the masks for BN gammas.
        Args:
            model: ultralytics Model (local)
            server_state: the global server state dict used to compute delta
            client_id, epochs, data_path: training params
            mask_threshold: threshold applied to per-element absolute delta to decide sparsity
            lam_bn_l1: L1 lambda applied to BN gamma during training
        Returns:
            sparse_update: OrderedDict of deltas (client - server) where BN deltas are masked
            masks: dict of bn param_name -> boolean mask tensor
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # Put server_state into client model before training (start from global)
        model.load_state_dict(server_state, strict=False)
        logging.info(f"Client {client_id}: loaded global model and will train locally")

        # Try to wrap loss to include BN L1
        FedWeg._wrap_loss_with_bn_l1(model, lam=lam_bn_l1)

        # Do training
        model.train(
            data=str(data_path),
            epochs=epochs,
            batch=8,
            save=False,
            project="fed_yolo",
            name=f"client_{client_id}_train",
            exist_ok=True,
            device=-1,
            cos_lr=True,
            warmup_epochs=3,
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )

        logging.info(f"Client {client_id}: local training complete")

        # After training: compute sparse delta relative to server_state
        client_state = model.state_dict()
        sparse_update, masks = FedWeg._make_sparse_update(client_state, server_state, mask_threshold=mask_threshold)

        # Additionally zero-out BN gamma in client's state where masked (keeps local consistency)
        with torch.no_grad():
            for k, mask in masks.items():
                # find corresponding param in model (if present)
                if k in client_state:
                    client_state[k].mul_(mask.to(dtype=client_state[k].dtype))

        # Return the sparse delta (client - server) and masks
        return sparse_update, masks

    @staticmethod
    def aggregate_sparse_updates(sparse_updates: List[OrderedDict],
                                 masks_list: List[Dict[str, torch.Tensor]],
                                 client_data_counts: List[int],
                                 server_state: OrderedDict) -> OrderedDict:
        """
        Aggregate a list of sparse_updates (deltas) returned by clients and the associated masks_list.
        Behavior:
          - For BN-gamma params (1D .weight and present in masks): do per-element weighted average only using clients
            that contributed that element (i.e., their mask element is True). Denominator is sum of client weights that
            had mask True for that element.
          - For other params: weighted average of deltas over all clients (standard FedAvg on deltas).
        Returns:
            new_server_state = server_state + aggregated_delta
        """
        assert len(sparse_updates) == len(client_data_counts) == len(masks_list)
        if not sparse_updates:
            raise ValueError("No updates to aggregate")

        total = int(sum(int(x) for x in client_data_counts))
        if total == 0:
            raise ValueError("total_data_count is zero; cannot aggregate")

        # weights per client
        weights = [float(n) / float(total) for n in client_data_counts]

        # initialize aggregated_delta zeros with server shapes/dtypes
        aggregated_delta = OrderedDict({k: torch.zeros_like(v) for k, v in server_state.items()})

        num_clients = len(sparse_updates)
        for k in aggregated_delta.keys():
            # BN gamma heuristic
            if k.endswith(".weight") and aggregated_delta[k].dim() == 1 and any((k in masks) for masks in masks_list):
                # We'll accumulate numerator and denominator per element
                numerator = torch.zeros_like(aggregated_delta[k], dtype=torch.float64)  # use higher precision for sum
                denom = torch.zeros_like(aggregated_delta[k], dtype=torch.float64)
                for client_idx, (upd, masks, w) in enumerate(zip(sparse_updates, masks_list, weights)):
                    client_delta = upd.get(k, None)
                    if client_delta is None:
                        continue
                    # Ensure types match
                    if client_delta.dtype != numerator.dtype:
                        client_delta = client_delta.to(dtype=numerator.dtype)

                    mask = masks.get(k, None)
                    if mask is None:
                        # this client did not provide mask for k -> does not contribute
                        continue
                    mask_f = mask.to(dtype=numerator.dtype)
                    # Add weighted contribution only where mask==1
                    numerator += (client_delta.to(dtype=numerator.dtype) * w)
                    denom += (mask_f * w)

                # safe divide: where denom > 0 set value = numerator/denom else 0
                denom_nonzero = denom > 0
                if denom_nonzero.any():
                    # convert back to server dtype
                    out = torch.zeros_like(aggregated_delta[k], dtype=aggregated_delta[k].dtype)
                    # compute only where denom nonzero
                    out_float = torch.zeros_like(aggregated_delta[k], dtype=numerator.dtype)
                    out_float[denom_nonzero] = numerator[denom_nonzero] / denom[denom_nonzero]
                    aggregated_delta[k] = out_float.to(dtype=aggregated_delta[k].dtype)
                else:
                    # no client contributed to this param (leave zeros)
                    aggregated_delta[k] = torch.zeros_like(aggregated_delta[k])
            else:
                # dense param: weighted average across all clients
                agg = torch.zeros_like(aggregated_delta[k], dtype=torch.float64)
                for upd, w in zip(sparse_updates, weights):
                    client_delta = upd.get(k, None)
                    if client_delta is None:
                        # treat as zero delta
                        continue
                    agg += client_delta.to(dtype=torch.float64) * w
                aggregated_delta[k] = agg.to(dtype=aggregated_delta[k].dtype)

        # apply aggregated_delta to server_state to form new state
        new_server_state = OrderedDict()
        for k, server_val in server_state.items():
            delta = aggregated_delta.get(k, torch.zeros_like(server_val))
            # ensure dtype/device match
            if delta.dtype != server_val.dtype:
                delta = delta.to(dtype=server_val.dtype)
            new_server_state[k] = server_val + delta

        return new_server_state
