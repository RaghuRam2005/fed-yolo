# strategy.py
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
            optimizer='AdamW',

            lr0=1e-4, 
            lrf=1e-5,
            cos_lr=True,
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
    @staticmethod
    def _add_l1_regularization_callback(model: Model, lam: float):
        """
        Robustly adds a callback to the model to apply L1 regularization on BN weights during training.
        This replaces the old, failing loss-wrapping method.
        """
        def on_loss_calculated(trainer):
            """Callback that receives the trainer object and adds the L1 penalty to its loss."""
            l1_penalty = torch.tensor(0., device=trainer.loss.device)
            # Find all BatchNorm2d layers and sum the absolute values of their weights (gamma)
            for module in trainer.model.modules():
                if isinstance(module, torch.nn.BatchNorm2d) and hasattr(module, 'weight') and module.weight is not None:
                    l1_penalty += torch.abs(module.weight).sum()
            
            # Add the scaled penalty directly to the trainer's loss attribute
            trainer.loss += lam * l1_penalty
            
        # Register the callback. Ultralytics' trainer will automatically call this function.
        model.add_callback("on_loss_calculated", on_loss_calculated)
        logging.info(f"Successfully added BN L1 regularization callback with lambda={lam}")
    
    @staticmethod
    def _calculate_adaptive_threshold(model:Model, sparsity_target:float = 0.5) -> float:
        """
        Calculates threshold based on sparcity

        Args:
            model (Model): our YOLO model
            sparcity_target (float, optional): Target Sparcity Ratio (0.5 means 50%). Defaults to 0.5.

        Returns:
            float: Calculated Threshold value
        """
        bn_weights = []
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d) and hasattr(module, 'weight') and module.weight is not None:
                bn_weights.append(module.weight.abs().flatten())
        
        if not bn_weights:
            logging.warning("No BatchNorm layers found for threshold calculation")
            return 1e-4
            
        all_weights = torch.cat(bn_weights)
        threshold = torch.quantile(all_weights, sparsity_target).item()
        logging.info(f"Calculated adaptive threshold: {threshold:.6f} for sparsity target: {sparsity_target}")
        return threshold
    
    @staticmethod
    def _apply_sparsification(model:Model, threshold:float) -> Dict[str, float]:
        """
        Apply sparsification to model and return sparsity statistics

        Args:
            model (Model): YOLO model instance
            threshold (float): Threshold to return 

        Returns:
            Dict[str, float]: sparced parameters
        """
        stats = {"total_params":0, "pruned_params":0, "bn_layers":0}

        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d) and hasattr(module, "weight"):
                    stats["bn_layers"] += 1

                    if module.weight is not None:
                        # count total parameters
                        stats["total_params"] += module.weight.numel()

                        # apply magnitude based pruning
                        weight_mask = module.weight.abs() > threshold
                        pruned_count = (~weight_mask).sum().item()
                        stats["pruned_params"] = pruned_count

                        if module.bias is not None:
                            module.bias *= weight_mask.float()
                    
                    logging.debug(f"Layer {name}: pruned {pruned_count}/{module.weight.numel()} weights")
            
        sparsity_ratio = stats["pruned_params"] / max(stats["total_params"], 1)
        stats["sparsity_ratio"] = sparsity_ratio
        
        logging.info(f"Applied sparsification: {sparsity_ratio:.3f} sparsity across {stats['bn_layers']} BN layers")
        return stats

    @staticmethod
    def client_train(
        model: Model,
        server_state: OrderedDict,
        client_id: int,
        epochs: int,
        data_path: Path,
        threshold: float = None,
        lam_bn_l1: float = 1e-4,
        sparsity_target: float = 0.5,
        enable_adaptive_threshold:bool = True
    ) -> Tuple[OrderedDict, Dict[str, torch.Tensor]]:
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        try:
            model.load_state_dict(server_state, strict=False)
            logging.info(f"Started Training the client {client_id}")
        except Exception as e:
            logging.error(f"client {client_id}: Failed to load server state with error: {e}")
            raise
        
        logging.info(f"client {client_id}: started training the model")

        # Add the L1 regularization callback to push BN weights towards zero
        try:
            FedWeg._add_l1_regularization_callback(model, lam=lam_bn_l1)
        except Exception as e:
            logging.error(f"client {client_id}: Failed to add L1 regularization callback, error: {e}")
            pass

        model.train(
            data=str(data_path), epochs=epochs, batch=8, save=False,
            project="fed_yolo", name=f"client_{client_id}_train",
            exist_ok=True, device=-1, optimizer='AdamW', lr0=1e-4, lrf=1e-5,
            cos_lr=True, warmup_epochs=1, augment=True,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
            scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.0, copy_paste=0.0
        )

        logging.info(f"Client {client_id}: Local training complete.")

        if threshold is None or enable_adaptive_threshold:
            threshold = FedWeg._calculate_adaptive_threshold(model, sparsity_target)
        
        sparsity_stats = FedWeg._apply_sparsification(model, threshold)

        try:
            pruned_state = OrderedDict({k:v.clone() for k, v in model.state_dict().items()} )
        except Exception as e:
            logging.error(f"Client {client_id}: Failed to extract model state: {e}")
            raise

        logging.info(f"Client {client_id} training complete with {sparsity_stats['sparsity_ratio']:.3f} sparsity")
        
        # Safe callback removal
        try:
            model.remove_callback("on_loss_calculated")
            logging.debug(f"Client {client_id}: Successfully removed L1 callback")
        except (AttributeError, KeyError) as e:
            logging.debug(f"Client {client_id}: Could not remove callback (this is normal): {e}")
        
        return pruned_state, sparsity_stats

    @staticmethod
    def aggregate_sparse_updates(base_params:OrderedDict,
                                 client_params_list:List[OrderedDict],
                                 client_data_sizes:List[int],
                                 client_sparsity_stats:List[Dict[str, float]]) -> OrderedDict:
        """
        Aggregates sparse updates from clients using FedWeg Algorithm

        Args:
            base_params (OrderedDict): server/global model before aggregation
            client_params_list (List[OrderedDict]): each client's pruned params
            client_data_sizes (List[int]): number of samples per client
            client_sparsity_stats (List[Dict[str, float]]): sparsity statistics per client

        Returns:
            OrderedDict: aggregated params
        """
        if not client_params_list:
            logging.warning("No client updates provided, returning base parameters")
            return base_params
        
        assert len(client_params_list) == len(client_data_sizes), "updates size doesn't match"
        
        aggregated = OrderedDict()
        total_data = sum(client_data_sizes)
        if total_data == 0:
            raise ValueError("Total data size cannot be zero")

        # iterate over each parameter
        inverse_sparsity_weights = None
        if client_sparsity_stats:
            try:
                sparsity_rates = [stats.get('sparsity_ratio', 0.0) for stats in client_sparsity_stats]
                # Inverse sparsity weighting: less sparse models get higher weight
                inverse_sparsity_weights = [(1.0 - s + 1e-6) for s in sparsity_rates]
                total_inv_sparsity = sum(inverse_sparsity_weights)
                inverse_sparsity_weights = [w / total_inv_sparsity for w in inverse_sparsity_weights]
                logging.info(f"Using FedWeg inverse sparsity weighting: {inverse_sparsity_weights}")
            except Exception as e:
                logging.warning(f"Failed to calculate inverse sparsity weights: {e}, falling back to data-size weighting")
                inverse_sparsity_weights = None
        
        # iterate over each parameter
        for k in base_params.keys():
            agg_param = torch.zeros_like(base_params[k])
            contributing_clients = 0
            total_weight = 0.0

            for i, (client_params, n_i) in enumerate(zip(client_params_list, client_data_sizes)):
                try:
                    if k not in client_params:
                        logging.debug(f"Parameter {k} missing from client {i}, skipping")
                        continue
                    
                    client_val = client_params[k]
                    
                    # Skip if parameter is all zeros (no contribution)
                    if torch.count_nonzero(client_val) == 0:
                        logging.debug(f"Parameter {k} from client {i} is all zeros, skipping")
                        continue
                    
                    # Calculate weight: combine data size and inverse sparsity (FedWeg approach)
                    if inverse_sparsity_weights:
                        weight = (n_i / total_data) * inverse_sparsity_weights[i]
                    else:
                        weight = n_i / total_data
                    
                    agg_param += weight * client_val
                    contributing_clients += 1
                    total_weight += weight
                    
                except Exception as e:
                    logging.error(f"Error processing parameter {k} from client {i}: {e}")
                    continue

            # Normalize by total contributing weight if needed
            if contributing_clients > 0 and total_weight > 0:
                if abs(total_weight - 1.0) > 1e-6:  # Renormalize if weights don't sum to 1
                    agg_param /= total_weight
                aggregated[k] = agg_param
                logging.debug(f"Parameter {k}: aggregated from {contributing_clients} clients")
            else:
                # No client contributed - use base parameters (FedWeg fallback)
                aggregated[k] = base_params[k].clone()
                logging.debug(f"Parameter {k}: no contributions, using base value")

        if len(aggregated) != len(base_params):
            logging.error(f"Aggregation size mismatch: {len(aggregated)} vs {len(base_params)}")
            raise RuntimeError("Aggregation failed: parameter count mismatch")
        
        logging.info(f"Successfully aggregated parameters from {len(client_params_list)} clients")
        return aggregated
        