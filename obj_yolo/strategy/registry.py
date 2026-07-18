""" Strategy registry: pick a federated algorithm via the `strategy-name` run-config key
instead of hardcoding a strategy class in server.py. """
from typing import Any, Callable

from obj_yolo.strategy.fedavg import CustomFedAvg
from obj_yolo.strategy.fedadam import CustomFedAdam
from obj_yolo.strategy.fedtag import FedTag


def _build_fedavg(*, fraction_train: float, dataset_name: str, run_config: Any) -> CustomFedAvg:
    return CustomFedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=1,
        min_train_nodes=3,
        min_evaluate_nodes=3,
        min_available_nodes=3,
        dataset_name=dataset_name,
    )


def _build_fedadam(*, fraction_train: float, dataset_name: str, run_config: Any) -> CustomFedAdam:
    return CustomFedAdam(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
        eta=float(run_config.get("fedadam-eta", 0.1)),
        eta_l=float(run_config.get("fedadam-eta-l", 0.3)),
        beta_1=float(run_config.get("fedadam-beta1", 0.9)),
        beta_2=float(run_config.get("fedadam-beta2", 0.9)),
        tau=float(run_config.get("fedadam-tau", 0.001)),
        dataset_name=dataset_name,
    )


def _build_fedtag(*, fraction_train: float, dataset_name: str, run_config: Any) -> FedTag:
    return FedTag(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_train_nodes=3,
        min_evaluate_nodes=3,
        min_available_nodes=3,
        dataset_name=dataset_name,
        l1_init=float(run_config.get("fedtag-l1-init", 0.2)),
        l1_min=float(run_config.get("fedtag-l1-min", 0.2)),
        l1_max=float(run_config.get("fedtag-l1-max", 0.8)),
        l1_step=float(run_config.get("fedtag-l1-step", 0.02)),
        convergence_window=int(run_config.get("fedtag-convergence-window", 5)),
        convergence_threshold=float(run_config.get("fedtag-convergence-threshold", 0.001)),
    )


# Add an entry here when a new algorithm strategy is merged (e.g. fedprox).
STRATEGY_BUILDERS: dict[str, Callable[..., Any]] = {
    "fedavg": _build_fedavg,
    "fedadam": _build_fedadam,
    "fedtag": _build_fedtag,
}


def build_strategy(name: str, *, fraction_train: float, dataset_name: str, run_config: Any) -> Any:
    """Instantiate the strategy registered under `name`.

    Args:
        name: value of the `strategy-name` run-config key (e.g. "fedavg").
        fraction_train: fraction of nodes sampled per training round.
        dataset_name: dataset-name run-config value, forwarded to the strategy.
        run_config: the full `context.run_config`, for algorithm-specific overrides.
    """
    try:
        builder = STRATEGY_BUILDERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown strategy-name '{name}'. Available: {list(STRATEGY_BUILDERS)}"
        ) from exc
    return builder(fraction_train=fraction_train, dataset_name=dataset_name, run_config=run_config)
