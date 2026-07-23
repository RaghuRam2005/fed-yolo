"""
Persistent per-client state, kept separate from the (conceptually ephemeral,
recreated-every-round) client compute step.

Plain FedAvg never touches this -- a client's local training in round R+1
only ever depends on the global weights it's handed, not on anything from
round R. It exists as the extension point a *stateful* algorithm would use
(e.g. the old FedTag's per-client BatchNorm running stats, persisted across
rounds so each client re-applies its own after loading the shared weights).
"""
from typing import Any, Optional


class ClientStateStore:
    """In-memory `{client_id: state}` store, alive for one simulation run."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, client_id: str, default: Optional[Any] = None) -> Any:
        return self._store.get(client_id, default)

    def set(self, client_id: str, state: Any) -> None:
        self._store[client_id] = state

    def has(self, client_id: str) -> bool:
        return client_id in self._store
