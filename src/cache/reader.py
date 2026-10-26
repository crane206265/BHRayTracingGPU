from operator import index

import h5py

from src.types import IntegrationState


class RayHistoryReader:
    def __init__(self, path):
        self.file = h5py.File(path, "r")
        try:
            self._validate_layout()
        except Exception:
            self.file.close()
            raise

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.file.close()

    def _require_open(self):
        if not self.file:
            raise RuntimeError("The ray history file is closed.")

    def _validate_layout(self):
        if self.file.attrs.get("format_version") != 1:
            raise ValueError("Unsupported ray history format version.")
        if self.file.attrs.get("history_includes_initial", True):
            raise ValueError("History must exclude the initial sample.")
        if "initial" not in self.file or "history" not in self.file:
            raise ValueError("The file must contain initial and history groups.")

        initial = self.file["initial"]
        for name in IntegrationState._fields:
            if name not in initial:
                raise ValueError(f"Missing initial state field: {name}.")

        y = initial["y"]
        if y.ndim != 2 or y.shape[1] != 8 or y.shape[0] == 0:
            raise ValueError("Initial states must have shape (B, 8) with B > 0.")
        self._batch_size = y.shape[0]
        for name in ("lam", "active", "status", "valid_length"):
            if initial[name].shape != (self._batch_size,):
                raise ValueError(f"Initial {name} must have shape (B,).")

        history = self.file["history"]
        for name in ("y", "lam", "valid"):
            if name not in history:
                raise ValueError(f"Missing history dataset: {name}.")

        y = history["y"]
        if y.ndim != 3 or y.shape[1:] != (self._batch_size, 8):
            raise ValueError("History states must have shape (S, B, 8).")
        for name in ("lam", "valid"):
            if history[name].shape != y.shape[:2]:
                raise ValueError(f"History {name} must have shape (S, B).")

    @property
    def batch_size(self):
        self._require_open()
        return self._batch_size

    @property
    def n_steps(self):
        self._require_open()
        return self.file["history/y"].shape[0]

    @property
    def has_final(self):
        self._require_open()
        return "final" in self.file

    def _read_state(self, name):
        self._require_open()
        if name not in self.file:
            raise RuntimeError(f"No {name} state has been written.")

        group = self.file[name]
        for field in IntegrationState._fields:
            if field not in group:
                raise ValueError(f"Missing {name} state field: {field}.")
            shape = (self._batch_size, 8) if field == "y" else (self._batch_size,)
            if group[field].shape != shape:
                raise ValueError(f"Invalid shape for {name} state field: {field}.")

        return IntegrationState(**{
            field: group[field][...]
            for field in IntegrationState._fields
        })

    def read_initial(self):
        return self._read_state("initial")

    def read_final(self):
        return self._read_state("final")

    def read_history(self, start, stop):
        self._require_open()
        start, stop = index(start), index(stop)
        if not 0 <= start <= stop <= self.n_steps:
            raise ValueError("Expected 0 <= start <= stop <= n_steps.")

        group = self.file["history"]
        return (
            group["y"][start:stop, ...],
            group["lam"][start:stop, ...],
            group["valid"][start:stop, ...],
        )
