from operator import index

import h5py
import numpy as np

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
        self._format_version = int(self.file.attrs.get("format_version", -1))
        if self._format_version not in (1, 2):
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
        required = ("y", "lam", "valid") if self._format_version == 1 else ("y", "lam", "step", "block_offsets", "saved_length")
        for name in required:
            if name not in history:
                raise ValueError(f"Missing history dataset: {name}.")
        y = history["y"]
        if self._format_version == 1:
            if y.ndim != 3 or y.shape[1:] != (self._batch_size, 8):
                raise ValueError("History states must have shape (S, B, 8).")
            for name in ("lam", "valid"):
                if history[name].shape != y.shape[:2]:
                    raise ValueError(f"History {name} must have shape (S, B).")
        else:
            if y.ndim != 2 or y.shape[1] != 8:
                raise ValueError("Packed history states must have shape (N, 8).")
            for name in ("lam", "step"):
                if history[name].shape != (y.shape[0],):
                    raise ValueError(f"Packed history {name} must have shape (N,).")
            offsets = history["block_offsets"]
            if offsets.ndim != 2 or offsets.shape[1] != self._batch_size + 1:
                raise ValueError("Block offsets must have shape (C, B + 1).")
            if history["saved_length"].shape != (self._batch_size,):
                raise ValueError("Saved lengths must have shape (B,).")
            lengths = history["saved_length"][:]
            if np.any(lengths < 0) or int(lengths.sum()) != y.shape[0]:
                raise ValueError("Saved lengths do not match the packed sample count.")
            if offsets.shape[0]:
                starts, stops = offsets[:, 0], offsets[:, -1]
                if starts[0] != 0 or stops[-1] != y.shape[0] or np.any(stops < starts) or np.any(starts[1:] != stops[:-1]):
                    raise ValueError("Block offsets do not cover the packed samples.")
            elif y.shape[0]:
                raise ValueError("Packed samples are missing block offsets.")
            if int(history.attrs.get("integrated_steps", -1)) < 0:
                raise ValueError("Missing integrated step count.")

    @property
    def format_version(self):
        self._require_open()
        return self._format_version

    @property
    def batch_size(self):
        self._require_open()
        return self._batch_size

    @property
    def n_steps(self):
        self._require_open()
        if self._format_version == 1:
            return self.file["history/y"].shape[0]
        return int(self.file["history"].attrs["integrated_steps"])

    @property
    def n_samples(self):
        self._require_open()
        if self._format_version == 2:
            return self.file["history/y"].shape[0]
        return int(self.saved_lengths.sum())

    @property
    def n_blocks(self):
        self._require_open()
        if self._format_version != 2:
            raise RuntimeError("Packed blocks are available only in format version 2.")
        return self.file["history/block_offsets"].shape[0]

    @property
    def saved_lengths(self):
        self._require_open()
        if self._format_version == 2:
            return self.file["history/saved_length"][:]
        if self.has_final:
            return self.file["final/valid_length"][:].astype(np.int64) - self.file["initial/valid_length"][:].astype(np.int64)
        counts = np.zeros(self._batch_size, dtype=np.int64)
        for start in range(0, self.n_steps, 256):
            counts += self.file["history/valid"][start:start + 256].sum(axis=0, dtype=np.int64)
        return counts

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
        return IntegrationState(**{field: group[field][...] for field in IntegrationState._fields})

    def read_initial(self):
        return self._read_state("initial")

    def read_final(self):
        return self._read_state("final")

    def read_history(self, start, stop):
        self._require_open()
        if self._format_version != 1:
            raise RuntimeError("Format version 2 is variable length; use read_ray() or read_block().")
        start, stop = index(start), index(stop)
        if not 0 <= start <= stop <= self.n_steps:
            raise ValueError("Expected 0 <= start <= stop <= n_steps.")
        group = self.file["history"]
        return (group["y"][start:stop, ...], group["lam"][start:stop, ...], group["valid"][start:stop, ...])

    def read_block(self, block_index):
        self._require_open()
        block_index = index(block_index)
        if not 0 <= block_index < self.n_blocks:
            raise ValueError("Block index is out of range.")
        group = self.file["history"]
        offsets = group["block_offsets"][block_index]
        if np.any(np.diff(offsets) < 0):
            raise ValueError("Block offsets must be nondecreasing.")
        start, stop = int(offsets[0]), int(offsets[-1])
        return (group["y"][start:stop], group["lam"][start:stop], group["step"][start:stop], offsets - start)

    def read_ray(self, ray_id, *, include_initial=False):
        self._require_open()
        ray_id = index(ray_id)
        if not 0 <= ray_id < self._batch_size:
            raise ValueError("Ray index is out of range.")
        group = self.file["history"]
        ys, lams, steps = [], [], []
        if include_initial:
            ys.append(self.file["initial/y"][ray_id:ray_id + 1])
            lams.append(self.file["initial/lam"][ray_id:ray_id + 1])
            steps.append(self.file["initial/valid_length"][ray_id:ray_id + 1].astype(np.int64))
        if self._format_version == 1:
            valid = group["valid"][:, ray_id]
            ys.append(group["y"][:, ray_id, :][valid])
            lams.append(group["lam"][:, ray_id][valid])
            initial_length = int(self.file["initial/valid_length"][ray_id])
            steps.append(initial_length + np.arange(1, int(valid.sum()) + 1, dtype=np.int64))
        else:
            for start, stop in group["block_offsets"][:, ray_id:ray_id + 2]:
                start, stop = int(start), int(stop)
                if start > stop:
                    raise ValueError("Block offsets must be nondecreasing.")
                if start != stop:
                    ys.append(group["y"][start:stop])
                    lams.append(group["lam"][start:stop])
                    steps.append(group["step"][start:stop])
        if not ys:
            return (np.empty((0, 8), dtype=group["y"].dtype), np.empty(0, dtype=group["lam"].dtype), np.empty(0, dtype=np.int64))
        return np.concatenate(ys), np.concatenate(lams), np.concatenate(steps)
