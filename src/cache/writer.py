# writer.py

import h5py
import jax
import numpy as np


class RayHistoryWriter:
    def __init__(self, path):
        self.file = h5py.File(path, "x")
        self.file.attrs["format_version"] = 1
        self.file.attrs["history_includes_initial"] = False
        self._batch_size = None
        self._next_chunk = 0
        self._finalized = False

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

    @staticmethod
    def _state_arrays(state):
        arrays = {
            name: np.asarray(value)
            for name, value in jax.device_get(state)._asdict().items()
        }
        y = arrays["y"]
        if y.ndim != 2 or y.shape[1] != 8 or y.shape[0] == 0:
            raise ValueError("state.y must have shape (B, 8) with B > 0.")
        for name in ("lam", "active", "status", "valid_length"):
            if arrays[name].shape != (y.shape[0],):
                raise ValueError(f"state.{name} must have shape (B,).")
        return arrays

    def write_initial(self, state):
        self._require_open()
        if "initial" in self.file:
            raise RuntimeError("The initial state has already been written.")

        arrays = self._state_arrays(state)
        B = arrays["y"].shape[0]

        initial = self.file.create_group("initial")
        for name, array in arrays.items():
            initial.create_dataset(name, data=array)

        history = self.file.create_group("history")
        for name, shape, dtype in (
            ("y", (0, B, 8), arrays["y"].dtype),
            ("lam", (0, B), arrays["lam"].dtype),
            ("valid", (0, B), np.dtype("bool")),
        ):
            history.create_dataset(
                name,
                shape=shape,
                maxshape=(None,) + shape[1:],
                dtype=dtype,
                chunks=True,
            )

        self._batch_size = B
        self.file.flush()

    def save_chunk(self, chunk_idx, history):
        self._require_open()
        if self._batch_size is None:
            raise RuntimeError("Call write_initial() before saving history.")
        if self._finalized:
            raise RuntimeError("The final state has already been written.")
        if chunk_idx != self._next_chunk:
            raise ValueError(f"Expected chunk index {self._next_chunk}.")

        y, lam, valid = (
            np.asarray(array) for array in jax.device_get(history)
        )
        B = self._batch_size
        if y.ndim != 3 or y.shape[1:] != (B, 8):
            raise ValueError("History states must have shape (K, B, 8).")
        if lam.shape != y.shape[:2] or valid.shape != y.shape[:2]:
            raise ValueError("History lam and valid must have shape (K, B).")

        group = self.file["history"]
        arrays = (("y", y), ("lam", lam), ("valid", valid))
        for name, array in arrays:
            if array.dtype != group[name].dtype:
                raise TypeError(f"History {name} dtype must match its dataset.")

        start = group["y"].shape[0]
        stop = start + y.shape[0]
        for name, array in arrays:
            dataset = group[name]
            dataset.resize(stop, axis=0)
            dataset[start:stop, ...] = array

        self.file.flush()
        self._next_chunk += 1

    def write_final(self, state):
        self._require_open()
        if self._batch_size is None:
            raise RuntimeError("Call write_initial() before writing the final state.")
        if self._finalized:
            raise RuntimeError("The final state has already been written.")

        arrays = self._state_arrays(state)
        if arrays["y"].shape[0] != self._batch_size:
            raise ValueError("The final state batch size must match the initial state.")

        final = self.file.create_group("final")
        for name, array in arrays.items():
            final.create_dataset(name, data=array)

        self.file.flush()
        self._finalized = True
