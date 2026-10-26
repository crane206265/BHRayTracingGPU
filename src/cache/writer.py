# writer.py

from operator import index
from pathlib import Path

import h5py
import jax
import numpy as np


class RayHistoryWriter:
    def __init__(self, path, *, radius_fn=None, dense_radius=None, outer_interval=1.0):
        if dense_radius is not None:
            dense_radius = float(dense_radius)
            if not np.isfinite(dense_radius) or dense_radius < 0:
                raise ValueError("dense_radius must be finite and nonnegative.")
            if radius_fn is None:
                raise ValueError("radius_fn is required when dense_radius is set.")
        outer_interval = float(outer_interval)
        if not np.isfinite(outer_interval) or outer_interval <= 0:
            raise ValueError("outer_interval must be finite and positive.")
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / "cache" / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "x")
        self.file.attrs["format_version"] = 2
        self.file.attrs["history_includes_initial"] = False
        self._radius_fn = radius_fn
        self._dense_radius = dense_radius
        self._outer_interval = outer_interval
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
        history.attrs["layout"] = "packed_ray_blocks"
        history.attrs["integrated_steps"] = 0
        history.attrs["dense_radius"] = np.inf if self._dense_radius is None else self._dense_radius
        history.attrs["outer_interval"] = self._outer_interval
        history.attrs["sampling"] = "all_valid" if self._dense_radius is None else "affine_bin_crossings"
        for name, shape, dtype in (
            ("y", (0, 8), arrays["y"].dtype),
            ("lam", (0,), arrays["lam"].dtype),
            ("step", (0,), np.dtype("int64")),
        ):
            history.create_dataset(
                name, shape=shape, maxshape=(None,) + shape[1:],
                dtype=dtype, chunks=(4096,) + shape[1:],
            )
        history.create_dataset(
            "block_offsets", shape=(0, B + 1), maxshape=(None, B + 1),
            dtype="int64", chunks=(1, min(B + 1, 4096)),
            compression="lzf", shuffle=True,
        )
        history.create_dataset("saved_length", data=np.zeros(B, dtype=np.int64))
        self._batch_size = B
        self._initial_lengths = arrays["valid_length"].astype(np.int64, copy=True)
        self._valid_steps = self._initial_lengths.copy()
        self._saved_lengths = np.zeros(B, dtype=np.int64)
        self._last_saved_steps = self._initial_lengths.copy()
        self._initial_lam = arrays["lam"].copy()
        self._last_y = arrays["y"].copy()
        self._last_lam = arrays["lam"].copy()
        self._has_previous = np.zeros(B, dtype=bool)
        self._last_dense = np.zeros(B, dtype=bool)
        self._last_outer_bin = np.zeros(B, dtype=float)
        self.file.flush()

    def _append_records(self, y, lam, steps, counts):
        n = len(lam)
        if not n:
            return
        ends = np.cumsum(counts, dtype=np.int64)
        if int(ends[-1]) != n:
            raise ValueError("Packed record counts do not match sample count.")
        group = self.file["history"]
        start = group["y"].shape[0]
        for name, array in (("y", y), ("lam", lam), ("step", steps)):
            dataset = group[name]
            dataset.resize(start + n, axis=0)
            dataset[start:start + n, ...] = array
        offsets = group["block_offsets"]
        row = offsets.shape[0]
        offsets.resize(row + 1, axis=0)
        offsets[row, :] = start + np.concatenate(([0], ends))
        present = counts > 0
        self._last_saved_steps[present] = steps[ends[present] - 1]
        self._saved_lengths += counts
        group["saved_length"][...] = self._saved_lengths

    def save_chunk(self, chunk_idx, history):
        self._require_open()
        if self._batch_size is None:
            raise RuntimeError("Call write_initial() before saving history.")
        if self._finalized:
            raise RuntimeError("The final state has already been written.")
        if index(chunk_idx) != self._next_chunk:
            raise ValueError(f"Expected chunk index {self._next_chunk}.")
        raw_y, raw_lam, raw_valid = history
        y, lam, valid = (np.asarray(a) for a in jax.device_get(history))
        B = self._batch_size
        if y.ndim != 3 or y.shape[1:] != (B, 8) or y.shape[0] == 0:
            raise ValueError("History states must have shape (K, B, 8) with K > 0.")
        if lam.shape != y.shape[:2] or valid.shape != y.shape[:2]:
            raise ValueError("History lam and valid must have shape (K, B).")
        group = self.file["history"]
        if y.dtype != group["y"].dtype or lam.dtype != group["lam"].dtype:
            raise TypeError("History state and affine dtypes must match their datasets.")
        if valid.dtype != np.dtype("bool"):
            raise TypeError("History valid must be boolean.")
        if not np.isfinite(y[valid]).all() or not np.isfinite(lam[valid]).all():
            raise ValueError("Valid history samples must be finite.")
        steps = self._valid_steps[None, :] + np.cumsum(valid, axis=0, dtype=np.int64)
        counts = valid.sum(axis=0, dtype=np.int64)
        if self._dense_radius is None:
            keep = valid.copy()
        else:
            radius = np.asarray(jax.device_get(
                self._radius_fn(raw_y[..., :4].reshape(-1, 4))
            )).reshape(valid.shape)
            if not np.isfinite(radius[valid]).all():
                raise ValueError("Radii of valid samples must be finite.")
            dense = radius <= self._dense_radius
            previous_dense = np.concatenate((self._last_dense[None, :], dense[:-1]), axis=0)
            with np.errstate(over="ignore", invalid="ignore"):
                bins = np.floor(np.abs(lam - self._initial_lam[None, :]) / self._outer_interval)
            if not np.isfinite(bins[valid]).all():
                raise ValueError("Outer sampling interval is too small for these affine parameters.")
            previous_bins = np.concatenate((self._last_outer_bin[None, :], bins[:-1]), axis=0)
            first = steps == self._initial_lengths[None, :] + 1
            keep = valid & (dense | previous_dense | (bins > previous_bins) | first)
            keep[:-1] |= valid[:-1] & valid[1:] & dense[1:]
            bridge = (
                valid[0] & dense[0] & self._has_previous
                & (self._last_saved_steps < self._valid_steps)
            )
            self._append_records(
                self._last_y[bridge], self._last_lam[bridge],
                self._valid_steps[bridge], bridge.astype(np.int64),
            )
            self._last_dense = dense[-1].copy()
            self._last_outer_bin = bins[-1].copy()
        self._append_records(
            y.swapaxes(0, 1)[keep.T], lam.T[keep.T], steps.T[keep.T],
            keep.sum(axis=0, dtype=np.int64),
        )
        self._valid_steps += counts
        self._last_y = y[-1].copy()
        self._last_lam = lam[-1].copy()
        self._has_previous |= counts > 0
        group.attrs.modify("integrated_steps", int(group.attrs["integrated_steps"]) + y.shape[0])
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
        if not np.array_equal(arrays["valid_length"], self._valid_steps):
            raise ValueError("Final valid_length does not match the received history.")
        group = self.file["history"]
        if arrays["y"].dtype != group["y"].dtype or arrays["lam"].dtype != group["lam"].dtype:
            raise TypeError("Final state and affine dtypes must match their datasets.")
        endpoint = (
            (self._valid_steps > self._initial_lengths)
            & (self._last_saved_steps < self._valid_steps)
        )
        self._append_records(
            arrays["y"][endpoint], arrays["lam"][endpoint],
            self._valid_steps[endpoint], endpoint.astype(np.int64),
        )
        final = self.file.create_group("final")
        for name, array in arrays.items():
            final.create_dataset(name, data=array)
        self.file.flush()
        self._finalized = True
