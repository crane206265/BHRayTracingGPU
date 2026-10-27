# writer.py

from operator import index
from pathlib import Path
from uuid import uuid4

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from src.types import IntegrationState, RayStatus


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
        if isinstance(path, h5py.File):
            self.file = path
        else:
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


class SelectiveRayHistoryWriter(RayHistoryWriter):
    """
    Terminal-state filtering with one CPU-memory ray batch at a time

    Parameters
    ----------
    disk_hit_fn : callable
        disk_hit_fn(x0, x1) -> boolean mask; positions: (segments, 4)
    disk_geometry : mapping
        plane, mass, spin, r_in, r_out for the omission decision

    Notes
    -----
    Omission rule: CAPTURED & no disk intersection on accepted segments
    Sampling: RayHistoryWriter, unchanged
    Buffer: in-memory HDF5, backing_store=False
    Final layout: format 3, packed ray blocks
    """

    def __init__(self, path, *, disk_hit_fn, disk_geometry, **sampling):
        geometry = dict(disk_geometry)
        required = {"plane", "mass", "spin", "r_in", "r_out"}
        if not required <= geometry.keys() or geometry["plane"] != "z=0":
            raise ValueError("Selective caching requires an aligned z=0 disk geometry.")
        M, a, r_in, r_out = (float(geometry[key]) for key in ("mass", "spin", "r_in", "r_out"))
        if not np.isfinite([M, a, r_in, r_out]).all() or M <= 0 or abs(a) > M or not 0 < r_in < r_out:
            raise ValueError("Invalid selective-cache disk parameters.")
        if not callable(disk_hit_fn):
            raise TypeError("disk_hit_fn must be callable.")
        super().__init__(path, **sampling)
        self.file.attrs["format_version"] = 3
        self.file.attrs["cache_history_mode"] = "omit_captured_no_hit"
        self.file.create_group("disk_geometry").attrs.update(geometry)
        self._disk_hit_fn = disk_hit_fn
        self._batch_open = False
        self._sampling = dict(sampling)

    def write_initial(self, state):
        super().write_initial(state)
        arrays = self._state_arrays(state)
        self._initial_arrays = {name: value.copy() for name, value in arrays.items()}
        self._final_arrays = {name: value.copy() for name, value in arrays.items()}
        self._committed = np.zeros(self._batch_size, dtype=bool)
        history = self.file["history"]
        history.create_dataset("omitted", data=np.zeros(self._batch_size, dtype=bool))
        reason = history.create_dataset("omission_reason", data=np.zeros(self._batch_size, dtype=np.uint8))
        reason.attrs["codes"] = "0=retained; 1=captured_without_disk_hit"
        history.create_dataset("disk_hit", data=np.zeros(self._batch_size, dtype=bool))
        history.create_dataset("unfiltered_saved_length", data=np.zeros(self._batch_size, dtype=np.int64))
        self.file.flush()

    def save_chunk(self, chunk_idx, history):
        raise RuntimeError("Selective caching requires writer.batch(...).save_chunk().")

    def batch(self, ray_ids, initial_state):
        self._require_open()
        if self._batch_size is None or self._finalized:
            raise RuntimeError("Write the initial state before opening an unfinished ray batch.")
        if self._batch_open:
            raise RuntimeError("Only one buffered trace batch may be open at a time.")
        ids = np.asarray(ray_ids)
        if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("ray_ids must be a nonempty integer vector.")
        if np.any(np.diff(ids) <= 0) or ids[0] < 0 or ids[-1] >= self._batch_size:
            raise ValueError("ray_ids must be sorted, unique, and within the initial batch.")
        ids = ids.astype(np.int64)
        if self._committed[ids].any():
            raise RuntimeError("A ray batch cannot be committed twice.")
        arrays = self._state_arrays(initial_state)
        if arrays["y"].shape[0] != len(ids) or any(
            arrays[name].dtype != value.dtype
            or not np.array_equal(arrays[name], value[ids], equal_nan=True)
            for name, value in self._initial_arrays.items()
        ):
            raise ValueError("Batch initial state does not match the corresponding global rays.")
        self._batch_open = True
        try:
            return _BufferedRayBatch(self, ids, initial_state)
        except Exception:
            self._batch_open = False
            raise

    def _commit_batch(self, pending, final_state):
        self._require_open()
        arrays = self._state_arrays(final_state)
        ids = pending.ray_ids
        if arrays["y"].shape[0] != len(ids) or arrays["active"].any():
            raise ValueError("Every buffered ray must have a terminal state before commit.")
        pending.writer.write_final(final_state)
        group = pending.writer.file["history"]
        lengths = group["saved_length"][:]
        omitted = (arrays["status"] == int(RayStatus.CAPTURED)) & ~pending.disk_hit
        keep = ~omitted
        kept_lengths = np.where(keep, lengths, 0)
        n = int(kept_lengths.sum())
        starts = np.cumsum(kept_lengths, dtype=np.int64) - kept_lengths
        cursors = np.zeros(len(ids), dtype=np.int64)
        y = np.empty((n, 8), dtype=group["y"].dtype)
        lam = np.empty(n, dtype=group["lam"].dtype)
        steps = np.empty(n, dtype=np.int64)
        for offsets in group["block_offsets"]:
            counts = np.diff(offsets)
            local_ids = np.repeat(np.arange(len(ids)), counts)
            selected = keep[local_ids]
            if not selected.any():
                continue
            first, last = int(offsets[0]), int(offsets[-1])
            within = np.arange(last - first) - np.repeat(offsets[:-1] - first, counts)
            destinations = (starts[local_ids] + cursors[local_ids] + within)[selected]
            y[destinations] = group["y"][first:last][selected]
            lam[destinations] = group["lam"][first:last][selected]
            steps[destinations] = group["step"][first:last][selected]
            cursors += np.where(keep, counts, 0)
        if not np.array_equal(cursors, kept_lengths):
            raise RuntimeError("Buffered sample counts are inconsistent.")
        global_counts = np.zeros(self._batch_size, dtype=np.int64)
        global_counts[ids] = kept_lengths
        self._append_records(y, lam, steps, global_counts)
        history = self.file["history"]
        history["omitted"][ids] = omitted
        history["omission_reason"][ids] = omitted.astype(np.uint8)
        history["disk_hit"][ids] = pending.disk_hit
        history["unfiltered_saved_length"][ids] = lengths
        history.attrs.modify("integrated_steps", max(
            int(history.attrs["integrated_steps"]), int(group.attrs["integrated_steps"])
        ))
        self._valid_steps[ids] = arrays["valid_length"]
        for name, value in arrays.items():
            self._final_arrays[name][ids] = value
        self._committed[ids] = True
        self.file.flush()

    def write_final(self, state=None):
        self._require_open()
        if self._batch_size is None or self._finalized or self._batch_open or not self._committed.all():
            raise RuntimeError("All ray batches must be committed and closed before finalization.")
        arrays = self._state_arrays(state) if state is not None else self._final_arrays
        if arrays["y"].shape[0] != self._batch_size or any(
            arrays[name].dtype != value.dtype
            or not np.array_equal(arrays[name], value, equal_nan=True)
            for name, value in self._final_arrays.items()
        ):
            raise ValueError("Final state differs from the committed ray results.")
        final = self.file.create_group("final")
        for name, array in arrays.items():
            final.create_dataset(name, data=array)
        self.file.flush()
        self._finalized = True


class _BufferedRayBatch:
    def __init__(self, owner, ray_ids, initial_state):
        self.owner = owner
        self.ray_ids = ray_ids
        memory_file = h5py.File(f"ray-buffer-{uuid4().hex}", "w", driver="core", backing_store=False)
        try:
            self.writer = RayHistoryWriter(memory_file, **owner._sampling)
            self.writer.write_initial(initial_state)
        except Exception:
            memory_file.close()
            raise
        self.disk_hit = np.zeros(len(ray_ids), dtype=bool)
        self._previous_x = jnp.asarray(initial_state.y[:, :4])
        self._committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.writer.close()
        finally:
            self.owner._batch_open = False
        if exc_type is None and not self._committed:
            raise RuntimeError("Buffered ray batch closed without committing its terminal state.")

    def save_chunk(self, chunk_idx, history):
        if self._committed:
            raise RuntimeError("Cannot append to a committed ray batch.")
        y, _, valid = history
        x0 = jnp.concatenate((self._previous_x[None, ...], y[:-1, :, :4]), axis=0)
        hits = self.owner._disk_hit_fn(x0.reshape(-1, 4), y[..., :4].reshape(-1, 4))
        hits = jnp.asarray(hits).reshape(valid.shape) & valid
        self.disk_hit |= np.asarray(jax.device_get(jnp.any(hits, axis=0)))
        self.writer.save_chunk(chunk_idx, history)
        self._previous_x = y[-1, :, :4]

    def commit(self, final_state):
        if self._committed:
            raise RuntimeError("A buffered ray batch cannot be committed twice.")
        self.owner._commit_batch(self, final_state)
        self._committed = True
