# render_NT_disk.py

# 전체 pipeline

import argparse
from pathlib import Path
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from src.types import KerrParams
from src.camera import KerrSchildCamera
from src.metrics import kerr_schild
from src.plasma.novikov_thorne import isco_radius_kerr
from src.observables import evaluate_nt_ray
from src.cache.reader import RayHistoryReader


parser = argparse.ArgumentParser(description="Render a bolometric NT disk image from a saved ray cache.")
parser.add_argument("--cache", required=True, help="Input HDF5 cache path, relative to the repository root or absolute.")
parser.add_argument("--output", help="Output NPZ path, relative to the repository root or absolute.")
args = parser.parse_args()

repo_root = Path(__file__).resolve().parents[2]
cache_path = Path(args.cache).expanduser()
if not cache_path.is_absolute():
    cache_path = repo_root / cache_path
if not cache_path.is_file():
    parser.error(f"Cache file not found: {cache_path}")

output_path = (
    Path(args.output).expanduser() if args.output
    else cache_path.with_name(f"render_{cache_path.stem.removeprefix('rays_')}.npz")
)
if not output_path.is_absolute():
    output_path = repo_root / output_path

render_started = time.perf_counter()
print(f"Cache: {cache_path}", flush=True)
print(f"Output: {output_path}", flush=True)

with RayHistoryReader(cache_path) as reader:
    parameters = dict(reader.file["parameters"].attrs)
    initial = reader.read_initial()

    # Connect the remaining pipeline here.
    M = parameters["mass"]
    a = parameters["spin"]
    FOV = parameters["fov"]
    PPD = parameters["ppd"]

    N = int(FOV * PPD)
    if N < 1 or reader.batch_size != N * N:
        raise ValueError("Cache ray count does not match the image dimensions from fov and ppd.")
    params = KerrParams(M=M, a=a)

    x_obs = initial.y[0, :4]
    tetrad = KerrSchildCamera.eulerian_tetrad(
        x_obs, params, kerr_schild.metric_cov,
    )
    u_obs = tetrad[0]

    r_in = isco_radius_kerr(params)
    r_out = 20*M
    mdot = 1

    img = np.full((N, N, 2), np.nan)
    order_map = np.full((N, N), -1, dtype=np.int32)
    hit_map = np.zeros((N, N), dtype=bool)

    evaluate_batch = jax.jit(
        jax.vmap(
            evaluate_nt_ray,
            in_axes=(0, None, None, None, None, None),
        )
    )

    BATCH_SIZE = 32
    T = max(2, int(reader.saved_lengths.max()) + 1)
    print(
        f"Image: {N}x{N} | Rays: {reader.batch_size} | Batch size: {BATCH_SIZE} | Padded length: {T}",
        flush=True,
    )
    print(f"Disk: r_in={float(r_in)}, r_out={float(r_out)}, mdot={mdot}", flush=True)

    batch = np.empty(
        (BATCH_SIZE, T, 8),
        dtype=initial.y.dtype,
    )

    for start in range(0, reader.batch_size, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, reader.batch_size)
        count = stop - start

        for j, ray_id in enumerate(range(start, stop)):
            history, _, _ = reader.read_ray(
                ray_id, include_initial=True,
            )
            length = len(history)

            batch[j, :length] = history
            batch[j, length:] = history[-1]

        batch[count:] = batch[0]

        I_values, g_values, orders, hits = jax.device_get(
            evaluate_batch(
                batch, params, u_obs, r_in, r_out, mdot,
            )
        )

        rows, cols = np.divmod(np.arange(start, stop), N)

        img[rows, cols, 0] = I_values[:count]
        img[rows, cols, 1] = g_values[:count]
        order_map[rows, cols] = orders[:count]
        hit_map[rows, cols] = hits[:count]

        batch_number = start // BATCH_SIZE + 1
        if start == 0 or batch_number % 10 == 0 or stop == reader.batch_size:
            print(
                f"Rendered {stop}/{reader.batch_size} rays | elapsed {time.perf_counter() - render_started:.1f}s",
                flush=True,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        intensity=img[:, :, 0],
        g=img[:, :, 1],
        image_order=order_map,
        found=hit_map,
    )
    print(
        f"Saved: {output_path} | Disk hits: {np.count_nonzero(hit_map)}/{reader.batch_size}"
        f" | Total elapsed: {time.perf_counter() - render_started:.1f}s",
        flush=True,
    )
