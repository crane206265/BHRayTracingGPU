# trace_Kerr.py

import argparse
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from src.types import KerrParams
from src.metrics import kerr_schild
from src.camera import KerrSchildCamera
from src.integrators.driver import initialize_state, integrate
from src.events import event_single_func
from src.integrators.rk4 import rk4_step_func
from src.integrators.batched import integrate_chunk_func
from src.hamiltonian import rhs_func
from src.cache.writer import RayHistoryWriter
from src.integrators.step_size import adaptive_step_size_func


parser = argparse.ArgumentParser(description="Trace Kerr photon geodesics and save their history.")
parser.add_argument("--mass", type=float, default=1.0)
parser.add_argument("--spin", type=float, default=0.0, help="Dimensional Kerr parameter a; abs(a) <= mass.")
parser.add_argument("--radius", type=float, default=50.0, help="Euclidean observer distance in code units.")
parser.add_argument("--inc", type=float, default=60.0, help="Observer inclination in degrees.")
parser.add_argument("--freq", type=float, default=1.0)
parser.add_argument("--fov", type=float, default=20.0, help="Field of view in degrees.")
parser.add_argument("--ppd", type=float, default=0.25, help="Pixels per degree; image width is int(fov * ppd).")
parser.add_argument("--escape-factor", type=float, default=2.0)
parser.add_argument("--epsilon", type=float, default=0.01, help="Dimensionless capture margin: r_stop = r_plus + epsilon * mass; zero selects the horizon.")
parser.add_argument("--h", type=float, default=-0.01)
parser.add_argument("--chunk-steps", type=int, default=256)
parser.add_argument("--max-chunks", type=int, default=100)
parser.add_argument("--output", default="rays.h5", help="Filename relative to repository cache/, or an absolute path.")
args = parser.parse_args()

numeric_values = [value for value in vars(args).values() if isinstance(value, (int, float))]
if not np.isfinite(numeric_values).all():
    parser.error("Numeric parameters must be finite.")
if args.mass <= 0 or abs(args.spin) > args.mass:
    parser.error("Require mass > 0 and abs(spin) <= mass.")
if args.radius <= 0 or args.freq <= 0:
    parser.error("Radius and frequency must be positive.")
if not 0 < args.inc < 180:
    parser.error("Require 0 < inc < 180; the current screen basis excludes the poles.")
if not 0 < args.fov < 180 or args.ppd <= 0 or int(args.fov * args.ppd) < 1:
    parser.error("Require 0 < fov < 180, ppd > 0, and at least one pixel per axis.")
if args.escape_factor <= 1 or args.h >= 0:
    parser.error("Require escape-factor > 1 and h < 0.")
if args.epsilon < 0:
    parser.error("Epsilon must be nonnegative.")
if args.chunk_steps < 1 or args.max_chunks < 1:
    parser.error("Chunk steps and max chunks must be positive integers.")

print("Parameters:", vars(args), flush=True)

BHparams = KerrParams(M=args.mass, a=args.spin)
R = args.radius
inc = args.inc
freq = args.freq
FOV = args.fov
PPD = args.ppd
r_esc_R = args.escape_factor
h = args.h
MAX_STEPS = args.chunk_steps
MAX_CHUNKS = args.max_chunks

# --------------------------

inc = np.deg2rad(inc)
x_obs = [R*np.sin(inc), 0, R*np.cos(inc)]
if h >= 0: raise ValueError("Integration step h should be negative")

# --------------------------

camera = KerrSchildCamera(
    x_obs = x_obs,
    freq = freq,
    FOV = FOV,
    PPD = PPD
)

metric_cov = kerr_schild.metric_cov
g = metric_cov(camera.init_pos, BHparams)

tetrad = KerrSchildCamera.eulerian_tetrad(
    x=camera.init_pos,
    params=BHparams,
    metric_cov=metric_cov,
)
p0 = camera.photonScreen(g, tetrad) # (4, N, N)
p0 = p0.reshape(4, -1).T
B = p0.shape[0]

x0 = jnp.broadcast_to(camera.init_pos, (B, 4))
y0 = jnp.concatenate([x0, p0], axis=1)

r_escape = jnp.full(
    (B,),
    r_esc_R * jnp.linalg.norm(camera.x_obs),
    dtype=y0.dtype,
)
safe_y = jnp.zeros((8,), dtype=y0.dtype)
safe_y = safe_y.at[1].set(100.0 * BHparams.M)
event_fn = event_single_func(
    partial(kerr_schild.capture_fn, epsilon=args.epsilon)
)

init_state = initialize_state(
    y0 = y0,
    params = BHparams,
    r_escape = r_escape,
    event_fn = event_fn,
    safe_y = safe_y
)

rhs = rhs_func(kerr_schild.metric_contra)
rk4 = rk4_step_func(rhs)
step_size_fn = adaptive_step_size_func(
    rhs,
    kerr_schild.radial_coordinate
)
integrate_chunk = integrate_chunk_func(
    rk4_step = rk4,
    event_single = event_fn,
    n_steps = MAX_STEPS,
    compute_step_size=step_size_fn
)

with RayHistoryWriter(args.output) as writer:
    parameters = writer.file.create_group("parameters")
    for name, value in vars(args).items():
        parameters.attrs[name] = value
    writer.write_initial(init_state)

    final_state = integrate(
        state=init_state,
        params=BHparams,
        h=h,
        r_escape=r_escape,
        integrate_chunk=integrate_chunk,
        max_chunks=MAX_CHUNKS,
        save_chunk=writer.save_chunk,
    )

    writer.write_final(final_state)