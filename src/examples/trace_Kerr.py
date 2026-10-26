# trace_Kerr.py

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


BHparams = KerrParams(M=1.0, a=0.0)

R = 50.0
inc = 60.0
freq = 1.0

FOV = 20.0
PPD = 0.25

r_esc_R = 2.0
h = -0.01
MAX_STEPS = 256
MAX_CHUNKS = 100

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
event_fn = event_single_func(kerr_schild.capture_fn)

init_state = initialize_state(
    y0 = y0,
    params = BHparams,
    r_escape = r_escape,
    event_fn = event_fn,
    safe_y = safe_y
)

rhs = rhs_func(kerr_schild.metric_contra)
rk4 = rk4_step_func(rhs)
integrate_chunk = integrate_chunk_func(
    rk4_step = rk4,
    event_single = event_fn,
    n_steps = MAX_STEPS
)

with RayHistoryWriter("rays.h5") as writer:
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