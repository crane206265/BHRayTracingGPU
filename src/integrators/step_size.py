# step_size.py

import math

import jax
import jax.numpy as jnp

from src.metrics import kerr_schild

def simple_step_size(y, params, h):
    return jnp.asarray(h, dtype=y.dtype)


def adaptive_step_size_func(rhs, radial_coordinate, *, far_h_max=None, growth_radius=30.0):
    growth_radius = float(growth_radius)
    if not math.isfinite(growth_radius) or growth_radius <= 0:
        raise ValueError("growth_radius must be finite and positive.")
    if far_h_max is not None:
        far_h_max = float(far_h_max)
        if not math.isfinite(far_h_max) or far_h_max <= 0:
            raise ValueError("far_h_max must be finite and positive.")

    grad_r = jax.grad(radial_coordinate, argnums=0)

    def adaptive_step_size(y, params, h_max):
        x = y[:4]
        p = y[4:]

        dy = rhs(y, params)
        xdot = dy[:4]
        pdot = dy[4:]

        M, a = params.M, params.a
        r = radial_coordinate(x, params)
        r_plus = M + jnp.sqrt(M**2 - a**2)

        # Geometry
        spatial_speed = jnp.linalg.norm(xdot[1:])
        h_geom = (
            0.02 * jnp.maximum(r, M)
            / jnp.maximum(spatial_speed, 1e-14)
        )

        # Horizon
        rdot = jnp.dot(grad_r(x, params), xdot)
        #r_stop = r_plus + 0.01 * M

        h_horizon = jnp.where(
            rdot > 0,
            #0.5 * jnp.maximum(r - r_stop, 0.0)
            0.5 * jnp.maximum(r - r_plus, 0.0)
            / jnp.maximum(jnp.abs(rdot), 1e-14),
            jnp.inf
        )

        # Momentum
        p_scale = jnp.maximum(jnp.linalg.norm(p[1:]), 1.0)
        pdot_norm = jnp.linalg.norm(pdot[1:])

        h_mom = (
            0.05 * p_scale
            / jnp.maximum(pdot_norm, 1e-14)
        )

        h_cap = jnp.abs(h_max)
        if far_h_max is not None:
            growth = jnp.clip((r - growth_radius) / growth_radius, 0.0, 1.0)
            far_cap = jnp.maximum(h_cap, far_h_max)
            h_cap = h_cap + growth * (far_cap - h_cap)

        h_abs = jnp.minimum(
            h_cap,
            jnp.minimum(
                h_geom,
                jnp.minimum(h_horizon, h_mom)
            )
        )

        return -h_abs

    return adaptive_step_size