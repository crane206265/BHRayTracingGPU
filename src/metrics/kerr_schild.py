# kerr_schild.py

import jax.numpy as jnp

ETA = jnp.diag(jnp.array(
    [-1.0, 1.0, 1.0, 1.0],
    dtype=jnp.float64
))


def ks_components(x, params):
    """Ingoing Cartesian Kerr-Schild coordinates."""
    _, X, Y, Z = x
    M, a = params.M, params.a
    R2 = X**2 + Y**2 + Z**2

    r2 = 0.5 * (
        R2 - a**2
        + jnp.sqrt((R2 - a**2)**2 + 4*a**2*Z**2)
    )
    r = jnp.sqrt(r2)

    H = M * r**3 / (r**4 + a**2 * Z**2)

    l_cov = jnp.array([
        1.0,
        (r*X + a*Y) / (r2 + a**2),
        (r*Y - a*X) / (r2 + a**2),
        Z / r
    ])

    l_contra = ETA @ l_cov

    return H, l_cov, l_contra


def metric_cov(x, params):
    H, l_cov, _ = ks_components(x, params)

    return ETA + 2*H*jnp.outer(l_cov, l_cov)


def metric_contra(x, params):
    H, _, l_contra = ks_components(x, params)

    return ETA - 2*H*jnp.outer(l_contra, l_contra)

# --------------------

def radial_coordinate(x, params):
    _, X, Y, Z = x
    a = params.a

    R2 = X**2 + Y**2 + Z**2

    r2 = 0.5 * (
        R2 - a**2
        + jnp.sqrt(
            (R2 - a**2)**2 + 4*(a**2)*(Z**2)
        )
    )

    return jnp.sqrt(r2)

def capture_fn(x, params):
    # Kerr BH Horizon
    M = params.M
    a = params.a

    r = radial_coordinate(x, params)

    r_plus = M + jnp.sqrt(M**2 - a**2)

    return r <= r_plus