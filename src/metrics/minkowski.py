# minkowski.py

import jax.numpy as jnp

ETA = jnp.diag(jnp.array(
    [-1.0, 1.0, 1.0, 1.0],
    dtype=jnp.float64
))

def metric_cov(x, params):
    # x: (4,)
    # return: (4, 4)
    # g_{\mu\nu}(x)
    return ETA

def metric_contra(x, params):
    # x: (4,)
    # return: (4, 4)
    # g^{\mu\nu}(x)
    return ETA