import jax
import jax.numpy as jnp

def hamiltonian_func(metric_contra):
    """
    Construct the null-geodesic Hamiltonian.

    y: (8,) = (x^mu, p_mu)
    params: metric parameters (PyTree)

    Returns:
        H: scalar
    """

    def hamiltonian(y, params):
        x = y[:4]
        p = y[4:]

        g_inv = metric_contra(x, params)

        return 0.5 * jnp.einsum(
            "i,ij,j->", p, g_inv, p
        )

    return hamiltonian

def rhs_func(metric_contra):
    """
    Construct Hamilton's equations.

    Returns:
        rhs(y, params): (8,)
    """
    H = hamiltonian_func(metric_contra)
    grad_H = jax.grad(H, argnums=0)

    def rhs(y, params):
        dH = grad_H(y, params)
        dx = dH[4:]
        dp = -dH[:4]

        return jnp.concatenate([dx, dp])

    return rhs