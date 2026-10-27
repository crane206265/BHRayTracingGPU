# camera.py

import jax.numpy as jnp


class KerrSchildCamera:
    def __init__(self, x_obs, freq, FOV, PPD):
        self.x_obs = jnp.asarray(x_obs)
        self.freq = freq
        self.FOV = FOV
        self.PPD = PPD
        
        self.init_pos = jnp.array([
            0.0,
            self.x_obs[0],
            self.x_obs[1],
            self.x_obs[2]
        ])
        self.N = int(FOV * PPD)

    def eulerian_tetrad(x, params, metric_cov):
        """
        Eulerian tetrad in Cartesian Kerr-Schild coordinates.

        Returns
        -------
        tetrad : (4, 4)
            tetrad[a, mu] = e_(a)^mu
        """

        g = metric_cov(x, params)
        gamma = g[1:, 1:]

        A = gamma - jnp.eye(3, dtype=g.dtype)
        H = 0.5*jnp.trace(A)
        
        alpha = 1.0 / jnp.sqrt(1 + 2*H)
        beta_cov = g[0, 1:]
        beta = jnp.linalg.solve(gamma, beta_cov)

        n = jnp.concatenate([
            jnp.array([1.0 / alpha], dtype=g.dtype),
            -beta / alpha
        ])

        s = jnp.sqrt(1.0 + 2 * H)
        coeff = -1.0 / (s * (1.0 + s))
        E = jnp.eye(3, dtype=g.dtype) + coeff * A

        tetrad = jnp.zeros((4, 4), dtype=g.dtype)
        tetrad = tetrad.at[0, :].set(n)
        tetrad = tetrad.at[1:, 1:].set(E.T)

        return tetrad


    def photonScreenPoint(self, alpha, beta, g, tetrad):
        to_local = tetrad[1:, 1:] @ g[1:, 1:]

        forward = to_local @ (-self.x_obs)
        forward = forward / jnp.linalg.norm(forward) # unit vector to BH direction in local frame

        up = to_local @ jnp.array([0.0, 0.0, 1.0], dtype=g.dtype)
        up = up - jnp.dot(up, forward) * forward
        up = up / jnp.linalg.norm(up) # Y("up") direction in local frame

        right = jnp.cross(forward, up)

        X = jnp.tan(alpha)
        Y = jnp.tan(beta)
        norm = jnp.sqrt(1 + X**2 + Y**2)

        screen_basis = jnp.stack(
            [forward, right, up], axis=1
        )
        screen_direction = jnp.stack(
            [jnp.ones_like(X), X, Y], axis=0
        )

        direction = jnp.einsum(
            "ij,j...->i...", screen_basis, screen_direction
        ) / norm

        p_local = self.freq * jnp.concatenate(
            [jnp.ones_like(X)[None, ...], -direction], axis=0
        )

        p_contra = jnp.einsum(
            "am,a...->m...", tetrad, p_local
        )
        p_cov = jnp.einsum(
            "mn,n...->m...", g, p_contra
        )

        return p_cov

    def photonScreen(self, g, tetrad):
        alpha = jnp.linspace(-self.FOV / 2, self.FOV / 2, self.N) * jnp.pi / 180
        beta = jnp.linspace(-self.FOV / 2, self.FOV / 2, self.N) * jnp.pi / 180
        alpha = jnp.tile(alpha, self.N).reshape((self.N, self.N))
        beta = jnp.repeat(beta, self.N).reshape((self.N, self.N))

        return self.photonScreenPoint(alpha, beta, g, tetrad)

