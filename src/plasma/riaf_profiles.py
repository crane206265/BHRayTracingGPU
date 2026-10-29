# riaf_profiles.py
# RIAF electron number density and temperature profiles

import jax.numpy as jnp


def num_density_electon(x, ne0, delta, H):
    """
    Electron number density in the semi-analytic RIAF model

    Parameters
    ----------
    x : array
        BL position (t, r, theta, phi); r in r_g=G*M_BH/c^2, angles in radians
    ne0 : scalar
        Nonnegative density normalization; cm^-3 for physical output
    delta : scalar
        Radial power-law index; fiducial value: 1.5
    H : scalar
        Positive dimensionless thickness parameter; fiducial value: 0.5

    Returns
    -------
    ne : scalar
        ne0*r^(-delta)*exp[-cot(theta)^2/(2*H^2)]
        Same units as ne0; polar axis: 0; invalid inputs: NaN

    Shapes
    ------
    x : (4,); ne0, delta, H, ne : scalar
    Batch evaluation: jax.vmap; model parameters shared or mapped as required

    Notes
    -----
    Coordinates: Boyer-Lindquist, not Cartesian Kerr-Schild
    Radius: dimensionless r/r_g; geometric M!=1 requires radius normalization
    Domain: finite r>0, 0<=theta<=pi; horizon / radial cutoffs external
    Equator: ne=ne0*r^(-delta); reflection symmetry: theta <-> pi-theta
    Polar limit: explicit zero; no artificial angular floor
    JIT: numeric parameters as dynamic arguments; no static_argnames required
    """
    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x, ne0, delta, H, 1.0))
    r, theta = x[1], x[2]
    valid_r = jnp.isfinite(r) & (r > 0.0)
    valid_H = jnp.isfinite(H) & (H > 0.0)
    safe_r = jnp.where(valid_r, r, 1.0)
    safe_H = jnp.where(valid_H, H, 1.0)
    on_axis = (theta == 0.0) | (theta == jnp.pi)
    # cot(theta)/H; safe denominator for the explicit polar-axis limit
    sin_theta = jnp.where(on_axis, 1.0, jnp.sin(theta))
    vertical_ratio = jnp.cos(theta) / (safe_H * sin_theta)
    ne = ne0 * safe_r**(-delta) * jnp.exp(-0.5 * vertical_ratio**2)
    ne = jnp.where(on_axis, 0.0, ne)
    valid = (
        valid_r & valid_H
        & jnp.isfinite(theta) & (theta >= 0.0) & (theta <= jnp.pi)
        & jnp.isfinite(ne0) & (ne0 >= 0.0)
        & jnp.isfinite(delta)
    )
    return jnp.where(valid, ne, jnp.nan)


def temperature_electon(x, te0, gamma):
    """
    Electron temperature in the semi-analytic RIAF model

    Parameters
    ----------
    x : array
        BL position (t, r, theta, phi); dimensionless r in gravitational radii
    te0 : scalar
        Nonnegative temperature normalization; Kelvin for physical output
    gamma : scalar
        Radial power-law index; fiducial value: 0.84

    Returns
    -------
    te : scalar
        te0*r^(-gamma); same units as te0; invalid inputs: NaN

    Shapes
    ------
    x : (4,); te0, gamma, te : scalar
    Batch evaluation: jax.vmap; model parameters shared or mapped as required

    Notes
    -----
    Coordinates: Boyer-Lindquist, not Cartesian Kerr-Schild
    Radius: dimensionless r/r_g; geometric M!=1 requires radius normalization
    Domain: finite r>0; horizon / radial cutoffs external
    Angular dependence: absent; density thickness parameter H not required
    JIT: numeric parameters as dynamic arguments; no static_argnames required
    """
    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x, te0, gamma, 1.0))
    r = x[1]
    valid_r = jnp.isfinite(r) & (r > 0.0)
    safe_r = jnp.where(valid_r, r, 1.0)
    te = te0 * safe_r**(-gamma)
    valid = (
        valid_r
        & jnp.isfinite(te0) & (te0 >= 0.0)
        & jnp.isfinite(gamma)
    )
    return jnp.where(valid, te, jnp.nan)
