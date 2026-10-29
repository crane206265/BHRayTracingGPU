# kerr_coordinates.py
# Cartesian Kerr-Schild <-> Boyer-Lindquist coordinates, vectors, and covectors

import jax.numpy as jnp

from src.metrics.kerr_schild import radial_coordinate
from src.types import KerrParams


def _cks_geometry(x_cks, params):
    """Kerr radius, polar angle factors, and exterior coordinate-domain mask."""
    x_cks = jnp.asarray(x_cks)
    x_cks = x_cks.astype(jnp.result_type(x_cks, params.M, params.a, 1.0))
    _, X, Y, Z = x_cks
    M, a = params.M, params.a
    r = radial_coordinate(x_cks, params)
    sin_theta = jnp.hypot(X, Y) / jnp.sqrt(r*r + a*a)
    cos_theta = Z / r
    r_plus = M + jnp.sqrt((M - a) * (M + a))
    valid = (
        jnp.all(jnp.isfinite(x_cks))
        & jnp.isfinite(M) & (M > 0.0)
        & jnp.isfinite(a) & (jnp.abs(a) <= M)
        & (r > r_plus) & (sin_theta > 0.0)
    )
    return x_cks, r, sin_theta, cos_theta, valid


def _radial_shifts(r, params):
    """
    Integrated ingoing KS time / azimuth shifts relative to BL coordinates

    Notes
    -----
    t_KS = t_BL + T(r); phi_KS = phi_BL + P(r)
    T' = 2*M*r/Delta; P' = a/Delta
    q = r/M - 1; d = sqrt(1 - (a/M)^2)
    A = atanh(d/q)/d; extremal limit: A = 1/q
    T = M*[log(q-d) + log(q+d) - 2*A]; P = -(a/M)*A
    Additive coordinate constants: fixed by these dimensionless primitives
    Domain: r>r_+, M>0, |a|<=M
    Reference: spectre-code.org/classgr_1_1KerrSchildCoords.html
    """
    M, a = params.M, params.a
    spin = a / M
    q = r / M - 1.0
    # Factored discriminant; signed spin and exact |a|=M limits
    d = jnp.sqrt((1.0 - spin) * (1.0 + spin))
    safe_d = jnp.where(d > 0.0, d, 1.0)
    A = jnp.where(d > 0.0, jnp.arctanh(d / q) / safe_d, 1.0 / q)
    time_shift = M * (jnp.log(q - d) + jnp.log(q + d) - 2.0 * A)
    phi_shift = -spin * A
    return time_shift, phi_shift


def _bl_to_cks_jacobian(x_cks, params):
    """
    K^mu_nu = partial x_CKS^mu / partial x_BL^nu at a CKS position

    Shapes
    ------
    x_cks : (4,); K : (4, 4)

    Notes
    -----
    Rows: (t_KS, X, Y, Z); columns: (t_BL, r, theta, phi_BL)
    Analytic local derivatives; no differentiation through angular wrapping
    Ingoing convention: X+iY = (r+i*a)*exp(i*phi_KS)*sin(theta)
    Invalid exterior / polar-axis domain: all NaN
    """
    x_cks, r, sin_theta, cos_theta, valid = _cks_geometry(x_cks, params)
    _, X, Y, _ = x_cks
    M, a = params.M, params.a
    # Delta=(r-r_+)*(r-r_-); reduced cancellation near extremal horizons
    horizon_gap = jnp.sqrt((M - a) * (M + a))
    delta = (r - M - horizon_gap) * (r - M + horizon_gap)
    azimuth_rate = a / delta
    polar_factor = cos_theta / sin_theta
    zero = jnp.zeros_like(r)
    one = jnp.ones_like(r)
    K = jnp.stack((
        jnp.stack((one, 2.0*M*r/delta, zero, zero)),
        jnp.stack((zero, (r*X + a*Y)/(r*r + a*a) - Y*azimuth_rate,
                   X*polar_factor, -Y)),
        jnp.stack((zero, (r*Y - a*X)/(r*r + a*a) + X*azimuth_rate,
                   Y*polar_factor, X)),
        jnp.stack((zero, cos_theta, -r*sin_theta, zero)),
    ))
    return jnp.where(valid, K, jnp.nan)


def cks2bl_position(x_cks, params: KerrParams):
    """
    Position from ingoing Cartesian Kerr-Schild to Boyer-Lindquist coordinates

    Parameters
    ----------
    x_cks : array
        Position (t_KS, X, Y, Z)
    params : KerrParams
        Positive mass M; signed dimensional spin a, |a|<=M

    Returns
    -------
    x_bl : array
        Position (t_BL, r, theta, phi_BL); invalid domain: all NaN

    Shapes
    ------
    x_cks, x_bl : (4,); single point; batching via jax.vmap

    Notes
    -----
    Units: G=c=1; lengths and times in the same units as M
    Domain: exterior r>r_+, away from the polar axis
    Angles: radians; theta in (0, pi); phi wrapped via atan2
    Time / azimuth origins: shared primitives in _radial_shifts
    Conventions: src.metrics.kerr_schild; stationary BH at origin, spin along z
    """
    x_cks, r, sin_theta, cos_theta, valid = _cks_geometry(x_cks, params)
    t, X, Y, _ = x_cks
    a = params.a
    time_shift, phi_shift = _radial_shifts(r, params)
    theta = jnp.arctan2(sin_theta, cos_theta)
    phi_ks = jnp.arctan2(r*Y - a*X, r*X + a*Y)
    phi_bl = phi_ks - phi_shift
    phi_bl = jnp.arctan2(jnp.sin(phi_bl), jnp.cos(phi_bl))
    x_bl = jnp.stack((t - time_shift, r, theta, phi_bl))
    return jnp.where(valid, x_bl, jnp.nan)


def bl2cks_position(x_bl, params: KerrParams):
    """
    Position from Boyer-Lindquist to ingoing Cartesian Kerr-Schild coordinates

    Parameters
    ----------
    x_bl : array
        Position (t_BL, r, theta, phi_BL); angles in radians
    params : KerrParams
        Positive mass M; signed dimensional spin a, |a|<=M

    Returns
    -------
    x_cks : array
        Position (t_KS, X, Y, Z); invalid domain: all NaN

    Shapes
    ------
    x_bl, x_cks : (4,); single point; batching via jax.vmap

    Notes
    -----
    Units: G=c=1; lengths and times in the same units as M
    Domain: exterior r>r_+, 0<theta<pi
    Ingoing map: X+iY = (r+i*a)*exp(i*phi_KS)*sin(theta)
    Time / azimuth origins: same primitives as cks2bl_position
    Round trip: phi_BL equal modulo 2*pi
    """
    x_bl = jnp.asarray(x_bl)
    x_bl = x_bl.astype(jnp.result_type(x_bl, params.M, params.a, 1.0))
    t, r, theta, phi = x_bl
    M, a = params.M, params.a
    time_shift, phi_shift = _radial_shifts(r, params)
    phi_ks = phi + phi_shift
    sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
    sin_phi, cos_phi = jnp.sin(phi_ks), jnp.cos(phi_ks)
    x_cks = jnp.stack((
        t + time_shift,
        (r*cos_phi - a*sin_phi) * sin_theta,
        (r*sin_phi + a*cos_phi) * sin_theta,
        r*cos_theta,
    ))
    r_plus = M + jnp.sqrt((M - a) * (M + a))
    valid = (
        jnp.all(jnp.isfinite(x_bl))
        & jnp.isfinite(M) & (M > 0.0)
        & jnp.isfinite(a) & (jnp.abs(a) <= M)
        & (r > r_plus) & (theta > 0.0) & (theta < jnp.pi)
    )
    return jnp.where(valid, x_cks, jnp.nan)


def cks2bl_contra(x_cks, v_cks, params: KerrParams):
    """
    Contravariant four-vector from Cartesian KS to BL components

    Parameters
    ----------
    x_cks : array
        CKS position of the vector
    v_cks : array
        Contravariant components (v^t, v^X, v^Y, v^Z)
    params : KerrParams
        M, a in geometric units

    Returns
    -------
    v_bl : array
        Contravariant components (v^t, v^r, v^theta, v^phi)

    Shapes
    ------
    x_cks, v_cks, v_bl : (4,); single vector; batching via jax.vmap

    Notes
    -----
    K = partial x_CKS / partial x_BL; solve K*v_BL = v_CKS
    Domain: cks2bl_position; invalid domain: all NaN
    Applicable to u^mu, k^mu, b^mu; covariant p_mu: cks2bl_cov
    """
    K = _bl_to_cks_jacobian(x_cks, params)
    return jnp.linalg.solve(K, jnp.asarray(v_cks))


def bl2cks_contra(x_bl, v_bl, params: KerrParams):
    """
    Contravariant four-vector from BL to Cartesian KS components

    Parameters
    ----------
    x_bl : array
        BL position of the vector
    v_bl : array
        Contravariant components (v^t, v^r, v^theta, v^phi)
    params : KerrParams
        M, a in geometric units

    Returns
    -------
    v_cks : array
        Contravariant components (v^t, v^X, v^Y, v^Z)

    Shapes
    ------
    x_bl, v_bl, v_cks : (4,); single vector; batching via jax.vmap

    Notes
    -----
    K = partial x_CKS / partial x_BL; v_CKS = K*v_BL
    Domain: bl2cks_position; invalid domain: all NaN
    Position argument: original BL position, not the vector components
    """
    x_cks = bl2cks_position(x_bl, params)
    K = _bl_to_cks_jacobian(x_cks, params)
    return K @ jnp.asarray(v_bl)


def cks2bl_cov(x_cks, p_cks, params: KerrParams):
    """
    Covariant four-vector from Cartesian KS to BL components

    Parameters
    ----------
    x_cks : array
        CKS position of the covector
    p_cks : array
        Covariant components (p_t, p_X, p_Y, p_Z); ray-history momentum convention
    params : KerrParams
        M, a in geometric units

    Returns
    -------
    p_bl : array
        Covariant components (p_t, p_r, p_theta, p_phi)

    Shapes
    ------
    x_cks, p_cks, p_bl : (4,); single covector; batching via jax.vmap

    Notes
    -----
    K = partial x_CKS / partial x_BL; p_BL = K.T*p_CKS
    Domain: cks2bl_position; invalid domain: all NaN
    Invariant contraction: p_CKS_mu*v_CKS^mu = p_BL_mu*v_BL^mu
    Metric lowering / raising: unnecessary for this coordinate transformation
    """
    K = _bl_to_cks_jacobian(x_cks, params)
    return K.T @ jnp.asarray(p_cks)


def bl2cks_cov(x_bl, p_bl, params: KerrParams):
    """
    Covariant four-vector from BL to Cartesian KS components

    Parameters
    ----------
    x_bl : array
        BL position of the covector
    p_bl : array
        Covariant components (p_t, p_r, p_theta, p_phi)
    params : KerrParams
        M, a in geometric units

    Returns
    -------
    p_cks : array
        Covariant components (p_t, p_X, p_Y, p_Z)

    Shapes
    ------
    x_bl, p_bl, p_cks : (4,); single covector; batching via jax.vmap

    Notes
    -----
    K = partial x_CKS / partial x_BL; solve K.T*p_CKS = p_BL
    Domain: bl2cks_position; invalid domain: all NaN
    Metric lowering / raising: unnecessary for this coordinate transformation
    """
    x_cks = bl2cks_position(x_bl, params)
    K = _bl_to_cks_jacobian(x_cks, params)
    return jnp.linalg.solve(K.T, jnp.asarray(p_bl))
