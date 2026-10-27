# novikov_thorne.py

# disk emission model

import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

from src.types import KerrParams


# 64-point Gauss-Legendre quadrature on [-1, 1]; fixed NumPy nodes / weights
# Runtime integration: JAX arrays
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = leggauss(64)

def isco_radius_kerr(params:KerrParams):
    """
    Kerr ISCO radius for +phi equatorial motion

    Parameters
    ----------
    params : KerrParams
        M: positive finite mass; a: signed spin parameter, |a/M|<=1

    Returns
    -------
    r_isco : scalar
        Innermost stable circular-orbit radius; same units as M
        Invalid parameters: NaN

    Notes
    -----
    Formula: standard Z1, Z2 expression
    Spin convention: a>0 prograde, a<0 retrograde
    Limits: a=0 -> 6M, a=M -> M, a=-M -> 9M
    """
    a, M = params.a, params.M
    spin = a / M
    z1 = 1.0 + jnp.cbrt(1.0 - spin**2) * (
        jnp.cbrt(1.0 + spin) + jnp.cbrt(1.0 - spin)
    )
    z2 = jnp.sqrt(3.0 * spin**2 + z1**2)
    # Roundoff protection for the nonnegative radicand
    radius = M * (
        3.0 + z2 - jnp.sign(spin) * jnp.sqrt(
            jnp.maximum((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0)
        )
    )
    valid = (M > 0.0) & jnp.isfinite(M) & (jnp.abs(spin) <= 1.0)
    return jnp.where(valid, radius, jnp.nan)

def nt_flux(r, params: KerrParams, mdot):
    """
    Novikov-Thorne disk surface flux

    Parameters
    ----------
    r : scalar or array
        Kerr radial coordinate
    params : KerrParams
        M, a in geometric units
    mdot : scalar
        Steady rest-mass accretion rate; positive for inflow

    Returns
    -------
    flux : scalar or array
        Local fluid-frame bolometric flux from one disk face
        r<=r_ISCO: 0; invalid inputs: NaN

    Shapes
    ------
    flux : shape of r; scalar M, a, mdot

    Notes
    -----
    Units: G=c=1
    Inner boundary: r_in=r_ISCO, zero torque
    D = E - Omega*L
    F = -mdot*Omega_,r / (4*pi*r*D^2) * integral[D*L_,r dr]
    Integration variable: x=sqrt(r/M)
    Quadrature: 64-point Gauss-Legendre
    Reference: Page & Thorne (1974), ApJ 191, 499, eqs. (11)-(15)
    """
    r = jnp.asarray(r)
    M, a = params.M, params.a
    spin = a / M
    r_in = isco_radius_kerr(params)
    # Integration coordinate x=sqrt(r/M), not a position four-vector
    x_in = jnp.sqrt(r_in / M)
    emitting = r > r_in
    # Safe evaluation radius for masked points r<=r_ISCO
    radius = jnp.where(emitting, r / M, r_in / M + 1.0)
    x = jnp.sqrt(radius)
    width = x - x_in
    nodes = jnp.asarray(_QUADRATURE_NODES, dtype=x.dtype)
    weights = jnp.asarray(_QUADRATURE_WEIGHTS, dtype=x.dtype)
    # Node mapping: [-1, 1] -> [x_in, x]; quadrature axis: last
    delta = 0.5 * width[..., None] * (nodes + 1.0)
    x_quad = x_in + delta
    # P(q)=q^4-6q^2+8*a_*q-3*a_*^2
    # Expansion at x_in: P(x_in)=0, delta=q-x_in; reduced cancellation
    numerator = delta * (
        4.0 * x_in**3 - 12.0 * x_in + 8.0 * spin
        + delta * (6.0 * x_in**2 - 6.0 + delta * (4.0 * x_in + delta))
    )
    # q*(q^3-3q+2*a_*); factored form near a_*=1, q=1
    denominator = x_quad * (
        (x_quad - 1.0)**2 * (x_quad + 2.0) + 2.0 * (spin - 1.0)
    )
    # I=integral P(q)/[q*(q^3-3q+2*a_*)] dq; limits: x_in -> x; Jacobian: width/2
    integral = 0.5 * width * jnp.sum(weights * numerator / denominator, axis=-1)
    orbit_factor = (x - 1.0)**2 * (x + 2.0) + 2.0 * (spin - 1.0)
    # One-face flux: 3*mdot*I/[8*pi*M^2*x^4*(x^3-3x+2*a_*)]
    flux = 3.0 * mdot * integral / (8.0 * jnp.pi * M**2 * x**4 * orbit_factor)
    # Nonnegative flux; zero emission for r<=r_ISCO
    flux = jnp.where(emitting, jnp.maximum(flux, 0.0), 0.0)
    valid = (
        (M > 0.0) & jnp.isfinite(M) & (jnp.abs(spin) <= 1.0)
        & (r >= 0.0) & jnp.isfinite(r) & (mdot >= 0.0) & jnp.isfinite(mdot)
    )
    return jnp.where(valid, flux, jnp.nan)

def eff_temp(flux, sigma_sb):
    """
    Effective temperature from the Stefan-Boltzmann law

    Parameters
    ----------
    flux : scalar or array
        Nonnegative finite local bolometric flux
    sigma_sb : scalar or array
        Positive finite Stefan-Boltzmann constant; units consistent with flux

    Returns
    -------
    temperature : scalar or array
        (flux / sigma_sb)^0.25
        Zero flux: 0; invalid inputs: NaN

    Shapes
    ------
    temperature : broadcast shape of flux, sigma_sb

    Notes
    -----
    Kelvin output: matching physical units for flux and sigma_sb
    Geometric flux + SI constant: unit conversion required
    """
    flux = jnp.asarray(flux)
    sigma_sb = jnp.asarray(sigma_sb)
    valid = (
        (flux >= 0.0) & jnp.isfinite(flux)
        & (sigma_sb > 0.0) & jnp.isfinite(sigma_sb)
    )
    return jnp.where(valid, (flux / sigma_sb)**0.25, jnp.nan)

