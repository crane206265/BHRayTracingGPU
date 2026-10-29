# transfer.py

# ray history로부터 redshift, 관측 intensity 계산

import jax
import jax.numpy as jnp

from src.types import KerrParams
from src.plasma.thin_disk import intersect_thin_disk, intersect_disk_plane

PI = jnp.pi

def find_first_disk_hit(y_history, params:KerrParams, r_in, r_out):
    """
    First valid disk hit along one backward-traced ray

    Parameters
    ----------
    y_history : array
        Observer-first ray history; rows (x^mu, p_mu)
        Cache input: read_ray(..., include_initial=True)
    params : KerrParams
        M, a in geometric units
    r_in, r_out : scalar
        Inner / outer disk radius

    Returns
    -------
    found : bool scalar
        Valid disk-hit mask
    y_hit : array
        Position and covariant momentum; same linear-interpolation weight
    image_order : integer scalar
        Equatorial-plane crossings through the selected hit minus one
        Direct image: 0; crossings outside disk radial bounds included

    Shapes
    ------
    y_history : (S, 8)
    y_hit     : (8,)

    Notes
    -----
    Status: unimplemented stub; intended interface only
    Surface: opaque, aligned equatorial disk
    Selection: first valid hit in backward-tracing order
    Output validity: found=True for y_hit and image_order
    """
    if y_history.shape[0] == 1:
        return (
            jnp.array(False),
            y_history[0],
            jnp.array(-1, dtype=jnp.int32),
        )
    
    x0 = y_history[:-1, :4]
    x1 = y_history[1:, :4]

    intersect_segments = jax.vmap(
        intersect_thin_disk,
        in_axes=(0, 0, None, None, None),
    )

    hits, weights, x_hits = intersect_segments(
        x0, x1, params, r_in, r_out,
    )

    found = jnp.any(hits)
    first_idx = jnp.argmax(hits)

    w = jnp.where(found, weights[first_idx], 0.0)
    interpolated = (1 - w) * y_history[first_idx] + w * y_history[first_idx + 1]
    y_hit = jnp.where(found, interpolated, y_history[0])

    plane_crosses, _, _ = jax.vmap(intersect_disk_plane)(x0, x1)
    orders = jnp.cumsum(plane_crosses, dtype=jnp.int32) - 1
    image_order = jnp.where(found, orders[first_idx], -1)

    return found, y_hit, image_order

def redshift_factor(p_obs, u_obs, p_em, u_em):
    """
    Photon frequency shift between emitter and observer

    Parameters
    ----------
    p_obs : array
        Covariant photon momentum p_mu at the observer
    u_obs : array
        Contravariant observer four-velocity u^mu
    p_em : array
        Covariant photon momentum p_mu at the disk emission point
    u_em : array
        Contravariant emitter four-velocity u^mu; four_vel_thin at the disk hit

    Returns
    -------
    g : scalar
        nu_obs / nu_em = E_obs / E_em
        g < 1: redshift; g > 1: blueshift

    Shapes
    ------
    p_obs, u_obs : (4,)
    p_em, u_em   : (4,)
    g            : scalar

    Notes
    -----
    Status: unimplemented stub; intended interface only
    Metric signature: (-, +, +, +)
    E_obs = -p_obs_mu * u_obs^mu
    E_em  = -p_em_mu  * u_em^mu
    Contraction: sum over mu; matching coordinate basis at each event
    Momentum: already covariant; no additional metric contraction
    Backward tracing: future-directed photon momentum, negative affine steps
    Physical domain: future-directed timelike velocities, E_obs > 0, E_em > 0
    Disk-hit validity: found=True from find_first_disk_hit
    """
    E_obs = -jnp.dot(p_obs, u_obs)
    E_em = -jnp.dot(p_em, u_em)
    g = E_obs/E_em
    return g


def observed_bolometric_intensity(flux_em, g):
    """
    Observed bolometric intensity from local disk surface flux

    Parameters
    ----------
    flux_em : scalar or array
        Evaluated local fluid-frame bolometric flux from one disk face
        Example source: nt_flux(r_hit, params, mdot); numeric value, not a callable
    g : scalar or array
        Positive frequency ratio nu_obs / nu_em from redshift_factor

    Returns
    -------
    I_obs : scalar or array
        Observed frequency-integrated intensity: g^4 * flux_em / pi
        Units: flux units per steradian

    Shapes
    ------
    I_obs : broadcast shape of flux_em, g

    Notes
    -----
    Angular emission: direction-independent intensity over the outward hemisphere
    Local surface flux: F_em = integral I_em*cos(theta) dOmega = pi*I_em
    Vacuum propagation: I_obs = g^4 * I_em
    Input assumptions: finite flux_em >= 0, finite g > 0
    Disk-hit mask: external; background intensity 0 for found=False
    """
    I_em = flux_em/PI
    I_obs = (g**4) * I_em
    return I_obs


# transfer.py
# Volume GRRT along ray histories using local emission and absorption coefficients
