# thin_disk.py

# disk 위치/방향/반경/ray 교차 탐지

import jax.numpy as jnp

from src.types import KerrParams
from src.metrics import kerr_schild

# 반경 범위 검사
def is_within_disk_radius(r, r_in, r_out):
    """
    Disk radius mask

    Parameters
    ----------
    r : array_like
        Kerr radial coordinate
    r_in, r_out : scalar
        Inner / outer disk radius; same units as r

    Returns
    -------
    within : bool or bool array
        (r >= r_in) & (r <= r_out)

    Shapes
    ------
    within : broadcast shape of r, r_in, r_out
    """
    return (r >= r_in) & (r <= r_out)

# 원반 평면 교점 근사
def intersect_disk_plane(x0, x1):
    """
    Single-segment intersection with z=0

    Parameters
    ----------
    x0, x1 : array
        Cartesian Kerr-Schild positions (t, x, y, z)

    Returns
    -------
    crosses_plane : bool scalar
        Opposite sides or arrival at z1=0; departure from z0=0 excluded
    weight : scalar
        z0 / (z0 - z1); -1 without a crossing
    x_hit : array
        (1 - weight)*x0 + weight*x1; x0 without a crossing

    Shapes
    ------
    x0, x1 : (4,)
    x_hit  : (4,)

    Notes
    -----
    Geometry: fixed plane z=0, straight-segment approximation
    Coplanar segment: crosses_plane=False
    """
    _, _, _, z0 = x0
    _, _, _, z1 = x1

    # Endpoint rule: z1=0 included, z0=0 excluded
    # Coplanar segment: no crossing
    crosses_plane = (
        ((z0 > 0) & (z1 <= 0))
        | ((z0 < 0) & (z1 >= 0))
    )
    # Safe denominator for non-crossing segments
    denominator = jnp.where(crosses_plane, z0 - z1, 1.0)
    weight = jnp.where(crosses_plane, z0 / denominator, -1.0)
    # Four-position interpolation; no-crossing fallback: x0
    x_hit = jnp.where(
        crosses_plane,
        (1-weight)*x0 + weight*x1,
        x0 # not effective value
    )

    return crosses_plane, weight, x_hit

# 원반 교차 판정
def intersect_thin_disk(x0, x1, params:KerrParams, r_in, r_out):
    """
    Single-segment intersection with a finite equatorial disk

    Parameters
    ----------
    x0, x1 : array
        Cartesian Kerr-Schild positions (t, x, y, z)
    params : KerrParams
        M, a in geometric units
    r_in, r_out : scalar
        Inner / outer Kerr radius; r_in typically r_ISCO

    Returns
    -------
    hit_disk : bool scalar
        Plane crossing & (r_in <= r_hit <= r_out)
    weight : scalar
        Plane-intersection weight; -1 without a plane crossing
    x_hit : array
        Plane-intersection position; x0 without a plane crossing

    Shapes
    ------
    x0, x1 : (4,)
    x_hit  : (4,)

    Notes
    -----
    Emission validity: hit_disk
    Plane crossings outside the radial bounds: hit_disk=False
    """
    crosses_plane, weight, x_hit = intersect_disk_plane(x0, x1)
    # Kerr radius for the radial mask
    r_hit = kerr_schild.radial_coordinate(x_hit, params)
    hit_disk = is_within_disk_radius(r_hit, r_in, r_out) & crosses_plane

    return hit_disk, weight, x_hit

