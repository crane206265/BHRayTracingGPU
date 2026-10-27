# disk_kinematics.py

# disk 유체의 4-velocity 계산

import jax.numpy as jnp

from src.types import KerrParams
from src.metrics.kerr_schild import radial_coordinate

def four_vel_thin(x, params: KerrParams):
    """
    Equatorial circular-orbit four-velocity in Cartesian Kerr-Schild coordinates

    Parameters
    ----------
    x : array
        Position (t, x, y, z)
    params : KerrParams
        M: mass in length units; a: signed spin parameter in length units

    Returns
    -------
    velocity : array
        Contravariant (u^t, u^x, u^y, u^z), g(u,u)=-1

    Shapes
    ------
    x        : (4,)
    velocity : (4,)

    Notes
    -----
    Units: G=c=1
    Orbit: z=0, dr=0, +phi rotation
    Spin convention: a>0 prograde, a<0 retrograde
    NT domain: r>=r_ISCO; plane and ISCO checks external
    Radial inflow: absent
    """
    x = jnp.asarray(x)
    r = radial_coordinate(x, params)
    # Dimensionless radius R=r/M, spin a_*=a/M
    radius = r / params.M
    spin = params.a / params.M
    radius_32 = radius ** 1.5
    # Omega=dphi/dt; dr=0 => identical BL / KS time and azimuth rates
    omega = 1.0 / (params.M * (radius_32 + spin))
    # u_t variable: contravariant u^t=dt/dtau, not covariant u_t
    u_t = (1.0 + spin / radius_32) / jnp.sqrt(
        1.0 - 3.0 / radius + 2.0 * spin / radius_32
    )
    # Cartesian tangent: dx/dt=-Omega*y, dy/dt=Omega*x, dz/dt=0
    velocity = jnp.stack((
        u_t,
        -u_t * omega * x[2],
        u_t * omega * x[1],
        jnp.zeros_like(u_t),
    ))
    # Basic parameter mask; circular-orbit existence required
    valid = (params.M > 0.0) & (jnp.abs(spin) <= 1.0) & (r > 0.0)
    return jnp.where(valid, velocity, jnp.nan)
