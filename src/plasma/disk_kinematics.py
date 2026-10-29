# disk_kinematics.py

# disk 유체의 4-velocity 계산

import jax.numpy as jnp

from src.types import KerrParams
from src.metrics.kerr_schild import radial_coordinate
from src.plasma.novikov_thorne import isco_radius_kerr

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


# disk_kinematics.py
# Kerr circular/plunging reference, free fall, and mixed-flow four-velocity normalization

def _valid_kerr_params(params):
    """Finite M>0 and signed dimensional spin |a|<=M."""
    return (
        jnp.isfinite(params.M) & (params.M > 0.0)
        & jnp.isfinite(params.a) & (jnp.abs(params.a) <= params.M)
    )


def _bl_flow_geometry(x, params):
    """Dimensionless BL geometry; single exterior point away from the axis."""
    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x, params.M, params.a, 1.0))
    r, theta = x[1], x[2]
    R, spin = r / params.M, params.a / params.M
    sin2 = jnp.sin(theta)**2
    sigma = R*R + spin*spin*jnp.cos(theta)**2
    horizon_gap = jnp.sqrt((1.0 - spin) * (1.0 + spin))
    # Delta/M^2; factored form near extremal horizons
    delta = (R - 1.0 - horizon_gap) * (R - 1.0 + horizon_gap)
    # A/M^4; positive-sum form outside the horizon
    area = (R*R + spin*spin)*sigma + 2.0*R*spin*spin*sin2
    valid = (
        _valid_kerr_params(params) & jnp.all(jnp.isfinite(x))
        & (R > 1.0 + horizon_gap)
        & (theta > 0.0) & (theta < jnp.pi)
    )
    return R, spin, sin2, sigma, delta, area, valid


def _normalize_bl_flow(x, params, u_r, omega):
    """
    Future-directed BL four-velocity from prescribed u^r and Omega

    Notes
    -----
    u^theta=0; u^phi=Omega*u^t
    u^t=sqrt[(1+g_rr*(u^r)^2)/(-g_tt-2*g_tphi*Omega-g_phiphi*Omega^2)]
    Denominator: alpha^2-g_phiphi*(Omega-omega_ZAMO)^2
    Dimensionless metric factors; M*Omega used for angular rates
    No clipping of non-timelike prescriptions; invalid domain: all NaN
    """
    _, spin, sin2, sigma, delta, area, valid = _bl_flow_geometry(x, params)
    R = jnp.asarray(x)[1] / params.M
    lapse2 = sigma * delta / area
    g_phiphi = area * sin2 / sigma
    omega_zamo = 2.0 * spin * R / area
    denominator = lapse2 - g_phiphi * (params.M*omega - omega_zamo)**2
    valid = (
        valid & jnp.isfinite(u_r) & jnp.isfinite(omega)
        & jnp.isfinite(denominator) & (denominator > 0.0)
        & (u_r <= 0.0)
    )
    safe_denominator = jnp.where(valid, denominator, 1.0)
    numerator = 1.0 + sigma / delta * u_r*u_r
    u_t = jnp.sqrt(jnp.where(valid, numerator, 1.0) / safe_denominator)
    velocity = jnp.stack((u_t, u_r, jnp.zeros_like(u_t), omega*u_t))
    valid = valid & jnp.all(jnp.isfinite(velocity)) & (u_t > 0.0)
    return jnp.where(valid, velocity, jnp.nan)


def const_kerr_circ(r, params: KerrParams):
    """
    Specific energy and angular momentum of a +phi equatorial circular orbit

    Parameters
    ----------
    r : scalar
        BL radius; same geometric length units as M
    params : KerrParams
        Positive mass M; signed dimensional spin a, |a|<=M

    Returns
    -------
    E_K : scalar
        Specific energy -u_t; dimensionless
    L_K : scalar
        Specific angular momentum u_phi; same length units as M

    Shapes
    ------
    r, E_K, L_K : scalar; batching via jax.vmap

    Notes
    -----
    R=r/M, s=a/M, z=sqrt(R)
    E=(z^3-2*z+s)/[z^(3/2)*sqrt(z^3-3*z+2*s)]
    L/M=(z^4-2*s*z+s^2)/[z^(3/2)*sqrt(z^3-3*z+2*s)]
    Orbit: +phi; signed a distinguishes prograde / retrograde
    Domain: timelike circular orbits outside r_+; stability checks external
    Exact a=M, r=M: limiting ISCO values E=1/sqrt(3), L=2*M/sqrt(3)
    Plunge constants: evaluate at r=isco_radius_kerr(params)
    Invalid domain: NaN; units: G=c=1
    """
    r = jnp.asarray(r)
    r = r.astype(jnp.result_type(r, params.M, params.a, 1.0))
    R, spin = r / params.M, params.a / params.M
    z = jnp.sqrt(R)
    dz = z - 1.0
    ds = spin - 1.0
    orbit_factor = dz*dz*(z + 2.0) + 2.0*ds
    denominator = z**1.5 * jnp.sqrt(jnp.where(orbit_factor > 0.0, orbit_factor, 1.0))
    energy = (dz*(z*z + z - 1.0) + ds) / denominator
    angular = (
        dz*(z**3 + z*z + z - 1.0) + ds*(spin + 1.0 - 2.0*z)
    ) / denominator
    # Cancel the common z-1 factor analytically for exact positive extremal spin.
    extremal_denominator = z**1.5 * jnp.sqrt(z + 2.0)
    extremal = (spin == 1.0) & (R >= 1.0)
    energy = jnp.where(extremal, (z*z + z - 1.0)/extremal_denominator, energy)
    angular = jnp.where(extremal, (z**3 + z*z + z - 1.0)/extremal_denominator, angular)
    r_plus = params.M + jnp.sqrt((params.M - params.a)*(params.M + params.a))
    valid = (
        _valid_kerr_params(params) & jnp.isfinite(r) & (r > 0.0)
        & (((r > r_plus) & (orbit_factor > 0.0)) | extremal)
    )
    return (
        jnp.where(valid, energy, jnp.nan),
        jnp.where(valid, params.M*angular, jnp.nan),
    )


def omega_kerr_circ(r, params: KerrParams):
    """
    Angular velocity of the +phi equatorial circular reference

    Parameters
    ----------
    r : scalar
        BL radius; same geometric length units as M
    params : KerrParams
        Positive M; signed dimensional spin a

    Returns
    -------
    omega : scalar
        dphi/dt_BL = 1/[M*((r/M)^(3/2)+a/M)]; inverse length units

    Shapes
    ------
    r, omega : scalar; batching via jax.vmap

    Notes
    -----
    Theta dependence: absent by prescription
    Domain: finite r>0, valid Kerr parameters, positive rate denominator
    Timelike circular-orbit existence / ISCO selection: external
    Invalid domain: NaN
    """
    r = jnp.asarray(r)
    r = r.astype(jnp.result_type(r, params.M, params.a, 1.0))
    denominator = params.M * ((r/params.M)**1.5 + params.a/params.M)
    valid = (
        _valid_kerr_params(params) & jnp.isfinite(r) & (r > 0.0)
        & (denominator > 0.0)
    )
    return jnp.where(valid, 1.0/denominator, jnp.nan)


def _plunge_components(r, params):
    """
    Theta-independent ISCO-plunge u^r and Omega reference

    Notes
    -----
    E, L: equatorial circular constants evaluated at ISCO
    Exact radial-potential factorization:
    R_potential=(1-E_ISCO^2)*r*(r_ISCO-r)^3
    u^r=-sqrt[2*M/(3*r_ISCO)]*((r_ISCO-r)/r)^(3/2)
    Evaluation above ISCO: clamp to ISCO for the unused circular/plunge branch
    """
    r_isco = isco_radius_kerr(params)
    energy, angular = const_kerr_circ(r_isco, params)
    r = jnp.minimum(jnp.asarray(r), r_isco)
    R, spin = r / params.M, params.a / params.M
    R_isco, ell = r_isco / params.M, angular / params.M
    gap = jnp.maximum((R_isco - R) / R, 0.0)
    u_r = -jnp.sqrt(2.0 / (3.0*R_isco)) * gap**1.5
    horizon_gap = jnp.sqrt((1.0 - spin)*(1.0 + spin))
    delta = (R - 1.0 - horizon_gap)*(R - 1.0 + horizon_gap)
    P = energy*(R*R + spin*spin) - spin*ell
    L_minus_aE = ell - spin*energy
    numerator = spin*P + delta*L_minus_aE
    denominator = (R*R + spin*spin)*P + spin*delta*L_minus_aE
    # At exact positive extremal ISCO: circular limiting Omega, unused outside ISCO.
    safe_denominator = jnp.where(denominator != 0.0, denominator, 1.0)
    omega = jnp.where(
        denominator != 0.0,
        numerator / (params.M*safe_denominator),
        omega_kerr_circ(r_isco, params),
    )
    return u_r, omega


def _keplerian_components(r, params):
    """Circular exterior / ISCO-plunge interior reference; no theta dependence."""
    r_isco = isco_radius_kerr(params)
    u_r_plunge, omega_plunge = _plunge_components(r, params)
    exterior = r >= r_isco
    return (
        jnp.where(exterior, 0.0, u_r_plunge),
        jnp.where(exterior, omega_kerr_circ(r, params), omega_plunge),
    )


def _free_fall_components(x, params):
    """E=1, L_z=Q=0 inward BL geodesic; theta-dependent u^r and Omega."""
    R, spin, _, sigma, _, area, valid = _bl_flow_geometry(x, params)
    u_r = -jnp.sqrt(2.0*R*(R*R + spin*spin)) / sigma
    omega = 2.0*spin*R / (params.M*area)
    return jnp.where(valid, u_r, jnp.nan), jnp.where(valid, omega, jnp.nan)


def four_vel_inner_plunge(x, params: KerrParams):
    """
    ISCO-plunge reference extended off the equator and normalized locally

    Parameters
    ----------
    x : array
        BL position (t, r, theta, phi); r in the same length units as M
    params : KerrParams
        M, a in geometric units; signed a, +phi circular reference

    Returns
    -------
    velocity : array
        Contravariant BL (u^t, u^r, u^theta, u^phi); g(u,u)=-1

    Shapes
    ------
    x, velocity : (4,); batching via jax.vmap

    Notes
    -----
    Domain: r_+<r<=r_ISCO, 0<theta<pi
    Equator: plunge preserving E_ISCO=-u_t and L_ISCO=u_phi
    Off equator: prescribed equatorial u^r(r), Omega(r); not a geodesic extension
    u^theta=0; u^t normalized with the metric at the actual theta
    Non-timelike prescriptions / invalid domain: all NaN
    Exact a=M: no exterior plunge interval
    """
    x = jnp.asarray(x)
    u_r, omega = _plunge_components(x[1], params)
    velocity = _normalize_bl_flow(x, params, u_r, omega)
    return jnp.where(x[1] <= isco_radius_kerr(params), velocity, jnp.nan)


def four_vel_ff(x, params: KerrParams):
    """
    Inward zero-angular-momentum free fall from rest at infinity

    Parameters
    ----------
    x : array
        BL position (t, r, theta, phi); r in the same length units as M
    params : KerrParams
        Positive M; signed dimensional a, |a|<=M

    Returns
    -------
    velocity : array
        Contravariant BL (u^t, u^r, 0, u^phi); future-directed, g(u,u)=-1

    Shapes
    ------
    x, velocity : (4,); batching via jax.vmap

    Notes
    -----
    Geodesic constants: E=-u_t=1, L_z=u_phi=0, Carter Q=0
    Sigma=r^2+a^2*cos(theta)^2; u^r=-sqrt[2*M*r*(r^2+a^2)]/Sigma
    Omega=-g_tphi/g_phiphi; u^theta=0 at any fixed theta
    Theta dependence: retained; Schwarzschild u^r=-sqrt(2*M/r)
    Domain: r>r_+, 0<theta<pi; invalid domain: all NaN
    """
    u_r, omega = _free_fall_components(x, params)
    return _normalize_bl_flow(x, params, u_r, omega)


def omega_ff(x, params: KerrParams):
    """
    Angular velocity of the zero-angular-momentum free-fall reference

    Parameters
    ----------
    x : array
        BL position (t, r, theta, phi)
    params : KerrParams
        M, a in geometric units

    Returns
    -------
    omega : scalar
        Omega_ff=-g_tphi/g_phiphi=2*M*a*r/A; inverse length units

    Shapes
    ------
    x : (4,); omega : scalar; batching via jax.vmap

    Notes
    -----
    A=(r^2+a^2)^2-a^2*Delta*sin(theta)^2
    Theta dependence: retained; a=0 -> 0
    Domain: r>r_+, 0<theta<pi; invalid domain: NaN
    """
    _, omega = _free_fall_components(x, params)
    return omega


def four_vel_mixed(x, params: KerrParams, mixing_coefs):
    """
    Locally normalized mixture of Keplerian / plunge and free-fall references

    Parameters
    ----------
    x : array
        BL position (t, r, theta, phi); r in the same length units as M
    params : KerrParams
        Positive M; signed dimensional a, |a|<=M
    mixing_coefs : tuple or array
        (kappa_ff, kappa_K); finite values in [0, 1], dynamic JAX parameters

    Returns
    -------
    velocity : array
        Contravariant BL (u^t, u^r, 0, u^phi); g(u,u)=-1

    Shapes
    ------
    x, velocity : (4,); mixing_coefs : (2,); batching via jax.vmap

    Notes
    -----
    u^r=(1-kappa_ff)*u_K^r+kappa_ff*u_ff^r
    Omega=kappa_K*Omega_K+(1-kappa_K)*Omega_ff
    Keplerian reference: theta-independent u_K^r and Omega_K; plunge below ISCO
    Free-fall reference: theta-dependent u_ff^r and Omega_ff
    Normalization: after mixing, using the metric at the actual theta
    Limits: (0,1) Keplerian/plunge; (1,0) free fall; (0.5,0.5) fiducial RIAF
    Domain: r>r_+, 0<theta<pi; no outflow
    Invalid coefficients / non-timelike prescriptions: all NaN
    Reference: Saurabh et al., arXiv:2508.11760, equations (3)-(4)
    """
    x = jnp.asarray(x)
    coefs = jnp.asarray(mixing_coefs)
    kappa_ff, kappa_K = coefs[0], coefs[1]
    u_r_K, omega_K = _keplerian_components(x[1], params)
    u_r_ff, omega_free = _free_fall_components(x, params)
    u_r = (1.0 - kappa_ff)*u_r_K + kappa_ff*u_r_ff
    omega = kappa_K*omega_K + (1.0 - kappa_K)*omega_free
    velocity = _normalize_bl_flow(x, params, u_r, omega)
    valid = jnp.all(jnp.isfinite(coefs) & (coefs >= 0.0) & (coefs <= 1.0))
    return jnp.where(valid, velocity, jnp.nan)
