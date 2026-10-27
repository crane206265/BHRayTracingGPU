# observables.py

# intensity, redshift -> image

import jax
import jax.numpy as jnp

from src.metrics import kerr_schild
from src.plasma.disk_kinematics import four_vel_thin
from src.plasma.novikov_thorne import nt_flux
from src.transfer import find_first_disk_hit, redshift_factor, observed_bolometric_intensity

def evaluate_nt_ray(y_history, params, u_obs, r_in, r_out, mdot):
    found, y_hit, image_order = find_first_disk_hit(
        y_history = y_history,
        params = params,
        r_in = r_in,
        r_out = r_out
    )

    x_hit = y_hit[:4]
    four_vel = four_vel_thin(x=x_hit, params=params)
    flux_em = nt_flux(
        r = kerr_schild.radial_coordinate(x_hit, params),
        params = params,
        mdot = mdot
    )

    g = redshift_factor(
        p_obs = y_history[0, 4:],
        u_obs = u_obs,
        p_em = y_hit[4:],
        u_em = four_vel,
    )
    intensity = observed_bolometric_intensity(flux_em=flux_em, g=g)
    
    intensity = jnp.where(found, intensity, 0.0)
    g = jnp.where(found, g, 0.0)

    return intensity, g, image_order, found