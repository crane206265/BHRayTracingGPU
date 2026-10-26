# events.py

import jax.numpy as jnp

from src.types import RayStatus

def event_single_func(capture_fn):
    """
    Construct a single-ray event classifier.

    Parameters
    ----------
    capture_fn : callable
        capture_fn(x, params) -> bool

    Returns
    -------
    event_single : callable
        event_single(y, params, r_escape, safe_y) -> int32 scalar

    Shapes
    ------
    y        : (8,)
    r_escape : scalar
    safe_y   : (8,), finite state at a regular metric point
    """
    def event_single(y, params, r_escape, safe_y):

        # numerical validity
        finite = jnp.all(jnp.isfinite(y))

        # Substitute a finite state before geometric tests.
        checked_y = jnp.where(
            finite,
            y,
            safe_y
        )

        x = checked_y[:4]

        # geometric termination conditions
        captured = capture_fn(x, params)

        R = jnp.linalg.norm(x[1:])
        escaped = R >= r_escape

        # priority : NONFINITE > CAPTURED > ESCAPED > ACTIVE
        status = jnp.where(
            ~finite,
            int(RayStatus.NONFINITE),
            jnp.where(
                captured,
                int(RayStatus.CAPTURED),
                jnp.where(
                    escaped,
                    int(RayStatus.ESCAPED),
                    int(RayStatus.ACTIVE)
                )
            )
        )

        return status.astype(jnp.int32)

    return event_single
