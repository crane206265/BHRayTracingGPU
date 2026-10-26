# integrators/batched.py

import jax
import jax.numpy as jnp

from src.types import IntegrationState, RayStatus


def integrate_chunk_func(rk4_step, event_single, n_steps):
    """
    Build a JIT-compiled batched RK4 integrator.

    Parameters
    ----------
    rk4_step : callable
        Single-ray step:
            rk4_step(y, h, params) -> y_next

    event_single : callable
        Single-ray event classifier:
            event_single(y, params, r_escape, safe_y) -> int32 scalar

    n_steps : int
        Number of integration steps per chunk.
        Fixed at compilation time.

    Returns
    -------
    integrate_chunk : callable
        integrate_chunk(state, params, h, r_escape)
            -> (final_state, history)

    Shapes
    ------
    state.y        : (B, 8)
    state.lam      : (B,)
    state.active   : (B,)
    state.status   : (B,)
    state.valid_length : (B,)
    h              : scalar (negative for backward tracing)
    r_escape       : (B,)

    history[0]     : (K, B, 8) states
    history[1]     : (K, B) affine parameters
    history[2]     : (K, B) valid-step mask

    Notes
    -----
    The safe state currently requires Kerr parameters with M > 0.
    History excludes the initial sample. Finite capture/escape endpoints
    are valid samples; failed and previously inactive steps are not.
    A nonfinite proposal preserves the last state and affine parameter.
    valid_length counts accepted steps, excluding the initial sample.
    Chunk completion leaves surviving rays ACTIVE; the driver handles
    MAX_STEPS when the total integration budget is exhausted.
    """

    rk4_batch = jax.vmap(
        rk4_step,
        in_axes=(0, None, None)
    )

    event_batch = jax.vmap(
        event_single,
        in_axes=(0, None, 0, None)
    )

    def integrate_chunk(state, params, h, r_escape):
        M = params.M

        safe_state = jnp.zeros((8,), dtype=state.y.dtype)
        safe_state = safe_state.at[1].set(100.0 * M)

        def scan_step(carry, _):
            # vmap still evaluates inactive lanes, so use a regular point
            # with zero momentum before computing any RK4 stages.
            safe_y = jnp.where(
                carry.active[:, None],
                carry.y,
                safe_state
            )

            proposed_y = rk4_batch(
                safe_y, h, params
            )

            finite = jnp.all(jnp.isfinite(proposed_y), axis=1)
            event_status = event_batch(
                proposed_y, params, r_escape, safe_state
            )
            valid_step = carry.active & finite

            next_y = jnp.where(
                valid_step[:, None],
                proposed_y,
                carry.y
            )

            next_lam = jnp.where(
                valid_step,
                carry.lam + h,
                carry.lam
            )

            next_active = (
                carry.active
                & (event_status == int(RayStatus.ACTIVE))
            )
            next_status = jnp.where(
                carry.active,
                event_status,
                carry.status
            )

            next_length = (
                carry.valid_length
                + valid_step.astype(carry.valid_length.dtype)
            )

            next_state = IntegrationState(
                y=next_y,
                lam=next_lam,
                active=next_active,
                status=next_status,
                valid_length=next_length
            )

            history_sample = (
                next_y,
                next_lam,
                valid_step
            )

            return next_state, history_sample

        final_state, history = jax.lax.scan(
            scan_step,
            state,
            xs=None,
            length=n_steps
        )

        return final_state, history

    return jax.jit(integrate_chunk)
