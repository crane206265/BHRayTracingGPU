# driver.py

import jax
import jax.numpy as jnp

from src.types import IntegrationState, RayStatus

def initialize_state(y0, params, r_escape, event_fn, safe_y):
    B = y0.shape[0]
    
    event_batch = jax.vmap(
        event_fn,
        in_axes=(0, None, 0, None)
    )
    init_status = event_batch(
        y0, params, r_escape, safe_y
    )
    
    init_state = IntegrationState(
        y = y0,
        lam = jnp.zeros((B,), dtype=y0.dtype),
        active = (init_status == int(RayStatus.ACTIVE)),
        status = init_status,
        valid_length = jnp.zeros((B,), dtype=jnp.int32)
    )

    return init_state

def integrate(
        state:IntegrationState,
        params,
        h,
        r_escape,
        integrate_chunk,
        max_chunks,
        save_chunk,
    ):

    for chunk_idx in range(max_chunks):
        # 1. Active ray check -> if not : terminate
        if not jnp.any(state.active):
            return state

        # 2. Integrate over chunk
        state, history = integrate_chunk(
            state, params, h, r_escape
        )

        # 3. Save history
        save_chunk(chunk_idx, history)

    # 4. Terminate the active rays that reached to MAX_STEPS
    state = state._replace(
        status=jnp.where(
            state.active,
            int(RayStatus.MAX_STEPS),
            state.status
        ),
        active=jnp.zeros_like(state.active)
    )

    return state
