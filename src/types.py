# types.py

from typing import NamedTuple
from enum import IntEnum
import jax


# ==========================================
# Spacetime parameters
# ==========================================

class MinkowskiParams(NamedTuple):
    pass


class SchwarzschildParams(NamedTuple):
    M: jax.Array


class KerrParams(NamedTuple):
    M: jax.Array
    a: jax.Array


# ==========================================
# Ray termination status
# ==========================================

class RayStatus(IntEnum):
    ACTIVE = 0
    CAPTURED = 1
    ESCAPED = 2
    MAX_STEPS = 3
    NONFINITE = 4
    INVALID = 5


# ==========================================
# Integration state
# ==========================================

class IntegrationState(NamedTuple):
    y: jax.Array
    lam: jax.Array
    active: jax.Array
    status: jax.Array
    valid_length: jax.Array