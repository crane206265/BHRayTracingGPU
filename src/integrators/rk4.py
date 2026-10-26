# integrators/rk4.py

def rk4_step_func(rhs):
    """
    Construct a single-ray RK4 step.

    Parameters
    ----------
    rhs : callable
        rhs(y, params) -> dy/dlambda

    Returns
    -------
    step : callable
        step(y, h, params) -> y_next

    Shapes
    ------
    y      : (8,)
    h      : scalar
    params : PyTree
    y_next : (8,)
    """

    def step(y, h, params):
        # backward tracing -> h < 0
        k1 = rhs(y, params)
        k2 = rhs(y + 0.5*h*k1, params)
        k3 = rhs(y + 0.5*h*k2, params)
        k4 = rhs(y + h*k3, params)

        return y + h*(k1 + 2*k2 + 2*k3 + k4)/6

    return step