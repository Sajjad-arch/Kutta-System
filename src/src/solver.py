"""
General ODE Solver Engine

Contributor: Member 3
Responsibility:
- Step-size controlled integration loop
- Method-agnostic solver interface
"""

import numpy as np

def solve_ode(step_method, f, y0, x_range, h):
    """
    Solve an ODE initial value problem using a fixed step size.

    Parameters
    ----------
    step_method : callable
        One-step integration method (RK2, RK3, RK4)
    f : callable
        ODE function dy/dx = f(x, y)
    y0 : float
        Initial condition
    x_range : tuple
        (x_start, x_end)
    h : float
        Step size

    Returns
    -------
    x_vals : numpy.ndarray
        Discretized x values
    y_vals : numpy.ndarray
        Numerical solution values
    """

    x_start, x_end = x_range

    # Number of integration steps
    n_steps = int((x_end - x_start) / h) + 1

    # Allocate arrays
    x_vals = np.linspace(x_start, x_end, n_steps)
    y_vals = np.zeros(n_steps)

    # Apply initial condition
    y_vals[0] = y0

    # Time-marching loop
    for i in range(n_steps - 1):
        x_current = x_vals[i]
        y_current = y_vals[i]

        y_vals[i + 1] = step_method(
            f,
            x_current,
            y_current,
            h
        )

    return x_vals, y_vals