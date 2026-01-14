Author: Fardous Rahman
Role: Core ODE solver engine
Description: Generic fixed-step ODE solver for Runge-Kutta methods
"""

import numpy as np

def solve_ode(step_function, f, x_range, y0, h):
    x0, xf = x_range
    n_steps = int((xf - x0) / h)

    x_values = np.zeros(n_steps + 1)
    y_values = np.zeros(n_steps + 1)

    x_values[0] = x0
    y_values[0] = y0

    for i in range(n_steps):
        x_values[i + 1] = x_values[i] + h
        y_values[i + 1] = step_function(f, x_values[i], y_values[i], h)

    return x_values, y_values
