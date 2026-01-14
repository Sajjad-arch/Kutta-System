Second-Order Runge-Kutta Method (Midpoint Method)

Contributor: Member 2
Responsibility:
- Numerical implementation of RK2
- Emphasis on mathematical clarity and extensibility
"""

def rk2_step(f, x, y, h):
    """
    Perform one RK2 (Midpoint) step.

    Parameters
    ----------
    f : callable
        Right-hand side of ODE dy/dx = f(x, y)
    x : float
        Current independent variable
    y : float
        Current solution value
    h : float
        Step size

    Returns
    -------
    float
        Approximate solution at x + h
    """

    # Stage 1: slope at the beginning of the interval
    k1 = f(x, y)

    # Predictor step: Euler half-step to midpoint
    x_mid = x + 0.5 * h
    y_mid = y + 0.5 * h * k1

    # Stage 2: slope at the midpoint
    k2 = f(x_mid, y_mid)

    # Corrector step using midpoint slope
    y_next = y + h * k2

    return y_next
