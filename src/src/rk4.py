"""
Classical Fourth-Order Runge-Kutta Method (RK4)

Contributor: Member 3
Responsibility:
- High-accuracy time integration
- Careful slope staging and averaging
"""

def rk4_step(f, x, y, h):
    """
    Perform one RK4 step.

    RK4 achieves fourth-order global accuracy by evaluating
    the derivative at four strategically chosen points.

    Parameters
    ----------
    f : callable
        ODE function dy/dx = f(x, y)
    x : float
        Current independent variable
    y : float
        Current dependent variable
    h : float
        Step size

    Returns
    -------
    float
        Approximate solution at x + h
    """

    # Stage 1: slope at start
    k1 = f(x, y)

    # Stage 2: slope at first midpoint
    x_half = x + 0.5 * h
    y_half_1 = y + 0.5 * h * k1
    k2 = f(x_half, y_half_1)

    # Stage 3: refined midpoint slope
    y_half_2 = y + 0.5 * h * k2
    k3 = f(x_half, y_half_2)

    # Stage 4: slope at endpoint
    x_end = x + h
    y_end = y + h * k3
    k4 = f(x_end, y_end)

    # Weighted average of all slopes
    y_next = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return y_next