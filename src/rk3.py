Third-Order Runge-Kutta Method

Contributor: Member 2
Responsibility:
- Higher-order Runge-Kutta integration
- Error-reducing weighted slope averaging
"""

def rk3_step(f, x, y, h):
    """
    Perform one RK3 step.

    This implementation achieves third-order global accuracy
    by cancelling lower-order truncation terms.

    Parameters
    ----------
    f : callable
        ODE function dy/dx = f(x, y)
    x : float
        Current x value
    y : float
        Current y value
    h : float
        Step size

    Returns
    -------
    float
        Approximate y at x + h
    """

    # Stage 1: initial slope
    k1 = f(x, y)

    # Stage 2: midpoint slope (based on k1)
    x_mid = x + 0.5 * h
    y_mid = y + 0.5 * h * k1
    k2 = f(x_mid, y_mid)

    # Stage 3: endpoint slope using linear combination
    x_end = x + h
    y_end = y - h * k1 + 2 * h * k2
    k3 = f(x_end, y_end)

    # Weighted average of slopes
    y_next = y + (h / 6.0) * (k1 + 4 * k2 + k3)

    return y_next