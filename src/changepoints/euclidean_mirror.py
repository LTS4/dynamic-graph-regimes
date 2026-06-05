import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog


def linf_piecewise_linear_fit(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    cp: float,
) -> tuple[float, float, float, float]:
    """Fit piecewise linear function using L-infinity norm via linear programming.

    Minimizes max_i |y_i - f(t_i)| where f is piecewise linear with a kink at cp.
    f(t) = alpha + bl * (t - cp) for t < cp
    f(t) = alpha + br * (t - cp) for t >= cp

    This is formulated as:
        min z
        s.t. z >= |y_i - f(t_i)| for all i

    Args:
        t (NDArray): Time points of shape (n,)
        y (NDArray): Values of shape (n,)
        cp (float): Change point location

    Returns:
        tuple: (z, alpha, bl, br) where z is optimal objective value (max absolute residual),
            alpha is intercept at change point, bl is left slope, and br is right slope
    """

    n = len(t)

    # Variables: [z, alpha, bl, br]
    # Objective: minimize z
    c = np.array([1.0, 0.0, 0.0, 0.0])

    # Build inequality constraints: A_ub @ x <= b_ub
    # For each point, we need:
    #   y_i - f(t_i) <= z  =>  -z + f(t_i) >= y_i
    #   f(t_i) - y_i <= z  =>  z - f(t_i) >= -y_i

    A_ub = []
    b_ub = []

    for i in range(n):
        if t[i] < cp:
            # f(t_i) = alpha + bl * (t_i - cp)
            # Constraint 1: -z + alpha + bl*(t-cp) >= y  =>  z - alpha - bl*(t-cp) <= -y
            A_ub.append([1.0, -1.0, -(t[i] - cp), 0.0])
            b_ub.append(-y[i])
            # Constraint 2: z - alpha - bl*(t-cp) >= -y  =>  z - alpha - bl*(t-cp) >= -y (redundant)
            A_ub.append([1.0, 1.0, (t[i] - cp), 0.0])
            b_ub.append(y[i])
        else:
            # f(t_i) = alpha + br * (t_i - cp)
            A_ub.append([1.0, -1.0, 0.0, -(t[i] - cp)])
            b_ub.append(-y[i])
            A_ub.append([1.0, 1.0, 0.0, (t[i] - cp)])
            b_ub.append(y[i])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds: z >= 0, others unbounded
    bounds = [(0, None), (None, None), (None, None), (None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        z, alpha, bl, br = result.x
        return z, alpha, bl, br
    else:
        return np.inf, 0.0, 0.0, 0.0


def localize_changepoint(
    iso_mirror: NDArray[np.float64],
    t: NDArray[np.float64] | None = None,
) -> tuple[int, NDArray[np.float64]]:
    """Localize change point in iso-mirror using L-infinity piecewise linear regression.

    For each candidate change point, fits a piecewise linear function and
    returns the one that minimizes the L-infinity objective.

    Args:
        iso_mirror (NDArray): 1D iso-mirror values of shape (m,)
        t (NDArray, optional): Time points. If None, uses 1 to m.

    Returns:
        tuple: (cp_idx, objective_values) where cp_idx is index of the estimated
            change point (0-indexed) and objective_values is L-infinity objective
            for each candidate change point
    """
    m = len(iso_mirror)
    if t is None:
        t = np.arange(1, m + 1, dtype=float)

    objective_values = np.full(m, np.inf)

    # Try each internal point as a change point
    for k in range(1, m - 1):
        obj, _, _, _ = linf_piecewise_linear_fit(t, iso_mirror, t[k])
        objective_values[k] = obj

    # Find change point that minimizes objective
    cp_idx = int(np.argmin(objective_values))

    return cp_idx, objective_values
