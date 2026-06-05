import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import eigh
from numpy.typing import NDArray


def eigh_spd(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parallel eigh.

    This is often faster than stacked eigh, but is somehow slower if falled within other Parallel.
    """
    if len(x.shape) == 2:
        return eigh(x)
    return map(
        np.stack, zip(*Parallel(n_jobs=-1, prefer="threads")(delayed(eigh)(mat) for mat in x))
    )  # type:ignore


def sqrt_pos(x: NDArray[np.float64], rcond=1e-8) -> NDArray[np.float64]:
    x[x < rcond] = 0
    return np.sqrt(x)


def sqrtm_spd(x: NDArray[np.float64], rcond=1e-8) -> NDArray[np.float64]:
    eigval, eigvec = eigh(x)
    return eigvec @ (sqrt_pos(eigval, rcond)[..., np.newaxis] * eigvec.swapaxes(-1, -2))


def inv_pos(x: NDArray[np.float64], rcond=1e-8) -> NDArray[np.float64]:
    return np.where(x > rcond, 1 / x, 0)


def pinv_spd(x: NDArray[np.float64], rcond=1e-8) -> NDArray[np.float64]:
    eigval, eigvec = eigh(x)
    return eigvec @ (inv_pos(eigval, rcond)[..., np.newaxis] * eigvec.swapaxes(-1, -2))
