"""Module implementing graph distances"""

import numpy as np
from numpy.linalg import eigvalsh
from numpy.typing import NDArray
from scipy.linalg import pinvh, sqrtm

from src.linalg import sqrt_pos
from src.utils import square_to_vec


def bw_dist_single(lapl0, lapl1, rcond=1e-8):
    r"""Bures-Wassertein distance between graphs.
    The distance is defined as
    .. math::
        \operatorname{tr} \left( L_0^\dagger  + L_1^\dagger \right )
        - 2 \operatorname{tr} \sqrt{L_0^{\dagger / 2} ̆L_1^\dagger L_0^{\dagger / 2}}

    Note that this method is slower than :method:`bw_dist`
    """
    inv0 = pinvh(lapl0)
    inv1 = pinvh(lapl1)
    invsqrt = np.real(sqrtm(inv0))

    out = (
        np.trace(inv0, axis1=-2, axis2=-1)
        + np.trace(inv1, axis1=-2, axis2=-1)
        - 2 * np.sum(sqrt_pos(eigvalsh(invsqrt @ inv1 @ invsqrt), rcond), axis=-1)
    )
    return sqrt_pos(out, rcond)


def bw_dist_decomposed(inv0: NDArray, inv1: NDArray, corr=0.0, rcond=0.0):
    """Compute BW distance between matrices.

    Args:
        inv0 (NDArray): First matrix
        inv1 (NDArray): Second matrix
        corr (float, optional): Correction to subtract before final sqrt. Defaults to 0.
        rcond (float, optional): Cutoff value before taking sqrt. Defaults to 0.

    Returns:
        _type_: _description_
    """
    invsqrt = np.real(sqrtm(inv0))
    out = (
        np.trace(inv0, axis1=-2, axis2=-1)
        + np.trace(inv1, axis1=-2, axis2=-1)
        - 2 * np.sum(sqrt_pos(eigvalsh(invsqrt @ inv1 @ invsqrt), rcond), axis=-1)
    )
    return sqrt_pos(out - corr, rcond)


def bw_dist(lapl0: NDArray, lapl1: NDArray, rcond=1e-8):
    r"""Bures-Wassertein distance between graphs.
    The distance is defined as
    .. math::
        \operatorname{tr} \left( L_0^\dagger  + L_1^\dagger \right )
        - 2 \operatorname{tr} \sqrt{L_0^{\dagger / 2} ̆L_1^\dagger L_0^{\dagger / 2}}

    """
    return bw_dist_decomposed(pinvh(lapl0, atol=rcond), pinvh(lapl1, atol=rcond), rcond)


def vec_distance(adj0: NDArray, adj1: NDArray, norm_ord: float):
    return np.linalg.norm(square_to_vec(adj0) - square_to_vec(adj1), ord=norm_ord, axis=-1)
