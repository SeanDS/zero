from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

from .base import BaseSolver

class ScipySolver(BaseSolver):
    """Scipy-based matrix solver"""

    # solver name
    NAME = "scipy-default"

    def full(self, dimensions):
        """Create new complex-valued full matrix

        Creates a Numpy full matrix.

        Parameters
        ----------
        dimensions : :class:`tuple`
            matrix shape

        Returns
        -------
        :class:`~np.ndmatrix`
            full matrix
        """

        return np.zeros(dimensions, dtype="complex64")

    def sparse(self, dimensions):
        """Create new complex-valued sparse matrix

        Creates a SciPy sparse matrix.

        Parameters
        ----------
        dimensions : :class:`tuple`
            matrix shape

        Returns
        -------
        :class:`~lil_matrix`
            sparse matrix
        """

        # complex64 gives real and imaginary parts each represented as 32-bit floats
        # with 8 bits exponent and 23 bits mantissa, giving between 6 and 7 digits
        # of precision; good enough for most purposes
        return lil_matrix(dimensions, dtype="complex128")

    def solve(self, A, b):
        """Solve linear system

        Parameters
        ----------
        A : :class:`~np.ndarray`, :class:`~scipy.sparse.spmatrix`
            square matrix
        B : :class:`~np.ndarray`, :class:`~scipy.sparse.spmatrix`
            matrix or vector representing right hand side of matrix equation

        Returns
        -------
        solution : :class:`~np.ndarray`, :class:`~scipy.sparse.spmatrix`
            x in the equation Ax = b
        """

        # permute specification chosen to minimise error with LISO
        return spsolve(A, b, permc_spec="MMD_AT_PLUS_A")
