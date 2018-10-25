from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

from .base import BaseSolver

class ScipySolver(BaseSolver):
    """Scipy-based matrix solver"""

    # solver name
    NAME = "scipy-default"

    # default data type
    # complex64 gives real and imaginary parts each represented as 32-bit floats
    # with 8 bits exponent and 23 bits mantissa, giving between 6 and 7 digits
    # of precision; not good enough for comparison to LISO
    DTYPE = "complex128"

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
        return np.zeros(dimensions, dtype=self.DTYPE)

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
        # of precision; not good enough for comparison to LISO
        return lil_matrix(dimensions, dtype=self.DTYPE)

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
        return spsolve(A, b)
