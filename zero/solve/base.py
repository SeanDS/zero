import abc

class BaseSolver(metaclass=abc.ABCMeta):
    """Base class for matrix solvers"""

    # solver name
    NAME = "base"

    @abc.abstractmethod
    def full(self, dimensions):
        """Create new complex-valued full matrix

        Parameters
        ----------
        dimensions : :class:`tuple`
            matrix shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sparse(self, dimensions):
        """Create new complex-valued sparse matrix

        Parameters
        ----------
        dimensions : :class:`tuple`
            matrix shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def solve(self, A, b):
        """Solve linear system"""
        raise NotImplementedError
