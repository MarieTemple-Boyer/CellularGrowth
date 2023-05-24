''' Class for the modelisation of cellular growth '''

import warnings
from typing import Callable
from scipy.linalg import solve_banded
import numpy as np

from cellulargrowth import step_from_array
from cellulargrowth import encode_tridiagonal_matrix

SMALL = 0.0001


class CellularGrowth:
    ''' Class to represent the solution of a cellular growth reaction-diffusion equation '''

    def __init__(self,
                 bacteria0: Callable[[np.ndarray], np.ndarray],
                 nutrient0: Callable[[np.ndarray], np.ndarray],
                 mu: float,
                 epsilon: float):
        ''' The equation modelised is
            d_t bacteria - epsilon d_xx bacteria = 1/epsilon * nutrient * (bacteria - mu)
            d_t nutrient - d_xx nutrient = -nutrient * bacteria

            The initial conditions are bacteria0 and nutrient0.
            The boundary conditions are Neumann boundary conditions.

        >>> def bacteria0(x, offset):
        ...     return 1/np.sqrt(2*np.pi) * np.exp(-(x-offset)**2/2)
        >>> def nutrient0(x):
        ...     return 1
        >>> model_sym = CellularGrowth(lambda space: bacteria0(space, offset=0),
        ...                            nutrient0=nutrient0,
        ...                            mu = 0.5,
        ...                            epsilon=1)
        '''

        self.epsilon = epsilon
        self.mu = mu
        self.bacteria0 = bacteria0
        self.nutrient0 = nutrient0

    def _symetric(self) -> bool:
        ''' Check if the initial conditions are symetric function.
        >>> model_sym._symetric()
        True
        >>> model_non_sym._symetric()
        False
        '''
        space = np.arange(0, 11, 1)
        return (self.bacteria0(space) == self.bacteria0(-space)).all() \
            and (self.nutrient0(space) == self.nutrient0(-space)).all()

    def solve_explicit(self,
                       time: np.ndarray,
                       space: np.ndarray) -> np.ndarray | np.ndarray:
        ''' Solve the equation with an explicit finite difference method
            on the domain space for a certain range of time.
        >>> space = np.arange(0, 5+0.1, 0.1)
        >>> time = np.arange(0, 1+0.004, 0.004)
        >>> bac, nut = model_sym.solve_explicit(time, space)
        >>> n, i = 10, 25
        >>> # at time time[n] at the postion space[i] the density of bacteria is
        >>> bac[n, i]
        0.05434032732745466
        >>> bac, nut = model_non_sym.solve_explicit(time, space)
        >>> bac[n, i]
        0.3463546751201988
        '''

        dt = time[1:] - time[:-1]
        dx = np.concatenate((space[1:]-space[:-1],
                             [np.average(space[1:]-space[:-1])]))

        if np.max(dt)/(np.min(dx)**2) >= 1/2:
            warnings.warn('The stability condition dt/(dx**2) < 1/2 should be verified.'
                          f'Yet we have dt/(dx**2) = {np.max(dt)/(np.min(dx)**2)} at some points.')

        bac = np.zeros((len(time), len(space)))
        nut = np.zeros((len(time), len(space)))

        bac[0] = self.bacteria0(space)
        nut[0] = self.nutrient0(space)

        for n in range(0, len(time)-1):

            for i in range(len(space)):

                j = i + (i+1 != len(space))  # Neumann boundary conditions
                if space[0] == 0 and self._symetric:
                    k = (i-1)*(i != 0) + (i == 0)  # symetry
                else:
                    k = i - (i-1 != 0)  # Neumann boundary conditions
                coeff = dt[n]/(dx[i]**2)
                bac[n+1, i] = bac[n, i] \
                    + self.epsilon*coeff * (bac[n, j] - 2*bac[n, i] + bac[n, k]) \
                    + dt[n]/self.epsilon * bac[n, i]*(nut[n, i] - self.mu)

                nut[n+1, i] = nut[n, i] \
                    + coeff * (nut[n, j] - 2*nut[n, i] + nut[n, k]) \
                    - dt[n] * bac[n, i]*nut[n, i]

        return bac, nut

    def _matrices_for_solve_implicit(self,
                                     dt: float,
                                     dx: float,
                                     space: np.ndarray,
                                     symetric: bool) -> np.ndarray:
        ''' Return the encoded matrices to solve the PDE with implicit finite difference scheme.'''

        coeff = dt/(dx**2)
        ones = np.ones(len(space)-1)

        mat_bac = (1+2*self.epsilon*coeff) * np.eye(len(space)) \
            - self.epsilon*coeff * np.diag(ones, 1) \
            - self.epsilon*coeff * np.diag(ones, -1)

        mat_nut = (1+2*coeff) * np.eye(len(space)) \
            - coeff * np.diag(ones, 1) \
            - coeff * np.diag(ones, -1)
        if symetric and space[0] == 0:
            # symetry on zero
            mat_bac[0, 1] += -self.epsilon*coeff
            mat_nut[0, 1] += -coeff
        else:
            # Neumann boundary conditions
            mat_bac[0, 0] += -self.epsilon*coeff
            mat_nut[0, 0] += -coeff
        # Neumann boundary conditions
        mat_bac[-1, -1] += -self.epsilon*coeff
        mat_nut[-1, -1] += -coeff

        code_bac = encode_tridiagonal_matrix(mat_bac)
        code_nut = encode_tridiagonal_matrix(mat_nut)

        return code_bac, code_nut

    def solve_implicit(self,
                       time: np.ndarray,
                       space: np.ndarray) -> np.ndarray | np.ndarray:
        ''' Solve the equation with an implicit finite difference method
        on the domain space for a certain range of time.
        >>> space = np.arange(0, 5+0.1, 0.1)
        >>> time = np.arange(0, 1+0.004, 0.004)
        >>> bac, nut = model_sym.solve_implicit(time, space)
        >>> bac[10, 25]
        0.05437187309253188
        >>> bac, nut = model_non_sym.solve_implicit(time, space)
        >>> bac[10, 25]
        0.3461369580166652
        >>> bac, nut = model_sym.solve_implicit(time,
        ...                                     space=np.array([0, 0.1, 0.2, 1]))
        Traceback (most recent call last):
            ...
        ValueError: The space discretisation has to be uniform
        '''

        dx = step_from_array(space, 'space')
        dt = step_from_array(time, 'time')

        bac = np.zeros((len(time), len(space)))
        nut = np.zeros((len(time), len(space)))

        bac[0] = self.bacteria0(space)
        nut[0] = self.nutrient0(space)

        code_bac, code_nut = self._matrices_for_solve_implicit(dt,
                                                               dx,
                                                               space,
                                                               self._symetric())

        for n in range(0, len(time)-1):
            bac[n+1] = solve_banded((1, 1),
                                    code_bac,
                                    (1 + dt/self.epsilon*(nut[n]-self.mu)) * bac[n])
            nut[n+1] = solve_banded((1, 1),
                                    code_nut,
                                    (1 - dt*bac[n]) * nut[n])

        return bac, nut

    def solve_long_time(self,
                        time_max: float,
                        dt: float,
                        space0: np.ndarray) -> np.ndarray | np.ndarray | np.ndarray:
        '''
        >>> space0 = np.arange(-5, 5+0.1, 0.1)
        >>> space, bac, nut = model_sym.solve_long_time(time_max=2,
        ...                                             dt=0.04,
        ...                                             space0=space0)
        >>> bac[10, 48]
        0.6949000892936333
        >>> bac, nut = model_sym.solve_long_time(time_max=2,
        ...                                      dt=0.04,
        ...                                      space0=np.array([0, 0.1, 0.2, 1]))
        Traceback (most recent call last):
            ...
        ValueError: The space discretisation has to be uniform
        '''

        dx = step_from_array(space0, 'space')

        nb_it_time = int(time_max/dt)+1

        space = np.zeros((nb_it_time, len(space0)))
        space[0] = space0

        bac = np.zeros((nb_it_time, len(space0)))
        nut = np.zeros((nb_it_time, len(space0)))

        bac[0] = self.bacteria0(space0)
        nut[0] = self.nutrient0(space0)

        if np.abs(bac[0][0]) > SMALL or np.abs(bac[0][-1]) > SMALL:
            warnings.warn(
                'The density of the initial bacteria should be very small on the   edage.')

        code_bac, code_nut = self._matrices_for_solve_implicit(
            dt, dx, space0, symetric=False)

        for n in range(0, nb_it_time-1):
            mean_bac = np.sum((space[n] >= 0) * bac[n]*space[n]) \
                / np.sum((space[n] >= 0) * bac[n])
            mean_space = (space[n][-1] + space[n][0])/2
            it_dif = round((mean_bac-mean_space)/dx)
            if it_dif > 0:
                space[n] = space[n]+it_dif*dx
                # TO DO:  find a good way to give a value on the boundary
                bac[n] = np.concatenate((
                    bac[n][it_dif:],
                    self.bacteria0(space[n][-it_dif:])))
                nut[n] = np.concatenate((
                    nut[n][it_dif:],
                    self.nutrient0(space[n][-it_dif:])))
            space[n+1] = space[n]

            bac[n+1] = solve_banded((1, 1),
                                    code_bac,
                                    (1 + dt/self.epsilon*(nut[n]-self.mu)) * bac[n])
            nut[n+1] = solve_banded((1, 1),
                                    code_nut,
                                    (1 - dt*bac[n]) * nut[n])

        return space, bac, nut


if __name__ == '__main__':

    def bac0(space, offset):
        ''' Initial density of becteria '''
        return np.exp(-(space-offset)**2/2)

    def nut0(space):
        ''' Initial density of nutrient '''
        return np.ones(len(space))

    model_sym = CellularGrowth(lambda space: bac0(space, offset=0),
                               nutrient0=nut0,
                               mu=0.5,
                               epsilon=1)

    model_non_sym = CellularGrowth(lambda space: bac0(space, offset=1),
                                   nutrient0=nut0,
                                   mu=0.5,
                                   epsilon=1)

    import doctest
    doctest.testmod()
