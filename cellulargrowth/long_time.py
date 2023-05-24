""" A COMPLETER
"""

import warnings
import numpy as np
from scipy.linalg import solve_banded

SMALL = 0.0001


def solve_1D_long_time(bacteria0: np.ndarray,
                       nutrient0: float,
                       time_max: float,
                       dt: float,
                       space0: np.ndarray,
                       mu: float = 0.5,
                       epsilon: float = 1):
    """ Solve the following equation :
            d_t u - epsilon d_xx u = 1/epsilon u (v-mu)
            d_t v - d_xx v = -uv
        where u is the density of bactery and v the density of nutrient.

        This solver use an implicite finite difference method.
        There are Neumann boundary conditions.
        The initial density are bacteria0 and nutrient0.
        The discretisation grid is defined by time and space. It has to be uniform.
        Note that the solution is symetric in space if the space discretisation begins with zero.
        The parameters of the equation are mu and epsilon.
    >>> space0 = np.arange(-5, 5+0.1, 0.1)
    >>> bacteria0 = np.exp(-space0**2/2)
    >>> nutrient0 = 1
    >>> space, bac, nut = solve_1D_long_time(bacteria0, nutrient0,
    ...                                      time_max=2, dt=0.04,
    ...                                      space0=space0,
    ...                                      epsilon=0.1)
    >>> bac[10,48]
    2.395638488173044
    """

    dx_tab = space0[1:] - space0[:-1]
    dx = np.mean(dx_tab)

    uniform_issue = (((dx_tab-dx)/dx) > SMALL).any()
    if uniform_issue:
        raise ValueError(
            'The initial space discretisation has to be uniform')

    if np.abs(bacteria0[-1]) > SMALL or np.abs(bacteria0[0]) > SMALL:
        warnings.warn(
            'The density of the initial bacteria should be very small on the edage.')

    nb_it_time = int(time_max/dt)

    space = np.zeros((nb_it_time, len(space0)))
    space[0] = space0

    bac = np.zeros((nb_it_time, len(space0)))
    nut = np.zeros((nb_it_time, len(space0)))

    bac[0] = bacteria0
    nut[0] = nutrient0*np.ones(len(space0))

    coeff = dt/(dx**2)

    ones = np.ones(len(space0)-1)

    mat_bac = (1+2*epsilon*coeff) * np.eye(len(space0)) \
        - epsilon*coeff * np.diag(ones, 1) \
        - epsilon*coeff * np.diag(ones, -1)

    mat_nut = (1+2*coeff) * np.eye(len(space0)) \
        - coeff * np.diag(ones, 1) \
        - coeff * np.diag(ones, -1)

    # Neumann boundary conditions
    mat_bac[0, 0] += -epsilon*coeff
    mat_nut[0, 0] += -coeff
    mat_bac[-1, -1] += -epsilon*coeff
    mat_nut[-1, -1] += -coeff

    # see documentation of scipy.solve_banded in order to understand
    code_bac = np.zeros((1+1+1, len(space0)))
    code_nut = np.zeros((1+1+1, len(space0)))
    for j in range(len(space0)):
        for i in range(len(space0)):
            if j-1 <= i <= j+1:
                code_bac[1+i-j, j] = mat_bac[i, j]
                code_nut[1+i-j, j] = mat_nut[i, j]

    for n in range(0, nb_it_time-1):
        
        mean_bac = np.sum((space[n] >= 0) * bac[n]*space[n]) \
                / np.sum((space[n] >= 0) * bac[n])
        mean_space = (space[n][-1] + space[n][0])/2
        it_dif = round((mean_bac-mean_space)/dx)
        if it_dif > 0 and n != 0:
            space[n] = space[n]+it_dif*dx
            # trouver une meilleure solution pour la valeur au bord
            bac[n] = np.concatenate((bac[n][it_dif:],
                                     [bac[n][-1]/(1.1**i) for i in range(it_dif)]))
            nut[n] = np.concatenate((nut[n][it_dif:],
                                     nutrient0*np.ones(it_dif)))
        space[n+1] = space[n]
        
        sec_bac = (1 + dt/epsilon*(nut[n]-mu)) * bac[n]
        sec_nut = (1 - dt*bac[n]) * nut[n]

        bac[n+1] = solve_banded((1, 1), code_bac, sec_bac)
        nut[n+1] = solve_banded((1, 1), code_nut, sec_nut)
    
    return space, bac, nut


if __name__ == "__main__":

    import doctest
    doctest.testmod()
