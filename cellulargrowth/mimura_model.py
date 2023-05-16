""" Resolution of a modelisation of cellular development 
d_t u - laplacian(u) = 1/epsilon u (v-mu)
d_t v - laplacian(v) = -uv
where u is the density of active bacteria and v the density of nutrient.
"""

import warnings
import numpy as np
from scipy.linalg import solve_banded

SMALL = 0.0001


def solve_1D_explicit(bacteria0: np.ndarray,
                      nutrient0: np.ndarray,
                      time: np.ndarray,
                      space: np.ndarray,
                      mu: float = 0.5,
                      epsilon: float = 1):
    """ Solve the following equation :
            d_t u - epsilon d_xx u = 1/epsilon u (v-mu)
            d_t v - d_xx v = -uv
        where u is the density of bactery and v the density of nutrient.

        This solver use an explicite finite difference method,
            and is robust to non uniform space or time discretisation.
        There are Neumann boundary conditions.
        The initial density are bacteria0 and nutrient0.
        Note that the solution is symetric in space if the space discretisation begins with zero.
        The parameters of the equation are mu and epsilon.
    >>> space = np.arange(0, 5+0.1, 0.1)
    >>> time = np.arange(0, 1, 0.004)
    >>> bacteria0 = np.exp(-space**2/2)
    >>> nutrient0 = np.ones(len(space))
    >>> bac, nut = solve_1D_explicit(bacteria0, nutrient0, time, space)
    >>> n, i = 10, 25
    >>> # at time time[n] at the postion space[i] the density of bacteria is
    >>> print(bac[n,i])
    0.05434032732745466
    >>> bac, nut = solve_1D_explicit(bacteria0=np.ones(5),
    ...                              nutrient0=np.ones(5),
    ...                              time=np.linspace(0,1,5),
    ...                              space=np.linspace(-5,5,5))
    """

    dt = time[1:] - time[:-1]
    dx = np.concatenate((space[1:]-space[:-1],
                         [np.average(space[1:]-space[:-1])]))

    if np.max(dt)/(np.min(dx)**2) >= 1/2:
        warnings.warn('The stability condition dt/(dx**2) < 1/2 should be verified.'
                      f'Yet we have dt/(dx**2) = {np.max(dt)/(np.min(dx)**2)} at some points.')

    bac = np.zeros((len(time), len(space)))
    nut = np.zeros((len(time), len(space)))

    bac[0] = bacteria0
    nut[0] = nutrient0

    for n in range(0, len(time)-1):

        for i in range(len(space)):

            j = i + (i+1 != len(space))  # Neumann boundary conditions
            if space[0] == 0:
                k = (i-1)*(i != 0) + (i == 0)  # symetrie
            else:
                k = i - (i-1 != 0)  # Neumann boundary conditions
            coeff = dt[n]/(dx[i]**2)
            bac[n+1, i] = bac[n, i] \
                + epsilon*coeff * (bac[n, j] - 2*bac[n, i] + bac[n, k]) \
                + dt[n]/epsilon * bac[n, i]*(nut[n, i] - mu)

            nut[n+1, i] = nut[n, i] \
                + coeff * (nut[n, j] - 2*nut[n, i] + nut[n, k]) \
                - dt[n] * bac[n, i]*nut[n, i]

    return bac, nut


def solve_1D_implicit(bacteria0: np.ndarray,
                      nutrient0: np.ndarray,
                      time: np.ndarray,
                      space: np.ndarray,
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
    >>> space = np.arange(0, 5+0.1, 0.1)
    >>> time = np.arange(0, 1, 0.004)
    >>> bacteria0 = np.exp(-space**2/2)
    >>> nutrient0 = np.ones(len(space))
    >>> bac, nut = solve_1D_implicit(bacteria0, nutrient0, time, space, epsilon=0.1)
    >>> print(bac[10,25])
    0.054666616059249024
    >>> bac, nut = solve_1D_implicit(bacteria0=np.ones(5),
    ...                              nutrient0=np.ones(5),
    ...                              time=np.linspace(0,1,5),
    ...                              space=np.linspace(-5,5,5))
    """

    dt_tab = time[1:] - time[:-1]
    dx_tab = space[1:] - space[:-1]
    dt = np.mean(dt_tab)
    dx = np.mean(dx_tab)

    uniform_issue = (((dx_tab-dx)/dx) > SMALL).any() \
        or (((dt_tab-dt)/dt) > SMALL).any()
    if uniform_issue:
        raise ValueError(
            'The space and time discretisations have to be uniform')

    bac = np.zeros((len(time), len(space)))
    nut = np.zeros((len(time), len(space)))

    bac[0] = bacteria0
    nut[0] = nutrient0

    coeff = dt/(dx**2)

    ones = np.ones(len(space)-1)

    mat_bac = (1+2*epsilon*coeff) * np.eye(len(space)) \
        - epsilon*coeff * np.diag(ones, 1) \
        - epsilon*coeff * np.diag(ones, -1)

    mat_nut = (1+2*coeff) * np.eye(len(space)) \
        - coeff * np.diag(ones, 1) \
        - coeff * np.diag(ones, -1)
    if space[0] == 0:
        # symetry on zero
        mat_bac[0, 1] += -epsilon*coeff
        mat_nut[0, 1] += -coeff
    else:
        # Neumann boundary conditions
        mat_bac[0, 0] += -epsilon*coeff
        mat_nut[0, 0] += -coeff
    # Neumann boundary conditions
    mat_bac[-1, -1] += -epsilon*coeff
    mat_nut[-1, -1] += -coeff

    # see documentation of scipy.solve_banded in order to understand
    code_bac = np.zeros((1+1+1, len(space)))
    code_nut = np.zeros((1+1+1, len(space)))
    for j in range(len(space)):
        for i in range(len(space)):
            if j-1 <= i <= j+1:
                code_bac[1+i-j, j] = mat_bac[i, j]
                code_nut[1+i-j, j] = mat_nut[i, j]

    for n in range(0, len(time)-1):
        sec_bac = (1 + dt/epsilon*(nut[n]-mu)) * bac[n]
        sec_nut = (1 - dt*bac[n]) * nut[n]

        bac[n+1] = solve_banded((1, 1), code_bac, sec_bac)
        nut[n+1] = solve_banded((1, 1), code_nut, sec_nut)

    return bac, nut


if __name__ == "__main__":

    import doctest
    doctest.testmod()
