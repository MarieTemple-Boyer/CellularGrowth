""" Resolution of a modelisation of cellular development 
d_t u - laplacian(u) = 1/epsilon u (v-mu)
d_t v - laplacian(v) = -uv
where u is the density of bacteria and v the density of nutrient.
"""

import numpy as np


def solve_1D(bacteria0: np.ndarray,
             nutrient0: np.ndarray,
             time: np.ndarray,
             space: np.ndarray,
             mu: float = 0.5,
             epsilon: float = 1):
    """ Solve the following equation :
            d_t u - epsilon d_xx u = 1/epsilon u (v-mu)
            d_t v - d_xx v = -uv
        where u is the density of bactery and v the density of nutrient.
    The initial density are bacteria0 and nutrient0.
    The discretisation grid is defined by time and space.
    The parameters of the equation are mu and epsilon.

    >>> space = np.arange(-5, 5+0.1, 0.1)
    >>> time = np.arange(0, 1+0.004, 0.004)
    >>> bacteria0 = np.exp(-space**2 / 2)
    >>> nutrient0 = np.ones(len(space))
    >>> bac, nut = solve_1D(bacteria0, nutrient0, time, space)
    >>> n, i = 10, 43
    >>> # at time time[n] at the postion space[i] the density of bacteria is
    >>> print(bac[n,i])
    0.7820090553382119
    """

    dt = time[1:] - time[:-1]
    dx = np.concatenate((space[1:]-space[:-1],
                         [np.average(space[1:]-space[:-1])]))

    if np.max(dt)/(np.min(dx)**2) >= 1/2:
        raise ValueError('The stability condition dt/(dx**2) < 1/2 shoul be verified.'
                         f'Yet we have dt/(dx**2) = {np.max(dt)/(np.min(dx)**2)} at some points.')

    bac = np.zeros((len(time), len(space)))
    nut = np.zeros((len(time), len(space)))

    bac[0] = bacteria0
    nut[0] = nutrient0

    for n in range(0, len(time)-1):

        for i in range(len(space)):

            j = i+1
            if j == len(space):  # periodic boundary conditions
                j = 0

            coeff = dt[n]/(dx[i]**2)

            bac[n+1, i] = bac[n, i] \
                + epsilon*coeff * (bac[n, j] - 2*bac[n, i] + bac[n, i-1]) \
                + dt[n]/epsilon * bac[n, i]*(nut[n, i] - mu)

            nut[n+1, i] = nut[n, i] \
                + coeff * (nut[n, j] - 2*nut[n, i] + nut[n, i-1]) \
                - dt[n] * bac[n, i]*nut[n, i]

    return bac, nut


if __name__ == "__main__":
    import doctest
    doctest.testmod()
