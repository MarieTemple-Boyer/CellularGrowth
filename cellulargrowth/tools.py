''' Technical tools '''

import numpy as np

SMALL = 0.0001


def step_from_array(array: np.ndarray,
                    name: str):
    '''
    Return the step of the array if it is uniform and raise an expection if not
    >>> unif = np.array([0, 1, 2, 3])
    >>> non_unif = np.array([0, 1, 3, 4])
    >>> step_from_array(unif, 'name_of_array')
    1.0
    >>> step_from_array(non_unif, 'name_of_array')
    Traceback (most recent call last):
        ...
    ValueError: The name_of_array discretisation has to be uniform
    '''
    d_array = array[1:] - array[:-1]
    step = np.mean(d_array)
    uniform_issue = uniform_issue = (((d_array-step)/step) > SMALL).any()
    if uniform_issue:
        raise ValueError('The ' + name + ' discretisation has to be uniform')
    return step

def encode_tridiagonal_matrix(mat):
    '''
    Return an encoding 'code_mat' of the triagonal matrix 'mat' where
    of i, j such that mat[i,j] != 0 we have code_mat[1+i-j, j] = a[i,j]

    >>> mat = np.array([[1, 1, 0, 0],
    ...                 [2, 1, 3, 0],
    ...                 [0, 2, 1, 3],
    ...                 [0, 0,-1, 1]])
    >>> encode_tridiagonal_matrix(mat)
    array([[ 0.,  1.,  3.,  3.],
           [ 1.,  1.,  1.,  1.],
           [ 2.,  2., -1.,  0.]])
    '''
    # see documentation of scipy.solve_banded in order to understand
    code_mat = np.zeros((3, len(mat[0])))
    for i in range(len(mat[0])):
        for j in range(len(mat[0])):
            if j-1 <= i<= j+1:
                code_mat[1+i-j, j] = mat[i,j]
            elif mat[i, j] != 0:
                raise ValueError('The matrix should be triadiagonal')
    return code_mat

if __name__ == '__main__':
    import doctest
    doctest.testmod()
