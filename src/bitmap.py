import numpy as np


def generate_bitmaps(m, n):
    """
    Generates all possible bitmaps of size m x n
    :param m: number of rows
    :param n: number of columns
    :return: generator of numpy matrices of size m x n
    """
    A = np.zeros(m * n, dtype=np.bool)
    for _ in range(2 ** (m * n)):
        yield np.copy(A.reshape((m, n)))
        for i in range(A.size):
            if A[i]:
                A[i] = 0
            else:
                A[i] = 1
                break
