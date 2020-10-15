import numpy as np
import itertools
from simulator import life_step

def generate_all(m, n):
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


# Data generator according to what they desribed here:
#
# https://www.kaggle.com/c/conways-reverse-game-of-life-2020/data
#
# An initial board was chosen by filling the board with a random density between 1% full (mostly zeros) and 99% full (mostly ones).
# This initial board was evolved 5 steps.
# The starting board's state was recorded after the 5 "warmup steps". These are the values in the start variables.
# The starting board was then evolved delta steps. Delta was chosen to be uniformly random between 1 and 5. If the stopping board was empty, the game was discarded.
# The stopping board's state was then recorded. These are the values in the stop variables.
def generate_test_set(set_size, seed, **kwargs):
    return generate_n_cases(False, set_size, seed, **kwargs)

def generate_train_set(set_size, seed, **kwargs):
    return generate_n_cases(True, set_size, seed, **kwargs)

def generate_n_cases(train, set_size, seed, **kwargs):
    return itertools.islice(
        generate_inf_cases(train, seed, **kwargs),
        set_size)

def generate_inf_cases(train, seed, board_size=25, min_dens=0.1, max_dens=0.99, warm_up=5, min_delta=1, max_delta=5, dtype=np.int):
    rs = np.random.RandomState(seed)
    zer = np.zeros(shape=(board_size, board_size), dtype=dtype)
    while True:
        density = rs.uniform(min_dens, max_dens)
        start = rs.choice([1, 0], size=(board_size, board_size), p=[density, 1.0-density])
        for _ in range(warm_up):
            start = life_step(start)

        delta = rs.randint(min_delta, max_delta+1)
        stop = start.copy()
        for _ in range(delta):
            stop = life_step(stop)

        if not (stop == zer).all():
            if train:
                yield delta, start, stop
            else:
                yield delta, stop
