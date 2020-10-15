import numpy as np


def const_zeros(delta, stop):
    return np.zeros_like(stop)


def mirror(delta, stop):
    return stop
