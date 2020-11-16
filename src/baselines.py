import numpy as np
from simulator import life_step
from scoring import score


def const_zeros(delta, stop):
    return np.zeros_like(stop)


def mirror(delta, stop):
    return stop


def mirror_or_zeros(delta, stop):
    # If stop is a still board, then return that. Otherwise, return zeros.
    n = life_step(stop)
    return stop if (stop == n).all() else np.zeros_like(stop)


def sim_density(delta, stop):
    # If stop is a still board, then return that. Otherwise, return zeros.
    n = life_step(stop)
    return stop if (stop == n).all() else np.zeros_like(stop)


def likely_starts(delta, stop):
    # Just test a couple of "likely" starting boards and pick the best one.
    starts = [
        np.zeros_like(stop),
        stop,
        life_step(stop)
    ]
    scores = [score(delta, start, stop) for start in starts]
    idx = np.argmax(scores)
    return starts[idx]
