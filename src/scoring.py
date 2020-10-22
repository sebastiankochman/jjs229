import numpy as np
from simulator import life_step

def score(delta, start, stop):
    X = start
    for i in range(delta):
        X = life_step(X)
    return np.count_nonzero(X == stop) / stop.size

def score_batch(delta_batch, start_batch, stop_batch):
    # TODO: this scoring method is slow, as we need to iterate through all examples.
    #   Maybe there is a way to score a batch of example in vectorized form? (something to investigate!)
    scores = np.zeros((delta_batch.shape[0]))
    for i in range(delta_batch.shape[0]):
        scores[i] = score(np.squeeze(delta_batch[i]), np.squeeze(start_batch[i]), np.squeeze(stop_batch[i]))
    return scores