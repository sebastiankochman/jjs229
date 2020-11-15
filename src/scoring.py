import numpy as np
from simulator import life_step
from tqdm import tqdm

def score(delta, start, stop):
    X = start
    for i in range(delta):
        X = life_step(X)
    return np.count_nonzero(X == stop) / stop.size

def score_batch(delta_batch, start_batch, stop_batch, show_progress=True):
    # TODO: this scoring method is slow, as we need to iterate through all examples.
    #   Maybe there is a way to score a batch of example in vectorized form? (something to investigate!)
    scores = np.zeros((delta_batch.shape[0]), dtype=np.float32)
    for i in tqdm(range(delta_batch.shape[0]), disable=not show_progress):
        scores[i] = score(np.squeeze(delta_batch[i]), np.squeeze(start_batch[i]), np.squeeze(stop_batch[i]))
    return scores