import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

import bitmap
import scoring
import itertools

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


# TODO: implement circular padding "by hand"
model = models.Sequential()
model.add(layers.Conv2D(9, (3, 3), activation='relu', input_shape=(25, 25, 1), padding='same'))
model.add(layers.Conv2D(9, (3, 3), activation='relu', input_shape=(25, 25, 1), padding='same'))
model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(10))


model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

def predict(deltas_batch, stops_batch):
    max_delta = np.max(deltas_batch)
    preds = [stops_batch]
    for _ in range(max_delta):
        pred_batch = model.predict(preds[-1]) > 0.5
        preds.append(pred_batch)

    # Use deltas as indices into preds.
    # TODO: I'm sure there's some clever way to do the same using numpy indexing/slicing.
    final_pred_batch = []
    for i in range(deltas_batch.size):
        final_pred_batch.append(preds[np.squeeze(deltas_batch[i])][i])

    return final_pred_batch

def cnnify_batch(batches):
    return (np.expand_dims(batch, -1) for batch in batches)

val_set = bitmap.generate_test_set(set_size=10000, seed=9568382)
deltas_val, stops_val = cnnify_batch(zip(*val_set))
ones_val = np.ones_like(deltas_val)

for i, batch in enumerate(grouper(bitmap.generate_inf_cases(True, 432341, return_one_but_last=True), 2048)):
    deltas, one_but_lasts, stops = zip(*batch)

    deltas_batch = np.expand_dims(deltas, -1)
    one_but_lasts_batch = np.expand_dims(one_but_lasts, -1)
    stops_batch = np.expand_dims(stops, -1)

    if i % 5 == 0:
        multi_step_pred_batch = predict(deltas_val, stops_val)
        multi_step_mean_err = 1 - np.mean(scoring.score_batch(deltas_val, multi_step_pred_batch, stops_val))

        one_step_pred_batch = model.predict(stops_val) > 0.5
        one_step_mean_err = 1 - np.mean(scoring.score_batch(ones_val, one_step_pred_batch, stops_val))
        print(f'Mean error: multi-step {multi_step_mean_err}, one step {one_step_mean_err}')

    model.fit(stops_batch, one_but_lasts_batch, epochs=1)
