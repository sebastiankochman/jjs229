import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
from tqdm import tqdm
from poutyne import Model

import bitmap
import scoring
import itertools

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

# TF:
#model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(25, 25, 1), padding='same'))
#model.add(layers.Conv2D(8, (5, 5), activation='relu', input_shape=(25, 25, 1), padding='same'))
#model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(25, 25, 1), padding='same'))
#model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in channels, out channels, kernel size
        #self.conv_a = nn.Conv2d(1, 4, (3, 3), padding=1, padding_mode='circular')
        #self.conv_b = nn.Conv2d(4, 1, (3, 3), padding=1, padding_mode='circular')
        self.conv1 = nn.Conv2d(1, 16, (5, 5), padding=2, padding_mode='circular')
        self.conv2 = nn.Conv2d(16, 8, (5, 5), padding=2, padding_mode='circular')
        self.conv3 = nn.Conv2d(8, 4, (3, 3), padding=1, padding_mode='circular')
        self.conv4 = nn.Conv2d(4, 1, (3, 3), padding=1, padding_mode='circular')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        #x = F.relu(self.conv_a(x))
        #x = torch.sigmoid(self.conv_b(x))
        return x

net = Net()

print(net)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters()) #, lr=0.001, momentum=0.9)


model = Model(net, optimizer, criterion)

def predict(deltas_batch, stops_batch):
    max_delta = np.max(deltas_batch)
    preds = [np.array(stops_batch, dtype=np.float)]
    for _ in range(max_delta):
        tens = torch.Tensor(preds[-1])
        pred_batch = net(tens) #np.array(net(tens) > 0.5, dtype=np.float)
        preds.append(pred_batch)

    # Use deltas as indices into preds.
    # TODO: I'm sure there's some clever way to do the same using numpy indexing/slicing.
    final_pred_batch = []
    for i in range(deltas_batch.size):
        final_pred_batch.append(preds[np.squeeze(deltas_batch[i])][i])

    return final_pred_batch

def cnnify_batch(batches):
    return (np.expand_dims(batch, 1) for batch in batches)

val_set = bitmap.generate_test_set(set_size=100, seed=9568382)
deltas_val, stops_val = cnnify_batch(zip(*val_set))
ones_val = np.ones_like(deltas_val)

multi_step_errors = []
one_step_errors = []
best_multi_step_error = 1.0
best_multi_step_idx = -1
best_one_step_error = 1.0
best_one_step_idx = -1

for i, batch in tqdm(enumerate(grouper(bitmap.generate_inf_cases(True, 432341, return_one_but_last=True), 2048))):
    deltas, one_but_lasts, stops = zip(*batch)

    deltas_batch = np.expand_dims(deltas, 1)
    one_but_lasts_batch = torch.Tensor(np.expand_dims(one_but_lasts, 1))
    stops_batch = torch.Tensor(np.expand_dims(stops, 1))

    if i % 10 == 0:
        multi_step_pred_batch = predict(deltas_val, stops_val)
        multi_step_mean_err = 1 - np.mean(scoring.score_batch(deltas_val, np.array(multi_step_pred_batch, dtype=np.bool), stops_val))

        one_step_pred_batch = net(torch.Tensor(stops_val)) > 0.5
        one_step_mean_err = 1 - np.mean(scoring.score_batch(ones_val, np.array(one_step_pred_batch, dtype=np.bool), stops_val))
        print(f'Mean error: multi-step {multi_step_mean_err}, one step {one_step_mean_err}')

        #if multi_step_mean_err < best_multi_step_error:
        #    best_multi_step_error = multi_step_mean_err
        #    best_multi_step_idx =

    model.fit(stops_batch, one_but_lasts_batch, epochs=1)

    """
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    #outputs = net(stops_batch)
    stops_batch.requires_grad_(True)
    loss = criterion(stops_batch, one_but_lasts_batch)
    loss.backward()
    optimizer.step()

    # print statistics
    print('[%5d] loss: %.3f' % (i, loss.item()))
    """
