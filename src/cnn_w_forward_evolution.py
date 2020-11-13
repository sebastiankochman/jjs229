import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.autograd import Variable
import torch
from tqdm import tqdm
# from poutyne import Model

import bitmap
import scoring
import itertools
from forward_prediction import forward_model

class ReverseNet(nn.Module):
    def __init__(self):
        super(ReverseNet, self).__init__()
        # in channels, out channels, kernel size
        self.conv1 = nn.Conv2d(1, 16, (5, 5), padding=(2, 2), padding_mode='circular')
        self.activ1 = nn.PReLU()
        self.conv2 = nn.Conv2d(16, 8, (5, 5), padding=(2, 2), padding_mode='circular')
        self.activ2 = nn.PReLU()
        self.conv3 = nn.Conv2d(8, 4, (3, 3), padding=(1, 1), padding_mode='circular')
        self.activ3 = nn.PReLU()
        self.conv4 = nn.Conv2d(4, 1, (3, 3), padding=(1, 1), padding_mode='circular')

    def forward(self, x):
        x = self.reverse(x)
        x = forward_model.forward(x)
        return x

    def reverse(self, x):
        x = self.activ1(self.conv1(x))
        x = self.activ2(self.conv2(x))
        x = self.activ3(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

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

if __name__ == '__main__':
    net = ReverseNet()
    print(net)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters()) #, lr=0.001, momentum=0.9)

    n = 12800
    input_data_gen = bitmap.generate_train_set(n, 41, max_delta=1)
    _, start_boards, stop_boards = map(list, zip(*list(input_data_gen)))

    X = Variable(torch.tensor(stop_boards).view(n, 1, 25, 25).float(), requires_grad=True)
    num_epochs = 100
    batch_size = 128
    for epoch in range(num_epochs): 
        permutation = torch.randperm(X.size()[0])
        running_loss = 0.0
        for i in range(0, X.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch = X[indices]
            print(batch)
            optimizer.zero_grad()
            outputs = net(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
        
