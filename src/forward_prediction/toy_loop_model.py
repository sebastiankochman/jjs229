import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch
from poutyne import Model


dim = 4

# Learnable backward net.
class BackwardNet(nn.Module):
    def __init__(self):
        super(BackwardNet, self).__init__()
        self.dense1 = nn.Linear(dim, dim)
        self.dense2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        return x

# Fixed forward net.
class ForwardNet(nn.Module):
    def __init__(self):
        super(ForwardNet, self).__init__()
        self.w1 = torch.rand(dim, dim)
        self.b1 = torch.rand(dim)
        self.w2 = torch.rand(dim, dim)
        self.b2 = torch.rand(dim)

    def forward(self, x):
        x = F.relu(x @ self.w1 + self.b1)
        x = torch.sigmoid(x @ self.w2 + self.b2)
        return x

class FullNet(nn.Module):
    def __init__(self):
        super(FullNet, self).__init__()
        self.backward_net = BackwardNet()
        self.forward_net = ForwardNet()

    def forward(self, x):
        x = self.backward_net.forward(x)
        x = self.forward_net.forward(x)
        return x

net = FullNet()
print(net)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters()) #, lr=0.001, momentum=0.9)

model = Model(net, optimizer, criterion)

m = 100
rs = np.random.RandomState(412112)
X = torch.tensor(rs.choice([1, 0], size=(m, dim)), dtype=torch.float)

mae_0 = torch.true_divide(torch.sum(X != (net(X) > 0.5)), m*dim)
print(f'MAE before training: {mae_0}')
model.fit(X, X, epochs=100)
mae_1 = torch.true_divide(torch.sum(X != (net(X) > 0.5)), m*dim)
print(f'MAE after training: {mae_1} (improvement: {mae_1 - mae_0})')
