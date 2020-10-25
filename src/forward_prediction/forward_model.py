import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

#sys.path.append('..')
sys.path.insert(0,'..')
import bitmap


# Forward step model using the weights defined in the appendix of https://arxiv.org/pdf/2009.01398.pdf.
def forward(x): 
    # Weights for layer 1
    weight1 = torch.tensor([[[1, 1, 1], [1, 0.1, 1], [1, 1, 1]],
                            [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]).view(2, 1, 3, 3).float()
    b1 = torch.tensor([-3, -2]).float()
    # Weights for layer 2
    weight2 = torch.tensor([-10, 1]).view(1, 2, 1, 1).float()
    # Weights for layer 3
    s = 20
    weight3 = torch.tensor([2*s]).view(1, 1, 1, 1).float()
    b3 = torch.tensor([-s]).float()
  
    x = F.pad(x.float(), (1, 1, 1, 1), mode='circular')
    x = F.relu(F.conv2d(x, weight1, b1))
    x = F.relu(F.conv2d(x, weight2))
    x = torch.sigmoid(F.conv2d(x, weight3, b3))
    return x

if __name__ == '__main__':
    n = 10000
    data_generator = bitmap.generate_train_set(n, 41)
    total_mae = 0
    for delta, start_board, stop_board in data_generator:
        start_board = torch.from_numpy(start_board).view(1, 1, 25, 25)
        stop_board = torch.from_numpy(stop_board)
        for _ in range(delta):
            start_board = forward(start_board)
     
        mae = ((start_board.view(25, 25) > 0.5).int() != stop_board).sum()
        if mae > 0:
            print("*** The CNN failed to reach the stop board from start board in {} steps ***".format(delta))
        total_mae += mae

    print("The average MAE is {}".format(total_mae / n))
