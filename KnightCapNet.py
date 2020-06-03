import torch
import torch.nn as nn
import torch.nn.functional as F


class KnightCapNet(nn.Module):

    def __init__(self, xg_size, xp_size, xs_size):
        '''
        xg_size: size of global features
        xp_size: size of piece-centric features
        xs_size: size of square-centric features
        '''
        super(KnightCapNet, self).__init__()
        self.xg_size = xg_size
        self.xp_size = xp_size
        self.xs_size = xs_size

        self.linear = nn.Linear(self.xg_size + self.xp_size + self.xs_size, 1)

    def forward(self, xg, xp, xs):
        x = torch.cat((xg, xp, xs), dim=1)
        x = torch.tanh(self.linear(x))
        return x

