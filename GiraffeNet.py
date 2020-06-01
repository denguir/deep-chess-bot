import torch
import torch.nn as nn
import torch.nn.functional as F


class GiraffeNet(nn.Module):

    def __init__(self, xg_size, xp_size, xs_size):
        '''
        xg_size: size of global features
        xp_size: size of piece-centric features
        xs_size: size of square-centric features
        '''
        super(GiraffeNet, self).__init__()
        self.xg_size = xg_size
        self.xp_size = xp_size
        self.xs_size = xs_size
        
        self.hidden_g = nn.Linear(self.xg_size, 2 * self.xg_size)
        self.hidden_p = nn.Linear(self.xp_size, 2 * self.xp_size)
        self.hidden_s = nn.Linear(self.xs_size, 2 * self.xs_size)

        self.hidden = nn.Linear(
            2 * (self.xg_size + self.xp_size + self.xs_size), 1)

    def forward(self, xg, xp, xs):
        xg = F.relu(self.hidden_g(xg))
        xp = F.relu(self.hidden_p(xp))
        xs = F.relu(self.hidden_s(xs))
        x = torch.cat((xg, xp, xs), dim=1)
        x = torch.tanh(self.hidden(x))
        return x

