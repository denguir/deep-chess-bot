from GiraffeNet import GiraffeNet
from functools import reduce
from tqdm import tqdm
import multiprocessing
import board_encoding as enc
import chess
import time
import torch
import torch.nn.functional as F
import torch.optim as optim 
import torch.nn as nn
import numpy as np
import pandas as pd
import random

# Testing parameters
BATCH_SIZE = 256
N_PROC = multiprocessing.cpu_count()

# Select hardware device to train on
device = "cpu"

# Instantiate model
giraffe_net = GiraffeNet(xg_size=15, xp_size=320, xs_size=128)
giraffe_net.to(device).float()

# Loading saved weights
model_name = 'model/giraffe_net_2.pt'
try:
    print(f'Loading model from {model_name}.')
    giraffe_net.load_state_dict(torch.load(model_name))
except FileNotFoundError as e:
    print(e)
    print('No model available.')
    print('Initilialisation of a new model with random weights.')

# Define optimizer + loss fct
optimizer = optim.Adadelta(giraffe_net.parameters())
criterion = nn.SmoothL1Loss()


# def get_inputs_and_target(batch):
#     global_features = map(enc.decode, batch['feature_g'])
#     piece_features = map(enc.decode, batch['feature_p'])
#     square_features = map(enc.decode, batch['feature_s'])
    
#     # inputs + target
#     xg = torch.cat(list(global_features), dim=0).to(device)
#     xp = torch.cat(list(piece_features), dim=0).to(device)
#     xs = torch.cat(list(square_features), dim=0).to(device)
#     targets = torch.Tensor(batch['value_norm'].values).unsqueeze(1).to(device)

#     return xg, xp, xs, targets


def get_inputs_and_target(batch):
    boards = [chess.Board(b) for b in batch['board']]
    global_features = map(enc.get_global_features, boards)
    piece_features = map(enc.get_piece_centric_features, boards)
    square_features = map(enc.get_square_centric_features, boards)

    global_features = map(torch.from_numpy, global_features)
    piece_features = map(torch.from_numpy, piece_features)
    square_features = map(torch.from_numpy, square_features)

    xg = reduce(lambda x,y: torch.cat((x,y), dim=0), global_features)
    xp = reduce(lambda x,y: torch.cat((x,y), dim=0), piece_features)
    xs = reduce(lambda x,y: torch.cat((x,y), dim=0), square_features)

    targets = torch.Tensor(batch['value_norm'].values).unsqueeze(1)
    return xg, xp, xs, targets


if __name__ == '__main__':

    # test set
    test = pd.read_csv('data/csv/test.csv')
    giraffe_net.eval()
    test_iter = len(test) // BATCH_SIZE

    running_loss = 0.0
    with torch.no_grad():
        for i in tqdm(range(test_iter)):
            with multiprocessing.Pool(processes=N_PROC) as pool:
                batch = test.sample(n=BATCH_SIZE)
                sub_batches = [batch[i*BATCH_SIZE//N_PROC:(i + 1)* BATCH_SIZE//N_PROC] for i in range(N_PROC)]
                inputs_and_targets = list(zip(*pool.map(get_inputs_and_target, sub_batches)))

                xg = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[0]).to(device).float()
                xp = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[1]).to(device).float()
                xs = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[2]).to(device).float()
                targets = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[3]).to(device).float()
            # forward pass
            values = giraffe_net(xg, xp, xs)
            loss = criterion(values, targets)
            running_loss += loss.item()

        print(f"Test_loss: {running_loss/test_iter}")