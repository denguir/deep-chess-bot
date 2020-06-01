from GiraffeNet import GiraffeNet
from functools import reduce
from tqdm import tqdm
import board_encoding as enc
import multiprocessing
import chess
import time
import torch
import torch.nn.functional as F
import torch.optim as optim 
import torch.nn as nn
import numpy as np
import pandas as pd
import random

# Training parameters
EPOCHS = 10
BATCH_SIZE = 256
N_PROC = multiprocessing.cpu_count()


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

    # Select hardware device to train on
    device = "cpu"

    # Instantiate model
    giraffe_net = GiraffeNet(xg_size=15, xp_size=320, xs_size=128)
    giraffe_net.to(device).float()

    val_giraffe_net = GiraffeNet(xg_size=15, xp_size=320, xs_size=128)
    val_giraffe_net.to(device).float()

    # Loading saved weights
    model_name = 'model/stockfish_net_4.pt'
    try:
        print(f'Loading model from {model_name}.')
        giraffe_net.load_state_dict(torch.load(model_name))
        val_giraffe_net.load_state_dict(torch.load(model_name))
    except FileNotFoundError as e:
        print(e)
        print('No model available.')
        print('Initilialisation of a new model with random weights.')

    # Define optimizer + loss fct
    optimizer = optim.Adadelta(giraffe_net.parameters())
    criterion = nn.SmoothL1Loss()
    
    # train-val split
    train_and_val = pd.read_csv('data/csv/train.csv')
    mask = np.random.rand(len(train_and_val)) < 0.85
    train = train_and_val[mask]
    val = train_and_val[~mask]

    iter_per_epoch = len(train) // BATCH_SIZE
    iter_per_val = len(val) // BATCH_SIZE
    giraffe_net.train()
    val_giraffe_net.eval()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i in tqdm(range(iter_per_epoch)):
            with multiprocessing.Pool(processes=N_PROC) as pool:
                batch = train.sample(n=BATCH_SIZE)
                sub_batches = [batch[i*BATCH_SIZE//N_PROC:(i + 1)* BATCH_SIZE//N_PROC] for i in range(N_PROC)]
                inputs_and_targets = list(zip(*pool.map(get_inputs_and_target, sub_batches)))

                xg = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[0]).to(device).float()
                xp = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[1]).to(device).float()
                xs = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[2]).to(device).float()
                targets = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[3]).to(device).float()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            values = giraffe_net(xg, xp, xs)
            loss = criterion(values, targets)
            # backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # print statistics
            if i % 1000 == 999:
                print(f"Epoch {epoch+1}, iter {i+1} \t train_loss: {running_loss/1000}")
                running_loss = 0.0
                val_running_loss = 0.0

                # validation process
                giraffe_net.eval()
                with torch.no_grad():
                    for j in tqdm(range(100)):
                        with multiprocessing.Pool(processes=N_PROC) as pool:
                            batch = val.sample(n=BATCH_SIZE)
                            sub_batches = [batch[i*BATCH_SIZE//N_PROC:(i + 1)* BATCH_SIZE//N_PROC] for i in range(N_PROC)]
                            inputs_and_targets = list(zip(*pool.map(get_inputs_and_target, sub_batches)))
                            
                            xg = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[0]).to(device).float()
                            xp = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[1]).to(device).float()
                            xs = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[2]).to(device).float()
                            targets = reduce(lambda x,y: torch.cat((x,y), dim=0), inputs_and_targets[3]).to(device).float()

                        # forward pass
                        values = giraffe_net(xg, xp, xs) # current net
                        val_values = val_giraffe_net(xg, xp, xs) # prev net

                        loss = criterion(values, targets)
                        val_loss = criterion(val_values, targets)

                        running_loss += loss.item()
                        val_running_loss += val_loss.item()

                    print(f"Epoch {epoch+1}, iter {i+1} \t val_loss: {running_loss/iter_per_val}")
                        
                    if running_loss < val_running_loss:
                        print(f"Validation loss decreased: \
                            {val_running_loss/iter_per_val} -> {running_loss/iter_per_val}")
                        print(f"Saving model to {model_name}")
                        torch.save(giraffe_net.state_dict(), model_name)
                        val_giraffe_net.load_state_dict(torch.load(model_name))

                running_loss = 0.0
                val_running_loss = 0.0
                val_giraffe_net.eval()
                giraffe_net.train()