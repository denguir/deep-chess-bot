from GiraffeNet import GiraffeNet
from functools import reduce, partial
from tqdm import tqdm
from minimax import find_best_move, giraffe_evaluation
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


# Training parameters
EPOCHS = 1
BATCH_SIZE = 256 
N_PROC = multiprocessing.cpu_count()


def push_move(board, move):
    if not board.is_game_over():
        board.push(move)
    return board


def push_random_move(board):
    if not board.is_game_over():
        legal_moves = list(board.legal_moves)
        n = random.randint(0, len(legal_moves) - 1)
        board.push(legal_moves[n])
    return board


def self_play_mc(batch, net, device, td_lambda):
    '''Self play on entire game'''
    boards = [chess.Board(b) for b in batch['board']]
    boards = list(map(push_random_move, boards))
    giraffe_move = partial(find_best_move, max_depth=0, 
                                evaluator=partial(giraffe_evaluation, net=net, device=device))
    err = torch.zeros(len(boards))
    for b, board in enumerate(boards):
        scores_board = []
        while not board.is_game_over():
            move, score = giraffe_move(board)
            scores_board.append(score)
            board.push(move)
        # when game is over, the score is the winner (1, -1, or 0)
        _, score = giraffe_move(board)
        scores_board.append(score)
        for t in range(len(scores_board)-1):
            discount = 1
            err_t = 0
            for j in range(t, len(scores_board)-1):
                dj = scores_board[j+1] - scores_board[j]
                discount *= td_lambda
                err_t += discount * dj
            err[b] -= scores_board[t] * err_t.detach()
    loss = torch.mean(err)
    return loss


def self_play(batch, net, device, n_moves):
    '''Self play on n_moves of a given game'''
    boards = [chess.Board(b) for b in batch['board']]
    boards = list(map(push_random_move, boards))
    giraffe_move = partial(find_best_move, max_depth=0, 
                                evaluator=partial(giraffe_evaluation, net=net, device=device))
    scores = []
    for _ in range(n_moves):
        moves_, scores_ = zip(*map(giraffe_move, boards))
        scores.append(torch.stack(scores_))
        boards = [push_move(board, move) for (board, move) in zip(boards, moves_)]
    #scores = list(zip(*scores))
    scores = torch.stack(scores)
    return torch.t(torch.squeeze(scores))


def td_loss(scores, td_lamdba):
    L, N = scores.size()
    err = torch.zeros((L, N))
    for t in range(N-1):
        discount = 1
        err_t = torch.zeros(L)
        for j in range(t, N-1):
            dj = scores[:, j+1] - scores[:, j]
            discount *= td_lamdba
            err_t += discount * dj
        err[:, t] = err_t
    # we include a minus sign because torch computes a gradient descent
    # by default, but we want to impose a custom update rule for the weights
    loss = torch.mean(torch.sum(-scores * err.detach(), dim=1))
    return loss


def n_steps_td_loss(scores, n_steps):
    # better use an even number for n_steps
    _, N = scores.size()
    criterion = nn.L1Loss()
    loss = criterion(scores[:, :N-n_steps], scores[:, n_steps:].detach())
    return loss


def self_learn(batch, net, device, n_moves, optimizer):
    #loss = self_play_mc(batch, net, device, 0.7)
    scores = self_play(batch, net, device, n_moves)
    loss = td_loss(scores, 0.7)
    # loss = n_steps_td_loss(scores, 6)
    with multiprocessing.Lock():
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()
    

if __name__ == '__main__':
    # Train on cpu, gpu wont work as we use multiprocessing
    device = "cpu"

    # Instantiate model
    giraffe_net = GiraffeNet(xg_size=15, xp_size=320, xs_size=128)
    giraffe_net.to(device).float()

    # Loading saved weights
    model_name = 'model/giraffe_net_td_lambda_07.pt'
    try:
        print(f'Loading model from {model_name}.')
        giraffe_net.load_state_dict(torch.load(model_name))
    except FileNotFoundError as e:
        try:
            print(e)
            print('Loading model from model/stockfish_net_5.pt')
            giraffe_net.load_state_dict(torch.load('model/stockfish_net_5.pt'))
        except FileNotFoundError as e:
            print(e)
            print('No model available.')
            print('Initilialisation of a new model with random weights.')

    # Define optimizer
    optimizer = optim.Adadelta(giraffe_net.parameters())

    # Load training data
    train = pd.read_csv('data/csv/train.csv')
    iter_per_epoch = len(train) // BATCH_SIZE

    # Activate memory sharing accross processes
    giraffe_net.train()
    giraffe_net.share_memory()

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        print(e)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i in tqdm(range(iter_per_epoch)):
            with multiprocessing.Pool(processes=N_PROC) as pool:
                batch = train.sample(n=BATCH_SIZE)
                sub_batches = [batch[i*BATCH_SIZE//N_PROC:(i + 1)* BATCH_SIZE//N_PROC] for i in range(N_PROC)]
                losses = pool.map(partial(self_learn, net=giraffe_net, device="cpu", n_moves=12, optimizer=optimizer), sub_batches)

            running_loss += reduce(lambda x,y: x+y, losses) / N_PROC

            if i % 50 == 49:
                print(f"Epoch {epoch+1}, iter {i+1} \t train_loss: {running_loss/(13)}")
                running_loss = 0.0

                print(f"Saving model to {model_name}")
                torch.save(giraffe_net.state_dict(), model_name)
