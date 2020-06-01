import os
from tqdm import tqdm
from os import listdir
from os.path import isfile, join, splitext, getsize
from functools import partial
import multiprocessing
import time
import chess
import chess.pgn
import chess.engine
import pandas as pd
import csv

N_PROC = multiprocessing.cpu_count()


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    return file_list


def build_data_and_target(pgn_file, max_game):
    engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
    pgn = open(pgn_file, 'r')
    csv_name = pgn_file.split('/')[2].split('.')[0] + '.csv'
    with open(join('data', 'csv', csv_name), 'a') as csv_file:
        fnames = ['board', 'value', 'value_norm']
        writer = csv.DictWriter(csv_file, fieldnames=fnames)
        if getsize(join('data', 'csv', csv_name)) == 0:
            writer.writeheader()
        i = 0
        game = True
        pbar = tqdm(total=max_game, initial=i)
        while i < max_game and game:
            try:
                game = chess.pgn.read_game(pgn)
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    board_fen = board.fen()

                    # get score from stockfish from white perspective
                    info = engine.analyse(board, chess.engine.Limit(time=0.1))
                    score = info["score"].white().score(mate_score=10_000)
                    score_norm = score / 10_000 # mate = 10000 score

                    writer.writerow({'board': board_fen,
                                     'value': score,
                                     'value_norm': score_norm})
            except BaseException as e:
                print(e)
                game = False
            i += 1
            pbar.update(1)
    pbar.close()
    engine.quit()


if __name__ == '__main__':
    pgn_files = get_file_list("data/pgn/", extension='.pgn')
    with multiprocessing.Pool(processes=N_PROC) as pool:
        # set max_game to a higher number to build a bigger database
        pool.map(partial(build_data_and_target, max_game=10), pgn_files)

    # concatenate all csv files into one single file
    csv_files = get_file_list("data/csv/", extension='.csv')
    df = pd.concat((pd.read_csv(f) for f in csv_files if not f.endswith(('train.csv', 'test.csv'))))

    # dropna, remove duplicates and shuffle
    df = df.dropna()
    df = df.drop_duplicates(subset='board', keep='first')
    df = df.sample(frac=1).reset_index(drop=True)

    # train-test split
    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]
    train.to_csv("data/csv/train.csv", index=False)
    test.to_csv("data/csv/test.csv", index=False)