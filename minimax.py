import chess
import torch
import time
import board_encoding as enc
from GiraffeNet import GiraffeNet
from functools import partial


def naive_evaluation(board):
    '''Naive evaluation function
    - Uppercase: white pieces
    - Lowercase: black pieces
    '''
    value_map = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
                 'p':-1, 'n':-3, 'b':-3, 'r':-5, 'q':-9, 'k': 0}
    val = 0.0
    val_max = 22.0
    # evaluate board by counting living pieces in both sides
    for char in board.board_fen():
        if char in value_map.keys():
            val += value_map[char]
    # attribute some points if side has castling rights
    if board.has_castling_rights(True):
        val += 1.0
    if board.has_castling_rights(False):
        val -= 1.0
    return val/val_max


def giraffe_evaluation(board, net, device):
    # use giraffe net as evaluator
    xg, xp, xs = enc.encode(board)
    xg = enc.decode(xg).to(device)
    xp = enc.decode(xp).to(device)
    xs = enc.decode(xs).to(device)
    val = net(xg, xp, xs)
    return torch.squeeze(val)


def alpha_beta(board, depth, color, alpha, beta, evaluator):
    '''
    Depth limited minimax tree search with alpha-beta pruning
    - depth: number of recursion before jumping into evaluation function
    - color: True (White), False (Black)
    - White is the max player (wants to maximize score)
    - Black is the min player (wants to minimize score)
    '''
    max_score = 10000.0

    if board.is_game_over():
        if board.is_checkmate():
            # if white to play, black won
            if color:
                return -1.0
            # if black to play, white won
            else:
                return 1.0
        else:
            # draw
            return 0.0

    if depth == 0:
        return evaluator(board)
    
    if color:
        best_score = -max_score
        for move in board.legal_moves:
            # make a move
            board.push(move)
            # call alpha_beta recusively and choose max value
            best_score = max(best_score, alpha_beta(board, depth-1, board.turn, alpha, beta, evaluator))
            alpha = max(alpha, best_score)
            # undo move
            board.pop()
            if alpha >= beta:
                break
        return best_score
    
    else:
        best_score = max_score
        for move in board.legal_moves:
            # make a move
            board.push(move)
            # call alpha_beta recusively and choose min value
            best_score = min(best_score, alpha_beta(board, depth-1, board.turn, alpha, beta, evaluator))
            beta = min(beta, best_score)
            # undo move
            board.pop()
            if beta <= alpha:
                break
        return best_score
            

def find_best_move(board, max_depth, evaluator):
    max_score = 10000.0
    best_move = None
    if board.turn:
        best_score = -max_score
    else:
        best_score = max_score
    
    if len(list(board.legal_moves)) > 0:
        for move in board.legal_moves:
            # make a move
            board.push(move)
            # evaluate this move
            score = alpha_beta(board, max_depth, board.turn, -max_score, max_score, evaluator)
            # undo move
            board.pop()

            if board.turn:
                # White want to maximize score
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                # Black want to minimize score
                if score < best_score:
                    best_score = score
                    best_move = move
    else:
        best_score = best_score / max_score
    
    if type(best_score) is float:
        best_score = torch.squeeze(torch.Tensor([best_score]))
    return best_move, best_score


if __name__ == '__main__':

    device = "cpu"
    giraffe_net = GiraffeNet(xg_size=15, xp_size=320, xs_size=128)
    giraffe_net.to(device).float()
    giraffe_net.eval()

    stockfish_net = GiraffeNet(xg_size=15, xp_size=320, xs_size=128)
    stockfish_net.to(device).float()
    stockfish_net.eval()

    # Loading saved weights
    white_model_name = 'model/giraffe_net_td_lambda_07.pt'
    black_model_name = 'model/stockfish_net_5.pt'
    try:
        print(f'Loading white model from {white_model_name}.')
        giraffe_net.load_state_dict(torch.load(white_model_name))
    except:
        print('No model available.')
        print('Initilialisation of a new model with random weights.')

    try:
        print(f'Loading black model from {black_model_name}.')
        stockfish_net.load_state_dict(torch.load(black_model_name))
    except:
        print('No model available.')
        print('Initilialisation of a new model with random weights.')

    board = chess.Board()
    score_board = {'White': 0, 'Black': 0}

    num_games = 10
    for i in range(num_games):
        while not board.is_game_over():
            with torch.no_grad():
                # white to play
                if board.turn:
                    # white is supposed to win if white depth > black depth
                    move, score = find_best_move(board, 1, partial(giraffe_evaluation, net=giraffe_net, device=device))
                    board.push(move)
                    print(score)
                    print('\nBLACK TO PLAY')
                # black to play
                else:
                    move, score = find_best_move(board, 1, partial(giraffe_evaluation, net=stockfish_net, device=device))
                    board.push(move)
                    print(score)
                    print('\nWHITE TO PLAY')
                print(board)
        
        if board.is_game_over():          
            if board.is_checkmate():
                # if white to play, black won
                if board.turn:
                    score_board['Black'] += 1
                    print(f'Game {i + 1}: BLACK WINS')
                # if black to play, white won
                else:
                    score_board['White'] += 1
                    print(f'Game {i + 1}: WHITE WINS')
            else:
                # draw
                score_board['White'] += 0.5
                score_board['Black'] += 0.5
                print(f'Game {i + 1}: DRAW')
            
            print('Score board:')
            print(f"White: {score_board['White']} \t Black: {score_board['Black']}")
            board.reset()

