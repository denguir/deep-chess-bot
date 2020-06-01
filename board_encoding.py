import chess
import torch
import ast
import numpy as np


ALL_PIECES = {'K': 1, 'Q': 1, 'R': 2, 'B': 2, 'N': 2, 'P': 8,
              'k': 1, 'q': 1, 'r': 2, 'b': 2, 'n': 2, 'p': 8}


VALUE_MAP = {'K': 100, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1,
             'k': 100, 'q': 9, 'r': 5, 'b': 3, 'n': 3, 'p': 1}


def get_side_to_move(board):
    t = np.zeros(1)
    if board.turn:
        t[0] = 1
    return t


def get_castling_rights(board):
    t = np.zeros(4)
    castling_rights = board.fen().split(' ')[2]
    for char in castling_rights:
        if char == 'K':
            t[0] = 1
        elif char == 'Q':
            t[1] = 1
        elif char == 'k':
            t[2] = 1
        elif char == 'q':
            t[3] = 1
    return t


def get_pieces_count(board):
    t = np.zeros(10)
    pieces = ['Q', 'R', 'B', 'N', 'P']
    for i, piece in enumerate(pieces):
        white_piece = board.board_fen().count(piece.upper())
        black_piece = board.board_fen().count(piece.lower())
        t[i] = white_piece / ALL_PIECES[piece]
        t[i + 5] = black_piece / ALL_PIECES[piece]
    return t


def get_global_features(board):
    t1 = get_side_to_move(board)
    t2 = get_castling_rights(board)
    t3 = get_pieces_count(board)
    t = np.concatenate((t1, t2, t3))
    return t[np.newaxis, :]


def coord_to_pos(x, y):
    return y * 8 + x


def pos_to_coord(pos):
    return pos % 8, pos // 8


def get_attackers_by(board, pos, color):
    pos_attackers = board.attackers(color, pos)
    attackers = map(lambda x: board.piece_at(x), pos_attackers)
    return attackers


def get_attackers_from_pos(board, pos):
    piece = board.piece_at(pos)
    if piece.symbol() in 'KQRBNP':
        attackers = get_attackers_by(board, pos, chess.BLACK)
    elif piece.symbol() in 'kqrbnp':
        attackers = get_attackers_by(board, pos, chess.WHITE)
    return attackers


def get_defenders_from_pos(board, pos):
    piece = board.piece_at(pos)
    if piece.symbol() in 'KQRBNP':
        defenders = get_attackers_by(board, pos, chess.WHITE)
    elif piece.symbol() in 'kqrbnp':
        defenders = get_attackers_by(board, pos, chess.BLACK)
    return defenders


def get_least_val_attacker_from_pos(board, pos):
    attackers = list(get_attackers_from_pos(board, pos))
    if len(attackers) > 0:
        least_val_attacker = min(attackers, key=lambda x: VALUE_MAP[x.symbol()])
        return VALUE_MAP[least_val_attacker.symbol()]
    else:
        return 0


def get_least_val_defender_from_pos(board, pos):
    defenders = list(get_defenders_from_pos(board, pos))
    if len(defenders) > 0:
        least_val_defender = min(defenders, key=lambda x: VALUE_MAP[x.symbol()])
        return VALUE_MAP[least_val_defender.symbol()]
    else:
        return 0


def get_mobility_from_pos(board, pos, color):
    # dir: from [1, 7, 8, 9]
    # dir=1: horizontal move (pos -> pos + k)
    # dir=8: vertical move (pos -> pos + k*8)
    # dir=7: SE-NW diagonal move (pos -> pos + k*7)
    # dir=9: SW-NE diagonal move (pos -> pos + k*9)
    turn = board.turn
    x, y = pos_to_coord(pos)
    dirs = {9: {'x':[0, 0], 'y':[0, 0]},
            7: {'x':[0, 0], 'y':[0, 0]},
            8: {'x':[0, 0], 'y':[0, 0]},
            1: {'x':[0, 0], 'y':[0, 0]}}

    # We dont want zero mobility output for the player
    # that can not play this turn (neural net might
    # think this piece will not be able to move afterwards)
    board.turn = color
    for move in board.legal_moves:
        if move.from_square == pos:
            horitzontal = True # flag to check horizontal move
            for dir in [9, 8, 7]:
                if (move.to_square - pos) % dir == 0:
                    xi, yi = pos_to_coord(move.to_square)
                    dirs[dir]['x'][0] = min(dirs[dir]['x'][0], xi - x)
                    dirs[dir]['x'][1] = max(dirs[dir]['x'][1], xi - x)
                    dirs[dir]['y'][0] = min(dirs[dir]['y'][0], yi - y)
                    dirs[dir]['y'][1] = max(dirs[dir]['y'][1], yi - y)
                    horitzontal = False
            if horitzontal:
                xi, yi = pos_to_coord(move.to_square)
                dirs[1]['x'][0] = min(dirs[1]['x'][0], xi - x)
                dirs[1]['x'][1] = max(dirs[1]['x'][1], xi - x)
                dirs[1]['y'][0] = min(dirs[1]['y'][0], yi - y)
                dirs[1]['y'][1] = max(dirs[1]['y'][1], yi - y)
    board.turn = turn
    return dirs


def get_piece_centric_features(board):
    t = np.zeros(320)
    rows = board.board_fen().split('/')[::-1]
    b = ''.join(rows)
    all_pieces_keys = 'KQRBNPkqrbnp'
    piece_to_bcoord = {k: [] for k in all_pieces_keys}

    k , i, pos = 0, 0, 0
    while pos < 64:
        if b[i] in all_pieces_keys:
            piece_to_bcoord[b[i]].append((i, pos))
            pos += 1
        else:
            pos += int(b[i])
        i += 1
    for piece in all_pieces_keys:
        n_piece = len(piece_to_bcoord[piece])
        diff = ALL_PIECES[piece] - n_piece
        for j in range(min(n_piece, ALL_PIECES[piece])):
            pos = piece_to_bcoord[piece][j][1]
            x, y = pos_to_coord(pos)
            
            t[k] = 1 # exist flag
            t[k+1] = (x + 1)/8 # row number
            t[k+2] = (y + 1)/8 # col number
            t[k+3] = get_least_val_attacker_from_pos(board, pos)/max(VALUE_MAP.values()) # least value attacker of piece
            t[k+4] = get_least_val_defender_from_pos(board, pos)/max(VALUE_MAP.values()) # least value defender of piece

            if piece in 'QqRrBb':
                color = True if piece in 'QRB' else False
                dirs = get_mobility_from_pos(board, pos, color)
                t[k+5] = dirs[9]['x'][0]/8
                t[k+6] = dirs[9]['x'][1]/8
                t[k+7] = dirs[9]['y'][0]/8
                t[k+8] = dirs[9]['y'][1]/8

                t[k+9] = dirs[7]['x'][0]/8
                t[k+10] = dirs[7]['x'][1]/8
                t[k+11] = dirs[7]['y'][0]/8
                t[k+12] = dirs[7]['y'][1]/8

                t[k+13] = dirs[1]['x'][0]/8
                t[k+14] = dirs[1]['x'][1]/8
                t[k+15] = dirs[1]['y'][0]/8
                t[k+16] = dirs[1]['y'][1]/8

                t[k+17] = dirs[8]['x'][0]/8
                t[k+18] = dirs[8]['x'][1]/8
                t[k+19] = dirs[8]['y'][0]/8
                t[k+20] = dirs[8]['y'][1]/8
                k += 21
            else:
                k += 5

        for j in range(diff):
            if piece in 'QqRrBb':
                k += 21
            else:
                k += 5
    return t[np.newaxis, :]


def get_attack_map(board):
    t = np.zeros(64)
    for pos in range(64):
        attackers = list(get_attackers_by(board, pos, board.turn))
        if len(attackers) > 0:
            least_val_attacker = min(attackers, key=lambda x: VALUE_MAP[x.symbol()])
            t[pos] = VALUE_MAP[least_val_attacker.symbol()]
        else:
            t[pos] = 0
    return t


def get_defend_map(board):
    t = np.zeros(64)
    for pos in range(64):
        defenders = list(get_attackers_by(board, pos, not board.turn))
        if len(defenders) > 0:
            least_val_defender = min(defenders, key=lambda x: VALUE_MAP[x.symbol()])
            t[pos] = VALUE_MAP[least_val_defender.symbol()]
        else:
            t[pos] = 0
    return t


def get_square_centric_features(board):
    t1 = get_attack_map(board)/max(VALUE_MAP.values())
    t2 = get_defend_map(board)/max(VALUE_MAP.values())
    t = np.concatenate((t1, t2))
    return t[np.newaxis, :]


def encode(board):
    xg = np.array2string(get_global_features(board), precision=4, separator=',')
    xp = np.array2string(get_piece_centric_features(board), precision=4, separator=',')
    xs = np.array2string(get_square_centric_features(board), precision=4, separator=',')
    return xg, xp, xs


def decode(x):
    t = torch.Tensor(ast.literal_eval(x))
    return t


if __name__ == '__main__':

    board = chess.Board('r3knQ1/1pR2pQ1/4p1b1/3pP3/p2p3b/q7/3B1PP1/6K1 b q - 3 29')
    print(board)

    xg, xp, xs = encode(board)
    print(xs)

    dec = decode(xs)
    print(dec)

    # print(list(get_attackers_from_pos(board, 13)))
    # print(get_least_val_attacker_from_pos(board, 13))

    # print(list(get_defenders_from_pos(board, 13)))
    # print(get_least_val_defender_from_pos(board, 13))

    # print(list(get_attackers_from_pos(board, 61)))
    # print(get_least_val_attacker_from_pos(board, 61))

    # print(list(get_defenders_from_pos(board, 61)))
    # print(get_least_val_defender_from_pos(board, 61))

    # print(get_attack_map(board)[18])
    # print(get_defend_map(board)[18])

    # print(get_attack_map(board)[62])
    # print(get_defend_map(board)[62])

    # print(get_mobility_from_pos(board, 62, chess.BLACK))
    # print(get_mobility_from_pos(board, 62, chess.WHITE))

    # print(get_mobility_from_pos(board, 31, chess.BLACK))
    # print(get_mobility_from_pos(board, 31, chess.WHITE))