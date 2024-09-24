from fianco_ai import negamax
from fianco import BLACK
import numpy as np
import threading
import time

def get_best_move(game_state, max_depth, pv_callback, weights):
    board = game_state.board.astype(np.int32)
    player = game_state.current_player
    best_move = None
    evaluation = 0

    # Run negamax in a separate thread
    for depth in range(1, max_depth + 1):
        best_move, evaluation, pv = negamax(board, depth, player, weights)

        # Adjust evaluation for consistency
        if player == BLACK:
            evaluation = -evaluation
        
        pv_moves = [((mv[0], mv[1]), (mv[2], mv[3])) for mv in pv]

        pv_callback(pv_moves)
        time.sleep(0.1)

    # If no move found, pick any valid move
    if best_move is None:
        valid_moves = game_state.get_valid_moves()
        if valid_moves:
            best_move = valid_moves[0][0] + valid_moves[0][1]
            # Convert to required format
            best_move = (best_move[0], best_move[1], best_move[2], best_move[3])
        else:
            # No valid moves, return None
            return None, evaluation
    from_pos = (best_move[0], best_move[1])
    to_pos = (best_move[2], best_move[3])
    return (from_pos, to_pos), evaluation
