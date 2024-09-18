# fianco.py

import numpy as np

BOARD_SIZE = 9
EMPTY = 0
BLACK = 1
WHITE = -1

START_POSITION = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0] * BOARD_SIZE,
    [0, 0, 0, -1, 0, -1, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, -1, 0, 0],
    [0, -1, 0, 0, 0, 0, 0, -1, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1]
]

DIRECTIONS = {
    BLACK: [(1, 0), (0, -1), (0, 1)],  # Forward, left, right
    WHITE: [(-1, 0), (0, -1), (0, 1)]
}

CAPTURE_DIRECTIONS = {
    BLACK: [(1, -1), (1, 1)],
    WHITE: [(-1, -1), (-1, 1)]
}

class GameState:
    def __init__(self):
        self.board = np.array(START_POSITION, dtype=int)
        self.current_player = WHITE
        self.move_history = []
        self.winner = None

    def is_valid_move(self, from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        if not self.is_within_bounds(to_row, to_col):
            return False
        if self.board[from_row, from_col] != self.current_player:
            return False
        if self.board[to_row, to_col] != EMPTY:
            return False

        row_diff = to_row - from_row
        col_diff = to_col - from_col

        # Normal move
        if (row_diff, col_diff) in DIRECTIONS[self.current_player]:
            return True

        # Capture move
        if (row_diff, col_diff) in [(d[0]*2, d[1]*2) for d in CAPTURE_DIRECTIONS[self.current_player]]:
            mid_row = (from_row + to_row) // 2
            mid_col = (from_col + to_col) // 2
            if self.board[mid_row, mid_col] == -self.current_player:
                return True

        return False

    def get_valid_moves(self):
        moves = []
        capture_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row, col] == self.current_player:
                    piece_moves, piece_capture_moves = self.get_piece_moves((row, col))
                    moves.extend(piece_moves)
                    capture_moves.extend(piece_capture_moves)
        if capture_moves:
            return capture_moves  # Only return capture moves if available
        else:
            return moves

    def get_piece_moves(self, pos):
        moves = []
        capture_moves = []
        row, col = pos

        # Capture moves
        for dr, dc in CAPTURE_DIRECTIONS[self.current_player]:
            mid_row, mid_col = row + dr, col + dc
            new_row, new_col = row + 2 * dr, col + 2 * dc
            if self.is_within_bounds(new_row, new_col) and self.board[new_row, new_col] == EMPTY:
                if self.board[mid_row, mid_col] == -self.current_player:
                    capture_moves.append((pos, (new_row, new_col)))

        if capture_moves:
            return [], capture_moves

        # Normal moves
        for dr, dc in DIRECTIONS[self.current_player]:
            new_row, new_col = row + dr, col + dc
            if self.is_within_bounds(new_row, new_col) and self.board[new_row, new_col] == EMPTY:
                moves.append((pos, (new_row, new_col)))

        return moves, []


    def make_move(self, move):
        from_pos, to_pos = move
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        # Check if it's a capture
        if abs(to_row - from_row) == 2:
            mid_row = (from_row + to_row) // 2
            mid_col = (from_col + to_col) // 2
            self.board[mid_row, mid_col] = EMPTY
            capture = True
        else:
            capture = False

        # Move the piece
        self.board[to_row, to_col] = self.board[from_row, from_col]
        self.board[from_row, from_col] = EMPTY

        # Record the move
        move_notation = self.get_move_notation(move, capture)
        self.move_history.append(move_notation)

        # Check for victory
        if (self.current_player == BLACK and to_row == BOARD_SIZE - 1) or \
           (self.current_player == WHITE and to_row == 0):
            self.winner = self.current_player

        # Switch player
        self.current_player *= -1

    def is_game_over(self):
        return self.winner is not None or not self.get_valid_moves()

    def evaluate(self):
        # Simple evaluation function
        return np.sum(self.board)

    def is_within_bounds(self, row, col):
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def get_move_notation(self, move, capture):
        from_pos, to_pos = move
        from_notation = self.pos_to_notation(from_pos)
        to_notation = self.pos_to_notation(to_pos)
        separator = 'x' if capture else '-'
        return f'{from_notation}{separator}{to_notation}'

    def pos_to_notation(self, pos):
        row, col = pos
        col_letter = chr(ord('A') + col)
        row_number = BOARD_SIZE - row
        return f'{col_letter}{row_number}'
    
    def copy(self):
        new_state = GameState()
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.move_history = self.move_history.copy()
        new_state.winner = self.winner
        return new_state

    def reset(self):
        self.__init__()
