# main.py

import pygame
from fianco import GameState, BLACK, WHITE, EMPTY
from ai import get_best_move
import sys
import threading


# Initialize Pygame
pygame.init()
SCREEN_SIZE = 600
INFO_HEIGHT = 150
CELL_SIZE = SCREEN_SIZE // 9
FONT_SIZE = 24
screen = pygame.display.set_mode((SCREEN_SIZE+30, SCREEN_SIZE + INFO_HEIGHT))
pygame.display.set_caption('Fianco')
font = pygame.font.Font(None, FONT_SIZE)

HIGHLIGHT_COLOR = (0, 255, 0)  # Green


def draw_board(screen, game_state, possible_moves=None):
    # Draw the board grid
    for row in range(10):
        pygame.draw.line(screen, (0, 0, 0), (0, row * CELL_SIZE), (SCREEN_SIZE, row * CELL_SIZE))
        pygame.draw.line(screen, (0, 0, 0), (row * CELL_SIZE, 0), (row * CELL_SIZE, SCREEN_SIZE))

    # Draw coordinates
    for i in range(9):
        # Columns
        col_label = font.render(chr(ord('A') + i), True, (0, 0, 0))
        screen.blit(col_label, (i * CELL_SIZE + CELL_SIZE // 2 - col_label.get_width() // 2, SCREEN_SIZE))
        # Rows
        row_label = font.render(str(9 - i), True, (0, 0, 0))
        screen.blit(row_label, (SCREEN_SIZE, i * CELL_SIZE + CELL_SIZE // 2 - row_label.get_height() // 2))

    # Highlight possible moves
    if possible_moves:
        for move in possible_moves:
            _, to_pos = move
            row, col = to_pos
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect)

    # Draw pieces
    for row in range(9):
        for col in range(9):
            piece = game_state.board[row, col]
            if piece != 0:
                center = (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2)
                if piece == 1:
                    color = (0, 0, 0)
                else:
                    color = (255, 255, 255)
                pygame.draw.circle(screen, color, center, CELL_SIZE // 2 - 5)
                pygame.draw.circle(screen, (0, 0, 0), center, CELL_SIZE // 2 - 5, 1)


def draw_move_history(screen, move_history):
    y_offset = SCREEN_SIZE + 20
    moves_text = ' '.join(move_history[-5:])  # Show last 5 moves
    text = font.render(f'Moves: {moves_text}', True, (0, 0, 0))
    screen.blit(text, (10, y_offset))

def draw_evaluation(screen, evaluation):
    y_offset = SCREEN_SIZE + 50
    text = font.render(f'AI Evaluation: {evaluation}', True, (0, 0, 0))
    screen.blit(text, (10, y_offset))

def draw_restart_button(screen):
    y_offset = SCREEN_SIZE + 80
    rect = pygame.Rect(10, y_offset, 100, 30)
    pygame.draw.rect(screen, (200, 200, 200), rect)
    text = font.render('Restart', True, (0, 0, 0))
    screen.blit(text, (rect.x + 10, rect.y + 5))
    return rect

def draw_principal_variation(screen, pv_moves, game_state):
    y_offset = SCREEN_SIZE + 120
    pv_text = 'Main Variation: ' + ' '.join(
        [f'{game_state.pos_to_notation(from_pos)}-{game_state.pos_to_notation(to_pos)}' for from_pos, to_pos in pv_moves]
    )
    text = font.render(pv_text, True, (0, 0, 0))
    screen.blit(text, (10, y_offset))


# def main():
#     game_state = GameState()
#     selected_piece = None
#     possible_moves = []
#     ai_enabled = True  # Set to False if you want to play both sides
#     running = True
#     evaluation = 0


#     ai_pv = []
#     ai_thread = None
#     evaluation = 0

#     ai_move_event = threading.Event()
#     ai_move = None

#     # ai_pv = []


#     # ai_thread = None
#     # pv_lock = threading.Lock()

#     def pv_callback(pv_moves):
#         nonlocal ai_pv
#         with pv_lock:
#             ai_pv = pv_moves


#     while running:
#         screen.fill((255, 255, 255))
#         draw_board(screen, game_state, possible_moves)
#         draw_move_history(screen, game_state.move_history)
#         draw_evaluation(screen, evaluation)
#         restart_button_rect = draw_restart_button(screen)
#         draw_principal_variation(screen, ai_pv, game_state)
#         pygame.display.flip()

#         if game_state.is_game_over():
#             winner = 'Black' if game_state.winner == 1 else 'White' if game_state.winner == -1 else 'No one'
#             print(f"Game Over! Winner: {winner}")
#             running = False
#             continue

#         # Get valid moves and check for captures
#         valid_moves = game_state.get_valid_moves()
#         capture_available = any(abs(move[0][0] - move[1][0]) == 2 for move in valid_moves)


#         if ai_enabled and game_state.current_player == BLACK and ai_thread is None:
#             # Start AI thinking in a separate thread
#             def ai_think():
#                 nonlocal evaluation, ai_move
#                 print("AI is thinking...")
#                 # Use a copy of the game state to prevent conflicts
#                 game_state_copy = game_state.copy()
#                 move, evaluation = get_best_move(game_state_copy, depth=3, pv_callback=pv_callback)
#                 print(f"AI move: {move}, evaluation: {evaluation}")
#                 ai_move = move
#                 ai_move_event.set()
#                 # Reset the thread variable
#                 nonlocal ai_thread
#                 ai_thread = None
#         # if ai_enabled and game_state.current_player == BLACK:
#         #     move, evaluation = get_best_move(game_state, depth=3)
#         #     if move:
#         #         game_state.make_move(move)
#         #     else:
#         #         # No valid moves, game over
#         #         game_state.winner = -game_state.current_player
#         #     continue


#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#                 sys.exit()

#             elif event.type == pygame.MOUSEBUTTONDOWN:
#                 mouse_pos = pygame.mouse.get_pos()
#                 if restart_button_rect.collidepoint(mouse_pos):
#                     game_state.reset()
#                     selected_piece = None
#                     possible_moves = []
#                     # evaluation = 0
#                     # Update valid moves and capture availability after reset
#                     valid_moves = game_state.get_valid_moves()
#                     capture_available = any(abs(move[0][0] - move[1][0]) == 2 for move in valid_moves)
#                     continue

#                 col = mouse_pos[0] // CELL_SIZE
#                 row = mouse_pos[1] // CELL_SIZE
#                 if game_state.is_within_bounds(row, col):
#                     if selected_piece:
#                         move = (selected_piece, (row, col))
#                         if move in possible_moves:
#                             game_state.make_move(move)
#                             selected_piece = None
#                             possible_moves = []
#                             # evaluation = 0
#                             # Update valid moves and capture availability after move
#                             valid_moves = game_state.get_valid_moves()
#                             capture_available = any(abs(move[0][0] - move[1][0]) == 2 for move in valid_moves)
#                         else:
#                             selected_piece = None
#                             possible_moves = []
#                     elif game_state.board[row, col] == game_state.current_player:
#                         piece_moves, piece_capture_moves = game_state.get_piece_moves((row, col))
#                         if capture_available:
#                             if piece_capture_moves:
#                                 selected_piece = (row, col)
#                                 possible_moves = piece_capture_moves
#                             else:
#                                 # Cannot select this piece as it cannot capture
#                                 selected_piece = None
#                                 possible_moves = []
#                                 # Optionally, display a message to the player
#                         else:
#                             selected_piece = (row, col)
#                             possible_moves = piece_moves
#                     else:
#                         selected_piece = None
#                         possible_moves = []
#             elif event.type == pygame.MOUSEBUTTONUP:
#                 pass
#         pygame.time.delay(10)
#     pygame.quit()

def main():
    game_state = GameState()
    selected_piece = None
    possible_moves = []
    ai_enabled = True  # Set to False if you want to play both sides
    running = True
    evaluation = 0

    ai_pv = []
    ai_thread = None
    ai_move_event = threading.Event()
    ai_move = None

    def pv_callback(pv_moves):
        with threading.Lock():
            ai_pv.clear()
            ai_pv.extend(pv_moves)

    def ai_think():
        nonlocal evaluation, ai_move
        print("AI is thinking...")
        game_state_copy = game_state.copy()
        move, evaluation = get_best_move(game_state_copy, 6, pv_callback=pv_callback)
        print(f"AI move: {move}, evaluation: {evaluation}")
        ai_move = move
        ai_move_event.set()
        nonlocal ai_thread
        ai_thread = None

    while running:
        screen.fill((255, 255, 255))
        draw_board(screen, game_state, possible_moves)
        draw_move_history(screen, game_state.move_history)
        draw_evaluation(screen, evaluation)
        restart_button_rect = draw_restart_button(screen)
        draw_principal_variation(screen, ai_pv, game_state)
        pygame.display.flip()

        if game_state.is_game_over():
            winner = 'Black' if game_state.winner == 1 else 'White' if game_state.winner == -1 else 'No one'
            print(f"Game Over! Winner: {winner}")
            running = False
            continue

        # Get valid moves and check for captures
        valid_moves = game_state.get_valid_moves()
        capture_available = any(abs(move[0][0] - move[1][0]) == 2 for move in valid_moves)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if restart_button_rect.collidepoint(mouse_pos):
                    game_state.reset()
                    selected_piece = None
                    possible_moves = []
                    evaluation = 0
                    ai_pv.clear()
                    continue

                col = mouse_pos[0] // CELL_SIZE
                row = mouse_pos[1] // CELL_SIZE
                if game_state.is_within_bounds(row, col):
                    if selected_piece:
                        move = (selected_piece, (row, col))
                        if move in possible_moves:
                            game_state.make_move(move)
                            selected_piece = None
                            possible_moves = []
                        else:
                            selected_piece = None
                            possible_moves = []
                    elif game_state.board[row, col] == game_state.current_player:
                        piece_moves, piece_capture_moves = game_state.get_piece_moves((row, col))
                        if capture_available:
                            if piece_capture_moves:
                                selected_piece = (row, col)
                                possible_moves = piece_capture_moves
                            else:
                                selected_piece = None
                                possible_moves = []
                        else:
                            selected_piece = (row, col)
                            possible_moves = piece_moves
                    else:
                        selected_piece = None
                        possible_moves = []
            elif event.type == pygame.MOUSEBUTTONUP:
                pass
        
        # Apply AI move if available
        if ai_move_event.is_set():
            ai_move_event.clear()
            if ai_move:
                game_state.make_move(ai_move)

            else:
                game_state.winner = -game_state.current_player
            ai_move = None

        if ai_enabled and game_state.current_player == BLACK and ai_thread is None:
            print("Starting AI thread")
            ai_thread = threading.Thread(target=ai_think)
            ai_thread.start()

        

        pygame.time.delay(10)

    pygame.quit()


if __name__ == "__main__":
    main()
