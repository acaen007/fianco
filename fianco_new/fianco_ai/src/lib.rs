// src/lib.rs

use pyo3::prelude::*;
use numpy::PyReadonlyArray2;
use ndarray::Array2;

const BOARD_SIZE: usize = 9;
const EMPTY: i32 = 0;
const BLACK: i32 = 1;
const WHITE: i32 = -1;

const WIN_SCORE: i32 = 1_000_000;
const LOSE_SCORE: i32 = -1_000_000;

const UNSTOPPABLE_PAWN_BONUS: i32 = 5000;
const UNSTOPPABLE_PAWN_PENALTY: i32 = -5000;



#[pyfunction]
fn negamax(
    _py: Python,
    board: PyReadonlyArray2<i32>,
    depth: i32,
    player: i32,
) -> PyResult<(Option<(i32, i32, i32, i32)>, i32, Vec<(i32, i32, i32, i32)>)> {
    let board_array = board.as_array().to_owned();

    let (evaluation, best_move, pv) =
        negamax_search(board_array, depth, player, i32::MIN + 1, i32::MAX - 1);

    let py_move = best_move.map(|(fr, fc, tr, tc)| {
        (fr as i32, fc as i32, tr as i32, tc as i32)
    });

    let py_pv = pv
        .into_iter()
        .map(|(fr, fc, tr, tc)| (fr as i32, fc as i32, tr as i32, tc as i32))
        .collect();

    Ok((py_move, evaluation, py_pv))
}


fn negamax_search(
    board: Array2<i32>,
    depth: i32,
    player: i32,
    mut alpha: i32,
    beta: i32,
) -> (i32, Option<(usize, usize, usize, usize)>, Vec<(usize, usize, usize, usize)>) {
    if depth == 0 || get_winner(&board).is_some() {
        let evaluation = evaluate_board(&board, player);
        return (evaluation, None, Vec::new());
    }

    let moves = get_valid_moves(&board, player);

    if moves.is_empty() {
        // No moves available, losing position
        return (LOSE_SCORE, None, Vec::new());
    }

    let mut max_eval = LOSE_SCORE;
    let mut best_move = None;
    let mut pv_line = Vec::new();

    for mv in moves {
        let mut new_board = board.clone();
        make_move(&mut new_board, &mv, player);

        let (eval, _, child_pv) = negamax_search(
            new_board,
            depth - 1,
            -player,
            -beta,
            -alpha,
        );
        let eval = -eval;

        if eval > max_eval {
            max_eval = eval;
            best_move = Some(mv);
            // Construct PV line
            pv_line = vec![mv];
            pv_line.extend(child_pv);
        }

        alpha = alpha.max(eval);
        if alpha >= beta {
            break;
        }
    }

    (max_eval, best_move, pv_line)
}


fn is_game_over(board: &Array2<i32>, player: i32) -> bool {
    // Check for victory condition: if a player's piece reaches the opponent's back row
    if player == BLACK {
        for col in 0..BOARD_SIZE {
            if board[[BOARD_SIZE - 1, col]] == BLACK {
                return true;
            }
        }
    } else {
        for col in 0..BOARD_SIZE {
            if board[[0, col]] == WHITE {
                return true;
            }
        }
    }
    false
}

fn evaluate_board(board: &Array2<i32>, player: i32) -> i32 {
    // Check if the current player has won
    if let Some(winner) = get_winner(board) {
        if winner == player {
            return WIN_SCORE;
        } else {
            return LOSE_SCORE;
        }
    }

    let mut score = 0;

    // Existing evaluation logic
    for ((row, _col), &piece) in board.indexed_iter() {
        if piece == player {
            // Reward for each piece
            score += 10;
            // Reward for advancement
            let advancement = if player == BLACK {
                row as i32
            } else {
                (BOARD_SIZE - 1 - row) as i32
            };
            score += advancement;
        } else if piece == -player {
            // Penalty for opponent's pieces
            score -= 10;
            // Penalty for opponent's advancement
            let advancement = if player == BLACK {
                (BOARD_SIZE - 1 - row) as i32
            } else {
                row as i32
            };
            score -= advancement;
        }
    }

    // Detect and evaluate unstoppable pawns
    // let ai_unstoppable_pawns = count_unstoppable_pawns(board, player);
    // let opponent_unstoppable_pawns = count_unstoppable_pawns(board, -player);

    // score += ai_unstoppable_pawns * UNSTOPPABLE_PAWN_BONUS;
    // score += opponent_unstoppable_pawns * UNSTOPPABLE_PAWN_PENALTY; // Note the sign

    score
}


fn get_winner(board: &Array2<i32>) -> Option<i32> {
    // Check if BLACK has won
    for col in 0..BOARD_SIZE {
        if board[[BOARD_SIZE - 1, col]] == BLACK {
            return Some(BLACK);
        }
    }

    // Check if WHITE has won
    for col in 0..BOARD_SIZE {
        if board[[0, col]] == WHITE {
            return Some(WHITE);
        }
    }

    // Check if either player has no pieces left
    let mut black_pieces = 0;
    let mut white_pieces = 0;
    for &piece in board.iter() {
        if piece == BLACK {
            black_pieces += 1;
        } else if piece == WHITE {
            white_pieces += 1;
        }
    }

    if black_pieces == 0 {
        return Some(WHITE);
    }
    if white_pieces == 0 {
        return Some(BLACK);
    }

    None
}



fn get_valid_moves(board: &Array2<i32>, player: i32) -> Vec<(usize, usize, usize, usize)> {
    let mut moves = Vec::new();
    let mut capture_moves = Vec::new();

    for row in 0..BOARD_SIZE {
        for col in 0..BOARD_SIZE {
            if board[[row, col]] == player {
                let (piece_moves, piece_capture_moves) = get_piece_moves(board, (row, col), player);
                moves.extend(piece_moves);
                capture_moves.extend(piece_capture_moves);
            }
        }
    }

    if !capture_moves.is_empty() {
        capture_moves
    } else {
        moves
    }
}

fn get_piece_moves(
    board: &Array2<i32>,
    pos: (usize, usize),
    player: i32,
) -> (
    Vec<(usize, usize, usize, usize)>,
    Vec<(usize, usize, usize, usize)>,
) {
    let mut moves = Vec::new();
    let mut capture_moves = Vec::new();
    let (row, col) = pos;

    let directions = match player {
        BLACK => vec![(1, 0), (0, -1), (0, 1)],
        WHITE => vec![(-1, 0), (0, -1), (0, 1)],
        _ => vec![],
    };
    let capture_directions = match player {
        BLACK => vec![(1, -1), (1, 1)],
        WHITE => vec![(-1, -1), (-1, 1)],
        _ => vec![],
    };

    // Capture moves
    for (dr, dc) in capture_directions {
        let mid_row = row as isize + dr;
        let mid_col = col as isize + dc;
        let new_row = row as isize + 2 * dr;
        let new_col = col as isize + 2 * dc;

        if is_within_bounds(mid_row, mid_col)
            && is_within_bounds(new_row, new_col)
            && board[[mid_row as usize, mid_col as usize]] == -player
            && board[[new_row as usize, new_col as usize]] == EMPTY
        {
            capture_moves.push((
                row,
                col,
                new_row as usize,
                new_col as usize,
            ));
        }
    }

    if !capture_moves.is_empty() {
        return (Vec::new(), capture_moves);
    }

    // Normal moves
    for (dr, dc) in directions {
        let new_row = row as isize + dr;
        let new_col = col as isize + dc;

        if is_within_bounds(new_row, new_col)
            && board[[new_row as usize, new_col as usize]] == EMPTY
        {
            moves.push((row, col, new_row as usize, new_col as usize));
        }
    }

    (moves, Vec::new())
}


fn make_move(board: &mut Array2<i32>, mv: &(usize, usize, usize, usize), _player: i32) {
    let (from_row, from_col, to_row, to_col) = *mv;

    // Check if it's a capture
    if (from_row as isize - to_row as isize).abs() == 2 {
        let mid_row = (from_row + to_row) / 2;
        let mid_col = (from_col + to_col) / 2;
        board[[mid_row, mid_col]] = EMPTY;
    }

    board[[to_row, to_col]] = board[[from_row, from_col]];
    board[[from_row, from_col]] = EMPTY;
}

fn count_unstoppable_pawns(board: &Array2<i32>, player: i32) -> i32 {
    let mut count = 0;

    for ((row, col), &piece) in board.indexed_iter() {
        if piece == player {
            if is_unstoppable_pawn(board, (row as isize, col as isize), player) {
                count += 1;
            }
        }
    }

    count
}

fn is_unstoppable_pawn(board: &Array2<i32>, pawn_pos: (isize, isize), player: i32) -> bool {
    let min_steps_to_goal = calculate_min_steps_to_goal(pawn_pos, player);
    let min_steps_for_opponent_to_block = calculate_min_steps_for_opponent_to_block(board, pawn_pos, player);

    min_steps_to_goal < min_steps_for_opponent_to_block
}

fn calculate_min_steps_to_goal(pawn_pos: (isize, isize), player: i32) -> isize {
    let (row, _) = pawn_pos;

    if player == BLACK {
        (BOARD_SIZE as isize - 1) - row
    } else {
        row
    }
}

fn calculate_min_steps_for_opponent_to_block(board: &Array2<i32>, pawn_pos: (isize, isize), player: i32) -> isize {
    let opponent = -player;
    let mut min_steps = std::isize::MAX;

    for ((opp_row, opp_col), &piece) in board.indexed_iter() {
        if piece == opponent {
            if let Some(steps) = calculate_steps_to_intercept(board, pawn_pos, (opp_row as isize, opp_col as isize), player) {
                if steps < min_steps {
                    min_steps = steps;
                }
            }
        }
    }

    min_steps
}

fn calculate_steps_to_intercept(
    board: &Array2<i32>,
    pawn_pos: (isize, isize),
    opp_pawn_pos: (isize, isize),
    player: i32,
) -> Option<isize> {
    let (pawn_row, pawn_col) = pawn_pos;
    let (opp_row, opp_col) = opp_pawn_pos;

    let pawn_direction = if player == BLACK { 1 } else { -1 };
    let opponent_direction = -pawn_direction;

    let pawn_steps_to_goal = calculate_min_steps_to_goal(pawn_pos, player);

    for step in 1..=pawn_steps_to_goal {
        // Pawn's future position at this step
        let pawn_future_row = pawn_row + step * pawn_direction;
        if pawn_future_row < 0 || pawn_future_row >= BOARD_SIZE as isize {
            break;
        }

        // Potential interception positions
        let interception_positions = vec![
            (pawn_future_row, pawn_col),             // Direct block
            (pawn_future_row, pawn_col - 1),         // Capture from left
            (pawn_future_row, pawn_col + 1),         // Capture from right
        ];

        for opp_step in 1..=step {
            let opp_future_row = opp_row + opp_step * opponent_direction;
            let opp_future_cols = vec![opp_col - opp_step, opp_col, opp_col + opp_step];

            for &opp_future_col in &opp_future_cols {
                let opp_future_pos = (opp_future_row, opp_future_col);
                if interception_positions.contains(&opp_future_pos) {
                    // Check if path is clear
                    if is_path_clear(board, opp_pawn_pos, opp_future_pos, -player) {
                        return Some(opp_step);
                    }
                }
            }
        }
    }

    None
}

fn is_path_clear(
    board: &Array2<i32>,
    start_pos: (isize, isize),
    end_pos: (isize, isize),
    player: i32,
) -> bool {
    let (start_row, start_col) = start_pos;
    let (end_row, end_col) = end_pos;

    let row_step = (end_row - start_row).signum();
    let col_step = (end_col - start_col).signum();

    let mut row = start_row + row_step;
    let mut col = start_col + col_step;

    while (row, col) != end_pos {
        if !is_within_bounds(row, col) {
            return false;
        }
        if board[[row as usize, col as usize]] != EMPTY {
            return false;
        }
        row += row_step;
        col += col_step;
    }

    true
}


fn is_within_bounds(row: isize, col: isize) -> bool {
    row >= 0 && row < BOARD_SIZE as isize && col >= 0 && col < BOARD_SIZE as isize
}

#[pymodule]
fn fianco_ai(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(negamax, m)?)?;
    Ok(())
}
