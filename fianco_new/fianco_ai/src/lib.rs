// src/lib.rs
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyKeyError;
use pyo3::FromPyObject;
use numpy::PyReadonlyArray2;
use ndarray::Array2;

const BOARD_SIZE: usize = 9;
const EMPTY: i32 = 0;
const BLACK: i32 = 1;
const WHITE: i32 = -1;

const WIN_SCORE: f64 = 1_000_000.0;
const LOSE_SCORE: f64 = -1_000_000.0;


#[derive(Debug, FromPyObject)]
struct Weights {
    piece_value: f64,
    advancement_value: f64,
    unstoppable_pawn_bonus: f64,
    opponent_unstoppable_pawn_penalty: f64,
    center_control_value: f64,
    mobility_value: f64,
    edge_pawn_bonus: f64,
    // Add more weights as needed
}

#[pyfunction]
fn negamax(
    _py: Python,
    board: PyReadonlyArray2<i32>,
    depth: i32,
    player: i32,
    weights: &PyAny,
) -> PyResult<(Option<(i32, i32, i32, i32)>, f64, Vec<(i32, i32, i32, i32)>)> {
    let board_array = board.as_array().to_owned();

    let weights: Weights = weights.extract()?;

    let (evaluation, best_move, pv) =
        negamax_search(board_array, depth, player, f64::NEG_INFINITY, f64::INFINITY, &weights);

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
    mut alpha: f64,
    beta: f64,
    weights: &Weights,
) -> (
    f64,
    Option<(usize, usize, usize, usize)>,
    Vec<(usize, usize, usize, usize)>,
) {
    if depth == 0 || get_winner(&board).is_some() {
        let evaluation = evaluate_board(&board, player, weights);
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
            weights, // Pass weights recursively
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

fn evaluate_board(board: &Array2<i32>, player: i32, weights: &Weights) -> f64 {
    // Check for game over
    if let Some(winner) = get_winner(board) {
        if winner == player {
            return WIN_SCORE;
        } else {
            return LOSE_SCORE;
        }
    }

    let mut score = 0.0;

    // Iterate over the board and calculate features
    for ((row, col), &piece) in board.indexed_iter() {
        if piece == player {
            // Material value
            score += weights.piece_value;

            // Advancement
            let advancement = if player == BLACK {
                row as f64
            } else {
                (BOARD_SIZE - 1 - row) as f64
            };
            score += weights.advancement_value * advancement;

            // Center control
            if is_center_square(row, col) {
                score += weights.center_control_value;
            }

            // Edge pawn bonus
            if is_edge_square(row, col) {
                score += weights.edge_pawn_bonus;
            }
        } else if piece == -player {
            // Opponent's material value
            score -= weights.piece_value;

            // Opponent's advancement
            let advancement = if player == BLACK {
                (BOARD_SIZE - 1 - row) as f64
            } else {
                row as f64
            };
            score -= weights.advancement_value * advancement;

            // Opponent's center control
            if is_center_square(row, col) {
                score -= weights.center_control_value;
            }

            // Opponent's edge pawn bonus
            if is_edge_square(row, col) {
                score -= weights.edge_pawn_bonus;
            }
        }
    }

    // Mobility
    let mobility = get_mobility(board, player) as f64;
    score += weights.mobility_value * mobility;

    let opponent_mobility = get_mobility(board, -player) as f64;
    score -= weights.mobility_value * opponent_mobility;

    // Unstoppable pawns
    let ai_unstoppable_pawns = count_unstoppable_pawns(board, player) as f64;
    let opponent_unstoppable_pawns = count_unstoppable_pawns(board, -player) as f64;

    score += ai_unstoppable_pawns * weights.unstoppable_pawn_bonus;
    score += opponent_unstoppable_pawns * weights.opponent_unstoppable_pawn_penalty;

    score
}


fn is_center_square(row: usize, col: usize) -> bool {
    // Define center squares (e.g., the middle 3x3 squares for a 9x9 board)
    let center_start = BOARD_SIZE / 3;
    let center_end = BOARD_SIZE - center_start;

    row >= center_start && row < center_end && col >= center_start && col < center_end
}

fn is_edge_square(row: usize, col: usize) -> bool {
    col == 0 || col == BOARD_SIZE - 1
}


fn get_mobility(board: &Array2<i32>, player: i32) -> i32 {
    let moves = get_valid_moves(board, player);
    moves.len() as i32
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
            if is_unstoppable_pawn(board, (row, col), player) {
                count += 1;
            }
        }
    }

    count
}

fn is_unstoppable_pawn(board: &Array2<i32>, pawn_pos: (usize, usize), player: i32) -> bool {
    let (row_pawn, col_pawn) = pawn_pos;
    let row_pawn = row_pawn as isize;
    let col_pawn = col_pawn as isize;
    let row_goal = if player == BLACK { BOARD_SIZE as isize - 1 } else { 0 };
    let direction = if player == BLACK { 1 } else { -1 };

    let steps_to_goal = (row_goal - row_pawn).abs();

    // For each opponent pawn
    for ((row_opp, col_opp), &piece) in board.indexed_iter() {
        if piece == -player {
            let row_opp = row_opp as isize;
            let col_opp = col_opp as isize;

            // Check if opponent pawn is ahead of the pawn
            let relative_row = (row_opp - row_pawn) * direction;
            if relative_row <= 0 {
                // Opponent pawn is not ahead
                continue;
            }

            let steps_to_opp = relative_row;
            let col_diff = (col_opp - col_pawn).abs();

            if col_diff <= steps_to_opp {
                // Opponent pawn is within triangle
                return false; // Pawn is stoppable
            }
        }
    }

    // No opponent pawns within the triangle
    return true; // Pawn is unstoppable
}


fn is_within_bounds(row: isize, col: isize) -> bool {
    row >= 0 && row < BOARD_SIZE as isize && col >= 0 && col < BOARD_SIZE as isize
}

#[pymodule]
fn fianco_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(negamax, m)?)?;
    Ok(())
}
