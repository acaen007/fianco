// src/lib.rs

use pyo3::prelude::*;
use pyo3::FromPyObject;
use numpy::PyReadonlyArray2;
use ndarray::Array2;
use std::collections::HashMap;
use rand::Rng;
use std::collections::HashSet;
use std::time::{Duration, Instant};


const BOARD_SIZE: usize = 9;
const EMPTY: i32 = 0;
const BLACK: i32 = 1;
const WHITE: i32 = -1;

const WIN_SCORE: f64 = 1_000_000.0;
const LOSE_SCORE: f64 = -1_000_000.0;

// Transposition Table Entry
struct TranspositionTableEntry {
    depth: i32,
    value: f64,
    flag: NodeType,
    best_move: Option<(usize, usize, usize, usize)>,
}

enum NodeType {
    Exact,
    LowerBound,
    UpperBound,
}

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
    max_depth: i32,
    player: i32,
    weights: &PyAny,
    time_limit: f64, // Time limit in seconds
) -> PyResult<(Option<(i32, i32, i32, i32)>, f64, Vec<(i32, i32, i32, i32)>)> {
    let board_array = board.as_array().to_owned();

    let weights: Weights = weights.extract()?;

    // Initialize Zobrist table
    let zobrist_table = initialize_zobrist_table();

    // Compute initial hash
    let initial_hash = compute_zobrist_hash(&board_array, &zobrist_table);

    // Initialize transposition table
    let mut transposition_table = HashMap::new();

    // Initialize position counts for threefold repetition detection
    let mut position_counts = HashMap::new();

    // Start timing
    let start_time = Instant::now();
    let time_limit = Duration::from_secs_f64(time_limit);

    let mut best_move = None;
    let mut evaluation = 0.0;
    let mut pv = Vec::new();

    // Get all valid moves in the current position
    let (capture_moves, normal_moves) = get_all_valid_moves(&board_array, player);

    if !capture_moves.is_empty() {
        // There are capture moves
        if capture_moves.len() == 1 {
            // Only one capture move, play it immediately
            let mv = capture_moves[0];
            let evaluation = evaluate_board(&board_array, player, &weights);
            let py_move = Some((mv.0 as i32, mv.1 as i32, mv.2 as i32, mv.3 as i32));
            let py_pv = vec![py_move.unwrap()];
            return Ok((py_move, evaluation, py_pv));
        }
    } else {
        // No capture moves
        if normal_moves.len() == 1 {
            // Only one normal move, play it immediately
            let mv = normal_moves[0];
            let evaluation = evaluate_board(&board_array, player, &weights);
            let py_move = Some((mv.0 as i32, mv.1 as i32, mv.2 as i32, mv.3 as i32));
            let py_pv = vec![py_move.unwrap()];
            return Ok((py_move, evaluation, py_pv));
        }
    }


    // Iterative Deepening Loop
    for depth in 1..=max_depth {
        // Check if time limit exceeded
        if start_time.elapsed() >= time_limit {
            break;
        }

        // Reset position counts for each iteration
        position_counts.clear();
        position_counts.insert(initial_hash, 1);

        let (eval, mv, principal_variation) = negamax_search(
            &board_array,
            depth,
            player,
            f64::NEG_INFINITY,
            f64::INFINITY,
            &weights,
            initial_hash,
            &zobrist_table,
            &mut transposition_table,
            &mut position_counts,
            &start_time,
            time_limit,
            best_move, // Pass the best move from previous iteration
        );

        // Check if time limit exceeded during search
        if start_time.elapsed() >= time_limit {
            break;
        }

        if mv.is_some() {
            evaluation = eval;
            best_move = mv;
            pv = principal_variation;
        } else {
            // If no move was found (possibly due to timeout), break
            break;
        }
    }

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
    board: &Array2<i32>,
    depth: i32,
    player: i32,
    mut alpha: f64,
    mut beta: f64,
    weights: &Weights,
    zobrist_hash: u64,
    zobrist_table: &[[[u64; 3]; BOARD_SIZE]; BOARD_SIZE],
    transposition_table: &mut HashMap<u64, TranspositionTableEntry>,
    position_counts: &mut HashMap<u64, i32>,
    start_time: &Instant,
    time_limit: Duration,
    first_move: Option<(usize, usize, usize, usize)>, // Best move from previous iteration
) -> (
    f64,
    Option<(usize, usize, usize, usize)>,
    Vec<(usize, usize, usize, usize)>,
) {
    // Check if time limit exceeded
    if start_time.elapsed() >= time_limit {
        return (0.0, None, Vec::new()); // Return default value on timeout
    }

    // Threefold repetition detection
    {
        let count = position_counts.entry(zobrist_hash).or_insert(0);
        *count += 1;
        if *count >= 3 {
            *count -= 1; // Decrement before returning
            if *count == 0 {
                position_counts.remove(&zobrist_hash);
            }
            return (0.0, None, Vec::new());
        }
    } // Mutable borrow ends here

    // Transposition Table Lookup
    if let Some(entry) = transposition_table.get(&zobrist_hash) {
        if entry.depth >= depth {
            match entry.flag {
                NodeType::Exact => {
                    // Decrement the position count before returning
                    {
                        let count = position_counts.get_mut(&zobrist_hash).unwrap();
                        *count -= 1;
                        if *count == 0 {
                            position_counts.remove(&zobrist_hash);
                        }
                    }
                    return (entry.value, entry.best_move, Vec::new());
                },
                NodeType::LowerBound => alpha = alpha.max(entry.value),
                NodeType::UpperBound => beta = beta.min(entry.value),
            }
            if alpha >= beta {
                // Decrement the position count before returning
                {
                    let count = position_counts.get_mut(&zobrist_hash).unwrap();
                    *count -= 1;
                    if *count == 0 {
                        position_counts.remove(&zobrist_hash);
                    }
                }
                return (entry.value, entry.best_move, Vec::new());
            }
        }
    }

    // Terminal Node Check
    if depth == 0 || get_winner(board).is_some() {
        let evaluation = evaluate_board(board, player, weights);
        // Decrement the position count before returning
        {
            let count = position_counts.get_mut(&zobrist_hash).unwrap();
            *count -= 1;
            if *count == 0 {
                position_counts.remove(&zobrist_hash);
            }
        }
        return (evaluation, None, Vec::new());
    }

    let alpha_orig = alpha;

    // Generate Valid Moves
    let moves = get_valid_moves(board, player);

    if moves.is_empty() {
        // No moves available, losing position
        // Decrement the position count before returning
        {
            let count = position_counts.get_mut(&zobrist_hash).unwrap();
            *count -= 1;
            if *count == 0 {
                position_counts.remove(&zobrist_hash);
            }
        }
        return (LOSE_SCORE, None, Vec::new());
    }

    // Move Ordering
    let mut ordered_moves = Vec::new();
    let mut added_moves = HashSet::new();

    // Convert moves to HashSet for quick lookup
    let moves_set: HashSet<_> = moves.iter().cloned().collect();

    // 1. Try first_move if provided
    if let Some(mv) = first_move {
        if moves_set.contains(&mv) {
            ordered_moves.push(mv);
            added_moves.insert(mv);
        }
    }

    // 2. Try best_move from transposition table
    if let Some(entry) = transposition_table.get(&zobrist_hash) {
        if let Some(best_move) = entry.best_move {
            if Some(best_move) != first_move && moves_set.contains(&best_move) {
                ordered_moves.push(best_move);
                added_moves.insert(best_move);
            }
        }
    }

    // 3. Separate remaining moves into capture and non-capture moves
    let mut capture_moves = Vec::new();
    let mut non_capture_moves = Vec::new();

    for mv in moves {
        if added_moves.contains(&mv) {
            continue; // Already added
        }

        if is_capture_move(board, &mv, player) {
            capture_moves.push(mv);
        } else {
            non_capture_moves.push(mv);
        }
    }

    // 4. Append capture moves and non_capture moves
    ordered_moves.extend(capture_moves);
    ordered_moves.extend(non_capture_moves);

    let mut max_eval = LOSE_SCORE;
    let mut best_move = None;
    let mut pv_line = Vec::new();

    // Search through ordered moves
    for mv in ordered_moves {
        // Check if time limit exceeded
        if start_time.elapsed() >= time_limit {
            break;
        }

        let mut new_board = board.clone();
        let mut new_hash = zobrist_hash;

        let _captured_piece = make_move(&mut new_board, &mv, player, &mut new_hash, zobrist_table);

        let (eval, _, child_pv) = negamax_search(
            &new_board,
            depth - 1,
            -player,
            -beta,
            -alpha,
            weights,
            new_hash,
            zobrist_table,
            transposition_table,
            position_counts,
            start_time,
            time_limit,
            None, // No specific move ordering in deeper levels
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

    // Store in Transposition Table
    let flag = if max_eval <= alpha_orig {
        NodeType::UpperBound
    } else if max_eval >= beta {
        NodeType::LowerBound
    } else {
        NodeType::Exact
    };

    let entry = TranspositionTableEntry {
        depth,
        value: max_eval,
        flag,
        best_move,
    };

    transposition_table.insert(zobrist_hash, entry);

    // Decrement the position count before returning
    {
        let count = position_counts.get_mut(&zobrist_hash).unwrap();
        *count -= 1;
        if *count == 0 {
            position_counts.remove(&zobrist_hash);
        }
    }

    (max_eval, best_move, pv_line)
}



fn initialize_zobrist_table() -> [[[u64; 3]; BOARD_SIZE]; BOARD_SIZE] {
    let mut zobrist_table = [[[0u64; 3]; BOARD_SIZE]; BOARD_SIZE];
    let mut rng = rand::thread_rng();
    for row in 0..BOARD_SIZE {
        for col in 0..BOARD_SIZE {
            for piece in 0..3 {
                zobrist_table[row][col][piece] = rng.gen();
            }
        }
    }
    zobrist_table
}

fn piece_index(piece: i32) -> usize {
    match piece {
        BLACK => 1,
        WHITE => 2,
        _ => 0, // EMPTY
    }
}

fn compute_zobrist_hash(board: &Array2<i32>, zobrist_table: &[[[u64; 3]; BOARD_SIZE]; BOARD_SIZE]) -> u64 {
    let mut hash: u64 = 0;
    for ((row, col), &piece) in board.indexed_iter() {
        let piece_idx = piece_index(piece);
        if piece_idx != 0 {
            hash ^= zobrist_table[row][col][piece_idx];
        }
    }
    hash
}


fn is_capture_move(board: &Array2<i32>, mv: &(usize, usize, usize, usize), player: i32) -> bool {
    let (from_row, _from_col, to_row, _to_col) = *mv;
    let delta_row = (to_row as isize - from_row as isize).abs();
    delta_row == 2 // Capture moves involve jumping over an opponent's piece
}

fn make_move(
    board: &mut Array2<i32>,
    mv: &(usize, usize, usize, usize),
    player: i32,
    zobrist_hash: &mut u64,
    zobrist_table: &[[[u64; 3]; BOARD_SIZE]; BOARD_SIZE],
) -> i32 {
    let (from_row, from_col, to_row, to_col) = *mv;

    let from_piece = board[[from_row, from_col]];
    let to_piece = board[[to_row, to_col]]; // Should be EMPTY

    // Remove piece from old position
    *zobrist_hash ^= zobrist_table[from_row][from_col][piece_index(from_piece)];
    // Place piece at new position
    *zobrist_hash ^= zobrist_table[to_row][to_col][piece_index(from_piece)];

    // Update the board
    board[[to_row, to_col]] = from_piece;
    board[[from_row, from_col]] = EMPTY;

    let mut captured_piece = EMPTY;

    // Check if it's a capture
    if (from_row as isize - to_row as isize).abs() == 2 {
        let mid_row = (from_row + to_row) / 2;
        let mid_col = (from_col + to_col) / 2;
        captured_piece = board[[mid_row, mid_col]];
        // Remove captured piece
        *zobrist_hash ^= zobrist_table[mid_row][mid_col][piece_index(captured_piece)];
        board[[mid_row, mid_col]] = EMPTY;
    }

    captured_piece
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

            // Opponent's edge pawn bonus
            if is_edge_square(row, col) {
                score -= weights.edge_pawn_bonus;
            }
        }
    }

    // Unstoppable pawns
    let ai_unstoppable_pawns = get_unstoppable_pawns_steps(board, player);
    let opponent_unstoppable_pawns = get_unstoppable_pawns_steps(board, -player);

    // Evaluate our unstoppable pawns
    for steps in ai_unstoppable_pawns.iter() {
        let bonus = weights.unstoppable_pawn_bonus / (*steps as f64 + 1.0);
        score += bonus;
    }

    // Evaluate opponent's unstoppable pawns
    for steps in opponent_unstoppable_pawns.iter() {
        let penalty = weights.opponent_unstoppable_pawn_penalty / (*steps as f64 + 1.0);
        score += penalty; // Since penalty is negative
    }

    // Additional logic to prioritize pawns that promote sooner
    if let Some(&min_ai_steps) = ai_unstoppable_pawns.iter().min() {
        if let Some(&min_opponent_steps) = opponent_unstoppable_pawns.iter().min() {
            if min_opponent_steps < min_ai_steps {
                // Opponent pawn promotes before ours
                score += weights.opponent_unstoppable_pawn_penalty * 2.0;
            } else if min_ai_steps < min_opponent_steps {
                // Our pawn promotes before opponent's
                score += weights.unstoppable_pawn_bonus * 2.0;
            }
        }
    } else if opponent_unstoppable_pawns.is_empty() && !ai_unstoppable_pawns.is_empty() {
        // Only we have unstoppable pawns
        score += weights.unstoppable_pawn_bonus * 2.0;
    } else if ai_unstoppable_pawns.is_empty() && !opponent_unstoppable_pawns.is_empty() {
        // Only opponent has unstoppable pawns
        score += weights.opponent_unstoppable_pawn_penalty * 2.0;
    }

    score
}

fn is_edge_square(row: usize, col: usize) -> bool {
    col == 0 || col == BOARD_SIZE - 1
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

fn get_all_valid_moves(
    board: &Array2<i32>,
    player: i32,
) -> (
    Vec<(usize, usize, usize, usize)>, // Capture moves
    Vec<(usize, usize, usize, usize)>, // Normal moves
) {
    let mut normal_moves = Vec::new();
    let mut capture_moves = Vec::new();

    for row in 0..BOARD_SIZE {
        for col in 0..BOARD_SIZE {
            if board[[row, col]] == player {
                let (piece_moves, piece_capture_moves) = get_piece_moves(board, (row, col), player);
                normal_moves.extend(piece_moves);
                capture_moves.extend(piece_capture_moves);
            }
        }
    }

    (capture_moves, normal_moves)
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

fn is_within_bounds(row: isize, col: isize) -> bool {
    row >= 0 && row < BOARD_SIZE as isize && col >= 0 && col < BOARD_SIZE as isize
}

fn get_opponent_pawns_by_row(board: &Array2<i32>, opponent: i32) -> Vec<Vec<usize>> {
    let mut pawns_by_row: Vec<Vec<usize>> = vec![Vec::new(); BOARD_SIZE];
    for ((row, col), &piece) in board.indexed_iter() {
        if piece == opponent {
            pawns_by_row[row].push(col);
        }
    }
    pawns_by_row
}

fn is_unstoppable_pawn(
    pawn_pos: (usize, usize),
    player: i32,
    opponent_pawns_by_row: &Vec<Vec<usize>>,
) -> Option<isize> {
    let (row_pawn, col_pawn) = pawn_pos;
    let row_pawn = row_pawn as isize;
    let col_pawn = col_pawn as isize;
    let row_goal = if player == BLACK { BOARD_SIZE as isize - 1 } else { 0 };
    let direction = if player == BLACK { 1 } else { -1 };

    let steps_to_goal = (row_goal - row_pawn).abs();

    // Only check rows ahead of the pawn
    let row_range = if player == BLACK {
        (row_pawn + 1) as usize..BOARD_SIZE
    } else {
        0..(row_pawn as usize)
    };

    for row in row_range {
        let relative_row = (row as isize - row_pawn) * direction;
        let steps_to_opp = relative_row;
        if steps_to_opp <= 0 {
            continue;
        }

        // If steps to opponent pawn exceed steps to goal, they can't stop us
        if steps_to_opp > steps_to_goal {
            break;
        }

        for &col_opp in &opponent_pawns_by_row[row] {
            let col_diff = (col_opp as isize - col_pawn).abs();
            if col_diff <= steps_to_opp {
                // Opponent pawn can reach our pawn
                return None;
            }
        }
    }

    // No opponent pawns can stop this pawn
    Some(steps_to_goal)
}

fn get_unstoppable_pawns_steps(
    board: &Array2<i32>,
    player: i32,
) -> Vec<isize> {
    let opponent = -player;
    let opponent_pawns_by_row = get_opponent_pawns_by_row(board, opponent);
    let mut steps_list = Vec::new();

    for ((row, col), &piece) in board.indexed_iter() {
        if piece == player {
            if let Some(steps_to_goal) = is_unstoppable_pawn(
                (row, col),
                player,
                &opponent_pawns_by_row,
            ) {
                steps_list.push(steps_to_goal);
            }
        }
    }

    steps_list
}

#[pymodule]
fn fianco_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(negamax, m)?)?;
    Ok(())
}
