#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include "core/position.hpp"
#include "core/move.hpp"
#include "core/movegen.hpp"
#include "core/zobrist.hpp"
#include "core/bitboard.hpp"
#include "search/search.hpp"
#include <thread>
#include <ctime>
#define _USE_MATH_DEFINES
#include <cmath>
#include <numbers>

double M_PI = 3.141592653589793238463;

// --- Helper Functions & Structs ---
//                                                                  
/*RUN WITH THIS COMMAND: .\bin\testpuzzles C:\path\to\puzzles\inside\computer 500 1000 */
//                                                                     puzzles ^   ^ time(ms)

struct Puzzle {
    std::string fen;
    std::string solution_uci;
    int elo;
};

//--
/* move_to_uci_string */
//--
// Converts a `hyperion::core::Move` object into its standard Universal Chess Interface (UCI)
// string representation. The function formats the move by concatenating the algebraic
// notation of the 'from' square and the 'to' square. If the move is a promotion, it
// appends the corresponding character for the promotion piece ('q', 'r', 'b', or 'n').
// This function does not modify any variables
std::string move_to_uci_string(const hyperion::core::Move& move) {
    using namespace hyperion::core;

    std::string uci_move = square_to_algebraic(static_cast<int>(move.from_sq)) +
                           square_to_algebraic(static_cast<int>(move.to_sq));

    if (move.is_promotion()) {
        switch (move.get_promotion_piece()) { 
            case P_QUEEN:  uci_move += 'q'; break;
            case P_ROOK:   uci_move += 'r'; break;
            case P_BISHOP: uci_move += 'b'; break;
            case P_KNIGHT: uci_move += 'n'; break;
            default: break;
        }
    }
    return uci_move;
}

//--
/* find_solution_uci_from_san */
//--
// Searches for a legal move in a given position that matches a move provided in
// Standard Algebraic Notation (SAN). It first sanitizes the input `san` string by
// removing check ('+') and checkmate ('#') characters. It then generates all legal
// moves for the given `pos` and iterates through them. The function matches the
// correct move by checking for castling notation or by comparing the piece type and
// destination square. If a match is found, it returns the move in UCI format using
// `move_to_uci_string`. If no match is found, it returns "NOT_FOUND"
std::string find_solution_uci_from_san(const hyperion::core::Position& pos, std::string san) {
    using namespace hyperion::core;
    san.erase(std::remove(san.begin(), san.end(), '+'), san.end());
    san.erase(std::remove(san.begin(), san.end(), '#'), san.end());
    MoveGenerator move_gen;
    std::vector<Move> legal_moves;
    move_gen.generate_legal_puzzle_moves(pos, legal_moves);

    for (const auto& move : legal_moves) {
        if (move.is_kingside_castle() && (san == "O-O" || san == "0-0")) return move_to_uci_string(move);
        if (move.is_queenside_castle() && (san == "O-O-O" || san == "0-0-0")) return move_to_uci_string(move);
        int mailbox_val = pos.get_piece_on_square(move.from_sq); 
        piece_type_e piece = pos.get_piece_type_from_mailbox_val(mailbox_val);
        std::string dest_sq_str = square_to_algebraic(static_cast<int>(move.to_sq));

        if (san.find(dest_sq_str) == std::string::npos) continue;

        bool piece_match = false;
        switch (piece) {
            case P_PAWN:   piece_match = (toupper(san[0]) < 'A' || toupper(san[0]) > 'H'); break;
            case P_KNIGHT: piece_match = (san[0] == 'N'); break;
            case P_BISHOP: piece_match = (san[0] == 'B'); break;
            case P_ROOK:   piece_match = (san[0] == 'R'); break;
            case P_QUEEN:  piece_match = (san[0] == 'Q'); break;
            case P_KING:   piece_match = (san[0] == 'K'); break;
            default: break;
        }

        if (piece_match) {
            return move_to_uci_string(move);
        }
    }
    return "NOT_FOUND";
}

//--
/* load_puzzles */
//--
// Reads a file containing chess puzzles, likely in a PGN-like format, and parses it
// into a vector of `Puzzle` structs. The function iterates through the file line by
// line, extracting the FEN string, the Elo rating, and the solution move in SAN.
// It then calls `find_solution_uci_from_san` to convert the solution to UCI format.
// Each successfully parsed puzzle is added to a vector, which is returned upon
// completion. The function will throw a `std::runtime_error` if it cannot open the
// specified file.
std::vector<Puzzle> load_puzzles(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open puzzle file: " + filename);
    }

    std::vector<Puzzle> puzzles;
    std::string line;
    
    std::string current_fen;
    int current_elo = 0;

    // This loop simply goes through the whole file.
    while (std::getline(file, line)) { 
        if (line.rfind("[FEN \"", 0) == 0) {
            current_fen = line.substr(6, line.length() - 8);
        } else if (line.rfind("[WhiteElo \"", 0) == 0) {
            std::string elo_str = line.substr(11, line.length() - 13);
            if (!elo_str.empty()) {
                current_elo = std::stoi(elo_str);
            }
        } else if (line.rfind("1...", 0) == 0 || line.rfind("1.", 0) == 0) {
            if (!current_fen.empty() && current_elo > 0) {
                std::stringstream ss(line);
                std::string move_num, san;
                ss >> move_num >> san;
                hyperion::core::Position temp_pos;
                temp_pos.set_from_fen(current_fen);
                std::string uci_solution = find_solution_uci_from_san(temp_pos, san);
                if (uci_solution != "NOT_FOUND") {
                    puzzles.push_back({current_fen, uci_solution, current_elo});
                }
                
                current_fen.clear();
                current_elo = 0;
            }
        }
    }
    return puzzles;
}

//--
/* glicko2_elo_change */
//--
// Calculates the dynamic "impact factor" for a single game based on the Glicko-2
// rating system. This factor is conceptually similar to the K-factor in the standard
// Elo system but is dynamic. It takes the player's Elo and Rating Deviation (RD),
// and the puzzle's Elo as input. The returned value represents how much a single game's
// outcome will influence the player's Elo. This value is later multiplied by the
// difference between the actual score (1.0 for a win, 0.0 for a loss) and the
// expected score to get the total Elo change for that game.
double glicko2_elo_change(double player_elo, double player_rd, double puzzle_elo) {
    // Glicko-2 system constants
    const double q = log(10.0) / 400.0;
    const double assumed_puzzle_rd = 70.0; // A reasonable, fixed RD for puzzles

    // Calculate the 'g' factor for the puzzle's RD.
    double g_puzzle = 1.0 / sqrt(1.0 + 3.0 * q * q * assumed_puzzle_rd * assumed_puzzle_rd / (M_PI * M_PI));

    // Calculate the expected score for this single matchup.
    double expected_score = 1.0 / (1.0 + pow(10.0, g_puzzle * (player_elo - puzzle_elo) / -400.0));

    // This is the key part of the Glicko-2 Elo update.
    // The "K-Factor" is effectively `q / (1/RD^2 + 1/d^2)`. It's dynamic and built-in.
    double d_squared = 1.0 / (q * q * g_puzzle * g_puzzle * expected_score * (1.0 - expected_score));
    
    // We will return the 'impact' of a single game, which is then multiplied by (actual - expected).
    return q / (1.0 / (player_rd * player_rd) + 1.0 / d_squared);
}

struct TestStats {
    int solved;
    int total_processed;
    size_t total_to_run;
    double engine_elo;
    double elo_change;
    double rating_deviation;
    int eta_seconds;
};

//--
/* format_duration */
//--
// Converts an integer representing a duration in total seconds into a human-readable
// string of the format "Xm YYs" (minutes and seconds). For example, 75 seconds would
// be formatted as "1m 15s". The seconds are padded with a leading zero if necessary.
// If the input `total_seconds` is negative, it returns "N/A".
std::string format_duration(int total_seconds) {
    if (total_seconds < 0) return "N/A";
    int minutes = total_seconds / 60;
    int seconds = total_seconds % 60;
    std::stringstream ss;
    ss << minutes << "m " << std::setw(2) << std::setfill('0') << seconds << "s";
    return ss.str();
}
//--
/* display_stats */
//--
// Clears the console screen and prints a real-time, formatted display of the puzzle
// test's progress. The output includes a progress bar, overall statistics like accuracy
// and ETA, the engine's current Elo and Rating Deviation, and detailed information
// about the last puzzle that was tested. It uses ANSI escape codes to clear the screen
// and to color the result of the last puzzle (green for correct, red for incorrect).
// This function does not change any program variables but modifies the console output.
void display_stats(const TestStats& stats, const Puzzle& current_puzzle, const std::string& engine_move, bool was_correct) {
    std::cout << "\033[2J\033[1;1H"; // Clear screen

    std::cout << "+----------------------------------------------------+\n";
    std::cout << "|              HYPERION PUZZLE TESTER                |\n";
    std::cout << "+----------------------------------------------------+\n\n";

    // Progress Bar
    int bar_width = 40;
    int progress = (stats.total_to_run > 0) ? (stats.total_processed * bar_width / stats.total_to_run) : 0;
    std::cout << "  Progress: [" << std::string(progress, '=') << (progress < bar_width ? ">" : "")
              << std::string(bar_width - progress - (progress < bar_width), ' ') << "]\n";

    // Overall Stats
    double accuracy = (stats.total_processed > 0) ? (static_cast<double>(stats.solved) / stats.total_processed) * 100.0 : 0.0;
    std::cout << "  Puzzles:       " << stats.total_processed << " / " << stats.total_to_run << "\n";
    std::cout << "  Accuracy:      " << std::fixed << std::setprecision(2) << accuracy << "%\n";
    std::cout << "  ETA:           " << format_duration(stats.eta_seconds) << "\n\n";

    // Elo Stats
    std::cout << "--- Elo Performance ---\n";
    std::cout << "  Current Elo:   " << std::fixed << std::setprecision(2) << stats.engine_elo 
              << " +- " << std::fixed << std::setprecision(1) << stats.rating_deviation
              << " (" << std::showpos << stats.elo_change << std::noshowpos << ")\n\n";

    // Last Puzzle Info
    std::cout << "--- Last Puzzle (Elo: " << current_puzzle.elo << ") ---\n";
    std::cout << "  Solution:      " << current_puzzle.solution_uci << "\n";
    std::cout << "  Engine's Move: " << engine_move << "\n";
    std::cout << "  Result:        ";
    if (was_correct) {
        std::cout << "\033[32mCORRECT\033[0m\n"; // Green text
    } else {
        std::cout << "\033[31mINCORRECT\033[0m\n"; // Red text
    }
    
    std::cout << "\n" << std::flush;
}

//--
/* update_rating_deviation */
//--
// Updates the player's (engine's) Rating Deviation (RD) after a rating period, according
// to the Glicko-2 rating system. It takes the current RD, current Elo, and a list of
// puzzles played and their results within the period. The function calculates the
// estimated variance ('v') of the player's performance against expectations and uses
// this to calculate the new RD. A lower RD indicates a more stable and reliable rating.
// If no games were played, the RD is slightly increased to reflect rating uncertainty
// over time.
double update_rating_deviation(double player_rd, double player_elo, const std::vector<Puzzle>& puzzles_played, const std::vector<bool>& results) {
    // Glicko-2 system constants
    const double q = log(10.0) / 400.0;
    const double assumed_puzzle_rd = 70.0;
    
    // If no games were played, RD increases slightly due to time passing (volatility)
    if (puzzles_played.empty()) {
        const double system_volatility = 0.5; // How much ratings fluctuate over time
        return std::min(350.0, sqrt(player_rd * player_rd + system_volatility * system_volatility));
    }

    // This is based on all the opponents (puzzles) faced in this rating period.
    double v_inverse = 0.0;
    for (const auto& puzzle : puzzles_played) {
        double g_puzzle = 1.0 / sqrt(1.0 + 3.0 * q * q * assumed_puzzle_rd * assumed_puzzle_rd / (M_PI * M_PI));
        double expected_score = 1.0 / (1.0 + pow(10.0, g_puzzle * (player_elo - puzzle.elo) / -400.0));
        v_inverse += g_puzzle * g_puzzle * expected_score * (1.0 - expected_score);
    }
    double v = 1.0 / (q * q * v_inverse);

    // This is how much the player over or under-performed expectations.
    double delta = 0.0;
    for (size_t i = 0; i < puzzles_played.size(); ++i) {
        double g_puzzle = 1.0 / sqrt(1.0 + 3.0 * q * q * assumed_puzzle_rd * assumed_puzzle_rd / (M_PI * M_PI));
        double expected_score = 1.0 / (1.0 + pow(10.0, g_puzzle * (player_elo - puzzles_played[i].elo) / -400.0));
        double actual_score = results[i] ? 1.0 : 0.0;
        delta += g_puzzle * (actual_score - expected_score);
    }
    delta *= (q * v);

   
    double new_rd = sqrt(1.0 / (1.0 / (player_rd * player_rd) + 1.0 / v));

    // A check to ensure RD doesn't become unrealistically small
    if (new_rd < 5) new_rd = 5;

    return new_rd;
}

//--
/* main */
//--
// The main entry point for the puzzle testing application. The function's primary
// responsibilities include:
// 1. Parsing command-line arguments to get the puzzle file, number of puzzles to run,
//    and time per move.
// 2. Initializing engine-specific components like attack tables and Zobrist keys.
// 3. Loading and sorting puzzles from the specified file.
// 4. Running the main test loop. In each iteration, it adaptively selects a puzzle
//    based on the engine's current Elo, has the engine find a move, and checks if the
//    move is correct.
// 5. Periodically updating the engine's Elo and Rating Deviation in batches using the
//    Glicko-2 system. This changes the `engine_elo` and `rating_deviation` variables.
// 6. Calling `display_stats` to provide a continuous, live view of the test's progress.
// 7. Printing a final summary of the engine's performance upon completion.
int main(int argc, char* argv[]) {
    // --- Step 1: Argument Parsing ---
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_puzzle_file> [num_puzzles_to_test] [time_per_move_ms]\n";
        return 1;
    }
    const std::string puzzle_file = argv[1];
    size_t num_puzzles_to_run = (argc > 2) ? std::stoul(argv[2]) : 1000;
    const int time_per_move_ms = (argc > 3) ? std::stoi(argv[3]) : 5000;

    // --- Step 2: Engine Initialization ---

    hyperion::core::Zobrist::initialize_keys();
    hyperion::core::initialize_attack_tables();

    // --- Step 3: Load and Prepare Puzzles ---
    std::cout << "Loading puzzles from " << puzzle_file << "...\n";
    std::vector<Puzzle> puzzles;
    try {
        puzzles = load_puzzles(puzzle_file);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    std::cout << "Loaded " << puzzles.size() << " valid puzzles.\n";

    // Sort puzzles by Elo rating to enable efficient adaptive selection.
    std::cout << "Sorting puzzles by difficulty...\n";
    std::sort(puzzles.begin(), puzzles.end(), [](const Puzzle& a, const Puzzle& b) {
        return a.elo < b.elo;
    });

    // Create a vector to track which puzzles have been used in this run.
    std::vector<bool> puzzle_used(puzzles.size(), false);
    size_t total_puzzles_in_set = puzzles.size();
    if (num_puzzles_to_run > total_puzzles_in_set || num_puzzles_to_run == 0) {
        num_puzzles_to_run = total_puzzles_in_set;
    }

    // Define the random number generator needed for adaptive selection volatility.
    std::random_device rd;
    std::mt19937 g(rd());

    std::cout << "Starting ADAPTIVE test with " << num_puzzles_to_run << " puzzles.\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));


    // --- Step 4: Main Testing Loop ---
    int puzzles_solved = 0;
    double engine_elo = 1500.0;
    double rating_deviation = 350.0; // Start with high uncertainty.
    auto test_start_time = std::chrono::steady_clock::now();

    const int RATING_PERIOD_LENGTH = 10; // Update stats every 10 puzzles
    std::vector<Puzzle> puzzles_in_period;
    std::vector<bool> results_in_period;

    // Create one search handler object to reuse for the entire test.
    hyperion::engine::Search search_handler;

    for (size_t i = 0; i < num_puzzles_to_run; ++i) {
        
        // --- ADAPTIVE PUZZLE SELECTION with VOLATILITY ---
        size_t best_puzzle_idx = -1;
        double search_radius = rating_deviation * 2.0; 
        double lower_bound_elo = engine_elo - search_radius;
        double upper_bound_elo = engine_elo + search_radius;

        auto start_it = std::lower_bound(puzzles.begin(), puzzles.end(), lower_bound_elo,
                                         [](const Puzzle& p, double elo){ return p.elo < elo; });
        auto end_it = std::upper_bound(puzzles.begin(), puzzles.end(), upper_bound_elo,
                                       [](double elo, const Puzzle& p){ return elo < p.elo; });

        std::vector<size_t> candidate_indices;
        for (auto it = start_it; it != end_it; ++it) {
            size_t idx = std::distance(puzzles.begin(), it);
            if (!puzzle_used[idx]) {
                candidate_indices.push_back(idx);
            }
        }

        if (!candidate_indices.empty()) {
            std::uniform_int_distribution<size_t> distrib(0, candidate_indices.size() - 1);
            best_puzzle_idx = candidate_indices[distrib(g)];
        } else {
            // FALLBACK: If the volatile range is empty, find the absolute closest unused puzzle.
            auto it_fallback = std::lower_bound(puzzles.begin(), puzzles.end(), engine_elo, 
                                                [](const Puzzle& p, double elo){ return p.elo < elo; });
            size_t start_idx_fallback = std::distance(puzzles.begin(), it_fallback);
            for (size_t offset = 0; ; ++offset) {
                size_t r = start_idx_fallback + offset;
                size_t l = start_idx_fallback - offset;
                if (r < total_puzzles_in_set && !puzzle_used[r]) { best_puzzle_idx = r; break; }
                if (offset > 0 && l < start_idx_fallback && !puzzle_used[l]) { best_puzzle_idx = l; break; }
                if (r >= total_puzzles_in_set && (offset == 0 || l >= start_idx_fallback)) { break; }
            }
        }

        if (best_puzzle_idx == static_cast<size_t>(-1)) {
            std::cout << "No more unused puzzles found. Ending test early.\n";
            break;
        }

        const auto& puzzle = puzzles[best_puzzle_idx];
        puzzle_used[best_puzzle_idx] = true;

         // --- Engine Search ---
    hyperion::core::Position pos;
    pos.set_from_fen(puzzle.fen);
    hyperion::core::Move best_move = search_handler.find_best_move(pos, time_per_move_ms);
    std::string engine_uci_move = move_to_uci_string(best_move);

    // --- Collect Results for this period ---
    bool is_correct = (engine_uci_move == puzzle.solution_uci);
    if (is_correct) {
        puzzles_solved++;
    }
    puzzles_in_period.push_back(puzzle);
    results_in_period.push_back(is_correct);

    double elo_change_for_display = 0.0;

    if (puzzles_in_period.size() >= RATING_PERIOD_LENGTH || (i + 1) == num_puzzles_to_run) {
    
    double old_elo = engine_elo;

    rating_deviation = update_rating_deviation(rating_deviation, old_elo, puzzles_in_period, results_in_period);
    
    // --- BATCH ELO UPDATE ---
    double total_elo_impact = 0;
    
    // Loop through each game in the period to calculate the total change.
    for (size_t j = 0; j < puzzles_in_period.size(); ++j) {
        const auto& puzzle_j = puzzles_in_period[j];
        const bool result_j = results_in_period[j];

        // Recalculate g and E for this specific matchup
        const double q = log(10.0) / 400.0;
        const double assumed_puzzle_rd = 70.0;
        double g_puzzle = 1.0 / sqrt(1.0 + 3.0 * q * q * assumed_puzzle_rd * assumed_puzzle_rd / (M_PI * M_PI));
        double expected_score = 1.0 / (1.0 + pow(10.0, g_puzzle * (old_elo - puzzle_j.elo) / -400.0));
        
        double actual_score = result_j ? 1.0 : 0.0;

        // Call the new helper to get the Glicko-2 "K-Factor" for this game
        double game_impact_factor = glicko2_elo_change(old_elo, rating_deviation, puzzle_j.elo);
        
        // The change for this one game is impact * (actual - expected)
        total_elo_impact += game_impact_factor * (actual_score - expected_score);
    }
    
    // Apply the total change for the entire rating period.
    engine_elo = old_elo + total_elo_impact;

    elo_change_for_display = engine_elo - old_elo;

    // Clear the buffers for the next rating period
    puzzles_in_period.clear();
    results_in_period.clear();
}

    // --- Timing and Display Logic ---
    auto now = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - test_start_time).count();
    double avg_time_per_puzzle = elapsed_seconds / (i + 1);
    int puzzles_remaining = num_puzzles_to_run - (i + 1);
    int eta_seconds = static_cast<int>(puzzles_remaining * avg_time_per_puzzle);

    TestStats current_stats = { puzzles_solved, static_cast<int>(i + 1), num_puzzles_to_run, engine_elo, elo_change_for_display, rating_deviation, eta_seconds };
    display_stats(current_stats, puzzle, engine_uci_move, is_correct);
}

    // --- Step 5: Final Summary ---
    std::cout << "\n--- TEST COMPLETE ---\n";
    std::cout << "Final Engine Elo: " << std::fixed << std::setprecision(2) << engine_elo << " +- " << rating_deviation << "\n";
    double final_accuracy = (num_puzzles_to_run > 0) ? (static_cast<double>(puzzles_solved) / num_puzzles_to_run) * 100.0 : 0.0;
    std::cout << "Total Accuracy:   " << final_accuracy << "% (" << puzzles_solved << " / " << num_puzzles_to_run << ")\n";

    return 0;
}