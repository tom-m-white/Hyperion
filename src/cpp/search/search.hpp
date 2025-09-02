#ifndef HYPERION_ENGINE_SEARCH_HPP
#define HYPERION_ENGINE_SEARCH_HPP

#include "../core/position.hpp"
#include "../core/move.hpp"
#include "eval.hpp"
#include "nn_inference/nn_inference.hpp"
#include "move_encoder.hpp" 
#include "tt.hpp"

#include <vector>
#include <memory>
#include <random>
#include <atomic>

namespace hyperion {
namespace engine {
//--
/* struct Node */
//--

struct Node {
    Node* parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;
    core::Move move;
    int visits = 0;
    double value = 0.0;
    Node() = default;
    Node(Node* p, core::Move m) : parent(p), move(m) {}
    bool is_fully_expanded(size_t num_legal_moves) const {
        return children.size() >= num_legal_moves;
    }
    double policy_prior = 0.01;
};
class Search {
public:
    Search();

    // The main function to find the best move
    core::Move find_best_move(core::Position& root_pos, int time_limit_ms);
    

private:
    std::unique_ptr<Node> root_node;
    TranspositionTable tt;
    std::mt19937 random_generator;

    // The Neural Network object
    std::unique_ptr<NeuralNetwork> nn_; 

    // Helper functions for MCTS
    Node* select(Node* node, core::Position& pos);
    // Expand will now also evaluate the position
    void expand_and_evaluate(Node* node, const core::Position& pos);
    void backpropagate(Node* node, double value);
    double uct_score(const Node* node, int parent_visits) const;
    core::Move get_best_move_from_root();
};

} // namespace engine
} // namespace hyperion

#endif // HYPERION_ENGINE_SEARCH_HPP
