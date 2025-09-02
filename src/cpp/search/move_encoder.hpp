#ifndef MOVE_ENCODER_HPP
#define MOVE_ENCODER_HPP

#include "../core/move.hpp"
#include "../core/position.hpp"
#include "../core/move.hpp"

namespace hyperion {
namespace engine {

// This function will be the C++ equivalent of the Python uci_to_policy_index
int get_policy_index(const core::Move& move, const core::Position& pos);

} // namespace engine
} // namespace hyperion

#endif // MOVE_ENCODER_HPP