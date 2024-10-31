#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "action.h"
#include "proto/statemachine/fsm_rule.pb.h"

namespace hozon {
namespace fsmcore {

class State;

// using fsm::action::Action;
using ActionVec = std::vector<std::shared_ptr<Action>>;
using StateVec = std::vector<std::shared_ptr<State>>;
//using fsm::action::FindActionCoreFunc;
using FindStateFunc =
    std::function<std::shared_ptr<State>(const hozon::fsm_rule::StateId& id)>;

/*****************************************************************************/
/* Used to descript defined state                                            */
/*****************************************************************************/
class State : public std::enable_shared_from_this<State> {
 public:
  State(const hozon::fsm_rule::FsmState& fsm_state,
        FindActionCoreFunc find_action_func, uint32_t level = 0);
  void set_parent(std::shared_ptr<State> parent);

  bool check_legal();
  std::string to_string(const std::string& indent = "");
  uint32_t get_level() { return _level; };
  std::string get_name() { return _name; };
  std::shared_ptr<State> get_parent() { return _parent; };
  StateVec get_substates() { return _sub_states; };
  ActionVec get_enter_actions() { return _enter_actions; };
  ActionVec get_exit_actions() { return _exit_actions; };
  bool is_groud_level() { return (_sub_states.size() == 0); };

 private:
  uint32_t _level;
  std::string _name;
  std::shared_ptr<State> _parent;
  StateVec _sub_states;
  ActionVec _enter_actions;
  ActionVec _exit_actions;
};

}  // namespace fsmcore
}  // namespace hozon