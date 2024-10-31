#include "state.h"
#include "fsm_utils.h"
#include "fsm.h"

namespace hozon {
namespace fsmcore {

///////////////////////////////////////////////////////////////////////////////

/*****************************************************************************/
/* 构造函数 */
/*****************************************************************************/
State::State(const hozon::fsm_rule::FsmState& fsm_state,
             FindActionCoreFunc find_action_func, uint32_t level) {
  auto state_id = fsm_state.id();
  _name = state_id.name();
  _level = level;

  // 添加子状态
  for (int ii = 0; ii < fsm_state.sub_states_size(); ++ii) {
    auto sub = fsm_state.sub_states(ii);
    _sub_states.emplace_back(
        // 子状态的层级，比主状态大 1
        std::make_shared<State>(sub, find_action_func, (_level + 1)));
  }

  // 添加进入本状态的动作
  for (int ii = 0; ii < fsm_state.enter_actions_size(); ++ii) {
    auto action = fsm_state.enter_actions(ii);
    _enter_actions.emplace_back(
        std::make_shared<Action>(action, find_action_func));
  }

  // 添加退出本状态的动作
  for (int ii = 0; ii < fsm_state.exit_actions_size(); ++ii) {
    auto action = fsm_state.exit_actions(ii);
    _exit_actions.emplace_back(
        std::make_shared<Action>(action, find_action_func));
  }
}

/*****************************************************************************/
/* 设置父 State，因为 shared_from_this 无法在构造函数中调用，而子 state 是递归构造   */
/*****************************************************************************/
void State::set_parent(std::shared_ptr<State> parent) {
  _parent = parent;

  for (auto&& state : _sub_states) {
    state->set_parent(shared_from_this());
  }
}

/*****************************************************************************/
/* 检查合法性 */
/*****************************************************************************/
bool State::check_legal() {
  for (auto&& action : _enter_actions) {
    if (!action->check_legal()) {
      FSMCORE_LOG_ERROR << "Transit action is illegal.";
      return false;
    }
  }

  for (auto&& action : _exit_actions) {
    if (!action->check_legal()) {
      FSMCORE_LOG_ERROR << "Transit action is illegal.";
      return false;
    }
  }

  for (auto&& sub : _sub_states) {
    if (!sub->check_legal()) {
      return false;
    }
  }

  return true;
}

/*****************************************************************************/
/* 输出 state 信息 */
/*****************************************************************************/
std::string State::to_string(const std::string& indent) {
  std::string result_str;
  std::string parent_str;

  if (!NULL_IPTR(_parent)) {
    parent_str = _parent->get_name();
    parent_str += "(";
    parent_str += std::to_string(_parent->get_level());
    parent_str += ")\n";
  } else {
    parent_str = "null\n";
  }

  result_str += indent;
  result_str += _name;
  result_str += "(";
  result_str += std::to_string(_level);
  result_str += ")\n";
  result_str += indent;
  result_str += "  parent: ";
  result_str += parent_str;

  // 添加子状态
  if (!_sub_states.empty()) {
    result_str += indent;
    result_str += "  sub state:\n";
    for (auto&& sub : _sub_states) {
      result_str += sub->to_string(indent + "    ");
    }
  }

  // 添加进入动作
  if (!_enter_actions.empty()) {
    result_str += indent;
    result_str += "enter state actions: \n";
    for (auto&& action : _enter_actions) {
      result_str += indent;
      result_str += "  ";
      result_str += action->to_string();
      result_str += "\n";
    }
  }
  if (!_exit_actions.empty()) {
    result_str += indent;
    result_str += "exit state actions: \n";
    for (auto&& action : _exit_actions) {
      result_str += indent;
      result_str += "  ";
      result_str += action->to_string();
      result_str += "\n";
    }
  }

  return result_str;
}

}  // namespace fsmcore
}  // namespace hozon
