#include "transit.h"
#include "fsm_utils.h"
#include "fsm.h"

namespace hozon {
namespace fsmcore {

///////////////////////////////////////////////////////////////////////////////

/*****************************************************************************/
/* 构造函数 */
/*****************************************************************************/
Transit::Transit(const hozon::fsm_rule::FsmTransit& fsm_transit,
                 FindStateFunc find_state_func,
                 FindActionCoreFunc find_action_func,
                 FindConditionCoreFunc find_condition_func) {
  _from = find_state_func(fsm_transit.from());
  _to = find_state_func(fsm_transit.to());

  // 填充跳转对象中的条件，这里的条件是一个组合条件，只有一个
  if (fsm_transit.has_condition()) {
    _condition = std::make_shared<Condition>(fsm_transit.condition(),
                                             find_condition_func);
  }

  // 填充跳转对象中的动作，动作是一个数组，包含一连串动作
  for (int ii = 0; ii < fsm_transit.actions_size(); ++ii) {
    auto one_action = fsm_transit.actions(ii);
    _actions.emplace_back(
        std::make_shared<Action>(one_action, find_action_func, true));
  }
}

/*****************************************************************************/
/* 合法的跳转必须具备： */
/*  1. 必须有源状态和目的状态，这就要求原状态和目的状态配置的名称和层级是对的 */
/*  2. 目的状态，必须是一个最底层的状态 */
/*****************************************************************************/
bool Transit::check_legal() {
  if (NULL_IPTR(_from)) {
    FSMCORE_LOG_ERROR << "Transit should has from state.";
    return false;
  }

  if (NULL_IPTR(_to)) {
    FSMCORE_LOG_ERROR << "Transit should has to state.";
    return false;
  }

  if (!(_to->is_groud_level())) {
    FSMCORE_LOG_ERROR
        << "Transit destination should be bottom level state, but state name: "
        << _to->get_name() << " has " << _to->get_substates().size()
        << " sub states.";
    return false;
  }

  for (auto&& action : _actions) {
    if (!action->check_legal()) {
      FSMCORE_LOG_ERROR << "Transit action is illegal.";
      return false;
    }
  }

  return true;
}

/*****************************************************************************/
/* 输出 transit  */
/*****************************************************************************/
std::string Transit::to_string() {
  std::string result_str;

  result_str += "  from: ";
  result_str += _from->get_name();
  result_str += "(";
  result_str += std::to_string(_from->get_level());
  result_str += ") ";
  result_str += "to: ";
  result_str += _to->get_name();
  result_str += "(";
  result_str += std::to_string(_to->get_level());
  result_str += ") \n";
  result_str += _condition->to_string("    ");
  result_str += "\n";

  return result_str;
}

}  // namespace fsmcore
}  // namespace hozon
