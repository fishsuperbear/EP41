#include "action.h"

#include "fsm.h"
#include "fsm_utils.h"

namespace hozon {
namespace fsmcore {

///////////////////////////////////////////////////////////////////////////////

/*****************************************************************************/
/* 构造函数 */
/*****************************************************************************/
ActionCore::ActionCore(const std::string& name, const DoActionFunc& func)
    : _core_name(name) {
  _do_action = std::make_unique<DoActionFunc>(func);
}

void ActionCore::do_action(const std::vector<std::string>& input_param) {
  if (!NULL_IPTR(_do_action)) {
    auto func = _do_action.get();
    (*func)(input_param);
  }
}

///////////////////////////////////////////////////////////////////////////////

/*****************************************************************************/
/* 构造函数 */
/*****************************************************************************/
Action::Action(const hozon::fsm_rule::FsmAction& fsm_action,
               FindActionCoreFunc find_func, bool from_transit)
    : _transit_member(from_transit) {
  for (auto param : fsm_action.params()) {
    _input_params.push_back(param);
  }

  auto core_name = fsm_action.name();
  _action_core = find_func(core_name);

  if (NULL_IPTR(_action_core)) {
    FSMCORE_LOG_FATAL << "Cannot find action core with core name: "
                      << core_name;
    exit(-1);
  }
}

/*****************************************************************************/
/* 检查合法性 */
/*****************************************************************************/
bool Action::check_legal() { return NULL_IPTR(_action_core) ? false : true; }

/*****************************************************************************/
/* 输出 action 的打印 */
/*****************************************************************************/
std::string Action::to_string() {
  std::string result_str;

  if (!NULL_IPTR(_action_core)) {
    result_str += _action_core->get_name();
  }
  if (_transit_member) {
    result_str += " is from transit: ";
  } else {
    result_str += " is from state: ";
  }
  for (auto&& param : _input_params) {
    result_str += "+";
    result_str += param;
  }

  return result_str;
}

/*****************************************************************************/
/* 动作执行函数 */
/*****************************************************************************/
void Action::do_action() {
  if (!NULL_IPTR(_action_core)) {
    _action_core->do_action(_input_params);
    FSMCORE_LOG_INFO << "Do action " << _action_core->get_name();
  }
}

}  // namespace fsmcore
}  // namespace hozon
