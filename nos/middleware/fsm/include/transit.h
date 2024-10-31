#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "action.h"
#include "condition.h"
#include "proto/statemachine/fsm_rule.pb.h"
#include "state.h"

namespace hozon {
namespace fsmcore {

/*****************************************************************************/
/* Used to descript defined transit                                          */
/*****************************************************************************/
class Transit {
 public:
  Transit(const hozon::fsm_rule::FsmTransit& fsm_transit,
          FindStateFunc find_state_func,
          FindActionCoreFunc find_action_func,
          FindConditionCoreFunc find_condition_func);

  bool check_legal();
  std::string to_string();
  std::shared_ptr<State> get_from() { return _from; }
  std::shared_ptr<State> get_to() { return _to; }
  std::shared_ptr<Condition> get_condition() { return _condition; }
  std::vector<std::shared_ptr<Action>> get_actions() { return _actions; }

 private:
  std::shared_ptr<State> _from;
  std::shared_ptr<State> _to;
  std::shared_ptr<Condition> _condition;
  std::vector<std::shared_ptr<Action>> _actions;
};

}  // namespace fsmcore
}  // namespace hozon