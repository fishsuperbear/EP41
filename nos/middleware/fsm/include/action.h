/*****************************************************************************/
/* Copyright [2023] <shenlinsen:80042544>                                    */
/*****************************************************************************/
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "proto/statemachine/fsm_rule.pb.h"

namespace hozon {
namespace fsmcore {

class ActionCore;

using DoActionFunc =
    std::function<void(const std::vector<std::string>&)>;
using FindActionCoreFunc =
    std::function<std::shared_ptr<ActionCore>(const std::string& name)>;

/*****************************************************************************/
/* Used to descript defined action                                           */
/*****************************************************************************/
class ActionCore {
 public:
  ActionCore(const std::string& name, const DoActionFunc& func);
  void do_action(const std::vector<std::string>& input_param);
  std::string get_name() { return _core_name; }

 private:
  std::unique_ptr<DoActionFunc> _do_action;
  std::string _core_name;
};

/*****************************************************************************/
/* same ActionCore different input parameter, is different Action            */
/*****************************************************************************/
class Action {
 public:
  Action(const hozon::fsm_rule::FsmAction& fsm_action,
            FindActionCoreFunc find_func, bool from_transit = false);

  bool check_legal();
  std::string to_string();
  void do_action();

 private:
  bool _transit_member;
  std::vector<std::string> _input_params;
  std::shared_ptr<ActionCore> _action_core;
};

}  // namespace fsmcore
}  // namespace hozon
