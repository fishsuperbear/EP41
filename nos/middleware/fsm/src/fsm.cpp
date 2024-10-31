#include "fsm.h"

#include <mutex>
#include <utility>

#include "fsm_utils.h"
#include "string_utils.h"

namespace hozon {
namespace fsmcore {

/*****************************************************************************/
/* called by Fsm::Start()                                                    */
/*****************************************************************************/
int32_t Fsm::AlgInit() {
  auto yaml_cfg = GetConfig();
  auto send_inst = yaml_cfg.send_instance_configs;
  for (auto& obj : send_inst) {
    if (obj.name == "output_fsmcore") {
      auto ret = _output_writer.Init(0, obj.topic);
      if (ret < 0) {
        FSMCORE_LOG_ERROR << "Fail to init fsmcore_output writer " << ret
                          << ", topic: " << obj.topic;
        return (-1);
      } else {
        FSMCORE_LOG_INFO << "Create Fsmcore Output writer, topic name: "
                         << obj.topic;
      }
    }
    if (obj.name == "output_canfd") {
      auto ret = _canfd_writer.Init(0, obj.topic);
      if (ret < 0) {
        FSMCORE_LOG_ERROR << "Fail to init canfd writer " << ret << ", topic: " << obj.topic;
        return (-1);
      } else {
        FSMCORE_LOG_INFO << "Create canfd Output writer, topic name: "
                         << obj.topic;
      }
    }
  }

  /***************************************************************************/
  /* 拉起状态机的消息，目前就是哨兵模式在用                                        */
  /**************************************************************************/
  auto ret = _statemachine_writer.Init(0, "call_perception");
  if (ret < 0) {
    FSMCORE_LOG_ERROR << "Fail to init call_perception writer " << ret;
    return (-1);
  }

  // 定时执行 process 函数，周期配在 period trigger 里面
  // 要求先执行 Fsm::init，再执行本函数，即先执行状态机 init，再执行 NodeBase 的
  // Start 函数
  RegistAlgProcessFunc("perioical",
                       std::bind(&Fsm::process, this, std::placeholders::_1));

  bool b_ret = parse_config();
  if (b_ret) {
    FSMCORE_LOG_INFO << "Fsm parse_config return success.";
    return 0;
  } else {
    FSMCORE_LOG_ERROR << "Fsm config file has some error, please check.";
    return (-1);
  }
}

void Fsm::AlgRelease() { _in_releasing.store(true); }

/*****************************************************************************/
/* init fsm, include :                                                       */
/*  1. init all condition name & function                                    */
/*  2. init all action name & functional                                     */
/*  3. read and parse config file                                            */
/*****************************************************************************/
bool Fsm::init(const std::string& cfg_path) {
  _fsm_path = cfg_path;
  auto phm_path = _fsm_path;
  auto str_vec = hozon::fsmcore::split(phm_path, "/");
  if (str_vec.size() > 0) {
    str_vec = hozon::fsmcore::split(phm_path, str_vec[str_vec.size() - 1]);
    if (str_vec.size() > 0) {
      phm_path = str_vec[0] + "/NodeConfig.yaml";
    } else {
      FSMCORE_LOG_ERROR << "Invalid input fsm_path: " << _fsm_path;
      phm_path = "./NodeConfig.yaml";
    }
  } else {
    phm_path = "./NodeConfig.yaml";
  }

  _phm_available.store(false);

  // 配置所有的 condition 和 action，需要注意这些 condition 和 action
  // 必须是全集，大于 config 文件中引用的 condition 和 action
  define_conditions();
  define_actions();

  auto ret = get_proto_from_file(cfg_path, &_fsm_rule);
  if (!ret) {
    FSMCORE_LOG_FATAL << "fsm config file is error, path " << cfg_path;
    return false;
  }

  // 故障暂未实现
#if 0
  // 故障监控，状态机核心处理，检测哪些故障，在业务 yaml 文件中配置
  auto avail_cb = std::bind(&Fsm::phm_service_available_callback, this,
                            std::placeholders::_1);
  auto fault_cb =
      std::bind(&Fsm::phm_receive_fault_callbak, this, std::placeholders::_1);
  _phm_client = std::make_shared<hozon::netaos::phm::PHMClient>();
  _phm_client->Init(phm_path, avail_cb, fault_cb);
#endif

  return true;
}

/*****************************************************************************/
/* parse fsm config file :                                                   */
/*  1. reading state:                    repeated FsmState states = 2;       */
/*  2. reading transit:                  repeated FsmTransit transits = 3;   */
/*  3. reading init state and check      required StateId init_state = 4;    */
/*****************************************************************************/
bool Fsm::parse_config() {
  std::vector<std::shared_ptr<State>> top_states;

  // add max_transit count
  _max_transit_count = _fsm_rule.max_transit_count();

  // add top level state, sub level state will be add in recursive
  for (int ii = 0; ii < _fsm_rule.states_size(); ++ii) {
    top_states.emplace_back(
        std::make_shared<State>(_fsm_rule.states(ii), _find_actioncore_func));
  }

  // check top state all legal
  for (auto&& top_state : top_states) {
    if (!top_state->check_legal()) {
      FSMCORE_LOG_ERROR << "State " << top_state->get_name() << "("
                        << top_state->get_level() << ") is illegal.";
      return false;
    }
  }

  // because sub state added in recursive, we should re-arrange to _state
  for (auto&& top_state : top_states) {
    std::shared_ptr<State> null_state;
    top_state->set_parent(null_state);
    auto ret = rearrange_state_to_levels(top_state);
    if (!ret) {
      FSMCORE_LOG_ERROR << "duplicate state config.";
      return false;
    }
  }

  // add transit
  for (int ii = 0; ii < _fsm_rule.transits_size(); ++ii) {
    _transits.emplace_back(std::make_shared<Transit>(
        _fsm_rule.transits(ii), _find_state_func, _find_actioncore_func,
        _find_conditioncore_func));
  }

  // check all transits
  for (auto&& one_transit : _transits) {
    auto ret = one_transit->check_legal();
    if (!ret) {
      FSMCORE_LOG_ERROR << "illegal transit finded.";
      return false;
    }
  }

  // add init state
  auto init_state = find_state(_fsm_rule.init_state());
  if (NULL_IPTR(init_state)) {
    FSMCORE_LOG_ERROR
        << "init state has not been defined in repeated FsmState states = 2";
    return false;
  }
  if (!init_state->is_groud_level()) {
    FSMCORE_LOG_ERROR
        << "illegal init state finded, which should be groud level, "
           "state name: "
        << init_state->get_name() << ".";
    return false;
  }

  // force fsm to init state
  change_state(init_state);

  // print all states
  for (auto&& top_state : top_states) {
    std::cout << "State: " << std::endl;
    std::cout << top_state->to_string("  ");
  }
  // print state arranged by level
  for (auto&& level : _states) {
    std::string str = "  level: ";
    str += std::to_string(level.first);
    str += ":";
    for (auto&& it : level.second) {
      str += " ";
      str += it.first;
    }
    std::cout << "Level: \n" << str << std::endl;
  }
  // print all transit
  for (auto transit : _transits) {
    std::cout << "Transit: \n" << transit->to_string();
  }
  // print init state
  std::cout << "Init state: " << init_state->get_name()
            << "(" + std::to_string(init_state->get_level()) << ")";
  std::cout << std::endl;

  return true;
}

/*****************************************************************************/
/* state has it's substate, should re-arrange in recursive                   */
/*****************************************************************************/
bool Fsm::rearrange_state_to_levels(std::shared_ptr<State> state) {
  auto level = state->get_level();
  auto name = state->get_name();

  if (_states.find(level) == _states.end()) {
    _states[level][name] = state;
  } else {
    auto level_states = _states[level];
    if (level_states.find(name) == level_states.end()) {
      _states[level][name] = state;
    } else {
      FSMCORE_LOG_ERROR << "duplicate configure of state, state name " << name
                        << ", level " << level;
      return false;
    }
  }

  // process sub state recursively
  for (auto&& sub : state->get_substates()) {
    auto ret = rearrange_state_to_levels(sub);
    if (!ret) {
      return false;
    }
  }

  return true;
}

/*****************************************************************************/
/* state changed from _prev_state to new_state, and update _curr_state       */
/*****************************************************************************/
void Fsm::change_state(const std::shared_ptr<State>& new_state) {
  if (!new_state->is_groud_level()) {
    FSMCORE_LOG_FATAL
        << "This should not happend, because load transit config already "
           "checked.";
    return;
  }

  // update preview and current state
  _prev_state = _curr_state;
  _curr_state = new_state;
  _curr_state_us = now_usec();
  _curr_state_frame = 0;

  // print from change to
  if (_prev_state.get()) {
    FSMCORE_LOG_ERROR << "#### change state from " << _prev_state->get_name()
                      << "(" << std::to_string(_prev_state->get_level())
                      << ") to " << _curr_state->get_name() << "("
                      << std::to_string(_curr_state->get_level()) << ") ####";
  } else {
    FSMCORE_LOG_ERROR << "#### change state from null to "
                      << _curr_state->get_name() << "("
                      << std::to_string(_curr_state->get_level()) << ") ####";
  }
  // change to new state need to do enter actions
  do_enter_actions();
}

/*****************************************************************************/
/* fill fsm state in recursive                                               */
/*****************************************************************************/
void Fsm::fill_state_stack(const std::shared_ptr<State>& state,
                           hozon::fsm_output::StateStack& state_stack) {
  auto state_id = state_stack.mutable_state_id();
  (*state_id).set_level(state->get_level());
  (*state_id).set_name(state->get_name());

  auto parent = state->get_parent();
  if (!NULL_IPTR(parent)) {
    auto recursiv_state_stack = state_stack.mutable_parent_state();
    fill_state_stack(parent, *recursiv_state_stack);
  }
}

/*****************************************************************************/
/* publish fsm state, once process() called, publish_fsm_state is call once  */
/* at least                                                                  */
/*****************************************************************************/
void Fsm::publish_fsm_state() {
  hozon::fsm_output::StateOutput output;
  output.set_into_time(now_usec());

  auto state_stack = output.mutable_stack();
  fill_state_stack(_curr_state, *state_stack);

  _output_writer.Write(output);
#if 0
  // MDC not support publish a protobuf data, here just print protobuf to string
  std::cout << "========Start Of " << _fsm_path << ":" << std::endl
            << output.DebugString() << std::endl
            << "End Of " << _fsm_path << "========" << std::endl;
#endif
}

/*****************************************************************************/
/* leaving _prev_state and entering _curr_state, do:                         */
/*   1. _curr_state enter_actions                                            */
/*   2. _curr_state parent enter_actions in recursive if _prev_state has     */
/*      different parent                                                     */
/*****************************************************************************/
void Fsm::do_enter_actions() {
  std::shared_ptr<State> curr;
  std::shared_ptr<State> prev;
  curr = _curr_state;
  prev = _prev_state;

  while (!NULL_IPTR(curr)) {
    // have same parent, donot execute same parent enter_actions
    if (SAME_ELEMENT_OF_IPTR(prev, curr)) {
      break;
    }

    auto enter_actions = curr->get_enter_actions();
    for (auto&& action : enter_actions) {
      action->do_action();
    }

    // get their parent
    curr = curr->get_parent();

    // if prev does not has parent, no need get parent
    if (!NULL_IPTR(prev)) {
      prev = prev->get_parent();
    }
  }
}

/*****************************************************************************/
/* leaving _curr_state, and entering new_state, do:                          */
/*   1. _curr_state exit_actions                                             */
/*   2. _curr_state parent exit_actions in recursive if new_state has        */
/*      different parent                                                     */
/*****************************************************************************/
void Fsm::do_exit_actions(const std::shared_ptr<State>& new_state) {
  std::shared_ptr<State> curr;
  std::shared_ptr<State> newly;
  curr = _curr_state;
  newly = new_state;

  while (!NULL_IPTR(curr)) {
    // have same parent, donot execute same parent exit_actions
    if (SAME_ELEMENT_OF_IPTR(newly, curr)) {
      break;
    }

    for (auto&& obj : curr->get_exit_actions()) {
      obj->do_action();
    }

    // get their parent
    curr = curr->get_parent();
    if (!NULL_IPTR(newly)) {
      newly = newly->get_parent();
    }
  }
}

/*****************************************************************************/
/* transit satify condition, do it's action                                  */
/*****************************************************************************/
void Fsm::do_transit_actions(const std::shared_ptr<Transit>& transit) {
  auto&& actions = transit->get_actions();
  for (auto&& act : actions) {
    act->do_action();
  }
}

/*****************************************************************************/
/* get fsm name, for example: parking fsm, top fsm ...                       */
/*****************************************************************************/
std::string Fsm::get_fsm_name() {
  if (_fsm_rule.has_name()) {
    return _fsm_rule.name();
  }

  return std::string("");
}

/*****************************************************************************/
/* get fsm state name, for example: off, selfcheck ...                       */
/*****************************************************************************/
std::shared_ptr<State> Fsm::find_state(const hozon::fsm_rule::StateId& id) {
  std::shared_ptr<State> null_state;
  auto name = id.name();

  if (id.has_level()) {
    auto level = id.level();

    if (_states.find(level) == _states.end()) {
      FSMCORE_LOG_ERROR << "state " << id.name()
                        << " has wrong level: " << level;
      return null_state;
    }

    auto level_states = _states[level];
    if (level_states.find(name) == level_states.end()) {
      FSMCORE_LOG_ERROR << "state name: " << id.name() << " cannot find.";
      return null_state;
    }

    return level_states[id.name()];
  }

  FSMCORE_LOG_WARN << "not appoint a state level, may has more than one state "
                      "which has a same name.";

  for (auto&& level_states : _states) {
    for (auto&& specific_state : level_states.second) {
      if (specific_state.first == name) {
        return specific_state.second;
      }
    }
  }

  return null_state;
}

/*****************************************************************************/
/* process is terminate for 2 kinds of situations:                           */
/*   1. no transit can be performed                                          */
/*   2. transit to many times exceeding _max_transit_count                   */
/*****************************************************************************/
int Fsm::process(NodeBundle* input) {
  _node_ctx.reset(input);
  auto counter = _max_transit_count;
  bool alread_publish = false;

  // 最多跳转计数器, case 2
  while (counter > 0) {
    auto once_ret = process_once();

    // 完成一次跳转，不需要本函数再输出状态机状态
    if (once_ret) {
      alread_publish = true;
    } else {  // 无法完成调转，结束本次 process, case 1
      break;
    }

    --counter;
  }

  // 到达最大跳转次数，记录一次日志
  if (!counter) {
    FSMCORE_LOG_ERROR << "Fsm transit reached configured max times "
                      << _max_transit_count;
  }

  // 一次 process 最少要 publish 一次 fsm state
  if (!alread_publish) {
    publish_fsm_state();
  }

  ++_curr_state_frame;

  // 不能直接 reset，否则 unique_ptr 会释放 input 对象；
  _node_ctx.release();

  return 0;
}

/*****************************************************************************/
/* process_once                                                              */
/*****************************************************************************/
bool Fsm::process_once() {
  auto state = _curr_state;

  // 从当前状态开始，逐渐往父状态查找
  while (!NULL_IPTR(state)) {
    // 从当前状态开始，检查所有以当前状态作为 from 的 transit
    for (auto&& transit : _transits) {
      if (SAME_ELEMENT_OF_IPTR(transit->get_from(), state)) {
        FSMCORE_LOG_ERROR << "==== Checking " << state->get_name() << " ----> "
                          << transit->get_to()->get_name() << " ====";
        // 条件满足，需要做四件事情
        if (transit->get_condition()->satisfy()) {
          // 第一件事情: 执行本跳转的动作
          do_transit_actions(transit);

          // 判断是否是自己跳到自己，如果是，只做 transit
          // action，不切状态，直至超出 _max_transit_count
          if (SAME_ELEMENT_OF_IPTR(transit->get_to(), state)) {
            FSMCORE_LOG_WARN << "transit from self to self, self state: "
                             << state->get_name();
            continue;
          }

          // 第二件事情，执行退出当前状态的动作
          do_exit_actions(transit->get_to());

          // 第三件事情，执行状态更新
          change_state(transit->get_to());

          // 第四件事情，输出当前状态
          publish_fsm_state();

          // 只要有一次跳转成功，本函数范围，自己跳到自己成功不算
          return true;
        }
      }
    }

    state = state->get_parent();
  }

  return false;
}

/*****************************************************************************/
/* 故障管理服务端可用回调 */
/*****************************************************************************/
void Fsm::phm_service_available_callback(bool bResult) {
  _phm_available.store(true);
}

/*****************************************************************************/
/* 故障管理上报故障产生、消失接口 */
/*****************************************************************************/
void Fsm::phm_receive_fault_callbak(const ReceiveFault_t& fault) {
  if (!_phm_available.load()) {
    return;
  }

  uint32_t fault_id = fault.faultId;
  uint8_t fault_obj = fault.faultObj;
  uint8_t fault_state = fault.faultStatus;

  std::lock_guard<std::mutex> lg(_fault_set.mtx);
  auto tmp_pair = std::make_pair(fault_id, fault_obj);
  auto one_set = _fault_set.fault_on.find(tmp_pair);

  // 故障消除
  if (0 == fault_state) {
    if (one_set == _fault_set.fault_on.end()) {
      FSMCORE_LOG_ERROR << "Faultid: " << fault_id << ", objid: " << fault_obj
                        << " has never occur.";
    } else {
      _fault_set.fault_on.erase(one_set);
    }
  } else if (1 == fault_state) {
    if (one_set == _fault_set.fault_on.end()) {
      _fault_set.fault_on.insert(tmp_pair);
    } else {
      FSMCORE_LOG_ERROR << "Faultid: " << fault_id << ", objid: " << fault_obj
                        << " has been in alarm.";
    }
  }

  return;
}

}  // namespace fsmcore
}  // namespace hozon
