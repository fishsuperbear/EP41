#include <memory>
#include <mutex>
#include <unordered_map>
#include "data_types/common/types.h"
#include "fsm.h"
#include "fsm_utils.h"
#include "node_base.h"
#include "node_bundle.h"
#include "proto/soc/chassis.pb.h"

namespace hozon {
namespace fsmcore {

/*****************************************************************************/
/* set out a counterdown timer                                               */
/*****************************************************************************/
int32_t Fsm::set_counterdown_timer(const std::string& name,
                                   uint32_t milliseconds) {
  if (_counterdown_timers.find(name) != _counterdown_timers.end()) {
    FSMCORE_LOG_WARN << "Counterdown timer " << name
                     << "has alread exist, clear and reset.";
    _counterdown_timers.erase(name);
  }

  std::shared_ptr<CounterdownTimer> timer =
      std::make_shared<CounterdownTimer>(name, milliseconds);
  timer->settering();
  _counterdown_timers[name] = timer;

  return 0;
}

/*****************************************************************************/
/* cancel a counterdown timer                                                */
/*****************************************************************************/
int32_t Fsm::cancel_counterdown_timer(const std::string& name) {
  if (_counterdown_timers.find(name) == _counterdown_timers.end()) {
    FSMCORE_LOG_WARN << "Counterdown timer " << name
                     << " no exist, do nothing.";
    return (-1);
  }

  // 取消倒计时器
  get_counterdown_timer(name)->canceling();
  FSMCORE_LOG_INFO << "Counterdown timer " << name << " is canceled.";

  return 0;
}

/*****************************************************************************/
/* save data to Fsm::_data_map, be careful: this need to lock for read/write */
/*****************************************************************************/
int32_t Fsm::save_data_callback(NodeBundle* node_bundle,
                                const std::string& topic_name,
                                uint32_t interval) {
  auto received_data = node_bundle->GetOne(topic_name);
  if (NULL_IPTR(received_data)) {
    FSMCORE_LOG_ERROR << "GetOne " << topic_name << " to be a null pointer.";
    return -1;
  }

  std::lock_guard<std::mutex> lg(_data_map.mtx);
  std::unordered_map<std::string, std::vector<BaseDataTypePtr>>& data_map =
      _data_map.data_map;

  if (data_map.find(topic_name) == data_map.end()) {
    std::vector<BaseDataTypePtr> datas;
    datas.push_back(received_data);
    data_map[topic_name] = datas;
  } else {
    auto& datas = data_map[topic_name];
    datas.push_back(received_data);

    uint64_t interval_us = (uint64_t)interval * 1000;
    uint64_t latest_us = received_data->__header.timestamp_real_us;

    /*************************************************************************/
    /* delete front data which is to older than interval                     */
    /*************************************************************************/
    while ((datas.size() > 1) &&
           (latest_us - datas[0]->__header.timestamp_real_us > interval_us)) {
      datas.erase(datas.begin());
    }
  }

  return 0;
}

/*****************************************************************************/
/* 根据禁止标记优先级，转换成 cdcs 的 forbidden sign */
/*****************************************************************************/
uint32_t Fsm::convert_forbidden_sign_to_cdcs(uint32_t priority) {
  enum TsrSubType {
    TSR_SUB_TYPE_UNKNOWN = 0,

    SPEED_MIN_LIMIT = 6,  // 最低限速标记
    SPEED_MAX_LIMIT = 7,  // 最高限速标志
    SPEED_NO_LIMIT = 8,   // 解除限速标志

    NO_OVER_TAKING = 40,          // 禁止超车
    NO_STRAIGHT = 41,             // 禁止直行
    NO_STRAIGHT_TURN_RIGHT = 42,  // 禁止直行和右转
    NO_STRAIGHT_TURN_LEFT = 43,   // 禁止直行和左转
    NO_TURN_RIGHT = 46,           // 禁止右转
    NO_TURN_LEFT_RIGHT = 48,      // 禁止左右转
    NO_TURN_LEFT = 51,            // 禁止左转
    NO_TURN_AROUND = 70,          // 禁止调头
    NO_PASSING = 82,              // 禁止通行
    NO_PARKING = 176,             // 禁止停车
    NO_ENTER = 177,               // 禁止进入
  };

  enum CDCSForbiddenSignType {
    CDCS_NO_FORBIDDEN_SIGN = 0,
    CDCS_NO_TURN_LEFT = 1,    // 禁止左转
    CDCS_NO_TURN_RIGHT,       // 禁止右转
    CDCS_NO_STRAIGHT,         // 禁止直行 gsc* 这个minieye没有的
    CDCS_NO_TURN_LEFT_RIGHT,  // 禁止左右转
    CDCS_NO_PARKING,          // 禁止停车
    CDCS_NO_ENTER,            // 禁止进入
    CDCS_NO_TURN_AROUND,      // 禁止调头
    CDCS_NO_PASSING,          // 禁止通行
  };

  static std::unordered_map<TsrSubType, uint32_t> sub_type_priority_map;
  static std::unordered_map<TsrSubType, CDCSForbiddenSignType>
      cdcs_forbidden_sin_map;

  sub_type_priority_map[TsrSubType::NO_PASSING] = 0;
  sub_type_priority_map[TsrSubType::NO_ENTER] = 1;
  sub_type_priority_map[TsrSubType::NO_TURN_LEFT] = 2;
  sub_type_priority_map[TsrSubType::NO_TURN_RIGHT] = 3;
  sub_type_priority_map[TsrSubType::NO_TURN_LEFT_RIGHT] = 4;
  sub_type_priority_map[TsrSubType::NO_TURN_AROUND] = 5;
  sub_type_priority_map[TsrSubType::NO_PARKING] = 6;

  cdcs_forbidden_sin_map[TsrSubType::NO_TURN_RIGHT] = CDCS_NO_TURN_RIGHT;
  cdcs_forbidden_sin_map[TsrSubType::NO_TURN_LEFT_RIGHT] =
      CDCS_NO_TURN_LEFT_RIGHT;
  cdcs_forbidden_sin_map[TsrSubType::NO_TURN_LEFT] = CDCS_NO_TURN_LEFT;
  cdcs_forbidden_sin_map[TsrSubType::NO_TURN_AROUND] = CDCS_NO_TURN_AROUND;
  cdcs_forbidden_sin_map[TsrSubType::NO_PASSING] = CDCS_NO_PASSING;
  cdcs_forbidden_sin_map[TsrSubType::NO_PARKING] = CDCS_NO_PARKING;
  cdcs_forbidden_sin_map[TsrSubType::NO_ENTER] = CDCS_NO_ENTER;

  TsrSubType forbidden_sign = TsrSubType::TSR_SUB_TYPE_UNKNOWN;

  for (auto&& obj : sub_type_priority_map) {
    if (obj.second == priority) {
      forbidden_sign = obj.first;
      break;
    }
  }

  if (forbidden_sign != TsrSubType::TSR_SUB_TYPE_UNKNOWN) {
    return cdcs_forbidden_sin_map[forbidden_sign];
  }

  FSMCORE_LOG_ERROR << "Invalid forbidden priority: " << priority;
  return std::numeric_limits<uint32_t>::max();
}

/*****************************************************************************/
/* 根据禁止标记优先级，转换成 cdcs 的 forbidden sign */
/*****************************************************************************/
int32_t Fsm::convert_speed_sign_to_cdcs(int32_t speed_sign) {
  static std::unordered_map<int32_t, int32_t> cdcs_speed_map = {
      {-1, -1},  {5, 0x1},   {10, 0x2},  {15, 0x3}, {20, 0x4}, {30, 0x5},
      {35, 0x6}, {40, 0x7},  {50, 0x8},  {60, 0x9}, {70, 0xA}, {80, 0xB},
      {90, 0xC}, {100, 0xD}, {110, 0xE}, {120, 0xF}};

  if (cdcs_speed_map.find(speed_sign) != cdcs_speed_map.end()) {
    return cdcs_speed_map[speed_sign];
  }

  return 0;
}

/*****************************************************************************/
/* 周期性将 tsr、tlr 数据，发送给 can */
/*****************************************************************************/
void Fsm::tsrtlr_send_can_message_thread(
    const std::vector<std::string>& params) {
#define CIRCLE_INTERVAL 100
  bool is_switch_on;
  bool is_in_warning;
  uint32_t cdcs_forbidden;
  int32_t cdcs_speed;
  int32_t speed_sign;
  uint32_t tlr_value = 0x3000000;
  uint32_t forbidden_priority;
  std::chrono::time_point<std::chrono::system_clock> last_warning_time;
  std::unordered_map<uint32_t, uint32_t> color_map = {{0, 0}, {1, 3}, {2, 1}, {3, 2}};

  if (params.size() < 2) {
    FSMCORE_LOG_ERROR << "Invalid input parameters count for "
                         "tsrtlr_send_can_message_thread, should be two.";
    return;
  }

  auto speed_warn_interval = std::stoi(params[0]);
  auto speed_warn_keep_max = std::stoi(params[1]);

  // 定时任务 periodic 没有主动退，就 100 毫秒发送一次 can
  while (!_in_releasing.load()) {
    // 这个条件理论上不会走到
    if (NULL_IPTR(_curr_state)) {
      FSMCORE_LOG_ERROR << "[TSR ERROR], Fsm has Null curr_state.";
      std::this_thread::sleep_for(std::chrono::milliseconds(CIRCLE_INTERVAL));
      continue;
    }

    auto state_name = _curr_state->get_name();
    if ("kOff" == state_name) {
      is_switch_on = false;
      is_in_warning = false;
      cdcs_forbidden = 0;
      cdcs_speed = 0;
    } else if (("kStandby" == state_name) || ("kSystemFailure" == state_name)) {
      is_switch_on = true;
      is_in_warning = false;
      cdcs_forbidden = 0;
      cdcs_speed = 0;
    } else {
      is_switch_on = true;
      is_in_warning = false;

      // 计算禁止标记
      auto ret = get_forbidden_sign_from_berkeley_db(forbidden_priority);
      if (ret) {
        cdcs_forbidden = convert_forbidden_sign_to_cdcs(forbidden_priority);
        FSMCORE_LOG_INFO << "Getted Forbidden Sign priority: "
                         << forbidden_priority
                         << ", CDCS forbidden code: " << cdcs_forbidden;
        if (cdcs_forbidden == std::numeric_limits<uint32_t>::max()) {
          cdcs_forbidden = 0;
        }
      } else {
        FSMCORE_LOG_INFO << "Get No Forbbidden Sign from bdb.";
        cdcs_forbidden = 0;
      }

      // 计算红绿灯
      ret = get_tlr_value_from_berkeley_db(tlr_value);
      if (!ret) {
        tlr_value = 0x3000000;  // 取不到红绿灯，给个特殊标记，表示没识别到
        FSMCORE_LOG_ERROR << "Get Tlr value Failed, Ret: " << ret;
      } else {
        FSMCORE_LOG_INFO << "Get Tlr value Success, value: " << tlr_value;
      }

      /***********************************************************************/
      /* 红绿灯在如下两种情况下，不加以显示 ： */
      /*   1. 车速大于 80kph              */
      /*   2. nnp 激活                   */
      /***********************************************************************/
      int32_t speed;
      ret = get_speed_from_berkeley_db(speed);
      if (ret && speed > 80) {
        FSMCORE_LOG_INFO << "Speed more than 80kpm, not display speed sign.";
        tlr_value = 0x3000000;  // 不显示红绿灯
      }

      uint32_t nnp_state;
      ret = get_nnp_state_from_berkeley_db(nnp_state);
      if (ret &&
          ((nnp_state == 0x2) || (nnp_state == 0x3) || (nnp_state == 0x4) ||
           (nnp_state == 0x5) || (nnp_state == 0x9) || (nnp_state == 0x10))) {
        FSMCORE_LOG_INFO << "In NNP state, nnp_state " << nnp_state
                         << ", not display speed sign.";
        tlr_value = 0x3000000;  // 不显示红绿灯
      }

      // 如果是 500 米推测限速，限速需要取有效限速牌
      if (!NULL_IPTR(_curr_state->get_parent()) &&
          ("SpeedLimitRidding" == _curr_state->get_parent()->get_name())) {
        ret = get_speed_sign_valid_from_berkeley_db(speed_sign);
      } else {
        ret = get_speed_sign_from_berkeley_db(speed_sign);
      }
      if (!ret) {
        FSMCORE_LOG_ERROR << "speed sign or speed sign valid from bdb "
                             "failed, speed_sign: "
                          << speed_sign;
        speed_sign = 0;
      }
      cdcs_speed = convert_speed_sign_to_cdcs(speed_sign);

      /***********************************************************************/
      /* 处理告警状态 */
      /***********************************************************************/
      if (("SpeedWarning" == state_name) || ("SpeedWarningR" == state_name)) {
        // 计算告警状态
        auto cur_time = std::chrono::system_clock::now();
        auto eclipsed_time =
            std::chrono::duration<double>(cur_time - last_warning_time).count();
        if (eclipsed_time > speed_warn_interval) {
          is_in_warning = true;
          last_warning_time = cur_time;
        } else {
          is_in_warning = eclipsed_time > speed_warn_keep_max ? false : true;
        }
      }
    }

    // 红绿灯通行情况
    uint32_t turn_left;
    uint32_t turn_straight;
    uint32_t turn_right;

    if (tlr_value >= 0x3000000) {
      turn_left = 0;
      turn_straight = 0;
      turn_right = 0;
    } else {
      turn_right = tlr_value & 0xff;
      turn_straight = tlr_value & 0xff00;
      turn_left = tlr_value & 0xff0000;

      turn_right = turn_right >> 0;
      turn_straight = turn_straight >> 8;
      turn_left = turn_left >> 16;

      turn_right = color_map[turn_right];
      turn_straight = color_map[turn_straight];
      turn_left = color_map[turn_left];
    }

    hozon::soc::Apa2Chassis canfd_msg;
    auto msg_fe = canfd_msg.mutable_canfd_msg233();
    msg_fe->set_soc_tsr_state(is_switch_on);
    msg_fe->set_soc_tsr_speed_warn_state(is_in_warning);
    msg_fe->set_soc_tsr_forbidden_sign(cdcs_forbidden);
    msg_fe->set_soc_tsr_speed_sign(cdcs_speed);
    msg_fe->set_soc_tsr_system_fault_status(0);
    msg_fe->set_soc_tsr_limit_overspeed_set(0);
    msg_fe->set_soc_tsr_left_light_color(turn_left);
    msg_fe->set_soc_tsr_str_light_color(turn_straight);
    msg_fe->set_soc_tsr_right_light_color(turn_right);

    _canfd_writer.Write(canfd_msg);

    // 等 100ms 进行下一轮发送
    std::this_thread::sleep_for(std::chrono::milliseconds(CIRCLE_INTERVAL));
  }

  FSMCORE_LOG_ERROR << "tsrtlr_send_can_message_thread is terminated.";
}

/*****************************************************************************/
/* 事件性将感知 mod 的检测结果发送给 can  */
/*****************************************************************************/
int32_t Fsm::mod_send_can_message(NodeBundle* node_bundle) {
  // 必须是状态机在 PerceptionOd 状态，才向 can 上发消息
  if (_curr_state->get_name() != "PerceptionOd") {
    return 0;
  }

  auto perception_data = node_bundle->GetOne("perception_workingstatus");
  if (NULL_IPTR(perception_data)) {
    FSMCORE_LOG_ERROR
        << "GetOne perception_workingstatus to be a null pointer.";
    return -1;
  }

  auto chassis_data = node_bundle->GetOne("chassis");
  if (NULL_IPTR(chassis_data)) {
    FSMCORE_LOG_ERROR << "GetOne chassis to be a null pointer.";
    return -1;
  }

  // 获取感知 pb 和 chassis pb
  auto perception_pb = std::static_pointer_cast<hozon::state::StateMachine>(
      perception_data->proto_msg);
  auto chassis_pb =
      std::static_pointer_cast<hozon::soc::Chassis>(chassis_data->proto_msg);

  // 检查车门是否有打开，如果车门打开，直接退出
  const auto& door_status = chassis_pb->door_status();
  if (door_status.has_fl_door() &&
      (door_status.fl_door() != hozon::soc::DoorStatus::CLOSED)) {
    FSMCORE_LOG_ERROR << "front left door is open, not send Mod information to can.";
    return 0;
  }

  if (door_status.has_fr_door() &&
      (door_status.fr_door() != hozon::soc::DoorStatus::CLOSED)) {
    FSMCORE_LOG_ERROR << "front right door is open, not send Mod information to can.";
    return 0;
  }

  uint32_t moving_direction = perception_pb->hpp_perception_status().adcs8_mod_object_moving_direction();
  uint32_t mod_warning = perception_pb->hpp_perception_status().adcs8_avm_mod_warning();

  // 创建 can 消息
  hozon::soc::Apa2Chassis canfd_msg;
  auto msg_fe = canfd_msg.mutable_canfd_msg194();
  msg_fe->set_adcs8_mod_object_movingdirection(moving_direction);
  msg_fe->set_adcs8_avm_modwarning(mod_warning);

  _canfd_writer.Write(canfd_msg);
  return 0;
}

/*****************************************************************************/
/* define all actions entrance                                               */
/*****************************************************************************/
void Fsm::define_actions() {
  action_print_log();
  action_set_countdown_timer();
  action_cancel_counterdown_timer();
  action_regist_data_buffering();

  // 通过命令拉起感知
  action_call_perception();

  // 以下动作为哨兵模式专有动作
  action_send_gm_state();
  action_set_gm_start_time();
  action_set_warn_stop_time();

  // 以下动作为 ihbc 专用动作
  action_send_ihbc_state();
  action_save_ihbc_soft_state();

  // 以下动作为 tsrtlr 专用动作
  action_send_tsrtlr_state();
  action_set_no_speed_sign_start_point();
  action_clear_no_speed_sign_start_point();

  // 以下为 mod 专用动作
  action_mod_send_can_message();
}

/*****************************************************************************/
/* print log action                                                          */
/*****************************************************************************/
void Fsm::action_print_log() {
  _action_cores.emplace(
      "print_log", std::make_shared<ActionCore>(
                       "print_log", [](const std::vector<std::string>& params) {
                         for (auto one_str : params) {
                           std::cout << one_str;
                         }
                         std::cout << std::endl;
                       }));
}

/*****************************************************************************/
/* set a counterdown timer action, if same name of counterdown timer exist   */
/* destruct it and reset                                                     */
/*****************************************************************************/
void Fsm::action_set_countdown_timer() {
  _action_cores.emplace(
      "set_counterdown_timer",
      std::make_shared<ActionCore>(
          "set_counterdown_timer", [&](const std::vector<std::string>& params) {
            if (params.size() < 2) {
              FSMCORE_LOG_ERROR << "Invalid input parameters count for "
                                   "set_counterdown_timer action.";
              return;
            }

            auto timer_name = params[0];
            auto milliseconds = std::stoul(params[1]);
            (void)set_counterdown_timer(timer_name, milliseconds);
            return;
          }));
}

/*****************************************************************************/
/* cancel a counterdown timer action, if counterdown timer not exist, do     */
/* nothing                                                                   */
/*****************************************************************************/
void Fsm::action_cancel_counterdown_timer() {
  _action_cores.emplace("cancel_counterdown_timer",
                        std::make_shared<ActionCore>(
                            "cancel_counterdown_timer",
                            [&](const std::vector<std::string>& params) {
                              if (params.size() < 1) {
                                FSMCORE_LOG_ERROR
                                    << "Invalid input parameters count for "
                                       "cancel_counterdown_timer action.";
                                return;
                              }
                              auto timer_name = params[0];
                              (void)cancel_counterdown_timer(timer_name);
                              return;
                            }));
}

/*****************************************************************************/
/* register data receiver , receiver will save data using copy               */
/* remeber : just save data, data processing use period processor            */
/*****************************************************************************/
void Fsm::action_regist_data_buffering() {
  _action_cores.emplace(
      "regist_data_buffering",
      std::make_shared<ActionCore>(
          "regist_data_buffering", [&](const std::vector<std::string>& params) {
            if (params.size() < 3) {
              FSMCORE_LOG_ERROR << "Invalid input parameters count for "
                                   "regist_data_buffering action.";
              return;
            }

            if (!NULL_IPTR(_prev_state)) {
              FSMCORE_LOG_INFO
                  << "save_data_callback can only registered when init state.";
              return;
            }

            auto trigger_name = params[0];
            auto topic_name = params[1];
            auto interval = std::stoi(params[2]);

            RegistAlgProcessFunc(
                trigger_name,
                std::bind(&Fsm::save_data_callback, this, std::placeholders::_1,
                          topic_name, interval));

            FSMCORE_LOG_INFO
                << "Action: regist_data_buffering is run ok, triger name :"
                << trigger_name << ", topic_name: " << topic_name;

            return;
          }));
}

/*****************************************************************************/
/* action_mod_send_can_message              */
/*****************************************************************************/
void Fsm::action_mod_send_can_message() {
  _action_cores.emplace(
      "mod_send_can_message",
      std::make_shared<ActionCore>(
          "mod_send_can_message", [&](const std::vector<std::string>& params) {

            static std::once_flag of;
            std::call_once(of, [&](){
              RegistAlgProcessFunc(
                "can_message_trigger",
                std::bind(&Fsm::mod_send_can_message, this, std::placeholders::_1));
            });

            FSMCORE_LOG_INFO << "Action: mod_send_can_message is run ok";

            return;
          }));
}

/*****************************************************************************/
/* sending output about guard mode state                                     */
/*****************************************************************************/
void Fsm::action_send_gm_state() {
  _action_cores.emplace(
      "send_gm_state",
      std::make_shared<ActionCore>(
          "send_gm_state", [&](const std::vector<std::string>& params) {
            if (params.size() < 3) {
              FSMCORE_LOG_ERROR << "Invalid input parameters count for "
                                   "send_gm_state action, At Least Three, but "
                                << params.size();
              return;
            }

            auto gm_state = std::stoi(params[0]);
            auto gm_workstate = std::stoi(params[1]);
            auto gm_text = std::stoi(params[2]);

            hozon::soc::Apa2Chassis canfd_msg;
            auto msg_fe = canfd_msg.mutable_canfd_msgfe();
            msg_fe->set_adcs4_gmstate(gm_state);
            msg_fe->set_adcs4_gmworkstate(gm_workstate);
            msg_fe->set_adcs4_gms_text(gm_text);

            // 如果有报警内容，增加报警内容输出
            if (params.size() > 3) {
              auto gm_warn = std::stoi(params[3]);
              msg_fe->set_adcs4_gmwarnstate(gm_warn);
            }

            _canfd_writer.Write(canfd_msg);

            return;
          }));
}

/*****************************************************************************/
/* sending output about ihbc state                                           */
/*****************************************************************************/
void Fsm::action_send_ihbc_state() {
  _action_cores.emplace(
      "ihbc_send_can_message",
      std::make_shared<ActionCore>(
          "ihbc_send_can_message", [&](const std::vector<std::string>& params) {
            uint32_t sys_state;
            uint32_t sys_option;

            if (NULL_IPTR(_curr_state)) {
              FSMCORE_LOG_ERROR << "[IHBC ERROR], Fsm has Null curr_state.";
              return;
            }

            auto state_name = _curr_state->get_name();
            // state has according sys_state and sys_option，本版本状态机
            // 比之前感知写的状态机，少一个 standby 状态，故 sys_state = 0x1
            // 没有
            if ("OFF" == state_name) {
              sys_state = 0x0;
              sys_option = 0x0;
            } else if ("HighBeamOff" == state_name) {
              sys_state = 0x2;
              sys_option = 0x1;
            } else if ("HighBeamOn" == state_name) {
              sys_state = 0x3;
              sys_option = 0x1;
            } else if ("ERROR" == state_name) {
              sys_state = 0x4;
              sys_option = 0x1;
            } else {
              FSMCORE_LOG_ERROR << "[IHBC ERROR], Invalid state name: "
                                << state_name;
              return;
            }

            hozon::soc::Apa2Chassis canfd_msg;
            auto msg_fe = canfd_msg.mutable_canfd_msg233();
            msg_fe->set_soc_adas_ihbc_sys_state(sys_state);
            msg_fe->set_soc_adas_ihbc_stat(sys_option);

            _canfd_writer.Write(canfd_msg);

            return;
          }));
}

/*****************************************************************************/
/* start perception with different command to start different functions      */
/*****************************************************************************/
void Fsm::action_call_perception() {
  _action_cores.emplace(
      "call_perception",
      std::make_shared<ActionCore>(
          "call_perception", [&](const std::vector<std::string>& params) {
            if (params.size() < 1) {
              FSMCORE_LOG_ERROR << "Invalid input parameters count for "
                                   "call_perception action, At Least One, but "
                                << params.size();
              return;
            }

            // 从输入参数中，获取 AutopilotStatus.processing_status
            // 从输入参数中，获取 Command.system_command
            auto is_reset = std::stoi(params[0]);
            if (is_reset) {
              hozon::state::StateMachine perception_cmd;
              auto hpp_command = perception_cmd.mutable_hpp_command();
              hpp_command->set_system_reset(is_reset);
              _statemachine_writer.Write(perception_cmd);
              return;
            }

            // construct hozon::state::StateMachine and send
            hozon::state::StateMachine perception_cmd;
            auto hpp_command = perception_cmd.mutable_hpp_command();
            auto system_command = std::stoi(params[1]);
            hpp_command->set_system_command(system_command);

            if (params.size() > 2) {
              auto pilot_status = perception_cmd.mutable_pilot_status();
              auto processing_status = std::stoi(params[2]);
              pilot_status->set_processing_status(processing_status);
            }

            _statemachine_writer.Write(perception_cmd);

            return;
          }));
}

/*****************************************************************************/
/* set gm start time(as perception been work)                                */
/*****************************************************************************/
void Fsm::action_set_gm_start_time() {
  _action_cores.emplace(
      "set_gm_start_time",
      std::make_shared<ActionCore>(
          "set_gm_start_time", [&](const std::vector<std::string>& params) {
            auto nowts = now_usec();
            auto ret = _bdb.set_basic("GmStartTime", nowts, false);

            if (!ret) {
              FSMCORE_LOG_ERROR
                  << "Save now time to Bdb failed, key: GmStartTime.";
            }

            return;
          }));
}

/*****************************************************************************/
/* set alarm stop time                                                       */
/*****************************************************************************/
void Fsm::action_set_warn_stop_time() {
  _action_cores.emplace(
      "set_warn_stop_time",
      std::make_shared<ActionCore>(
          "set_warn_stop_time", [&](const std::vector<std::string>& params) {
            auto nowts = now_usec();
            auto ret = _bdb.set_basic("WarnStopTime", nowts, false);

            if (!ret) {
              FSMCORE_LOG_ERROR
                  << "Save now time to Bdb failed, key: WarnStopTime.";
            }

            return;
          }));
}

/*****************************************************************************/
/* save ihbc soft state to paramserver                                       */
/*****************************************************************************/
void Fsm::action_save_ihbc_soft_state() {
  _action_cores.emplace(
      "save_ihbc_soft_state",
      std::make_shared<ActionCore>(
          "save_ihbc_soft_state", [&](const std::vector<std::string>& params) {
            if (params.size() < 1) {
              FSMCORE_LOG_ERROR << "Invalid input parameters count for "
                                   "save_ihbc_soft_state action.";
              return;
            }

            // if params is "on", save param_server 1
            // if params is others string, save param_server 0
            bool is_on = (params[0] == "on");

            auto ret = _bdb.set_basic("ihbc_mem", is_on, true);
            if (!ret) {
              FSMCORE_LOG_ERROR
                  << "[IHBC_MEM], Bdb save ihbc soft state failed.";
            }

            return;
          }));
}

/*****************************************************************************/
/* save light intense fsm state to paramserver                               */
/*****************************************************************************/
void Fsm::action_set_no_speed_sign_start_point() {
  _action_cores.emplace("set_no_speed_sign_start_point",
                        std::make_shared<ActionCore>(
                            "set_no_speed_sign_start_point",
                            [&](const std::vector<std::string>& params) {
                              auto latest_data = _node_ctx->GetOne("chassis");
                              auto timestamp_us =
                                  latest_data->__header.timestamp_real_us;
                              float dis = 0.0f;

                              save_ridding_time_us_to_berkeley_db(timestamp_us);
                              save_ridding_distance_to_berkeley_db(dis);

                              return;
                            }));
}

/*****************************************************************************/
/* save light intense fsm state to paramserver                               */
/*****************************************************************************/
void Fsm::action_clear_no_speed_sign_start_point() {
  _action_cores.emplace(
      "clear_no_speed_sign_start_point",
      std::make_shared<ActionCore>("clear_no_speed_sign_start_point",
                                   [&](const std::vector<std::string>& params) {
                                     del_ridding_param_from_berkeley_db();
                                   }));
}

/*****************************************************************************/
/* send can message about tsr\tlr state                                      */
/*****************************************************************************/
void Fsm::action_send_tsrtlr_state() {
  _action_cores.emplace(
      "tsr_send_can_message",
      std::make_shared<ActionCore>(
          "tsr_send_can_message", [&](const std::vector<std::string>& params) {
            if (params.size() < 2) {
              FSMCORE_LOG_ERROR << "Invalid input parameters count for "
                                   "tsr_send_can_message action, should be six";
              return;
            }

            // 这个线程只能有一个实例，当 tsr 状态机启动后，就存在了
            static bool has_run = false;
            if (!has_run) {
              std::thread glob_thread(&Fsm::tsrtlr_send_can_message_thread,
                                      this, params);
              glob_thread.detach();
              has_run = true;
            }
          }));
}

}  // namespace fsmcore
}  // namespace hozon
