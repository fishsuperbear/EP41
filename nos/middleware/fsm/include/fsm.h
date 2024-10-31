#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "action.h"
#include "bdb_wrapper.h"
#include "cm/include/proto_cm_writer.h"
#include "condition.h"
#include "counterdown_timer.h"
#include "data_types/common/types.h"
#include "log.h"
#include "logging.h"
#include "node_base.h"
#include "node_bundle.h"
#include "phm/include/phm_client.h"
#include "proto/perception/perception_measurement.pb.h"
#include "proto/perception/perception_obstacle.pb.h"
#include "proto/perception/transport_element.pb.h"
#include "proto/soc/apa2mcu_chassis.pb.h"
#include "proto/statemachine/fsm_output.pb.h"
#include "proto/statemachine/fsm_rule.pb.h"
#include "proto/statemachine/state_machine.pb.h"
#include "proto/soc/chassis.pb.h"
#include "proto/fsm/function_manager.pb.h"
#include "state.h"
#include "transit.h"

namespace hozon {
namespace fsmcore {

using hozon::netaos::adf::BaseDataTypePtr;
using hozon::netaos::adf::NodeBase;
using hozon::netaos::adf::NodeBundle;
using hozon::netaos::phm::PHMClient;
using hozon::netaos::phm::ReceiveFault_t;

#define FSMCORE_LOG_FATAL   (NODE_LOG_CRITICAL)
#define FSMCORE_LOG_ERROR   (NODE_LOG_ERROR)
#define FSMCORE_LOG_WARN    (NODE_LOG_WARN)
#define FSMCORE_LOG_INFO    (NODE_LOG_INFO)
#define FSMCORE_LOG_DEBUG   (NODE_LOG_DEBUG)
#define FSMCORE_LOG_TRACE   (NODE_LOG_TRACE)

class Fsm : public NodeBase {
 public:
  Fsm() {
    _node_ctx.reset();
    _in_releasing.store(false);
  }
  ~Fsm(){};

  int32_t AlgInit() override;
  void AlgRelease() override;

  bool init(const std::string& cfg_path);
  int process(NodeBundle* input);

  std::string get_fsm_name();
  std::shared_ptr<State> find_state(const hozon::fsm_rule::StateId& id);
  void fill_state_stack(const std::shared_ptr<State>& state,
                        hozon::fsm_output::StateStack& state_stack);
  void publish_fsm_state();
  bool check_condition(Condition condition);
  void do_enter_actions();
  void do_exit_actions(const std::shared_ptr<State>& new_state);
  void do_transit_actions(const std::shared_ptr<Transit>& transit);

 private:
  bool process_once();
  bool parse_config();
  int transit_once();
  bool rearrange_state_to_levels(std::shared_ptr<State> state);
  void change_state(const std::shared_ptr<State>& new_state);

  // protobuf 反射获取字段
  std::shared_ptr<google::protobuf::Message> get_topic_data(
      const std::string& topic_name);
  bool get_topic_field(const std::vector<std::string>& params,
                       const google::protobuf::Message** o_msg,
                       const google::protobuf::Reflection** o_reflect,
                       const google::protobuf::FieldDescriptor** o_field);

  // 定义条件
  void define_conditions();

  // 状态机通行的一些条件
  void condition_time_is_up();
  void condition_couterdowntimer_is_timeup();
  void condition_couterdowntimer_is_canceled();
  void condition_get_from_state_prev();
  void condition_curr_state_elapse_us();
  void condition_curr_state_maintain_frames();
  void condition_has_fault();

  // 借用 protobuf 的反射，写的一些通用条件
  void condition_topic_bool();
  void condition_topic_uint32();
  void condition_topic_string();
  void condition_topic_float();
  void condition_topic_double();
  void condition_topic_uint64();
  void condition_topic_int32();
  void condition_topic_int64();
  void condition_topic_enum();

  // ihbc 功能用的一些条件
  void condition_highbeam_req_last();
  void condition_has_target_light();
  void condition_has_target_obstacle();
  void condition_wiper_work_last();
  void condition_is_light_high();

  // tsr 功能用的一些条件
  void condition_tsr_open();
  void condition_speed_sign();
  void condition_speed_over_limit();
  void condition_distance_no_speed_sign();

  // 哨兵模式功能用的一些条件
  void condition_vibration_g();
  void condition_gm_work_time();
  void condition_last_warn_diff();

  // 定义动作
  void define_actions();

  // 状态机通行的一些动作
  void action_print_log();                 // 打印日志动作
  void action_set_countdown_timer();       // 设定倒计时器动作
  void action_cancel_counterdown_timer();  // 取消倒计时动作
  void action_regist_data_buffering();     // 注册数据接收器，用于缓存，需要数据触发源

  // ihbc 功能用的一些动作
  void action_save_ihbc_soft_state();  // 保存 ihbc 软开关状态到参数服务器
  void action_send_ihbc_state();  // 将 ihbc 状态机状态发送到 can 上

  // tsr tlr 功能用的一些动作
  void action_set_no_speed_sign_start_point();  // 没有限速标志时候，累加里程
  void
  action_set_no_speed_sign_last();  // 没有限速标志时候，以上一个限速标志限速
  void action_clear_no_speed_sign_start_point();  // 离开限速区域，清掉里程
  void action_send_tsrtlr_state();  // 将 tsr tlr 状态机状态发送到 can 上

  // 哨兵模式功能用的一些动作
  void action_call_perception();     // 哨兵模式拉起感知的动作
  void action_send_gm_state();       // 发送哨兵模式状态
  void action_set_gm_start_time();   // 记录哨兵模式开始 work 的时间
  void action_set_warn_stop_time();  // 记录告警消失的时间

  // MOD 功能用的一些动作
  void action_mod_send_can_message();// 转发感知mod告警

  int32_t set_counterdown_timer(const std::string& name, uint32_t milliseconds);
  std::shared_ptr<CounterdownTimer> get_counterdown_timer(
      const std::string& name);
  int32_t cancel_counterdown_timer(const std::string& name);
  int32_t save_data_callback(NodeBundle* node_bundle,
                             const std::string& topic_name, uint32_t interval);

  // berkeley 操作的相关的函数
  void save_speed_to_berkeley_db(int32_t speed);
  void save_speed_sign_to_berkeley_db(int32_t speed_sign);
  void save_forbidden_sign_to_berkeley_db(uint32_t forbidden_sign);
  void save_tlr_value_to_berkeley_db(uint32_t tlr_value);
  void save_light_intense_to_berkeley_db(bool is_high);
  void save_nnp_state_to_berkeley_db(uint32_t nnp_state);
  void save_speed_sign_valid_to_berkeley_db(int32_t speed_sign);
  void save_tsr_open_to_berkeley_db(bool tsr_open);
  void save_ridding_time_us_to_berkeley_db(uint64_t time);
  void save_ridding_distance_to_berkeley_db(float distance);

  bool get_speed_from_berkeley_db(int32_t& speed);
  bool get_speed_sign_from_berkeley_db(int32_t& speed_sign);
  bool get_forbidden_sign_from_berkeley_db(uint32_t& forbidden_sign);
  bool get_tlr_value_from_berkeley_db(uint32_t& tlr_value);
  bool get_light_intense_from_berkeley_db(bool& is_high);
  bool get_nnp_state_from_berkeley_db(uint32_t& nnp_state);
  bool get_speed_sign_valid_from_berkeley_db(int32_t& speed_sign);
  bool get_tsr_open_from_berkeley_db(bool& tsr_open);
  bool get_ridding_time_us_from_berkeley_db(uint64_t& time);
  bool get_ridding_distance_from_berkeley_db(float& distance);

  void del_ridding_param_from_berkeley_db();

  bool get_speed_forbidden_from_tsr_ihbc_detection(
      const std::vector<std::string>& params, int32_t& speed,
      uint32_t& forbidden_priority);

  bool get_tlr_value_from_junction_passable(uint32_t& passable);
  bool get_nnp_state_from_mcu2ego(uint32_t& nnp_state);

  // 故障 phm 回调函数
  void phm_service_available_callback(bool bResult);
  void phm_receive_fault_callbak(const ReceiveFault_t& fault);

  // tsr tlr 定时发送 can message 的函数
  void tsrtlr_send_can_message_thread(const std::vector<std::string>& params);
  int32_t mod_send_can_message(NodeBundle* node_bundle);
  uint32_t convert_forbidden_sign_to_cdcs(uint32_t priority);
  int32_t convert_speed_sign_to_cdcs(int32_t speed_sign);
  bool is_legal_speed_sign(int32_t speed);
  bool is_legal_forbidden_sign(uint8_t forbidden_sign);
  uint32_t get_forbidden_priority(uint8_t forbidden_sub_type);

 private:
  std::string _fsm_path;  // 单个进程可能运行多个状态机，从这里区分
  bool _is_first_process{false};  // process 函数第一次被调
  std::atomic<bool> _in_releasing;
  std::unique_ptr<NodeBundle> _node_ctx;
  hozon::fsm_rule::FsmRule _fsm_rule;
  uint32_t _max_transit_count;
  std::string
      _trigger_name;  // 每一个状态机，其process函数，需要绑定一个yaml中的trigger

  std::shared_ptr<State> _prev_state;  // 上一个状态
  std::shared_ptr<State> _curr_state;  // 当前状态
  uint64_t _curr_state_us;             // 进入当前状态时刻
  uint32_t _curr_state_frame{0};       // 保持当前状态的帧数

  FindStateFunc _find_state_func{
      [&](const hozon::fsm_rule::StateId& id) -> std::shared_ptr<State> {
        return find_state(id);
      }};
  FindActionCoreFunc _find_actioncore_func{
      [&](const std::string& name) -> std::shared_ptr<ActionCore> {
        return _action_cores[name];
      }};
  FindConditionCoreFunc _find_conditioncore_func{
      [&](const std::string& name) -> std::shared_ptr<ConditionCore> {
        return _condition_cores[name];
      }};
  std::unordered_map<uint32_t,
                     std::unordered_map<std::string, std::shared_ptr<State>>>
      _states;
  std::vector<std::shared_ptr<Transit>> _transits;
  std::unordered_map<std::string, std::shared_ptr<ConditionCore>>
      _condition_cores;
  std::unordered_map<std::string, std::shared_ptr<ActionCore>> _action_cores;
  std::unordered_map<std::string, std::shared_ptr<CounterdownTimer>>
      _counterdown_timers;

  struct threadsafe_unordered_map {
    std::unordered_map<std::string, std::vector<BaseDataTypePtr>> data_map;
    std::mutex mtx;
  };
  threadsafe_unordered_map _data_map;

  hozon::fsmcore::BdbWrapper _bdb{"fsmcore"};
  hozon::netaos::cm::ProtoCMWriter<hozon::fsm_output::StateOutput>
      _output_writer;
  hozon::netaos::cm::ProtoCMWriter<hozon::soc::Apa2Chassis> _canfd_writer;
  hozon::netaos::cm::ProtoCMWriter<hozon::state::StateMachine>
      _statemachine_writer;

  // 故障管理相关
  std::shared_ptr<PHMClient> _phm_client;
  std::atomic_bool _phm_available;
  struct threadsafe_set {
    std::set<std::pair<uint32_t, uint8_t>> fault_on;
    std::mutex mtx;
  };
  threadsafe_set _fault_set;
};

}  // namespace fsmcore
}  // namespace hozon