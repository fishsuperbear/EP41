#include "fsm.h"
#include "yaml_node.h"
#include "proto/soc/chassis.pb.h"
#include "proto/soc/apa2mcu_chassis.pb.h"
#include "proto/statemachine/state_machine.pb.h"

int main() {
  std::string yaml_path = "./NodeModConfig.yaml";
  std::string fsm_rule_path;

  /***************************************************************************/
  /* 获取主状态机的配置文件路径，和 tsr/tlr 配置文件路径 */
  /***************************************************************************/
  auto yaml_config = hozon::fsmcore::YamlNode(yaml_path);
  yaml_config.GetValue<std::string>("fsm_rule_path", fsm_rule_path);

  hozon::fsmcore::Fsm mod_fsm;
  auto ret = mod_fsm.init(fsm_rule_path);
  if (!ret) {
    std::cout << "Fsm config file has some error, please check, file name: "
              << fsm_rule_path << std::endl;
    return (-1);
  }

  /***************************************************************************/
  /* 各种开关来自于 chassis                                                   */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("chassis", hozon::soc::Chassis)

  /***************************************************************************/
  /* 感知发送运动物体检测、感知发送MOD模式已经开启                                  */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("perception_workingstatus", hozon::state::StateMachine)

  /***************************************************************************/
  /* 跑 mod 状态机 */
  /***************************************************************************/
  mod_fsm.Start(yaml_path);

  /***************************************************************************/
  /* 根据主状态机是否结束，决定进程是否结束 */
  /***************************************************************************/
  while (!mod_fsm.NeedStop()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  mod_fsm.Stop();

  return 0;
}