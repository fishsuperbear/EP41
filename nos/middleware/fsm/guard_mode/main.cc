#include "fsm.h"
#include "yaml_node.h"
#include "proto/soc/chassis.pb.h"
#include "proto/soc/sensor_imu_ins.pb.h"
#include "proto/statemachine/state_machine.pb.h"

int main() {
  std::string yaml_path = "./NodeConfig.yaml";
  std::string fsm_rule_path;

  /***************************************************************************/
  /* 获取主状态机的配置文件路径，和 tsr/tlr 配置文件路径 */
  /***************************************************************************/
  auto yaml_config = hozon::fsmcore::YamlNode(yaml_path);
  yaml_config.GetValue<std::string>("fsm_rule_path", fsm_rule_path);

  hozon::fsmcore::Fsm gm_fsm;
  auto ret = gm_fsm.init(fsm_rule_path);
  if (!ret) {
    std::cout << "Fsm config file has some error, please check, file name: "
              << fsm_rule_path << std::endl;
    return (-1);
  }

  /***************************************************************************/
  /* 使用最近十帧 /soc/imuinsinfo 中的 imu 数据，判断震动报警，注册是为了缓存十帧 */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("imu_lineacc", hozon::soc::ImuIns)

  /***************************************************************************/
  /* 各种开关来自于 chassis                                                   */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("chassis", hozon::soc::Chassis)

  /***************************************************************************/
  /* 感知发送靠近报警、感知发送哨兵模式已经开启                                  */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("perception_workingstatus", hozon::state::StateMachine)

  /***************************************************************************/
  /* 跑哨兵状态机 */
  /***************************************************************************/
  gm_fsm.Start(yaml_path);

  /***************************************************************************/
  /* 根据主状态机是否结束，决定进程是否结束 */
  /***************************************************************************/
  while (!gm_fsm.NeedStop()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  gm_fsm.Stop();

  return 0;
}