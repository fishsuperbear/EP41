#include "fsm.h"
#include "proto/map/junction_passable.pb.h"
#include "proto/perception/perception_measurement.pb.h"
#include "proto/perception/perception_obstacle.pb.h"
#include "proto/perception/transport_element.pb.h"
#include "proto/fsm/function_manager.pb.h"
#include "proto/soc/chassis.pb.h"
#include "proto/soc/mcu2ego.pb.h"
#include "yaml_node.h"

int main() {
  std::string yaml_path = "/app/runtime_service/ihbc_tsr_tlr/conf/NodeIhbcConfig.yaml";
  std::string fsm_rule_path;

  /***************************************************************************/
  /* 缓存 同时周期性任务也收 Chassis */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("chassis", hozon::soc::Chassis)

  /***************************************************************************/
  /* 周期性任务收障碍物 */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("ihbc_obstacle",
                              hozon::perception::PerceptionObstacles)

  /***************************************************************************/
  /* 周期性任务收车灯和光强信息 */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("ihbc_light_and_intense",
                              hozon::perception::Vision2dDetection)

  /***************************************************************************/
  /* 周期性soc2ego信息 */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("nnp_state", hozon::functionmanager::FunctionManagerIn)

  /***************************************************************************/
  /* 周期性任务收限速标记和禁止标记信息 */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("forbidden_and_limit",
                              hozon::perception::Vision2dDetection)

  /***************************************************************************/
  /* 周期性任务收交通灯信息 */
  /***************************************************************************/
  REGISTER_PROTO_MESSAGE_TYPE("traffic_light", hozon::hdmap::JunctionPassable)

  /***************************************************************************/
  /* 获取 ihbc 状态机的配置文件路径 */
  /***************************************************************************/
  auto yaml_config = hozon::fsmcore::YamlNode(yaml_path);
  yaml_config.GetValue<std::string>("fsm_rule_path", fsm_rule_path);

  hozon::fsmcore::Fsm ihbc_fsm;
  auto ret = ihbc_fsm.init(fsm_rule_path);
  if (!ret) {
    std::cout << "Fsm config file has some error, please check, file name: "
              << fsm_rule_path << std::endl;
    return (-1);
  }

  /***************************************************************************/
  /* 跑 ihbc 状态机                                                           */
  /***************************************************************************/
  ihbc_fsm.Start(yaml_path);

  /***************************************************************************/
  /* 获取 tsr tlr 状态机的配置文件路径 */
  /***************************************************************************/
  yaml_path = "/app/runtime_service/ihbc_tsr_tlr/conf/NodeTsrTlrConfig.yaml";
  yaml_config = hozon::fsmcore::YamlNode(yaml_path);
  yaml_config.GetValue<std::string>("fsm_rule_path", fsm_rule_path);

  hozon::fsmcore::Fsm tsrtlr_fsm;
  ret = tsrtlr_fsm.init(fsm_rule_path);
  if (!ret) {
    std::cout << "Fsm config file has some error, please check, file name: "
              << fsm_rule_path << std::endl;
    return (-1);
  }

  /***************************************************************************/
  /* 跑 tsr tlr 状态机                                                        */
  /***************************************************************************/
  tsrtlr_fsm.Start(yaml_path);

  /***************************************************************************/
  /* 根据主状态机是否结束，决定进程是否结束 */
  /***************************************************************************/
  while (!ihbc_fsm.NeedStop()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  ihbc_fsm.Stop();

  return 0;
}