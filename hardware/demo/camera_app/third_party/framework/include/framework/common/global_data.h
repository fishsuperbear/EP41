#ifndef CYBER_COMMON_GLOBAL_DATA_H_
#define CYBER_COMMON_GLOBAL_DATA_H_

#include <string>
#include <unordered_map>

#include "framework/base/atomic_hash_map.h"
#include "framework/base/atomic_rw_lock.h"
#include "framework/common/log.h"
#include "framework/common/macros.h"
#include "framework/common/util.h"
#include "framework/proto/netaos_conf.pb.h"

namespace netaos {
namespace framework {
namespace common {

using ::netaos::framework::base::AtomicHashMap;
using ::netaos::framework::proto::ClockMode;
using ::netaos::framework::proto::NetaosConfig;
using ::netaos::framework::proto::RunMode;

class GlobalData {
 public:
  ~GlobalData();

  int ProcessId() const;

  void SetProcessGroup(const std::string& process_group);
  const std::string& ProcessGroup() const;

  void SetComponentNums(const int component_nums);
  int ComponentNums() const;

  void SetSchedName(const std::string& sched_name);
  const std::string& SchedName() const;

  const std::string& HostIp() const;

  const std::string& HostName() const;

  const NetaosConfig& Config() const;

  void EnableSimulationMode();
  void DisableSimulationMode();

  bool IsRealityMode() const;
  bool IsMockTimeMode() const;

  static uint64_t GenerateHashId(const std::string& name) {
    return common::Hash(name);
  }

  static uint64_t RegisterNode(const std::string& node_name);
  static std::string GetNodeById(uint64_t id);

  static uint64_t RegisterChannel(const std::string& channel);
  static std::string GetChannelById(uint64_t id);

  static uint64_t RegisterService(const std::string& service);
  static std::string GetServiceById(uint64_t id);

  static uint64_t RegisterTaskName(const std::string& task_name);
  static std::string GetTaskNameById(uint64_t id);

 private:
  void InitHostInfo();
  bool InitConfig();

  // global config
  NetaosConfig config_;

  // host info
  std::string host_ip_;
  std::string host_name_;

  // process info
  int process_id_;
  std::string process_group_;

  int component_nums_ = 0;

  // sched policy info
  std::string sched_name_ = "CYBER_DEFAULT";

  // run mode
  RunMode run_mode_;
  ClockMode clock_mode_;

  static AtomicHashMap<uint64_t, std::string, 512> node_id_map_;
  static AtomicHashMap<uint64_t, std::string, 256> channel_id_map_;
  static AtomicHashMap<uint64_t, std::string, 256> service_id_map_;
  static AtomicHashMap<uint64_t, std::string, 256> task_id_map_;

  DECLARE_SINGLETON(GlobalData)
};

}  // namespace common
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_COMMON_GLOBAL_DATA_H_
