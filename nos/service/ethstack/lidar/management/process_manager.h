#ifndef LIDAR_PROCESS_MANAGER_H
#define LIDAR_PROCESS_MANAGER_H

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <thread>

#include "transport/udp_base.h"
#include "common/logger.h"
#include "publish/point_cloud_pub.h"
#include "pointcloud/point_cloud_parser.h"
#include "protocol/point_cloud.h"

#include "json/json.h"


// constexpr uint16_t LOCAL_POINT_CLOUD_PORT = 2368;
// constexpr uint16_t REMOTE_POINT_CLOUD_PORT = 58005;

// constexpr uint16_t LOCAL_FAULT_MESSAGE_PORT = 2369;
// constexpr uint16_t REMOTE_FAULT_MESSAGE_PORT = 58003;

// const constexpr char* POINT_CLOUD_MULTICAST_ADDRESS   = ("239.255.0.1");

// const constexpr char* LOCAL_DEV_IP                    = ("172.16.80.11");
// const constexpr char* LIDAR_DEV_ADDRESS               = ("172.16.80.20");



namespace hozon {
namespace ethstack {
namespace lidar {

class ProcessManager {
 public:
  static ProcessManager& Instance();
  virtual ~ProcessManager();

  void Init();
  void Start();
  void Stop();

  void SetIfName(const std::string& ifName);
  void SetLidarFrame(const std::string& lidarFrame);
  bool GetLidarExtrinsicsPara();


 private:
  ProcessManager();

  int32_t GetLidarDiagVersion();

 private:
  static ProcessManager* s_instance;
  std::shared_ptr<UdpBase> udp_pointcloud;
  std::shared_ptr<UdpBase> udp_fault_message;

  std::string m_ifName;
  std::string m_lidarFrame;
  std::string m_localAddr;

  bool stop_;
};

}  // namespace lidar
}  // namespace ethstack
}  // namespace hozon
#endif  // LIDAR_PROCESS_MANAGER_H