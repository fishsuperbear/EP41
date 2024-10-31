#ifndef LIDAR_UDP_DISPATCH_H
#define LIDAR_UDP_DISPATCH_H

#include <memory>
#include <shared_mutex>
#include <vector>

#include "common/lidar_types.h"
#include "message/message_base.h"
#include "common/logger.h"
#include "pointcloud/point_cloud_parser.h"
#include "faultmessage/fault_message_parse.h"
#include "protocol/point_cloud.h"


namespace hozon {
namespace ethstack {
namespace lidar {


class UdpDispatcher {
 public:
  UdpDispatcher();
  virtual ~UdpDispatcher();

  int32_t Parse(uint16_t recvPort, std::shared_ptr<EthernetPacket> packet);


 private:
  std::shared_timed_mutex m_mutex;
  std::vector<std::shared_ptr<MessageBase>> m_parserList;
};

}  // namespace lidar
}  // namespace ethstack
}  // namespace hozon
#endif  // LIDAR_UDP_DISPATCH_H