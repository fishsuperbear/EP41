#ifndef LIDAR_MESSAGE_BASE_H
#define LIDAR_MESSAGE_BASE_H

#include <memory>
#include "common/lidar_types.h"

namespace hozon {
namespace ethstack {
namespace lidar {

class MessageBase {
 public:
  MessageBase(uint32_t messageId);
  virtual ~MessageBase();

  uint32_t GetMessageId() const;
  uint16_t GetMessgaeType() const;

  virtual int32_t Parse(std::shared_ptr<EthernetPacket> packet);
  virtual int32_t Serialize(std::shared_ptr<EthernetPacket>& packet);

 private:
  uint32_t m_messageId;
  uint16_t m_messageType;
};

}  // namespace lidar
}  // namespace ethstack
}  // namespace hozon
#endif  // LIDAR_MESSAGE_BASE_H