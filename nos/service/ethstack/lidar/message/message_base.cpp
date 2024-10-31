#include "message_base.h"
// #include "protocol/lidar_message.h"


namespace hozon {
namespace ethstack {
namespace lidar {

MessageBase::MessageBase(uint32_t messageId) {
    m_messageId = messageId;
}

MessageBase::~MessageBase() {

}


uint32_t MessageBase::GetMessageId() const
{
    return m_messageId;
}

uint16_t MessageBase::GetMessgaeType() const
{
    return m_messageType;
}

int32_t MessageBase::Parse(std::shared_ptr<EthernetPacket> packet)
{
    return 0;
}

int32_t MessageBase::Serialize(std::shared_ptr<EthernetPacket>& packet)
{
    return 0;
}

}
}
}