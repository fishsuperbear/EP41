#ifndef TOOLS_CVT_MONITOR_CYBER_TOPOLOGY_MESSAGE_H_
#define TOOLS_CVT_MONITOR_CYBER_TOPOLOGY_MESSAGE_H_

#include <map>
#include <string>

#include "topic_manager.hpp"

#include "monitor/cyber_topology_message.h"
#include "monitor/renderable_message.h"

namespace hozon {
namespace netaos {
namespace topic {

using namespace hozon::netaos::data_tool_common;

class ChangeMsg;
class RoleAttributes;

class GeneralChannelMessage;

// class GeneralMessage;

class CyberTopologyMessage : public RenderableMessage {
   public:
    explicit CyberTopologyMessage(const std::string& channel, std::shared_ptr<hozon::netaos::data_tool_common::TopicManager> topic_manager);
    ~CyberTopologyMessage();

    int Render(const Screen* s, int key) override;
    RenderableMessage* Child(int index) const override;

    void TopologyChanged(const TopicInfo& changeMsg);
    void AddReaderWriter(const std::string topic_name, const std::string msgTypeName);

   protected:
    std::shared_ptr<hozon::netaos::data_tool_common::TopicManager> topic_manager_;

   private:
    CyberTopologyMessage(const CyberTopologyMessage&) = delete;
    CyberTopologyMessage& operator=(const CyberTopologyMessage&) = delete;

    void ChangeState(const Screen* s, int key);

    std::map<std::string, GeneralChannelMessage*>::const_iterator FindChild(int index) const;

    enum class SecondColumnType { ProtoType, MessageFrameRatio };
    SecondColumnType second_column_;

    int pid_;
    int col1_width_;
    const std::string& specified_channel_;
    std::map<std::string, GeneralChannelMessage*> all_channels_map_;

    friend class GeneralChannelMessage;
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon

#endif  // TOOLS_CVT_MONITOR_CYBER_TOPOLOGY_MESSAGE_H_
