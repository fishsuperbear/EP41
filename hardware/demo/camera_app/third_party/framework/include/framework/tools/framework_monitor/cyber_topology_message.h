#ifndef TOOLS_CVT_MONITOR_CYBER_TOPOLOGY_MESSAGE_H_
#define TOOLS_CVT_MONITOR_CYBER_TOPOLOGY_MESSAGE_H_

#include <map>
#include <string>

#include "framework/tools/framework_monitor/renderable_message.h"

namespace netaos {
namespace framework {
namespace proto {
class ChangeMsg;
class RoleAttributes;
}  // namespace proto
}  // namespace framework
}  // namespace netaos

class GeneralChannelMessage;
// class GeneralMessage;

class CyberTopologyMessage : public RenderableMessage {
 public:
  explicit CyberTopologyMessage(const std::string& channel);
  ~CyberTopologyMessage();

  int Render(const Screen* s, int key) override;
  RenderableMessage* Child(int index) const override;

  void TopologyChanged(const netaos::framework::proto::ChangeMsg& change_msg);
  void AddReaderWriter(const netaos::framework::proto::RoleAttributes& role,
                       bool isWriter);

 private:
  CyberTopologyMessage(const CyberTopologyMessage&) = delete;
  CyberTopologyMessage& operator=(const CyberTopologyMessage&) = delete;

  void ChangeState(const Screen* s, int key);
  bool IsFromHere(const std::string& node_name);

  std::map<std::string, GeneralChannelMessage*>::const_iterator FindChild(
      int index) const;

  enum class SecondColumnType { MessageType, MessageFrameRatio };
  SecondColumnType second_column_;

  int pid_;
  int col1_width_;
  const std::string& specified_channel_;
  std::map<std::string, GeneralChannelMessage*> all_channels_map_;
};

#endif  // TOOLS_CVT_MONITOR_CYBER_TOPOLOGY_MESSAGE_H_
