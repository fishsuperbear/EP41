#ifndef CYBER_MESSAGE_PROTOBUF_TRAITS_H_
#define CYBER_MESSAGE_PROTOBUF_TRAITS_H_

#include <cassert>
#include <memory>
#include <string>

#include "framework/message/protobuf_factory.h"

namespace netaos {
namespace framework {
namespace message {

template <typename MessageT,
          typename std::enable_if<
              std::is_base_of<google::protobuf::Message, MessageT>::value,
              int>::type = 0>
inline std::string MessageType() {
  return MessageT::descriptor()->full_name();
}

template <typename MessageT,
          typename std::enable_if<
              std::is_base_of<google::protobuf::Message, MessageT>::value,
              int>::type = 0>
std::string MessageType(const MessageT& message) {
  return message.GetDescriptor()->full_name();
}

template <typename MessageT,
          typename std::enable_if<
              std::is_base_of<google::protobuf::Message, MessageT>::value,
              int>::type = 0>
inline void GetDescriptorString(const MessageT& message,
                                std::string* desc_str) {
  ProtobufFactory::Instance()->GetDescriptorString(message, desc_str);
}

template <typename MessageT,
          typename std::enable_if<
              std::is_base_of<google::protobuf::Message, MessageT>::value,
              int>::type = 0>
inline void GetDescriptorString(const std::string& type,
                                std::string* desc_str) {
  ProtobufFactory::Instance()->GetDescriptorString(type, desc_str);
}

template <typename MessageT,
          typename std::enable_if<
              std::is_base_of<google::protobuf::Message, MessageT>::value,
              int>::type = 0>
bool RegisterMessage(const MessageT& message) {
  return ProtobufFactory::Instance()->RegisterMessage(message);
}

}  // namespace message
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_MESSAGE_PROTOBUF_TRAITS_H_
