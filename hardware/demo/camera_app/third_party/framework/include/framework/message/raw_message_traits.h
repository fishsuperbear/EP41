#ifndef CYBER_MESSAGE_RAW_MESSAGE_TRAITS_H_
#define CYBER_MESSAGE_RAW_MESSAGE_TRAITS_H_

#include <cassert>
#include <memory>
#include <string>

#include "framework/message/raw_message.h"

namespace netaos {
namespace framework {
namespace message {

// Template specialization for RawMessage
inline bool SerializeToArray(const RawMessage& message, void* data, int size) {
  return message.SerializeToArray(data, size);
}

inline bool ParseFromArray(const void* data, int size, RawMessage* message) {
  return message->ParseFromArray(data, size);
}

inline int ByteSize(const RawMessage& message) { return message.ByteSize(); }

}  // namespace message
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_MESSAGE_RAW_MESSAGE_TRAITS_H_
