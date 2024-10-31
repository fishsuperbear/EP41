#ifndef CYBER_MESSAGE_PY_MESSAGE_TRAITS_H_
#define CYBER_MESSAGE_PY_MESSAGE_TRAITS_H_

#include <cassert>
#include <memory>
#include <string>

#include "framework/message/py_message.h"

namespace netaos {
namespace framework {
namespace message {

// Template specialization for RawMessage
inline bool SerializeToArray(const PyMessageWrap& message, void* data,
                             int size) {
  return message.SerializeToArray(data, size);
}

inline bool ParseFromArray(const void* data, int size, PyMessageWrap* message) {
  return message->ParseFromArray(data, size);
}

inline int ByteSize(const PyMessageWrap& message) { return message.ByteSize(); }

}  // namespace message
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_MESSAGE_PY_MESSAGE_TRAITS_H_
