#ifndef CYBER_TRANSPORT_MESSAGE_HISTORY_ATTRIBUTES_H_
#define CYBER_TRANSPORT_MESSAGE_HISTORY_ATTRIBUTES_H_

#include <cstdint>

#include "framework/proto/qos_profile.pb.h"

namespace netaos {
namespace framework {
namespace transport {

struct HistoryAttributes {
  HistoryAttributes()
      : history_policy(proto::QosHistoryPolicy::HISTORY_KEEP_LAST),
        depth(1000) {}
  HistoryAttributes(const proto::QosHistoryPolicy& qos_history_policy,
                    uint32_t history_depth)
      : history_policy(qos_history_policy), depth(history_depth) {}

  proto::QosHistoryPolicy history_policy;
  uint32_t depth;
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_MESSAGE_HISTORY_ATTRIBUTES_H_
