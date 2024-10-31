#ifndef CYBER_TRANSPORT_RTPS_ATTRIBUTES_FILLER_H_
#define CYBER_TRANSPORT_RTPS_ATTRIBUTES_FILLER_H_

#include <string>

#include "framework/proto/qos_profile.pb.h"
#include "fastrtps/attributes/PublisherAttributes.h"
#include "fastrtps/attributes/SubscriberAttributes.h"

namespace netaos {
namespace framework {
namespace transport {

using proto::QosDurabilityPolicy;
using proto::QosHistoryPolicy;
using proto::QosProfile;
using proto::QosReliabilityPolicy;

class AttributesFiller {
 public:
  AttributesFiller();
  virtual ~AttributesFiller();

  static bool FillInPubAttr(const std::string& channel_name,
                            const QosProfile& qos,
                            eprosima::fastrtps::PublisherAttributes* pub_attr);

  static bool FillInSubAttr(const std::string& channel_name,
                            const QosProfile& qos,
                            eprosima::fastrtps::SubscriberAttributes* sub_attr);
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_RTPS_ATTRIBUTES_FILLER_H_
