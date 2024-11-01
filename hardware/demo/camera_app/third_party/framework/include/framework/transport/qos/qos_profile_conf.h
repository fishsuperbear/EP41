#ifndef CYBER_TRANSPORT_QOS_QOS_PROFILE_CONF_H_
#define CYBER_TRANSPORT_QOS_QOS_PROFILE_CONF_H_

#include <cstdint>

#include "framework/proto/qos_profile.pb.h"

namespace netaos {
namespace framework {
namespace transport {

using framework::proto::QosDurabilityPolicy;
using framework::proto::QosHistoryPolicy;
using framework::proto::QosProfile;
using framework::proto::QosReliabilityPolicy;

class QosProfileConf {
 public:
  QosProfileConf();
  virtual ~QosProfileConf();

  static QosProfile CreateQosProfile(const QosHistoryPolicy& history,
                                     uint32_t depth, uint32_t mps,
                                     const QosReliabilityPolicy& reliability,
                                     const QosDurabilityPolicy& durability);

  static const uint32_t QOS_HISTORY_DEPTH_SYSTEM_DEFAULT;
  static const uint32_t QOS_MPS_SYSTEM_DEFAULT;

  static const QosProfile QOS_PROFILE_DEFAULT;
  static const QosProfile QOS_PROFILE_SENSOR_DATA;
  static const QosProfile QOS_PROFILE_PARAMETERS;
  static const QosProfile QOS_PROFILE_SERVICES_DEFAULT;
  static const QosProfile QOS_PROFILE_PARAM_EVENT;
  static const QosProfile QOS_PROFILE_SYSTEM_DEFAULT;
  static const QosProfile QOS_PROFILE_TF_STATIC;
  static const QosProfile QOS_PROFILE_TOPO_CHANGE;
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_QOS_QOS_PROFILE_CONF_H_
