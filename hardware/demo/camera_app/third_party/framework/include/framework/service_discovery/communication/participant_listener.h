#ifndef CYBER_SERVICE_DISCOVERY_COMMUNICATION_PARTICIPANT_LISTENER_H_
#define CYBER_SERVICE_DISCOVERY_COMMUNICATION_PARTICIPANT_LISTENER_H_

#include <functional>
#include <mutex>

#include "fastrtps/Domain.h"
#include "fastrtps/participant/Participant.h"
#include "fastrtps/participant/ParticipantListener.h"

namespace netaos {
namespace framework {
namespace service_discovery {

class ParticipantListener : public eprosima::fastrtps::ParticipantListener {
 public:
  using ChangeFunc = std::function<void(
      const eprosima::fastrtps::ParticipantDiscoveryInfo& info)>;

  explicit ParticipantListener(const ChangeFunc& callback);
  virtual ~ParticipantListener();

  virtual void onParticipantDiscovery(
      eprosima::fastrtps::Participant* p,
      eprosima::fastrtps::ParticipantDiscoveryInfo info);

 private:
  ChangeFunc callback_;
  std::mutex mutex_;
};

}  // namespace service_discovery
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SERVICE_DISCOVERY_COMMUNICATION_PARTICIPANT_LISTENER_H_
