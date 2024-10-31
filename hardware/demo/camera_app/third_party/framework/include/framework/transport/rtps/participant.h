#ifndef CYBER_TRANSPORT_RTPS_PARTICIPANT_H_
#define CYBER_TRANSPORT_RTPS_PARTICIPANT_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "framework/transport/rtps/underlay_message_type.h"
#include "fastrtps/Domain.h"
#include "fastrtps/attributes/ParticipantAttributes.h"
#include "fastrtps/participant/Participant.h"
#include "fastrtps/participant/ParticipantListener.h"
#include "fastrtps/rtps/common/Locator.h"

namespace netaos {
namespace framework {
namespace transport {

class Participant;
using ParticipantPtr = std::shared_ptr<Participant>;

class Participant {
 public:
  Participant(const std::string& name, int send_port,
              eprosima::fastrtps::ParticipantListener* listener = nullptr);
  virtual ~Participant();

  void Shutdown();

  eprosima::fastrtps::Participant* fastrtps_participant();
  bool is_shutdown() const { return shutdown_.load(); }

 private:
  void CreateFastRtpsParticipant(
      const std::string& name, int send_port,
      eprosima::fastrtps::ParticipantListener* listener);

  std::atomic<bool> shutdown_;
  std::string name_;
  int send_port_;
  eprosima::fastrtps::ParticipantListener* listener_;
  UnderlayMessageType type_;
  eprosima::fastrtps::Participant* fastrtps_participant_;
  std::mutex mutex_;
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_RTPS_PARTICIPANT_H_
