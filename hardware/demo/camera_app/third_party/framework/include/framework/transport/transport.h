#ifndef CYBER_TRANSPORT_TRANSPORT_H_
#define CYBER_TRANSPORT_TRANSPORT_H_

#include <atomic>
#include <memory>
#include <string>

#include "framework/proto/transport_conf.pb.h"

#include "framework/common/macros.h"
#include "framework/transport/dispatcher/intra_dispatcher.h"
#include "framework/transport/dispatcher/rtps_dispatcher.h"
#include "framework/transport/dispatcher/shm_dispatcher.h"
#include "framework/transport/qos/qos_profile_conf.h"
#include "framework/transport/receiver/hybrid_receiver.h"
#include "framework/transport/receiver/intra_receiver.h"
#include "framework/transport/receiver/receiver.h"
#include "framework/transport/receiver/rtps_receiver.h"
#include "framework/transport/receiver/shm_receiver.h"
#include "framework/transport/rtps/participant.h"
#include "framework/transport/shm/notifier_factory.h"
#include "framework/transport/transmitter/hybrid_transmitter.h"
#include "framework/transport/transmitter/intra_transmitter.h"
#include "framework/transport/transmitter/rtps_transmitter.h"
#include "framework/transport/transmitter/shm_transmitter.h"
#include "framework/transport/transmitter/transmitter.h"

namespace netaos {
namespace framework {
namespace transport {

using netaos::framework::proto::OptionalMode;

class Transport {
 public:
  virtual ~Transport();

  void Shutdown();

  template <typename M>
  auto CreateTransmitter(const RoleAttributes& attr,
                         const OptionalMode& mode = OptionalMode::HYBRID) ->
      typename std::shared_ptr<Transmitter<M>>;

  template <typename M>
  auto CreateReceiver(const RoleAttributes& attr,
                      const typename Receiver<M>::MessageListener& msg_listener,
                      const OptionalMode& mode = OptionalMode::HYBRID) ->
      typename std::shared_ptr<Receiver<M>>;

  ParticipantPtr participant() const { return participant_; }

 private:
  void CreateParticipant();

  std::atomic<bool> is_shutdown_ = {false};
  ParticipantPtr participant_ = nullptr;
  NotifierPtr notifier_ = nullptr;
  IntraDispatcherPtr intra_dispatcher_ = nullptr;
  ShmDispatcherPtr shm_dispatcher_ = nullptr;
  RtpsDispatcherPtr rtps_dispatcher_ = nullptr;

  DECLARE_SINGLETON(Transport)
};

template <typename M>
auto Transport::CreateTransmitter(const RoleAttributes& attr,
                                  const OptionalMode& mode) ->
    typename std::shared_ptr<Transmitter<M>> {
  if (is_shutdown_.load()) {
    AINFO << "transport has been shut down.";
    return nullptr;
  }

  std::shared_ptr<Transmitter<M>> transmitter = nullptr;
  RoleAttributes modified_attr = attr;
  if (!modified_attr.has_qos_profile()) {
    modified_attr.mutable_qos_profile()->CopyFrom(
        QosProfileConf::QOS_PROFILE_DEFAULT);
  }

  switch (mode) {
    case OptionalMode::INTRA:
      transmitter = std::make_shared<IntraTransmitter<M>>(modified_attr);
      break;

    case OptionalMode::SHM:
      transmitter = std::make_shared<ShmTransmitter<M>>(modified_attr);
      break;

    case OptionalMode::RTPS:
      transmitter =
          std::make_shared<RtpsTransmitter<M>>(modified_attr, participant());
      break;

    default:
      transmitter =
          std::make_shared<HybridTransmitter<M>>(modified_attr, participant());
      break;
  }

  RETURN_VAL_IF_NULL(transmitter, nullptr);
  if (mode != OptionalMode::HYBRID) {
    transmitter->Enable();
  }
  return transmitter;
}

template <typename M>
auto Transport::CreateReceiver(
    const RoleAttributes& attr,
    const typename Receiver<M>::MessageListener& msg_listener,
    const OptionalMode& mode) -> typename std::shared_ptr<Receiver<M>> {
  if (is_shutdown_.load()) {
    AINFO << "transport has been shut down.";
    return nullptr;
  }

  std::shared_ptr<Receiver<M>> receiver = nullptr;
  RoleAttributes modified_attr = attr;
  if (!modified_attr.has_qos_profile()) {
    modified_attr.mutable_qos_profile()->CopyFrom(
        QosProfileConf::QOS_PROFILE_DEFAULT);
  }

  switch (mode) {
    case OptionalMode::INTRA:
      receiver =
          std::make_shared<IntraReceiver<M>>(modified_attr, msg_listener);
      break;

    case OptionalMode::SHM:
      receiver = std::make_shared<ShmReceiver<M>>(modified_attr, msg_listener);
      break;

    case OptionalMode::RTPS:
      receiver = std::make_shared<RtpsReceiver<M>>(modified_attr, msg_listener);
      break;

    default:
      receiver = std::make_shared<HybridReceiver<M>>(
          modified_attr, msg_listener, participant());
      break;
  }

  RETURN_VAL_IF_NULL(receiver, nullptr);
  if (mode != OptionalMode::HYBRID) {
    receiver->Enable();
  }
  return receiver;
}

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_TRANSPORT_H_
