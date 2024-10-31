#ifndef CYBER_TRANSPORT_DISPATCHER_RTPS_DISPATCHER_H_
#define CYBER_TRANSPORT_DISPATCHER_RTPS_DISPATCHER_H_

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "framework/common/log.h"
#include "framework/common/macros.h"
#include "framework/message/message_traits.h"
#include "framework/transport/dispatcher/dispatcher.h"
#include "framework/transport/rtps/attributes_filler.h"
#include "framework/transport/rtps/participant.h"
#include "framework/transport/rtps/sub_listener.h"

namespace netaos {
namespace framework {
namespace transport {

struct Subscriber {
  Subscriber() : sub(nullptr), sub_listener(nullptr) {}

  eprosima::fastrtps::Subscriber* sub;
  SubListenerPtr sub_listener;
};

class RtpsDispatcher;
using RtpsDispatcherPtr = RtpsDispatcher*;

class RtpsDispatcher : public Dispatcher {
 public:
  virtual ~RtpsDispatcher();

  void Shutdown() override;

  template <typename MessageT>
  void AddListener(const RoleAttributes& self_attr,
                   const MessageListener<MessageT>& listener);

  template <typename MessageT>
  void AddListener(const RoleAttributes& self_attr,
                   const RoleAttributes& opposite_attr,
                   const MessageListener<MessageT>& listener);

  void set_participant(const ParticipantPtr& participant) {
    participant_ = participant;
  }

 private:
  void OnMessage(uint64_t channel_id,
                 const std::shared_ptr<std::string>& msg_str,
                 const MessageInfo& msg_info);
  void AddSubscriber(const RoleAttributes& self_attr);
  // key: channel_id
  std::unordered_map<uint64_t, Subscriber> subs_;
  std::mutex subs_mutex_;

  ParticipantPtr participant_;

  DECLARE_SINGLETON(RtpsDispatcher)
};

template <typename MessageT>
void RtpsDispatcher::AddListener(const RoleAttributes& self_attr,
                                 const MessageListener<MessageT>& listener) {
  auto listener_adapter = [listener](
                              const std::shared_ptr<std::string>& msg_str,
                              const MessageInfo& msg_info) {
    auto msg = std::make_shared<MessageT>();
    RETURN_IF(!message::ParseFromString(*msg_str, msg.get()));
    listener(msg, msg_info);
  };

  Dispatcher::AddListener<std::string>(self_attr, listener_adapter);
  AddSubscriber(self_attr);
}

template <typename MessageT>
void RtpsDispatcher::AddListener(const RoleAttributes& self_attr,
                                 const RoleAttributes& opposite_attr,
                                 const MessageListener<MessageT>& listener) {
  auto listener_adapter = [listener](
                              const std::shared_ptr<std::string>& msg_str,
                              const MessageInfo& msg_info) {
    auto msg = std::make_shared<MessageT>();
    RETURN_IF(!message::ParseFromString(*msg_str, msg.get()));
    listener(msg, msg_info);
  };

  Dispatcher::AddListener<std::string>(self_attr, opposite_attr,
                                       listener_adapter);
  AddSubscriber(self_attr);
}

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_DISPATCHER_RTPS_DISPATCHER_H_
