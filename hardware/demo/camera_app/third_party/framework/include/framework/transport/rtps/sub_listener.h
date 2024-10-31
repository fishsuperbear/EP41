#ifndef CYBER_TRANSPORT_RTPS_SUB_LISTENER_H_
#define CYBER_TRANSPORT_RTPS_SUB_LISTENER_H_

#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "framework/transport/message/message_info.h"
#include "framework/transport/rtps/underlay_message.h"
#include "framework/transport/rtps/underlay_message_type.h"
#include "fastrtps/Domain.h"
#include "fastrtps/subscriber/SampleInfo.h"
#include "fastrtps/subscriber/Subscriber.h"
#include "fastrtps/subscriber/SubscriberListener.h"

namespace netaos {
namespace framework {
namespace transport {

class SubListener;
using SubListenerPtr = std::shared_ptr<SubListener>;

class SubListener : public eprosima::fastrtps::SubscriberListener {
 public:
  using NewMsgCallback = std::function<void(
      uint64_t channel_id, const std::shared_ptr<std::string>& msg_str,
      const MessageInfo& msg_info)>;

  explicit SubListener(const NewMsgCallback& callback);
  virtual ~SubListener();

  void onNewDataMessage(eprosima::fastrtps::Subscriber* sub);
  void onSubscriptionMatched(eprosima::fastrtps::Subscriber* sub,
                             eprosima::fastrtps::MatchingInfo& info);  // NOLINT

 private:
  NewMsgCallback callback_;
  MessageInfo msg_info_;
  std::mutex mutex_;
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_RTPS_SUB_LISTENER_H_
