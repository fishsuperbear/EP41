#ifndef CYBER_SERVICE_DISCOVERY_COMMUNICATION_SUBSCRIBER_LISTENER_H_
#define CYBER_SERVICE_DISCOVERY_COMMUNICATION_SUBSCRIBER_LISTENER_H_

#include <functional>
#include <mutex>
#include <string>

#include "fastrtps/Domain.h"
#include "fastrtps/subscriber/SampleInfo.h"
#include "fastrtps/subscriber/Subscriber.h"
#include "fastrtps/subscriber/SubscriberListener.h"

namespace netaos {
namespace framework {
namespace service_discovery {

class SubscriberListener : public eprosima::fastrtps::SubscriberListener {
 public:
  using NewMsgCallback = std::function<void(const std::string&)>;

  explicit SubscriberListener(const NewMsgCallback& callback);
  virtual ~SubscriberListener();

  void onNewDataMessage(eprosima::fastrtps::Subscriber* sub);
  void onSubscriptionMatched(eprosima::fastrtps::Subscriber* sub,
                             eprosima::fastrtps::MatchingInfo& info);  // NOLINT

 private:
  NewMsgCallback callback_;
  std::mutex mutex_;
};

}  // namespace service_discovery
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SERVICE_DISCOVERY_COMMUNICATION_SUBSCRIBER_LISTENER_H_
