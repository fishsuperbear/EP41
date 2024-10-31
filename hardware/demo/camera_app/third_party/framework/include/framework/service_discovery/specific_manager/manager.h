#ifndef CYBER_SERVICE_DISCOVERY_SPECIFIC_MANAGER_MANAGER_H_
#define CYBER_SERVICE_DISCOVERY_SPECIFIC_MANAGER_MANAGER_H_

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

#include "fastrtps/Domain.h"
#include "fastrtps/attributes/PublisherAttributes.h"
#include "fastrtps/attributes/SubscriberAttributes.h"
#include "fastrtps/participant/Participant.h"
#include "fastrtps/publisher/Publisher.h"
#include "fastrtps/subscriber/Subscriber.h"

#include "framework/base/signal.h"
#include "framework/proto/topology_change.pb.h"
#include "framework/service_discovery/communication/subscriber_listener.h"

namespace netaos {
namespace framework {
namespace service_discovery {

using proto::ChangeMsg;
using proto::ChangeType;
using proto::OperateType;
using proto::RoleAttributes;
using proto::RoleType;

/**
 * @class Manager
 * @brief Base class for management of Topology elements.
 * Manager can Join/Leave the Topology, and Listen the topology change
 */
class Manager {
 public:
  using ChangeSignal = base::Signal<const ChangeMsg&>;
  using ChangeFunc = std::function<void(const ChangeMsg&)>;
  using ChangeConnection = base::Connection<const ChangeMsg&>;

  using RtpsParticipant = eprosima::fastrtps::Participant;
  using RtpsPublisherAttr = eprosima::fastrtps::PublisherAttributes;
  using RtpsSubscriberAttr = eprosima::fastrtps::SubscriberAttributes;

  /**
   * @brief Construct a new Manager object
   */
  Manager();

  /**
   * @brief Destroy the Manager object
   */
  virtual ~Manager();

  /**
   * @brief Startup topology discovery
   *
   * @param participant is used to create rtps Publisher and Subscriber
   * @return true if start successfully
   * @return false if start fail
   */
  bool StartDiscovery(RtpsParticipant* participant);

  /**
   * @brief Stop topology discovery
   */
  void StopDiscovery();

  /**
   * @brief Shutdown module
   */
  virtual void Shutdown();

  /**
   * @brief Join the topology
   *
   * @param attr is the attributes that will be sent to other Manager(include
   * ourselves)
   * @param role is one of RoleType enum
   * @return true if Join topology successfully
   * @return false if Join topology failed
   */
  bool Join(const RoleAttributes& attr, RoleType role,
            bool need_publish = true);

  /**
   * @brief Leave the topology
   *
   * @param attr is the attributes that will be sent to other Manager(include
   * ourselves)
   * @param role if one of RoleType enum.
   * @return true if Leave topology successfully
   * @return false if Leave topology failed
   */
  bool Leave(const RoleAttributes& attr, RoleType role);

  /**
   * @brief Add topology change listener, when topology changed, func will be
   * called.
   *
   * @param func the callback function
   * @return ChangeConnection Store it to use when you want to stop listening.
   */
  ChangeConnection AddChangeListener(const ChangeFunc& func);

  /**
   * @brief Remove our listener for topology change.
   *
   * @param conn is the return value of `AddChangeListener`
   */
  void RemoveChangeListener(const ChangeConnection& conn);

  /**
   * @brief Called when a process' topology manager instance leave
   *
   * @param host_name is the process's host's name
   * @param process_id is the process' id
   */
  virtual void OnTopoModuleLeave(const std::string& host_name,
                                 int process_id) = 0;

 protected:
  bool CreatePublisher(RtpsParticipant* participant);
  bool CreateSubscriber(RtpsParticipant* participant);

  virtual bool Check(const RoleAttributes& attr) = 0;
  virtual void Dispose(const ChangeMsg& msg) = 0;
  virtual bool NeedPublish(const ChangeMsg& msg) const;

  void Convert(const RoleAttributes& attr, RoleType role, OperateType opt,
               ChangeMsg* msg);

  void Notify(const ChangeMsg& msg);
  bool Publish(const ChangeMsg& msg);
  void OnRemoteChange(const std::string& msg_str);
  bool IsFromSameProcess(const ChangeMsg& msg);

  std::atomic<bool> is_shutdown_;
  std::atomic<bool> is_discovery_started_;
  int allowed_role_;
  ChangeType change_type_;
  std::string host_name_;
  int process_id_;
  std::string channel_name_;
  eprosima::fastrtps::Publisher* publisher_;
  std::mutex lock_;
  eprosima::fastrtps::Subscriber* subscriber_;
  SubscriberListener* listener_;

  ChangeSignal signal_;
};

}  // namespace service_discovery
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SERVICE_DISCOVERY_SPECIFIC_MANAGER_MANAGER_H_
