#ifndef CYBER_SERVICE_DISCOVERY_TOPOLOGY_MANAGER_H_
#define CYBER_SERVICE_DISCOVERY_TOPOLOGY_MANAGER_H_

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "framework/base/signal.h"
#include "framework/common/macros.h"
#include "framework/service_discovery/communication/participant_listener.h"
#include "framework/service_discovery/specific_manager/channel_manager.h"
#include "framework/service_discovery/specific_manager/node_manager.h"
#include "framework/service_discovery/specific_manager/service_manager.h"
#include "framework/transport/rtps/participant.h"

namespace netaos {
namespace framework {
namespace service_discovery {

class NodeManager;
using NodeManagerPtr = std::shared_ptr<NodeManager>;

class ChannelManager;
using ChannelManagerPtr = std::shared_ptr<ChannelManager>;

class ServiceManager;
using ServiceManagerPtr = std::shared_ptr<ServiceManager>;

/**
 * @class TopologyManager
 * @brief elements in netaos framework -- Node, Channel, Service, Writer, Reader, Client
 * and Server's relationship is presented by Topology. You can Imagine that a
 * directed graph -- Node is the container of Server/Client/Writer/Reader, and
 * they are the vertice of the graph and Channel is the Edge from Writer flow to
 * the Reader, Service is the Edge from Server to Client. Thus we call Writer
 * and Server `Upstream`, Reader and Client `Downstream` To generate this graph,
 * we use TopologyManager, it has three sub managers -- NodeManager: You can
 * find Nodes in this topology ChannelManager: You can find Channels in this
 * topology, and their Writers and Readers ServiceManager: You can find Services
 * in this topology, and their Servers and Clients TopologyManager use
 * fast-rtps' Participant to communicate. It can broadcast Join or Leave
 * messages of those elements. Also, you can register you own `ChangeFunc` to
 * monitor topology change
 */
class TopologyManager {
 public:
  using ChangeSignal = base::Signal<const ChangeMsg&>;
  using ChangeFunc = std::function<void(const ChangeMsg&)>;
  using ChangeConnection = base::Connection<const ChangeMsg&>;
  using PartNameContainer =
      std::map<eprosima::fastrtps::rtps::GUID_t, std::string>;
  using PartInfo = eprosima::fastrtps::ParticipantDiscoveryInfo;

  virtual ~TopologyManager();

  /**
   * @brief Shutdown the TopologyManager
   */
  void Shutdown();

  /**
   * @brief To observe the topology change, you can register a `ChangeFunc`
   *
   * @param func is the observe function
   * @return ChangeConnection is the connection that connected to
   * `change_signal_`. Used to Remove your observe function
   */
  ChangeConnection AddChangeListener(const ChangeFunc& func);

  /**
   * @brief Remove the observe function connect to `change_signal_` by `conn`
   */
  void RemoveChangeListener(const ChangeConnection& conn);

  /**
   * @brief Get shared_ptr for NodeManager
   */
  NodeManagerPtr& node_manager() { return node_manager_; }

  /**
   * @brief Get shared_ptr for ChannelManager
   */
  ChannelManagerPtr& channel_manager() { return channel_manager_; }

  /**
   * @brief Get shared_ptr for ServiceManager
   */
  ServiceManagerPtr& service_manager() { return service_manager_; }

 private:
  bool Init();

  bool InitNodeManager();
  bool InitChannelManager();
  bool InitServiceManager();

  bool CreateParticipant();
  void OnParticipantChange(const PartInfo& info);
  bool Convert(const PartInfo& info, ChangeMsg* change_msg);
  bool ParseParticipantName(const std::string& participant_name,
                            std::string* host_name, int* process_id);

  std::atomic<bool> init_;             /// Is TopologyManager inited
  NodeManagerPtr node_manager_;        /// shared ptr of NodeManager
  ChannelManagerPtr channel_manager_;  /// shared ptr of ChannelManager
  ServiceManagerPtr service_manager_;  /// shared ptr of ServiceManager
  /// rtps participant to publish and subscribe
  transport::ParticipantPtr participant_;
  ParticipantListener* participant_listener_;
  ChangeSignal change_signal_;           /// topology changing signal,
                                         ///< connect to `ChangeFunc`s
  PartNameContainer participant_names_;  /// other participant in the topology

  DECLARE_SINGLETON(TopologyManager)
};

}  // namespace service_discovery
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SERVICE_DISCOVERY_TOPOLOGY_MANAGER_H_
