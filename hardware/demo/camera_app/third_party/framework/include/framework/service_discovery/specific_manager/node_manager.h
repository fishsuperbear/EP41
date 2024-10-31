#ifndef CYBER_SERVICE_DISCOVERY_SPECIFIC_MANAGER_NODE_MANAGER_H_
#define CYBER_SERVICE_DISCOVERY_SPECIFIC_MANAGER_NODE_MANAGER_H_

#include <memory>
#include <string>
#include <vector>

#include "framework/service_discovery/container/single_value_warehouse.h"
#include "framework/service_discovery/role/role.h"
#include "framework/service_discovery/specific_manager/manager.h"

namespace netaos {
namespace framework {
namespace service_discovery {

class TopologyManager;

/**
 * @class NodeManager
 * @brief Topology Manager of Node related
 */
class NodeManager : public Manager {
  friend class TopologyManager;

 public:
  using RoleAttrVec = std::vector<RoleAttributes>;
  using NodeWarehouse = SingleValueWarehouse;

  /**
   * @brief Construct a new Node Manager object
   */
  NodeManager();

  /**
   * @brief Destroy the Node Manager object
   */
  virtual ~NodeManager();

  /**
   * @brief Checkout whether we have `node_name` in topology
   *
   * @param node_name Node's name we want to inquire
   * @return true if this node found
   * @return false if this node not exits
   */
  bool HasNode(const std::string& node_name);

  /**
   * @brief Get the Nodes object
   *
   * @param nodes result RoleAttr vector
   */
  void GetNodes(RoleAttrVec* nodes);

 private:
  bool Check(const RoleAttributes& attr) override;
  void Dispose(const ChangeMsg& msg) override;
  void OnTopoModuleLeave(const std::string& host_name, int process_id) override;

  void DisposeJoin(const ChangeMsg& msg);
  void DisposeLeave(const ChangeMsg& msg);

  NodeWarehouse nodes_;
};

}  // namespace service_discovery
}  // namespace framework
}  // namespace netaos

#endif  //  CYBER_SERVICE_DISCOVERY_SPECIFIC_MANAGER_NODE_MANAGER_H_
