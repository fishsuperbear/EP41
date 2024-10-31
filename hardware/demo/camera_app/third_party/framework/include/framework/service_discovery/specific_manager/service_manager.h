#ifndef CYBER_SERVICE_DISCOVERY_SPECIFIC_MANAGER_SERVICE_MANAGER_H_
#define CYBER_SERVICE_DISCOVERY_SPECIFIC_MANAGER_SERVICE_MANAGER_H_

#include <memory>
#include <string>
#include <vector>

#include "framework/service_discovery/container/multi_value_warehouse.h"
#include "framework/service_discovery/container/single_value_warehouse.h"
#include "framework/service_discovery/role/role.h"
#include "framework/service_discovery/specific_manager/manager.h"

namespace netaos {
namespace framework {
namespace service_discovery {

class TopologyManager;

/**
 * @class ServiceManager
 * @brief Topology Manager of Service related
 */
class ServiceManager : public Manager {
  friend class TopologyManager;

 public:
  using RoleAttrVec = std::vector<RoleAttributes>;
  using ServerWarehouse = SingleValueWarehouse;
  using ClientWarehouse = MultiValueWarehouse;

  /**
   * @brief Construct a new Service Manager object
   */
  ServiceManager();

  /**
   * @brief Destroy the Service Manager object
   */
  virtual ~ServiceManager();

  /**
   * @brief Inquire whether `service_name` exists in topology
   *
   * @param service_name the name we inquire
   * @return true if service exists
   * @return false if service not exists
   */
  bool HasService(const std::string& service_name);

  /**
   * @brief Get the All Server in the topology
   *
   * @param servers result RoleAttr vector
   */
  void GetServers(RoleAttrVec* servers);

  /**
   * @brief Get the Clients object that subscribes `service_name`
   *
   * @param service_name Name of service you want to get
   * @param clients result vector
   */
  void GetClients(const std::string& service_name, RoleAttrVec* clients);

 private:
  bool Check(const RoleAttributes& attr) override;
  void Dispose(const ChangeMsg& msg) override;
  void OnTopoModuleLeave(const std::string& host_name, int process_id) override;

  void DisposeJoin(const ChangeMsg& msg);
  void DisposeLeave(const ChangeMsg& msg);

  ServerWarehouse servers_;
  ClientWarehouse clients_;
};

}  // namespace service_discovery
}  // namespace framework
}  // namespace netaos

#endif  //  CYBER_SERVICE_DISCOVERY_SPECIFIC_MANAGER_SERVICE_MANAGER_H_
