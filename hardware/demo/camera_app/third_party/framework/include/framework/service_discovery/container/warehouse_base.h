#ifndef CYBER_SERVICE_DISCOVERY_CONTAINER_WAREHOUSE_BASE_H_
#define CYBER_SERVICE_DISCOVERY_CONTAINER_WAREHOUSE_BASE_H_

#include <cstdint>
#include <vector>

#include "framework/service_discovery/role/role.h"

namespace netaos {
namespace framework {
namespace service_discovery {

class WarehouseBase {
 public:
  WarehouseBase() {}
  virtual ~WarehouseBase() {}

  virtual bool Add(uint64_t key, const RolePtr& role, bool ignore_if_exist) = 0;

  virtual void Clear() = 0;
  virtual std::size_t Size() = 0;

  virtual void Remove(uint64_t key) = 0;
  virtual void Remove(uint64_t key, const RolePtr& role) = 0;
  virtual void Remove(const proto::RoleAttributes& target_attr) = 0;

  virtual bool Search(uint64_t key) = 0;
  virtual bool Search(uint64_t key, RolePtr* first_matched_role) = 0;
  virtual bool Search(uint64_t key,
                      proto::RoleAttributes* first_matched_role_attr) = 0;
  virtual bool Search(uint64_t key, std::vector<RolePtr>* matched_roles) = 0;
  virtual bool Search(
      uint64_t key, std::vector<proto::RoleAttributes>* matched_roles_attr) = 0;

  virtual bool Search(const proto::RoleAttributes& target_attr) = 0;
  virtual bool Search(const proto::RoleAttributes& target_attr,
                      RolePtr* first_matched) = 0;
  virtual bool Search(const proto::RoleAttributes& target_attr,
                      proto::RoleAttributes* first_matched_role_attr) = 0;
  virtual bool Search(const proto::RoleAttributes& target_attr,
                      std::vector<RolePtr>* matched_roles) = 0;
  virtual bool Search(
      const proto::RoleAttributes& target_attr,
      std::vector<proto::RoleAttributes>* matched_roles_attr) = 0;

  virtual void GetAllRoles(std::vector<RolePtr>* roles) = 0;
  virtual void GetAllRoles(std::vector<proto::RoleAttributes>* roles_attr) = 0;
};

}  // namespace service_discovery
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SERVICE_DISCOVERY_CONTAINER_WAREHOUSE_BASE_H_
