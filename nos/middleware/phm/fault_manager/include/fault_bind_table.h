#ifndef FAULT_BIND_TABLE_H
#define FAULT_BIND_TABLE_H

#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <functional>
#include "phm/include/phm_def.h"
#include "phm/fault_manager/include/module_config.h"
#include "phm/common/include/phm_config.h"
#include "phm/common/include/phm_logger.h"

namespace hozon {
namespace netaos {
namespace phm {

struct FaultReceiveMap {
    std::string module_name;
    std::shared_ptr<ModuleConfig> cfg;
    std::function<void(ReceiveFault_t)> recv_callback;
};

class FaultReceiveTable {

public:
    static FaultReceiveTable *getInstance();

    void Set(std::shared_ptr<ModuleConfig> cfg, std::function<void(ReceiveFault_t)> recv_callback);
    void GetMap(const std::string& module_name, FaultReceiveMap& map);
    void GetAllMap(ReceiveFault_t& fault, std::vector<FaultReceiveMap>& maps);

private:
    FaultReceiveTable();
    ~FaultReceiveTable();

    static FaultReceiveTable *instancePtr_;
    static std::mutex mtx_;

    std::unordered_map<std::string, FaultReceiveMap> recv_table_;
};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon

#endif  // FAULT_BIND_TABLE_H