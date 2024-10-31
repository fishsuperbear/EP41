#include "phm/common/include/phm_logger.h"
#include "phm/fault_manager/include/fault_bind_table.h"

namespace hozon {
namespace netaos {
namespace phm {

FaultReceiveTable *FaultReceiveTable::instancePtr_ = nullptr;
std::mutex FaultReceiveTable::mtx_;
std::mutex tableMtx;

FaultReceiveTable *FaultReceiveTable::getInstance()
{
    if (nullptr == instancePtr_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instancePtr_) {
            instancePtr_ = new FaultReceiveTable();
        }
    }
    return instancePtr_;
}

FaultReceiveTable::FaultReceiveTable()
{
}

FaultReceiveTable::~FaultReceiveTable()
{
}

void
FaultReceiveTable::Set(std::shared_ptr<ModuleConfig> cfg, std::function<void(ReceiveFault_t)> recv_callback)
{
    std::unique_lock<std::mutex> lck(tableMtx);
    FaultReceiveMap fault_map;
    fault_map.module_name = cfg->GetModuleName();
    fault_map.cfg = cfg;
    fault_map.recv_callback = recv_callback;

    if (recv_table_.count(fault_map.module_name) > 0) {
        PHM_ERROR << "FaultReceiveTable::Set app name repeat! name:" << fault_map.module_name;
        return;
    }

    recv_table_[fault_map.module_name] = fault_map;
    return ;
}

void
FaultReceiveTable::GetMap(const std::string& module_name, FaultReceiveMap& map)
{
    if (recv_table_.count(module_name) > 0) {
        map = recv_table_[module_name];
    }
}

void
FaultReceiveTable::GetAllMap(ReceiveFault_t& fault, std::vector<FaultReceiveMap>& maps)
{
    std::unique_lock<std::mutex> lck(tableMtx);
    for (auto& item : recv_table_) {
        if (item.second.cfg == nullptr) {
            continue;
        }

        if (item.second.cfg->IsRegistFault(fault)) {
            maps.emplace_back(item.second);
        }
    }
}

}  // namespace phm
}  // namespace netaos
}  // namespace hozon