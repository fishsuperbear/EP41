
#ifndef PHM_CLIENT_IMPL_H
#define PHM_CLIENT_IMPL_H

#include "phm/fault_manager/include/fault_reporter.h"
#include "phm/fault_manager/include/fault_collection_file.h"
#include "phm/task_monitor/include/phm_task_manager.h"
#include "phm/include/phm_def.h"
#include "phm/fault_manager/include/module_config.h"
#include "phm/fault_manager/include/fault_bind_table.h"

#include <stdint.h>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>

namespace hozon {
namespace netaos {
namespace phm {

class FaultReporter;
class PHMClientImpl {
public:
    PHMClientImpl();
    ~PHMClientImpl();

    int32_t Init(const std::string& phmConfigPath,
                 std::function<void(bool)> service_available_callback,
                 std::function<void(ReceiveFault_t)> fault_receive_callback,
                 const std::string& processName);
    int32_t Start(uint32_t delayTime);
    void Stop();
    void Deinit();

    int32_t ReportCheckPoint(const uint32_t checkPointId);
    int32_t ReportFault(const SendFault_t& faultInfo);
    void InhibitFault(const std::vector<uint32_t>& faultKeys);
    void RecoverInhibitFault(const std::vector<uint32_t>& faultKeys);
    void InhibitAllFault();
    void RecoverInhibitAllFault();
    int32_t GetDataCollectionFile(std::function<void(std::vector<std::string>&)> collectionFileCb);

private:
    PHMClientImpl(const PHMClientImpl &);
    PHMClientImpl & operator = (const PHMClientImpl &);

    bool is_stoped_;
    std::atomic<bool> m_isInit;
    std::unique_ptr<PHMTaskManager> task_manager_;
    std::shared_ptr<ModuleConfig> module_cfg_;
    std::thread thread_init_;
    std::shared_ptr<FaultCollectionFile> m_spFaultCollectionFile;
    std::shared_ptr<FaultReporter> m_spFaultReporter;
};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_CLIENT_IMPL_H
