#ifndef FAULT_AUTO_RECOVERY_H
#define FAULT_AUTO_RECOVERY_H

#include <mutex>
#include <memory>
#include <vector>
#include <unordered_map>

#include "phm/include/phm_def.h"
#include "phm/common/include/timer_manager.h"

namespace hozon {
namespace netaos {
namespace phm {

class FaultAutoRecovery {
public:
    FaultAutoRecovery(std::function<void(const SendFault_t&)> cb);
    ~FaultAutoRecovery();

    void Init();
    void DeInit();

    bool NotifyFaultInfo(const SendFault_t& faultInfo);

    void StartTimer(const uint32_t faultKey, unsigned int msTime);

    void StopTimer(const uint32_t faultKey);

    bool DealWithFault(const SendFault_t& faultInfo);

    bool ReportFault(const SendFault_t& faultInfo);

private:
    void RecoveryTimeoutCallback(void* data);

    // map<faultkey, timefd>
    std::unordered_map<uint32_t, int> fault_map_;
    // map<faultkey, faultStatus>
    std::unordered_map<uint32_t, uint8_t> fault_once_map_;

    std::function<void(const SendFault_t&)> m_faultReporterCb;
};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // FAULT_AUTO_RECOVERY_H
