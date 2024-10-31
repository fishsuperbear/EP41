#ifndef FAULT_REPORTER_H
#define FAULT_REPORTER_H

#include <mutex>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>

#include "phm/include/phm_def.h"
#include "phm/fault_manager/include/fault_debounce_base.h"
#include "phm/fault_manager/include/module_config.h"
#include "phm/fault_manager/include/fault_auto_recovery.h"
#include "sm/include/state_client.h"
#include "sm/include/state_client_impl.h"

namespace hozon {
namespace netaos {
namespace phm {

typedef enum FaultStat {
    RECOVER = 0,
    MATURE,
    IMMATURE,
} FaultStat_t;


class Fault;
class FaultDispatcher;
class FaultReporter final {
public:
    FaultReporter();
    ~FaultReporter();

    int32_t Init(std::function<void(ReceiveFault_t)> fault_receive_callback, const std::string& processName);
    void DeInit();

    int32_t ReportFault(const SendFault_t& faultInfo, std::shared_ptr<ModuleConfig> cfg);
    Fault* GenFault(const std::uint32_t faultId, const std::uint8_t faultObj, std::string moduleName);
    bool ReportFaultImmediate(Fault* fault, const std::uint8_t faultStatus);
    bool IsPrefixMatch(std::string& str, const std::string& prefix);
    void FaultAutoRecoveryCallback(const SendFault_t& faultInfo);

private:
    static std::mutex mtx_;
    std::unique_ptr<hozon::netaos::sm::StateClient> state_client_;

    std::unordered_map<std::uint32_t, Fault*> faultMap_;
    std::unique_ptr<FaultDispatcher> faultDispatcher_;
    std::shared_ptr<FaultAutoRecovery> m_spFaultAutoRecovery;
};

class Fault {
public:
    /**  使用基于计数的消抖策略
     @param[in]  maxCount 最大抖动次数
     @param[in]  timeoutMs 超时时间 [单位:毫秒]
     @param[out] none
     @return     void
     @warning    如果application自行消抖，不必使用该接口
     @note       timeout时间内，如果上报故障的次数达到maxCount，则认为故障成熟
    */
    void UseCountBaseDebouncePolicy(const std::uint32_t maxCount, const std::uint32_t timeoutMs);

    /**  使用基于时间的消抖策略
     @param[in]  timeoutMs 超时时间 [单位:毫秒]
     @param[out] none
     @return     void
     @warning    如果application自行消抖，不必使用该接口
     @note       timeout时间内，如果未收到故障恢复的上报，则认为故障成熟
    */
    void UseTimeBaseDebouncePolicy(const std::uint32_t timeoutMs);

    /**  获取当前的FaultID
     @param[in]  none
     @param[out] none
     @return     FaultID
     @warning    无
     @note       无
    */
    inline std::uint32_t GetFaultId() { return faultId_; }

    /**  获取当前的FaultObj
     @param[in]  none
     @param[out] none
     @return     FaultObj
     @warning    无
     @note       无
    */
    inline std::uint8_t GetFaultObj() { return faultObj_; }

    /**  获取当前的ModuleName
     @param[in]  none
     @param[out] none
     @return     获取当前的ModuleName
     @warning    无
     @note       无
    */
    inline std::string& GetModuleName() { return moduleName_; }

private:
    friend class FaultReporter;
    Fault() = default;
    Fault(const std::uint32_t faultId, const std::uint8_t faultObj, std::string moduleName, std::function<bool(Fault*, const std::uint8_t)> reportFaultCb);
    ~Fault();
    FaultStat_t Report(const std::uint8_t faultStatus);
    void TimeBaseDebouncepolicyTimeoutCallback();

    std::shared_ptr<DebounceBase> debouncePolicyPtr_;
    std::uint32_t faultId_;
    std::uint8_t faultObj_;
    std::string moduleName_;
    std::function<bool(Fault*, const std::uint8_t)> m_reportFaultCb;
};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // FAULT_REPORTER_H
