#ifndef PHM_CLIENT_SAMPLE_H
#define PHM_CLIENT_SAMPLE_H

#include <mutex>

#include "phm/include/phm_client.h"

using namespace hozon::netaos::phm;


class PhmClientInstance {

public:
    static PhmClientInstance* getInstance();

    void Init();
    void DeInit();

    // 故障上报
    int32_t ReportFault(const SendFault_t& faultInfo);
    // 监控任务上报检查点
    int32_t ReportCheckPoint(uint32_t checkPointId);

    // 故障抑制
    void InhibitFault(const std::vector<uint32_t>& faultKeys);
    // 故障抑制的恢复
    void RecoverInhibitFault(const std::vector<uint32_t>& faultKeys);

    // 抑制所有故障
    void InhibitAllFault();
    // 所有故障抑制的恢复
    void RecoverInhibitAllFault();

private:
    PhmClientInstance();
    PhmClientInstance(const PhmClientInstance &);
    PhmClientInstance & operator = (const PhmClientInstance &);

private:
    void ServiceAvailableCallback(const bool bResult);
    void FaultReceiveCallback(const ReceiveFault_t& fault);

private:
    static PhmClientInstance* instance_;
    static std::mutex mtx_;

    PHMClient* phm_client_ptr_;
};

#endif  // #define PHM_CLIENT_SAMPLE_H