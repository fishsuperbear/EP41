#ifndef PHM_CLIENT_SAMPLE_H
#define PHM_CLIENT_SAMPLE_H

#include <mutex>
#include <iostream>

#include "phm/include/phm_def.h"
#include "phm/include/phm_client.h"

using namespace hozon::netaos::phm;

class PhmClientInstance {

public:
    static PhmClientInstance* getInstance();

    void Init();
    void DeInit();

    int32_t ReportCheckPoint(uint32_t checkPointId);
    int32_t ReportFault(const SendFault_t& faultInfo);
    int32_t Start();
    
    void Stop();

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