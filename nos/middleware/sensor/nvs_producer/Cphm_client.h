#ifndef PHM_CLIENT_SAMPLE_H
#define PHM_CLIENT_SAMPLE_H

#include <iostream>
#include <mutex>

#include "phm/include/phm_client.h"

using namespace hozon::netaos::phm;

class CameraPhmClient {

public:
    static CameraPhmClient* getInstance();

    void Init();
    void DeInit();

    int32_t CameraReportFault(const SendFault_t& faultInfo);

private:
    CameraPhmClient();
    CameraPhmClient(const CameraPhmClient&);
    CameraPhmClient& operator=(const CameraPhmClient&);

    void CameraServiceAvailableCallback(const bool bResult);
    void CameraFaultReceiveCallback(const ReceiveFault_t& fault);

    static CameraPhmClient* instance_;
    static std::mutex mtx_;

    PHMClient* phm_client_ptr_;
};

#endif  // #define PHM_CLIENT_SAMPLE_H