/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: adf-lite
 * Description: phm adapter
 * Created on: Nov 11, 2023
 *
 */
#ifndef ADFLITE_INCLUDE_PHM_CLIENT_INSTANCE_H_
#define ADFLITE_INCLUDE_PHM_CLIENT_INSTANCE_H_

#include <iostream>
#include <mutex>
#include <vector>

#include "phm/include/phm_client.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
class PhmClientInstance {
 public:
    static PhmClientInstance* getInstance();
    void Init();
    void DeInit();
    // 故障上报
    int32_t ReportFault(const hozon::netaos::phm::SendFault_t& faultInfo);
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
    PhmClientInstance(const PhmClientInstance&);
    PhmClientInstance& operator=(const PhmClientInstance&);

    static PhmClientInstance* instance_;
    static std::mutex mtx_;

    hozon::netaos::phm::PHMClient* phm_client_ptr_;
};
}
}
}

#endif  // ADFLITE_INCLUDE_PHM_CLIENT_INSTANCE_H_
