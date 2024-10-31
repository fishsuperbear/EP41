/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-08-21 15:38:46
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-09-13 17:19:41
 * @FilePath: /nos/service/config_server/include/phm_client_instance.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: phm adapter
 * Created on: Feb 7, 2023
 *
 */
#ifndef SERVICE_CONFIG_SERVER_INCLUDE_PHM_CLIENT_INSTANCE_H_
#define SERVICE_CONFIG_SERVER_INCLUDE_PHM_CLIENT_INSTANCE_H_

#include <iostream>
#include <mutex>
#include <vector>

#include "include/cfg_logger.h"
#include "include/phm_client.h"
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

#endif  // SERVICE_CONFIG_SERVER_INCLUDE_PHM_CLIENT_INSTANCE_H_
