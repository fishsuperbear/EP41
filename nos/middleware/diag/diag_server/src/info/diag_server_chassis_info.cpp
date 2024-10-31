/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server chassis info
*/

#include "diag/diag_server/include/info/diag_server_chassis_info.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag_server_transport_cm.h"

namespace hozon {
namespace netaos {
namespace diag {

DiagServerChassisInfo* DiagServerChassisInfo::instance_ = nullptr;
std::mutex DiagServerChassisInfo::mtx_;

DiagServerChassisInfo::DiagServerChassisInfo()
{
    memset(&chassis_data_, 0x00, sizeof(chassis_data_));
}

DiagServerChassisInfo*
DiagServerChassisInfo::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerChassisInfo();
        }
    }

    return instance_;
}

void
DiagServerChassisInfo::Init()
{
    DG_INFO << "DiagServerChassisInfo::Init";
}

void
DiagServerChassisInfo::DeInit()
{
    DG_INFO << "DiagServerChassisInfo::DeInit";
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
DiagServerChassisInfo::UpdateChassisInfo()
{
    DG_INFO << "DiagServerChassisInfo::UpdateChassisInfo";
    cm_transport::DiagServerTransPortCM::getInstance()->ChassisMethodSend();
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon