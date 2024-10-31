/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fault event inhibit
*/

#include "phm/common/include/phm_logger.h"
#include "phm/fault_manager/include/fault_inhibit_judge.h"



namespace hozon {
namespace netaos {
namespace phm {


FaultInhibitJudge* FaultInhibitJudge::instancePtr_ = nullptr;
std::mutex FaultInhibitJudge::mtx_;

FaultInhibitJudge*
FaultInhibitJudge::getInstance()
{
    if (nullptr == instancePtr_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instancePtr_) {
            instancePtr_ = new FaultInhibitJudge();
        }
    }
    return instancePtr_;
}

FaultInhibitJudge::FaultInhibitJudge()
{
}

FaultInhibitJudge::~FaultInhibitJudge()
{
}

uint8_t
FaultInhibitJudge::CheckReportCondition()
{
    std::unique_lock<std::mutex> lck(mtx_);
    uint8_t result = PHM_INHIBIT_TYPE_NONE;
    switch (m_inhibitType) {
    case PHM_INHIBIT_TYPE_OTA:
    case PHM_INHIBIT_TYPE_CALIBRATION:
    case PHM_INHIBIT_TYPE_PARKING:
    case PHM_INHIBIT_TYPE_85:
    case PHM_INHIBIT_TYPE_POWERMODE_OFF:
    case PHM_INHIBIT_TYPE_RUNNING_MODE:
        {
            result = m_inhibitType;
        }
        break;
    default:
        break;
    }

    return result;
}

void
FaultInhibitJudge::SetInhibitType(const uint32_t type)
{
    std::unique_lock<std::mutex> lck(mtx_);
    m_inhibitType = type;
}


}  // namespace phm
}  // namespace netaos
}  // namespace hozon