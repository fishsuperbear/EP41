#pragma once

#include <mutex>


namespace hozon {
namespace netaos {
namespace phm {

enum PHM_INHIBIT_TYPE
{
    PHM_INHIBIT_TYPE_NONE = 0,
    PHM_INHIBIT_TYPE_OTA,
    PHM_INHIBIT_TYPE_CALIBRATION,
    PHM_INHIBIT_TYPE_PARKING,
    PHM_INHIBIT_TYPE_85,
    PHM_INHIBIT_TYPE_POWERMODE_OFF,
    PHM_INHIBIT_TYPE_RUNNING_MODE
};


class FaultInhibitJudge
{
public:
    static FaultInhibitJudge* getInstance();
    uint8_t CheckReportCondition();
    void SetInhibitType(const uint32_t type);

private:
    FaultInhibitJudge();
    ~FaultInhibitJudge();
    static FaultInhibitJudge* instancePtr_;
    static std::mutex mtx_;

    uint32_t m_inhibitType;
};


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
