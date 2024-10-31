/**
 * @file DiagServerEventStatus.h
 * @brief Implementation file of class DiagServerEventStatus.
 */
#pragma once
#include "diag/diag_server/include/common/diag_server_def.h"
#include <iostream>
#include <mutex>

namespace hozon {
namespace netaos {
namespace diag {

 // dtc status change condition
enum DIAG_DTCSTSCHGCON
{
    DIAG_DTCSTSCHGCON_OCCUR,              // dtc occor, TestResult(Failed)
    DIAG_DTCSTSCHGCON_RECOVER,            // dtc recover, TestResult(Passed)
    DIAG_DTCSTSCHGCON_CLEAR,              // dtc clear DiagnosticInformation
    DIAG_DTCSTSCHGCON_CIRCLE_CHANGE,      // dtc operation circle change
    DIAG_DTCSTSCHGCON_MAX,
};

union DiagDtcBitStatus
{
    struct
    {
        uint8_t byTestFailed : 1;              // testFailed (bit0)
        uint8_t byTestFaiThOpCycle : 1;        // testFailedThisOperationCycle (bit1)
        uint8_t byPendingDTC : 1;              // pendingDTC (bit2)
        uint8_t byConfirmedDTC : 1;            // confirmedDTC (bit3)
        uint8_t byTestNCompSinLaClear : 1;     // testNotCompletedSinceLastClear (bit4)
        uint8_t byTestFaiSinLaClear : 1;       // testFailedSinceLastClear (bit5)
        uint8_t byTestNCompThOpCycle : 1;      // testNotCompletedThisOperationCycle (bit6)
        uint8_t byWarningIndicatorReq : 1;     // warningIndicatorRequeseted (bit7)
    };
    uint8_t byDTCStatus;
};

class DiagServerEventStatus
{
public:

    DiagServerEventStatus();
    ~DiagServerEventStatus();

    // set TripCounter of Dtc
    void setTripCounter(uint8_t iTripCounter);

    void setConfigTripCounter(uint8_t iConfigTripCounter) { m_iConfigTripCounter = iConfigTripCounter; };

    // get TripCounter of Dtc
    uint8_t getTripCounter() { return m_uTripCounter; };

    uint8_t getConfigTripCounter() { return m_iConfigTripCounter; };

    // set Status of Dtc
    void setDTCStatus(uint8_t byDtcStatus);

    // get Status of Dtc
    uint8_t getDTCStatus();

    // set Status setting of Dtc, default on
    void setStatusSetting(const DIAG_CONTROLDTCSTATUSTYPE bStatusSetting);

    // get Status setting of Dtc, default on
    bool getStatusSetting();

    // dtc status changes some reasons : dtc occor, recover, clear or operation circle change
    uint8_t onStatusChange(DIAG_DTCSTSCHGCON eStaChaCon);

    // set dtc code
    void setDtcCode(const uint32_t iCode) { m_iDtc = iCode; }

    // get dtc code
    uint32_t getDtcCode() { return m_iDtc; }

    // set dtc occur/recover status
    void setEventStatus(const uint8_t eventStatus) { m_eventStatus = eventStatus; }

    // get dtc occur/recover status
    uint8_t getEventStatus() { return m_eventStatus; }

    void setDtcExist(const bool isExist) { m_bDtcExist = isExist; }

    bool getDtcExist() { return m_bDtcExist; }

    void setNotifyHmi(const bool isNotifyHmi) { m_bNotifyHmi = isNotifyHmi; }

    bool getNotifyHmi() { return m_bNotifyHmi; }

    void setCurFaultKey(const uint32_t faultKey) { m_curFaultKey = faultKey; }

    uint32_t getCurFaultKey() { return m_curFaultKey; }

    void setLastFaultObj(const uint64_t faultObj) { m_lastFaultObj = faultObj; }

    uint64_t getLastFaultObj() { return m_lastFaultObj; }

private:

    bool dealWithDTCChange(DIAG_DTCSTSCHGCON eStaChaCon);

    bool onDTCOccur();

    bool onDTCRecover();

    bool onDTCClear();

    bool onDTCCircleChange();

    bool isTripThreshold();

    uint32_t m_curFaultKey;
    uint64_t m_lastFaultObj;
    uint32_t m_iDtc;
    uint8_t m_uTripCounter;
    uint8_t m_iConfigTripCounter;
    uint8_t m_eventStatus;
    bool m_bDtcExist;
    bool m_bNotifyHmi;
    std::mutex m_syncTrip;
    std::mutex m_syncStatus;
    DiagDtcBitStatus m_dtcBitStatus;
    static bool s_bStatusSetting;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
