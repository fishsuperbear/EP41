/**
 * @file DiagServerEventStatus.cpp
 * @brief Implementation file of class DiagServerEventStatus.
 */

#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/event_manager/diag_server_event_status.h"

namespace hozon {
namespace netaos {
namespace diag {

// init status value
const uint8_t byStatusInitVal = 0b00000000;
bool DiagServerEventStatus::s_bStatusSetting = true;


DiagServerEventStatus::DiagServerEventStatus()
: m_curFaultKey(0)
, m_lastFaultObj(0)
, m_iDtc(0)
, m_uTripCounter(0)
, m_iConfigTripCounter(0)
, m_eventStatus(0)
, m_bDtcExist(true)
, m_bNotifyHmi(false)
, m_syncTrip()
, m_syncStatus()
{
    m_dtcBitStatus.byDTCStatus = byStatusInitVal;
    DG_INFO << "DiagServerEventStatus::DiagServerEventStatus";
}

DiagServerEventStatus::~DiagServerEventStatus()
{

}

void DiagServerEventStatus::setTripCounter(uint8_t iTripCounter)
{
    // DG_INFO << "DiagServerEventStatus::setTripCounter iTripCounter:" << iTripCounter;
    m_uTripCounter = iTripCounter;
}

void DiagServerEventStatus::setDTCStatus(uint8_t byDtcStatus)
{
    DG_INFO << "DiagServerEventStatus::setDTCStatus byDtcStatus:" << byDtcStatus;
    m_dtcBitStatus.byDTCStatus = byDtcStatus;
}

uint8_t DiagServerEventStatus::getDTCStatus()
{
    DG_INFO << "DiagServerEventStatus::getDTCStatus m_dtcBitStatus.byDTCStatus:" << m_dtcBitStatus.byDTCStatus;
    return m_dtcBitStatus.byDTCStatus;
}

void DiagServerEventStatus::setStatusSetting(const DIAG_CONTROLDTCSTATUSTYPE bStatusSetting)
{
    std::unique_lock<std::mutex> lck(m_syncStatus);
    if (DIAG_CONTROLDTCSTATUSTYPE::kDTCSettingOn == bStatusSetting) {
        s_bStatusSetting = true;
    }
    else if (DIAG_CONTROLDTCSTATUSTYPE::kDTCSettingOff == bStatusSetting) {
        s_bStatusSetting = false;
    }
}

bool DiagServerEventStatus::getStatusSetting()
{
    std::unique_lock<std::mutex> lck(m_syncStatus);
    return s_bStatusSetting;
}

uint8_t DiagServerEventStatus::onStatusChange(DIAG_DTCSTSCHGCON eStaChaCon)
{
    DG_INFO << "DiagServerEventStatus::onStatusChange eStaChaCon:" << eStaChaCon;
    {
        // when s_bStatusSetting is false, and eStaChaCon is OCCUR or RECOVER, does not update status
        std::unique_lock<std::mutex> lck(m_syncStatus);
        if (false == s_bStatusSetting && eStaChaCon != DIAG_DTCSTSCHGCON_CLEAR) {
            DG_INFO << "DiagServerEventStatus::onStatusChange, dtc setting off, does not update status:" << m_dtcBitStatus.byDTCStatus;
            return m_dtcBitStatus.byDTCStatus;
        }
    }

    dealWithDTCChange(eStaChaCon);
    return m_dtcBitStatus.byDTCStatus;
}


bool DiagServerEventStatus::dealWithDTCChange(DIAG_DTCSTSCHGCON eStaChaCon)
{
    DG_INFO << "DiagServerEventStatus::dealWithDTCChange +,  dtc: status:" << m_dtcBitStatus.byDTCStatus;
    bool bResult = false;

    switch (eStaChaCon) {
    case DIAG_DTCSTSCHGCON_OCCUR :
        bResult = onDTCOccur();
        break;
    case DIAG_DTCSTSCHGCON_RECOVER :
        bResult = onDTCRecover();
        break;
    case DIAG_DTCSTSCHGCON_CLEAR :
        bResult = onDTCClear();
        break;
    case DIAG_DTCSTSCHGCON_CIRCLE_CHANGE :
        bResult = onDTCCircleChange();
        break;
    default :
        DG_INFO << "DiagServerEventStatus::dealWithDTCChange eStaChaCon error";
        bResult = false;
        break;
    }

    DG_INFO << "DiagServerEventStatus::dealWithDTCChange -,  dtc: status:" << m_dtcBitStatus.byDTCStatus;
    return bResult;
}

bool DiagServerEventStatus::onDTCOccur()
{
    // DG_INFO << "DiagServerEventStatus::onDTCOccur";

    if (0 == m_dtcBitStatus.byTestFaiThOpCycle) {
        ++m_uTripCounter;
        // DG_INFO << "DiagServerEventStatus::onDTCOccur byTestFaiThOpCycle is 0 and ++m_uTripCounter:" << m_uTripCounter;
        if(isTripThreshold()) {
            // DG_INFO << "DiagServerEventStatus::onDTCOccur byTestFaiThOpCycle tripcounter is topper and turn to 0, the byConfirmedDTC status turn to 1";
            m_uTripCounter = 0;
            m_dtcBitStatus.byConfirmedDTC = 1;
        }
    }

    m_dtcBitStatus.byTestFailed = 1;
    m_dtcBitStatus.byTestFaiThOpCycle = 0;
    m_dtcBitStatus.byPendingDTC = 0;
    m_dtcBitStatus.byTestNCompSinLaClear = 0;
    m_dtcBitStatus.byTestFaiSinLaClear = 0;
    m_dtcBitStatus.byTestNCompThOpCycle = 0;
    m_dtcBitStatus.byWarningIndicatorReq = 0;

    if (getNotifyHmi()) {
        m_dtcBitStatus.byWarningIndicatorReq = 1;
    }

    return true;
}

bool DiagServerEventStatus::onDTCRecover()
{
    // DG_INFO << "DiagServerEventStatus::onDTCRecover";
    m_dtcBitStatus.byTestFailed = 0;
    m_dtcBitStatus.byTestNCompSinLaClear = 0;
    m_dtcBitStatus.byTestNCompThOpCycle = 0;

    if (0 == m_dtcBitStatus.byTestFaiThOpCycle && 0 == m_dtcBitStatus.byTestNCompThOpCycle) {
        m_dtcBitStatus.byPendingDTC = 0;
    }
    return true;
}

bool DiagServerEventStatus::onDTCClear()
{
    // DG_INFO << "DiagServerEventStatus::onDTCClear";
    m_dtcBitStatus.byTestFailed = 0;
    m_dtcBitStatus.byTestFaiThOpCycle = 0;
    m_dtcBitStatus.byPendingDTC = 0;
    m_dtcBitStatus.byConfirmedDTC = 0;
    m_dtcBitStatus.byTestNCompSinLaClear = 0;
    m_dtcBitStatus.byTestFaiSinLaClear = 0;
    m_dtcBitStatus.byTestNCompThOpCycle = 0;
    m_dtcBitStatus.byWarningIndicatorReq = 0;
    return true;
}

bool DiagServerEventStatus::onDTCCircleChange()
{
    DG_INFO << "DiagServerEventStatus::onDTCCircleChange";
    m_dtcBitStatus.byTestFailed = 0;
    m_dtcBitStatus.byTestFaiThOpCycle = 0;
    m_dtcBitStatus.byTestNCompThOpCycle = 0;
    return true;
}

bool
DiagServerEventStatus::isTripThreshold()
{
    std::unique_lock<std::mutex> lck(m_syncTrip);
    DG_INFO << "DiagServerEventStatus::isTripThreshold m_iConfigTripCounter:" << m_iConfigTripCounter;
    if (m_iConfigTripCounter <= m_uTripCounter) {
        return true;
    }

    return false;
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
