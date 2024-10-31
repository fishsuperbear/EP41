
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/event_manager/diag_server_event_handler.h"
#include "diag/diag_server/include/event_manager/diag_server_event_status.h"
#include "diag/diag_server/include/event_manager/diag_server_event_task.h"
#include "diag/diag_server/include/event_manager/diag_server_event_impl.h"

namespace hozon {
namespace netaos {
namespace diag {

int DiagServerClearDTCInfo::Run()
{
    DG_INFO << "DiagServerClearDTCInfo task run";
    bool isdeleted = DiagServerEventImpl::getInstance()->clearDTCInformation(m_dtcGroup);
    DiagServerEventHandler::getInstance()->replyClearAllDtc(isdeleted);
    return 0;
}

int DiagServerReportDTCNumByStatusMask::Run()
{
    DG_INFO << "DiagServerReportDTCNumByStatusMask task run";
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->reportDTCByStatusMask(m_dtcStatusMask, dtcInfos);
    DiagServerEventHandler::getInstance()->replyNumberOfDTCByStatusMask(dtcInfos.size());
    return 0;
}

int DiagServerReportDTCByStatusMask::Run()
{
    DG_INFO << "DiagServerReportDTCByStatusMask task run";
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->reportDTCByStatusMask(m_dtcStatusMask, dtcInfos);
    DiagServerEventHandler::getInstance()->replyDTCByStatusMask(dtcInfos);
    return 0;
}

int DiagServerReportDTCSnapshotIdentification::Run()
{
    DG_INFO << "DiagServerReportDTCSnapshotIdentification task run";
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->reportDTCSnapshotIdentification(dtcInfos);
    DiagServerEventHandler::getInstance()->replyDTCSnapshotIdentification(dtcInfos);
    return 0;
}

int DiagServerReportDTCSnapshotRecordByDTCNumber::Run()
{
    DG_INFO << "DiagServerReportDTCSnapshotRecordByDTCNumber task run";
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->reportDTCSnapshotRecordByDTCNumber(m_dtcValue, m_number, dtcInfos);
    DiagServerEventHandler::getInstance()->replyDTCSnapshotRecordByDTCNumber(dtcInfos, m_number);
    return 0;
}

int DiagServerReportSupportedDTC::Run()
{
    DG_INFO << "DiagServerReportSupportedDTC task run";
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->reportSupportedDTC(dtcInfos);
    DiagServerEventHandler::getInstance()->replySupportedDTC(dtcInfos);
    return 0;
}

int DiagServerControlDTCStatusType::Run()
{
    DG_INFO << "DiagServerControlDTCStatusType task run";
    DiagServerEventStatus controlSetting;
    controlSetting.setStatusSetting(m_controlDtcStatusType);
    // DiagServerEventImpl::getInstance()->notifyDtcControlSetting(m_controlDtcStatusType);
    DiagServerEventHandler::getInstance()->replyControlDTCStatusType(m_controlDtcStatusType);
    return 0;
}

int DiagServerReportDTCEvent::Run()
{
    DG_INFO << "DiagServerReportDTCEvent task run";
    DiagServerEventImpl::getInstance()->reportDTCEvent(m_dtcValue, m_dtcStatus);
    return 0;
}

int DiagServerSessionChange::Run()
{
    DG_INFO << "DiagServerSessionChange task run";
    DiagServerEventStatus controlSetting;
    controlSetting.setStatusSetting(DIAG_CONTROLDTCSTATUSTYPE::kDTCSettingOn);
    return 0;
}

int DiagServerRequestOutputDtcInfo::Run()
{
    DG_INFO << "DiagServerRequestOutputDtcInfo task run";
    DiagServerEventImpl::getInstance()->requestOutputDtcInfo();
    DiagServerEventHandler::getInstance()->replyOutputDtcInfo(true);
    return 0;
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
