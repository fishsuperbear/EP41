/*!
 * @file diag_server_event_pub.cpp
 * This file contains the implementation of the subscriber functions.
 */

#include "idl/generated/phmPubSubTypes.h"
#include "idl/generated/diagPubSubTypes.h"
#include "diag/common/include/format.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/event_manager/diag_server_event_pub.h"

namespace hozon {
namespace netaos {
namespace diag {


DiagServerEventPub::DiagServerEventPub()
{
}

DiagServerEventPub::~DiagServerEventPub()
{
}

void DiagServerEventPub::init()
{
    m_spFaultEventPub = std::make_shared<fault_eventPubSubType>();
    m_spSkeleton = std::make_shared<hozon::netaos::cm::Skeleton>(m_spFaultEventPub);
    m_spSkeleton->Init(0, "fault_eventTopic");

    m_spDtcControlSettingSwPub = std::make_shared<dtcControlSettingSwPubSubType>();
    m_spDtcControlSettingSwSkeleton = std::make_shared<hozon::netaos::cm::Skeleton>(m_spDtcControlSettingSwPub);
    m_spDtcControlSettingSwSkeleton->Init(0, "dtcControlSettingSw");
}

void DiagServerEventPub::deInit()
{
    DG_INFO << "DiagServerEventPub::deInit";
    m_spSkeleton->Deinit();
}

void DiagServerEventPub::sendFaultEvent(uint32_t faultKey, uint8_t status)
{
    timespec curDate;
    clock_gettime(CLOCK_REALTIME, &curDate);
    // DG_INFO << "DiagServerEventPub::sendFaultEvent cur sec:" << curDate.tv_sec;

    std::shared_ptr<fault_event> data = std::make_shared<fault_event>();
    data->domain("dtc_recover");
    data->occur_time(curDate.tv_sec);
    data->fault_id(faultKey / 100);
    data->fault_obj(faultKey % 100);
    data->fault_status(status);
    DG_INFO << Format("DiagServerEventPub::sendFaultEvent domain:{0}, time:{1}, faultid:{2}, faultobj:{3}, faultstatus:{4}",
                data->domain(), data->occur_time(), data->fault_id(), (int)data->fault_obj(), (int)data->fault_status());

    if (!m_spSkeleton) {
        DG_ERROR << "DiagServerEventPub::sendFaultEvent m_spSkeleton null";
        return;
    }

    if (!m_spSkeleton->IsMatched()) {
        DG_ERROR << "DiagServerEventPub::sendFaultEvent not matched";
        // return;
    }

    if (m_spSkeleton->Write(data) != 0) {
        DG_ERROR << "DiagServerEventPub::sendFaultEvent write failed";
    }

    return;
}

void DiagServerEventPub::notifyDtcControlSetting(const uint8_t dtcControlSetting)
{
    DG_INFO << "DiagServerEventPub::notifyDtcControlSetting dtcControlSetting:" << dtcControlSetting;
    std::shared_ptr<dtcControlSettingSw> data = std::make_shared<dtcControlSettingSw>();
    data->controlSettingSw(dtcControlSetting);
    if (!m_spDtcControlSettingSwSkeleton) {
        DG_ERROR << "DiagServerEventPub::notifyDtcControlSetting m_spDtcControlSettingSwSkeleton null";
        return;
    }

    if (!m_spDtcControlSettingSwSkeleton->IsMatched()) {
        DG_ERROR << "DiagServerEventPub::notifyDtcControlSetting not matched";
        // return;
    }

    if (m_spDtcControlSettingSwSkeleton->Write(data) != 0) {
        DG_ERROR << "DiagServerEventPub::notifyDtcControlSetting write failed";
    }

    return;
}

void DiagServerEventPub::notifyHmi()
{
    DG_INFO << "DiagServerEventPub::notifyHmi";
    // TODO notifyHmi
    return;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
