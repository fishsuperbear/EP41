/*!
 * @file diag_server_event_sub.cpp
 * This file contains the implementation of the subscriber functions.
 */

#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/event_manager/diag_server_event_sub.h"
#include "diag/diag_server/include/event_manager/diag_server_event_handler.h"

namespace hozon {
namespace netaos {
namespace diag {


DiagServerEventSub::DiagServerEventSub()
{
}

DiagServerEventSub::~DiagServerEventSub()
{
}

void DiagServerEventSub::init()
{
    DG_INFO << "DiagServerEventSub::init";
    m_spPubsubtype = std::make_shared<reportDemEventPubSubType>();
    m_spProxy = std::make_shared<hozon::netaos::cm::Proxy>(m_spPubsubtype);
    m_spProxy->Init(0, "reportDemEvent");

    m_spDemData = std::make_shared<reportDemEvent>();
}

void DiagServerEventSub::deInit()
{
    m_spProxy->Deinit();
}

void DiagServerEventSub::registCallback()
{
    m_spProxy->Listen(std::bind(&DiagServerEventSub::recvCallback, this));
}

void DiagServerEventSub::recvCallback()
{
    if (m_spProxy->IsMatched()) {
        m_spProxy->Take(m_spDemData);
        DG_INFO << "DiagServerEventSub::recvCallback faultKey : " << m_spDemData->dtcValue()
                << " alarmStatus : " << m_spDemData->alarmStatus();
        DiagServerEventHandler::getInstance()->reportDTCEvent(m_spDemData->dtcValue(), m_spDemData->alarmStatus());
    }
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
