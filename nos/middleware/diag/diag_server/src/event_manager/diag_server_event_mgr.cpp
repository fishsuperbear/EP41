/*!
 * @file diag_server_event_mgr.cpp
 * This file contains the implementation of the subscriber functions.
 *
 * This file was generated by the tool fastcdrgen.
 */

#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/event_manager/diag_server_event_impl.h"
#include "diag/diag_server/include/event_manager/diag_server_event_sub.h"
#include "diag/diag_server/include/event_manager/diag_server_event_pub.h"
#include "diag/diag_server/include/event_manager/diag_server_event_test.h"
#include "diag/diag_server/include/event_manager/diag_server_event_mgr.h"

namespace hozon {
namespace netaos {
namespace diag {

DiagServerEventMgr* DiagServerEventMgr::m_pInstance = nullptr;
std::mutex DiagServerEventMgr::m_mtx;


DiagServerEventMgr::DiagServerEventMgr()
:m_spEventSub(new DiagServerEventSub())
,m_spEventPub(new DiagServerEventPub())
,m_spEventTest(new DiagServerEventTest())
{
    DG_INFO << "DiagServerEventMgr::DiagServerEventSub";
}

DiagServerEventMgr::~DiagServerEventMgr()
{
    DG_INFO << "DiagServerEventMgr::~DiagServerEventSub";
}

DiagServerEventMgr*
DiagServerEventMgr::getInstance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new DiagServerEventMgr();
        }
    }

    return m_pInstance;
}

void DiagServerEventMgr::Init()
{
    DG_INFO << "DiagServerEventMgr::Init";
    if (nullptr != m_spEventSub) {
        m_spEventSub->init();
        m_spEventSub->registCallback();
    }

    if (nullptr != m_spEventPub) {
        m_spEventPub->init();
    }

    if (nullptr != m_spEventTest) {
        m_spEventTest->init();
    }

    DiagServerEventImpl::getInstance()->newCircle();
    DiagServerEventImpl::getInstance()->setPubHandle(m_spEventPub);
}

void DiagServerEventMgr::DeInit()
{
    DG_INFO << "DiagServerEventMgr::DeInit";
    DiagServerEventImpl::getInstance()->destroy();
    if (nullptr != m_spEventTest) {
        m_spEventTest->deInit();
    }

    if (nullptr != m_spEventPub) {
        m_spEventPub->deInit();
    }

    if (nullptr != m_spEventSub) {
        m_spEventSub->deInit();
    }
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon