/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskWaitPending implement
 */

#include "docan_sttask_wait_pending.h"
#include "diag/docan/manager/docan_event_sender.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanSTTaskWaitPending::DocanSTTaskWaitPending(STObject* pParent, STObject::TaskCB pfnCallback,
        const TaskReqInfo& reqInfo, const TaskResInfo& resInfo)
        : CommandTaskBase(DOCAN_COMMAND_WAIT_FC + reqInfo.reqEcu, pParent, pfnCallback, reqInfo, resInfo)
    {
    }

    DocanSTTaskWaitPending::~DocanSTTaskWaitPending()
    {
    }

    uint32_t DocanSTTaskWaitPending::doCommand()
    {
        std::vector<uint8_t> pendingFrame;
        pendingFrame.push_back(0x7F);
        pendingFrame.push_back(m_reqInfo.reqContent[0]);
        pendingFrame.push_back(0x78);
        DocanEventSender::instance()->sendUdsIndication(m_reqInfo.N_TA, m_reqInfo.N_SA, pendingFrame);

        if (waitEvent(DOCAN_TIMER_P2StartServer)) {
            return eContinue;
        }

        return N_TIMEOUT_P2StarServer;
    }

    bool DocanSTTaskWaitPending::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            setTaskResult(N_TIMEOUT_P2StarServer);
            return true;
        }

        DocanTaskEvent* taskevent = static_cast<DocanTaskEvent*>(event);
        if (nullptr == taskevent) {
            return false;
        }

        if (DocanTaskEvent::checkEvent(DOCAN_EVENT_ECU, m_reqInfo.reqEcu, m_resInfo.resCanid, N_PCItype_SF, event)) {
            // the response is a SF frame
            if (taskevent->getEvtData()[1] != 0x7F && (taskevent->getEvtData()[1] != m_reqInfo.reqContent[0] + 0x40)) {
                // no match response
                DOCAN_LOG_D("not match response.");
                return false;
            }
            m_resInfo.resLen = (uint8_t)(taskevent->getEvtData()[0] & 0x0F);
            uint8_t* ptr = &(taskevent->getEvtData()[1]);
            m_resInfo.resContent = std::vector<uint8_t>(ptr, ptr + m_resInfo.resLen);
            setTaskResult(N_OK);
            return true;
        }

        if (DocanTaskEvent::checkEvent(DOCAN_EVENT_ECU, m_reqInfo.reqEcu, m_resInfo.resCanid, N_PCItype_FF, event)) {
            // the response is a FF frame
            if (taskevent->getEvtData()[2] != m_reqInfo.reqContent[0] + 0x40) {
                // not match response
                DOCAN_LOG_D("not match response.");
                return false;
            }
            m_resInfo.resLen = (uint16_t)((taskevent->getEvtData()[0] & 0x0F) << 8 | taskevent->getEvtData()[1]);
            m_resInfo.resCompletedSize = 0x06;
            uint8_t* ptr = &(taskevent->getEvtData()[2]);
            m_resInfo.resContent = std::vector<uint8_t>(ptr, ptr + 6);
            setTaskResult(N_OK);
            return true;
        }

        return false;
    }


} // end of diag
} // end of netaos
} // end of hozon
/* EOF */
