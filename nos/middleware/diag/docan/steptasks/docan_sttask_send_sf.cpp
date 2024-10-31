/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskSendSF implement
 */

#include "docan_sttask_send_sf.h"
#include "diag/docan/manager/docan_sys_interface.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanSTTaskSendSF::DocanSTTaskSendSF(STObject* pParent, STObject::TaskCB pfnCallback,
        const TaskReqInfo& reqInfo, const TaskResInfo& resInfo)
        : CommandTaskBase(DOCAN_COMMAND_SEND_SF + reqInfo.reqEcu, pParent, pfnCallback, reqInfo, resInfo)
    {
    }

    DocanSTTaskSendSF::~DocanSTTaskSendSF()
    {
    }

    uint32_t DocanSTTaskSendSF::doCommand()
    {
        int32_t ret = -1;
        // check status

        // send L_Data_Request to ecu
        CanPacket packet;
        packet.STmin = m_reqInfo.reqSTmin;
        packet.frame.__pad = 0xAA;
        packet.frame.can_id = m_reqInfo.reqCanid;
        packet.frame.can_dlc = 0x08;
        m_reqInfo.reqBsIndexExpect = 1;
        memset(packet.frame.data, 0xAA, packet.frame.can_dlc);
        packet.frame.data[0] = m_reqInfo.reqContent.size();
        memcpy(&packet.frame.data[1], &m_reqInfo.reqContent[0], m_reqInfo.reqContent.size());
        std::vector<CanPacket> queue;
        queue.push_back(packet);

        if (DIAG_FUNCTIONAL_ADDR_DOCAN == m_reqInfo.reqCanid) {
            DocanSysInterface::instance()->AddAllCanSendQueue(queue);
        }
        else {
            ret = DocanSysInterface::instance()->AddCanSendQueue(m_reqInfo.reqEcu, queue);
            if (ret < 0) {
                DOCAN_LOG_E("AddCanSendQueue failed.");
                return N_TIMEOUT_A;
            }
        }

        m_reqInfo.reqCompletedSize = m_reqInfo.reqContent.size();
        uint32_t timeout = DOCAN_TIMER_100ms + DOCAN_TIMER_As + DOCAN_TIMER_Ar;
        if (m_reqInfo.suppressPosRsp || DIAG_FUNCTIONAL_ADDR_DOCAN == m_reqInfo.reqCanid) {
            timeout = 100;
        }
        if (waitEvent(timeout)) {
            DOCAN_LOG_D("wait event for timer timeout : %d ms.", timeout);
            return eContinue;
        }

        DOCAN_LOG_D("wait event failed.");
        return N_TIMEOUT_A;
    }

    bool DocanSTTaskSendSF::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            DOCAN_LOG_D("onEventAction timeout");
            if (m_reqInfo.suppressPosRsp || DIAG_FUNCTIONAL_ADDR_DOCAN == m_reqInfo.reqCanid) {
                // suppress positive response, no need wait response
                DOCAN_LOG_D("suppress positive response, no need wait response.");
                setTaskResult(N_OK);
            }
            else {
                setTaskResult(N_TIMEOUT_A);
            }
            return true;
        }

        DocanTaskEvent* taskevent = static_cast<DocanTaskEvent*>(event);
        if (nullptr == taskevent) {
            return false;
        }

        // all data send completed, wait response
        if (DocanTaskEvent::checkEvent(DOCAN_EVENT_ECU, m_reqInfo.reqEcu, m_resInfo.resCanid, N_PCItype_SF, event)) {
            // the response is a SF frame
            if (taskevent->getEvtData()[1] != 0x7F && (taskevent->getEvtData()[1] != m_reqInfo.reqContent[0] + 0x40)) {
                // no match response
                DOCAN_LOG_D("onEventAction SF no matched response");
                return false;
            }
            DOCAN_LOG_D("onEventAction SF matched response");
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
                DOCAN_LOG_D("onEventAction FF no matched response");
                return false;
            }
            DOCAN_LOG_D("onEventAction FF matched response");
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
