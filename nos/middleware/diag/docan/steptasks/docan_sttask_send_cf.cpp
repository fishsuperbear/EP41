/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskSendCF implement
 */

#include "docan_sttask_send_cf.h"
#include "diag/docan/manager/docan_sys_interface.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanSTTaskSendCF::DocanSTTaskSendCF(STObject* pParent, STObject::TaskCB pfnCallback,
        const TaskReqInfo& reqInfo, const TaskResInfo& resInfo)
        : CommandTaskBase(DOCAN_COMMAND_SEND_CF + reqInfo.reqEcu, pParent, pfnCallback, reqInfo, resInfo)
    {
    }

    DocanSTTaskSendCF::~DocanSTTaskSendCF()
    {
    }

    uint32_t DocanSTTaskSendCF::doCommand()
    {
        int32_t ret = -1;
        // check status

        // send L_Data_Request to ecu
        CanPacket packet;
        packet.STmin = m_resInfo.resSTmin;
        packet.frame.__pad = 0xAA;
        packet.frame.can_id = m_reqInfo.reqCanid;
        packet.frame.can_dlc = 0x08;
        m_reqInfo.reqBsIndexExpect = 1;
        std::vector<CanPacket> queue;
        for (uint16_t it = m_reqInfo.reqCompletedSize; it < m_reqInfo.reqContent.size(); it+=7) {
            memset(packet.frame.data, 0xAA, packet.frame.can_dlc);

            if (it + 7 >= (uint16_t)m_reqInfo.reqContent.size()) {
                // all data send completed.
                DOCAN_LOG_D("all data need send add queue completed.");
                packet.frame.data[0] = (uint8_t)(0x20 | (m_reqInfo.reqBsIndexExpect & 0x0F));
                memcpy(&packet.frame.data[1], &m_reqInfo.reqContent[it], m_reqInfo.reqContent.size() - it);
                queue.push_back(packet);
                m_reqInfo.reqCompletedSize = m_reqInfo.reqContent.size();
                break;
            }

            // add each packet to the queue to send
            packet.frame.data[0] = (uint8_t)(0x20 | (m_reqInfo.reqBsIndexExpect & 0x0F));
            memcpy(&packet.frame.data[1], &m_reqInfo.reqContent[it], 0x07);
            m_reqInfo.reqCompletedSize += 7;
            m_reqInfo.reqBsIndexExpect += 1;
            queue.push_back(packet);

            if (m_resInfo.resBs > 0 && m_reqInfo.reqBsIndexExpect >= m_resInfo.resBs) {
                // BS buff is completed need to send.
                DOCAN_LOG_D("BS buff is completed need to send, resBs: %d, bsIndex: %d.", m_resInfo.resBs, m_reqInfo.reqBsIndexExpect);
                break;
            }
        }

        ret = DocanSysInterface::instance()->AddCanSendQueue(m_reqInfo.reqEcu, queue);
        if (ret < 0) {
            return N_TIMEOUT_Cr;
        }

        uint32_t timeMs = 0;
        if (m_reqInfo.reqCompletedSize >= m_reqInfo.reqContent.size()) {
            if (m_reqInfo.suppressPosRsp) {
                // suppress positive response
                DOCAN_LOG_D("wait for suppress positive response.");
                timeMs = 100;
            }
            else {
                // wait for ecu response SF or FF
                DOCAN_LOG_D("wait for ecu response SF or FF, resBs: %d, resSTmin: %d.", m_resInfo.resBs, m_resInfo.resSTmin);
                timeMs = DOCAN_TIMER_100ms + DOCAN_TIMER_As + DOCAN_TIMER_Ar + (uint32_t)queue.size() * m_resInfo.resSTmin * 10;
            }
        }
        else {
            // wait for ecu another FC frame to continue send
            DOCAN_LOG_D("wait for ecu another FC frame to caontinue send, reqCompletedSize: %d, resBs: %d, resSTmin: %d.", m_reqInfo.reqCompletedSize, m_resInfo.resBs, m_resInfo.resSTmin);
            timeMs = DOCAN_TIMER_100ms + DOCAN_TIMER_As + DOCAN_TIMER_Cs + (uint32_t)queue.size() * m_resInfo.resSTmin * 10;
        }

        if (waitEvent(timeMs)) {
            return eContinue;
        }
        return N_TIMEOUT_Cr;
    }

    bool DocanSTTaskSendCF::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            if (m_reqInfo.reqCompletedSize >= m_reqInfo.reqContent.size()) {
                // all data send completed
                if (m_reqInfo.suppressPosRsp) {
                    // suppress positive response
                    DOCAN_LOG_D("suppress positive response.");
                    setTaskResult(N_OK);
                }
                else {
                    // wait response timeout
                    DOCAN_LOG_D("wait response timeout.");
                    setTaskResult(N_TIMEOUT_Cr);
                }
            }
            else {
                // wait next FC frame to continue send, while timeout
                DOCAN_LOG_D("wait next FC frame to continue send, while timeout N_TIMEOUT_Bs: %d.", N_TIMEOUT_Bs);
                setTaskResult(N_TIMEOUT_Bs);
            }
            return true;
        }

        DocanTaskEvent* taskevent = static_cast<DocanTaskEvent*>(event);
        if (nullptr == taskevent) {
            return false;
        }

        if (m_reqInfo.reqCompletedSize >= m_reqInfo.reqContent.size()) {
            // all data send completed, wait response
            if (DocanTaskEvent::checkEvent(DOCAN_EVENT_ECU, m_reqInfo.reqEcu, m_resInfo.resCanid, N_PCItype_SF, event)) {
                // the response is a SF frame
                DOCAN_LOG_D("the response is a SF frame.");
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
                DOCAN_LOG_D("the response is a FF frame.");
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
        }
        else {
            // wait next FC frame to continue send
            DOCAN_LOG_D("wait next FC frame to continue send.");
            if (DocanTaskEvent::checkEvent(DOCAN_EVENT_ECU, m_reqInfo.reqEcu, m_resInfo.resCanid, N_PCItype_FC, event)) {
                m_resInfo.resContent.clear();
                m_resInfo.resLen = 0;
                m_resInfo.resCompletedSize = 0;
                setTaskResult(N_OK);
                return true;
            }
        }
        return false;
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */
