/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskSendFC implement
 */

#include "docan_sttask_send_fc.h"
#include "diag/docan/manager/docan_sys_interface.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanSTTaskSendFC::DocanSTTaskSendFC(STObject* pParent, STObject::TaskCB pfnCallback,
        const TaskReqInfo& reqInfo, const TaskResInfo& resInfo)
        : CommandTaskBase(DOCAN_COMMAND_SEND_FC + reqInfo.reqEcu, pParent, pfnCallback, reqInfo, resInfo)
    {
    }

    DocanSTTaskSendFC::~DocanSTTaskSendFC()
    {
    }

    uint32_t DocanSTTaskSendFC::doCommand()
    {
        int32_t ret = -1;
        // check status

        // send L_Data_Request to ecu
        CanPacket packet;
        packet.STmin = m_reqInfo.reqSTmin;
        packet.frame.__pad = 0xAA;
        packet.frame.can_id = m_reqInfo.reqCanid;
        packet.frame.can_dlc = 0x08;
        memset(packet.frame.data, 0xAA, packet.frame.can_dlc);
        packet.frame.data[0] = (uint8_t)(0x30);   // FS
        packet.frame.data[1] = (uint8_t)(m_reqInfo.reqBs);   // BS
        packet.frame.data[2] = (uint8_t)(m_reqInfo.reqSTmin); // STmin
        std::vector<CanPacket> queue;
        queue.push_back(packet);

        ret = DocanSysInterface::instance()->AddCanSendQueue(m_reqInfo.reqEcu, queue);
        if (ret < 0) {
            return N_TIMEOUT_Bs;
        }

        m_resInfo.resBsIndexExpect = 1;

        uint32_t timeMs = 0;
        if (0 == m_reqInfo.reqBs) {
            timeMs = DOCAN_TIMER_100ms + DOCAN_TIMER_As + DOCAN_TIMER_Cr + (m_resInfo.resLen / 0x07) * m_reqInfo.reqSTmin;
        }
        else {
            timeMs = DOCAN_TIMER_100ms + DOCAN_TIMER_As + DOCAN_TIMER_Cr + m_reqInfo.reqBs * m_reqInfo.reqSTmin;
        }

        if (waitEvent(timeMs)) {
            return eContinue;
        }

        return N_TIMEOUT_Cr;
    }

    bool DocanSTTaskSendFC::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            setTaskResult(N_TIMEOUT_Cr);
            return true;
        }

        DocanTaskEvent* taskevent = static_cast<DocanTaskEvent*>(event);
        if (nullptr == taskevent) {
            return false;
        }

        if (DocanTaskEvent::checkEvent(DOCAN_EVENT_ECU, m_reqInfo.reqEcu, m_resInfo.resCanid, N_PCItype_CF, event)) {
            if ((m_resInfo.resBsIndexExpect % 0x10) != ((uint8_t)(taskevent->getEvtData()[0]) & 0x0F)) {
                DOCAN_LOG_D("SN is not ordered, sn: %d, expected: %d.", ((uint8_t)(taskevent->getEvtData()[0]) & 0x0F), (m_resInfo.resBsIndexExpect % 0x10));
                setTaskResult(N_UNEXP_PDU);
                return true;
            }

            if (m_resInfo.resCompletedSize + 7 >= m_resInfo.resLen) {
                // the last packet may be not full for 7 bytes
                DOCAN_LOG_D("the last packet may be not full for 7 bytes.");
                uint8_t* ptr = &(taskevent->getEvtData()[1]);
                m_resInfo.resContent.insert(m_resInfo.resContent.end(), ptr, ptr + (m_resInfo.resLen - m_resInfo.resCompletedSize));
                m_resInfo.resCompletedSize = m_resInfo.resLen;
                setTaskResult(N_OK);
                return true;
            }

            uint8_t* cfptr = &(taskevent->getEvtData()[1]);
            m_resInfo.resContent.insert(m_resInfo.resContent.end(), cfptr, cfptr + 7);
            m_resInfo.resBsIndexExpect += 1;
            m_resInfo.resCompletedSize += 7;

            if (m_reqInfo.reqBs != 0 && m_resInfo.resBsIndexExpect == m_reqInfo.reqBs) {
                DOCAN_LOG_D("read BS buff completed, need send another FC frame.");
                setTaskResult(N_OK);
            }
            return true;
        }

        return false;
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */
