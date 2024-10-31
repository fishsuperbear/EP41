/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskSendFF implement
 */

#include "docan_sttask_send_ff.h"
#include "diag/docan/manager/docan_sys_interface.h"
#include "diag/docan/common/docan_internal_def.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanSTTaskSendFF::DocanSTTaskSendFF(STObject* pParent, STObject::TaskCB pfnCallback,
        const TaskReqInfo& reqInfo, const TaskResInfo& resInfo)
        : CommandTaskBase(DOCAN_COMMAND_SEND_FF + reqInfo.reqEcu, pParent, pfnCallback, reqInfo, resInfo)
    {
    }

    DocanSTTaskSendFF::~DocanSTTaskSendFF()
    {
    }

    uint32_t DocanSTTaskSendFF::doCommand()
    {
        int32_t ret = -1;
        // check status

        // send L_Data_Request to ecu
        CanPacket packet;
        packet.STmin = 0;
        packet.frame.__pad = 0xAA;
        packet.frame.can_id = m_reqInfo.reqCanid;
        packet.frame.can_dlc = 0x08;
        memset(packet.frame.data, 0xAA, packet.frame.can_dlc);
        uint16_t len = m_reqInfo.reqContent.size();
        packet.frame.data[0] = (uint8_t)(0x10 | ((uint8_t)(len >> 8)));
        packet.frame.data[1] = (uint8_t)(len & 0xFF);
        m_reqInfo.reqCompletedSize = 0x06;
        memcpy(&packet.frame.data[2], &m_reqInfo.reqContent[0], m_reqInfo.reqCompletedSize);
        std::vector<CanPacket> queue;
        queue.push_back(packet);

        ret = DocanSysInterface::instance()->AddCanSendQueue(m_reqInfo.reqEcu, queue);
        if (ret < 0) {
            return N_TIMEOUT_A;
        }

        uint32_t timeout = DOCAN_TIMER_100ms + DOCAN_TIMER_As + DOCAN_TIMER_Bs;
        if (m_reqInfo.suppressPosRsp || DIAG_FUNCTIONAL_ADDR_DOCAN == m_reqInfo.reqCanid) {
            timeout = 100;
        }
        if (waitEvent(timeout)) {
            return eContinue;
        }

        return N_TIMEOUT_A;
    }

    bool DocanSTTaskSendFF::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            setTaskResult(N_TIMEOUT_Bs);
            return true;
        }

        DocanTaskEvent* taskevent = static_cast<DocanTaskEvent*>(event);
        if (nullptr == taskevent) {
            return false;
        }

        if (DocanTaskEvent::checkEvent(DOCAN_EVENT_ECU, m_reqInfo.reqEcu, m_resInfo.resCanid, N_PCItype_FC, event)) {
            m_resInfo.resContent = taskevent->getEvtData();
            m_resInfo.resFs = m_resInfo.resContent[0] & 0x0F;
            m_resInfo.resBs = m_resInfo.resContent[1];
            m_resInfo.resSTmin = m_resInfo.resContent[2];
            if (m_resInfo.resFs > 2 || m_resInfo.resSTmin > 0x7F) {
                // invalid FC frame
                DOCAN_LOG_E("invalid FC frame.");
                setTaskResult(N_INVALID_FS);
            }
            else {
                DOCAN_LOG_D("recv FC frame resFs: %x, resBs: %d, resSTmin: %d.", m_resInfo.resFs, m_resInfo.resBs, m_resInfo.resSTmin);
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
