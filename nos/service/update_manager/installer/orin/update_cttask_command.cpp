/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskCommand implement
 */

#include "update_cttask_command.h"
#include "update_manager/agent/diag_agent.h"
#include "update_manager/taskbase/update_task_event.h"
#include "update_manager/common/data_def.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

    UpdateCTTaskCommand::UpdateCTTaskCommand(STObject* pParent, STObject::TaskCB pfnCallback,
                                   const TaskReqInfo& reqInfo, const TaskResInfo& resInfo)
        : CommandTaskBase(OTA_CTTASK_SEND_COMMAND + reqInfo.reqTa, pParent, pfnCallback)
        , m_suppressPositiveResponse(false)
        , m_functionAddr(false)
        , m_reqInfo(reqInfo)
        , m_resInfo(resInfo)
    {
        if (0x10 == m_reqInfo.reqContent[0] || 0x11 == m_reqInfo.reqContent[0]
           || 0x28 == m_reqInfo.reqContent[0] || 0x3E == m_reqInfo.reqContent[0]
           || 0x85 == m_reqInfo.reqContent[0]) {
            m_suppressPositiveResponse = (0x80 == (m_reqInfo.reqContent[1] & 0x80));
        }

        if ((reqInfo.reqUpdateType == 1 && reqInfo.reqTa == UPDATE_MANAGER_FUNCTIONAL_ADDR_DOCAN)
            || (reqInfo.reqUpdateType == 2 && reqInfo.reqTa == UPDATE_MANAGER_FUNCTIONAL_ADDR_DOIP)) {
            m_functionAddr = true;
        }
    }

    UpdateCTTaskCommand::~UpdateCTTaskCommand()
    {
    }

    uint32_t UpdateCTTaskCommand::doCommand()
    {
        if (m_reqInfo.reqUpdateType == 5) {
            m_reqInfo.reqUpdateType = 2;
        }
        std::unique_ptr<uds_raw_data_req_t> rawDataReq = std::make_unique<uds_raw_data_req_t>();
        rawDataReq->ta = m_reqInfo.reqTa;
        rawDataReq->sa = m_reqInfo.reqSa;
        rawDataReq->bus_type = m_reqInfo.reqUpdateType;
        rawDataReq->data_vec = m_reqInfo.reqContent;
        rawDataReq->data_len = m_reqInfo.reqContent.size();

        int32_t ret = DiagAgent::Instance()->SendUdsRawData(rawDataReq);
        if (ret <= 0) {
            return N_ERROR;
        }

        uint32_t timeout = OTA_TIMER_P2START_CLIENT;
        if (m_reqInfo.reqWaitTime > 0) {
            timeout = m_reqInfo.reqWaitTime;
        }

        if (m_suppressPositiveResponse || m_functionAddr) {
            // suppressPosRes and function addr there is no response, just wait 100ms
            timeout = OTA_TIMER_P2_CLIENT;
        }

        if (waitEvent(timeout)) {
            return eContinue;
        }
        return N_ERROR;
    }

    bool UpdateCTTaskCommand::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            UPDATE_LOG_D("wait event time out!~");
            if (m_functionAddr || m_suppressPositiveResponse) {
                // there is no response
                m_resInfo.resResult = N_OK;
                setTaskResult(N_OK);
            }
            else {
                m_resInfo.resResult = N_TIMEOUT_P2START_CLIENT;
                setTaskResult(N_TIMEOUT_P2START_CLIENT);
            }
            return true;
        }

        UpdateTaskEvent* taskevent = static_cast<UpdateTaskEvent*>(event);
        if (nullptr == taskevent) {
            return false;
        }

        // UPDATE_LOG_D("evtType: %x, evtId: %x, evtVal1: %x, evtVal2: %x, evtSize: %ld!~",
        //      taskevent->getEventKind(), taskevent->getEventId(), taskevent->getEvtVal1(), taskevent->getEvtVal2(), taskevent->getEvtData().size());

        if (UpdateTaskEvent::checkEvent(m_reqInfo.reqUpdateType, m_reqInfo.reqTa, m_reqInfo.reqSa, taskevent)) {
            UPDATE_LOG_D("checkEvent match, evtSize: %ld!~", taskevent->getEvtData().size());
            if (taskevent->getEvtVal2() != N_OK) {
                setTaskResult(N_ERROR);
                return true;
            }

            if ((m_functionAddr || m_suppressPositiveResponse) && taskevent->getEvtData().size() == 0) {
                setTaskResult(N_OK);
                return true;
            }

            if (taskevent->getEvtData().size() == 3 && taskevent->getEvtData()[0] == 0x7F) {
                // NRC response
                if (taskevent->getEvtData()[2] != 0x78) {
                    UPDATE_LOG_D("NRC response, NRC: %X", taskevent->getEvtData()[2]);
                    m_resInfo.resContent = taskevent->getEvtData();
                    m_resInfo.resResult = taskevent->getEvtData()[2];
                    setTaskResult(N_NRC);
                }
                // pending frame no need to handle
                return true;
            }

            if (taskevent->getEvtData().size() >= m_reqInfo.reqExpectContent.size()) {
                bool match  = true;
                for (uint32_t index = 0; index < m_reqInfo.reqExpectContent.size(); ++index) {
                    if (taskevent->getEvtData()[index] != m_reqInfo.reqExpectContent[index]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    UPDATE_LOG_D("matched response");
                    m_resInfo.resContent = taskevent->getEvtData();
                    m_resInfo.resResult = N_OK;
                    setTaskResult(eOK);
                    return true;
                }
            }

            m_resInfo.resResult = N_UNEXP_PDU;
            setTaskResult(N_UNEXP_PDU);
            return true;
        }

        return false;
    }

    TaskReqInfo&
    UpdateCTTaskCommand::GetReqInfo()
    {
        return m_reqInfo;
    }

    TaskResInfo&
    UpdateCTTaskCommand::GetResInfo()
    {
        return m_resInfo;
    }

} // end of update
} // end of netaos
} // end of hozon
/* EOF */
