/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanNTTaskSendCommand implement
 */

#include "docan_nttask_sendcommand.h"
#include "docan_nttask_commumonitor.h"
#include "diag/docan/steptasks/docan_sttask_send_ff.h"
#include "diag/docan/steptasks/docan_sttask_send_sf.h"
#include "diag/docan/steptasks/docan_sttask_send_cf.h"
#include "diag/docan/steptasks/docan_sttask_wait_fc.h"
#include "diag/docan/steptasks/docan_sttask_wait_pending.h"
#include "diag/docan/steptasks/docan_sttask_send_fc.h"
#include "diag/docan/manager/docan_event_receiver.h"
#include "diag/docan/manager/docan_event_sender.h"
#include "diag/docan/config/docan_config.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanNTTaskSendCommand::DocanNTTaskSendCommand(NormalTaskBase* pParent,
                                                    STObject::TaskCB pfnCallback,
                                                    const TaskReqInfo& reqInfo,
                                                    bool isTopTask)
            : NormalTaskBase(DOCAN_NTTASK_SEND_COMMAND + reqInfo.reqEcu, pParent, pfnCallback, isTopTask)
            , m_reqInfo(reqInfo)
    {
        DocanConfig::instance()->getEcuInfo(reqInfo.reqEcu, m_ecuInfo);
        m_reqInfo.reqCanid = m_ecuInfo.canid_tx;
        if (DIAG_FUNCTIONAL_ADDR_DOCAN == m_reqInfo.reqCanid) {
            // functional request no need response
            m_reqInfo.suppressPosRsp = true;
        }

        m_reqInfo.reqBsIndexExpect = 1;
        m_reqInfo.reqCompletedSize = 0;
        m_reqInfo.reqBs = m_ecuInfo.BS;
        m_reqInfo.reqSTmin = m_ecuInfo.STmin;
        m_reqInfo.suppressPosRsp = false;
        if (0x10 == m_reqInfo.reqContent[0] || 0x11 == m_reqInfo.reqContent[0]
           || 0x28 == m_reqInfo.reqContent[0] || 0x3E == m_reqInfo.reqContent[0]
           || 0x85 == m_reqInfo.reqContent[0]) {
            m_reqInfo.suppressPosRsp = (0x80 == (m_reqInfo.reqContent[1] & 0x80));
        }

        m_resInfo.resCanid = m_ecuInfo.canid_rx;
        m_resInfo.resLen = 0;
        m_resInfo.resBsIndexExpect = 1;
        m_resInfo.resCompletedSize = 0;
    }

    DocanNTTaskSendCommand::~DocanNTTaskSendCommand()
    {
    }

    uint32_t DocanNTTaskSendCommand::doAction()
    {
        // interrupt ecu monitor task
        if (getContext()->interruptOperation(DOCAN_NTTASK_COMMU_MONITOR + m_reqInfo.reqEcu, N_TASK_INTERRUPT)) {
            // DOCAN_LOG_D("ecu: %X reqCanid: %X, communication task: %d interrupted.", m_reqInfo.reqEcu, m_reqInfo.reqCanid, DOCAN_NTTASK_COMMU_MONITOR + m_reqInfo.reqEcu);
        }

        // handle current task.
        if (m_reqInfo.reqContent.size() > 4095) {
            return N_BUFFER_OVFLW;
        }

        if (m_reqInfo.reqContent.size() > 0x07) {
            if (DIAG_FUNCTIONAL_ADDR_DOCAN == m_reqInfo.reqCanid) {
                DOCAN_LOG_E("Functional request invalid.");
                return N_BUFFER_OVFLW;
            }
            return startToSendFF();
        }
        else {
            return startToSendSF();
        }
    }

    void DocanNTTaskSendCommand::onCallbackAction(uint32_t taskResult)
    {
        if (isTopTask()) {
            DocanEventSender::instance()->sendUdsResponse(m_reqInfo.who, m_reqInfo.N_TA, m_reqInfo.N_SA, reqId(), taskResult, m_resInfo.resContent);
        }

        if (getContext()->curOperationCount(DOCAN_NTTASK_SEND_COMMAND + m_reqInfo.reqEcu) > 1) {
            DOCAN_LOG_D("curOperationCount %d.", getContext()->curOperationCount(DOCAN_NTTASK_SEND_COMMAND + m_reqInfo.reqEcu));
            return;
        }

        if (getContext()->isOperationExist(DOCAN_NTTASK_COMMU_MONITOR + m_reqInfo.reqEcu)) {
            getContext()->restoreOperation(DOCAN_NTTASK_COMMU_MONITOR + m_reqInfo.reqEcu);
        }
        else {
            // can communication finished, 1 min monitor to close channel.
            DocanNTTaskCommuMonitor *task = new DocanNTTaskCommuMonitor(nullptr, nullptr, m_reqInfo, true);
            taskResult = post(task);
            if (eContinue != taskResult) {
                DOCAN_LOG_E("post monitor task failed.");
            }
        }
    }

    uint32_t DocanNTTaskSendCommand::startToSendSF()
    {
        DOCAN_LOG_D("startToSendSF reqEcu: %d, reqCanid: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid);
        DocanSTTaskSendSF* task = new DocanSTTaskSendSF(this,
                                                        CAST_TASK_CB(&DocanNTTaskSendCommand::onSendSFResult),
                                                        m_reqInfo, m_resInfo);
        return post(task);
    }

    void DocanNTTaskSendCommand::onSendSFResult(STTask *task, uint32_t result)
    {
        DOCAN_LOG_D("onSendSFResult reqEcu: %d, reqCanid: %X, result: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
        if (N_OK == result) {
            DocanSTTaskSendSF* sftask = static_cast<DocanSTTaskSendSF*>(task);
            if (nullptr == sftask) {
                result = N_TIMEOUT_A;
                onCallbackResult(result);
                return;
            }

            m_reqInfo.reqCompletedSize = sftask->getReqInfo().reqCompletedSize;
            m_resInfo.resContent = sftask->getResInfo().resContent;
            m_resInfo.resLen = sftask->getResInfo().resLen;
            m_resInfo.resCompletedSize = sftask->getResInfo().resCompletedSize;

            if (m_reqInfo.suppressPosRsp) {
                result = N_OK;
                // suppress positive response
                DOCAN_LOG_D("suppress positive response, reqEcu: %d, reqCanid: %X, result N_OK: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
                onCallbackResult(result);
                return;
            }

            // positive response
            if (sftask->getResInfo().resLen <= 0x07) {
                // SF frame received
                if (sftask->getResInfo().resContent[0] == m_reqInfo.reqContent[0] + 0x40) {
                    // match req service normal response
                    result = N_OK;
                    DOCAN_LOG_D("positive SF response math request, reqEcu: %d, reqCanid: %X, result N_OK: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
                    onCallbackResult(result);
                    return;
                }

                // negativ req service response, eg 7f xx 31/78/xx ...
                if (sftask->getResInfo().resContent[0] != 0x7F ||
                    sftask->getResInfo().resContent[1] != m_reqInfo.reqContent[0]
                    || sftask->getResInfo().resContent.size() > 3) {
                    // invalid natative req service response format
                    result = N_UNEXP_PDU;
                    DOCAN_LOG_D("invalid natative req service response format, reqEcu: %d, reqCanid: %X, result N_UNEXP_PDU: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
                    onCallbackResult(result);
                    return;
                }

                if (sftask->getResInfo().resContent[2] != 0x78) {
                    // negative req service response
                    result = N_OK;
                    DOCAN_LOG_D("negativ req service response, reqEcu: %d, reqCanid: %X, NRC: %X, result N_OK: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, sftask->getResInfo().resContent[2], result);
                    onCallbackResult(result);
                    return;
                }
                // received 78 pending
                result = startToWaitPending();
            }
            else {
                // FF frame received
                result = startToSendFC();
            }

            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
        else {
            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
    }

    uint32_t DocanNTTaskSendCommand::startToSendFF()
    {
        DOCAN_LOG_D("startToSendFF reqEcu: %d, reqCanid: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid);
        DocanSTTaskSendFF* task = new DocanSTTaskSendFF(this,
                                                        CAST_TASK_CB(&DocanNTTaskSendCommand::onSendFFResult),
                                                        m_reqInfo, m_resInfo);
        return post(task);
    }

    void DocanNTTaskSendCommand::onSendFFResult(STTask *task, uint32_t result)
    {
        DOCAN_LOG_D("onSendFFResult reqEcu: %d, reqCanid: %X, result: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
        if (N_OK == result) {
            DocanSTTaskSendFF* fftask = static_cast<DocanSTTaskSendFF*>(task);
            if (nullptr == fftask) {
                result = N_ERROR;
                onCallbackResult(result);
                return;
            }

            m_resInfo.resFs = fftask->getResInfo().resFs;
            m_reqInfo.reqCompletedSize = fftask->getReqInfo().reqCompletedSize;
            if (CTS == m_resInfo.resFs) {
                m_resInfo.resBs = fftask->getResInfo().resBs;
                m_resInfo.resSTmin = fftask->getResInfo().resSTmin;
                result = startToSendCF();
            } else if (WAIT == m_resInfo.resFs) {
                result = startToWaitFC();
            }
            else {
                result = N_BUFFER_OVFLW;
            }

            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
        else {
            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
    }

    uint32_t DocanNTTaskSendCommand::startToWaitFC()
    {
        DOCAN_LOG_D("startToWaitFC reqEcu: %d, reqCanid: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid);
        if (m_ecuInfo.N_WFTmax <= m_waitFcRetry) {
            DOCAN_LOG_E("reqEcu: %d, reqCanid: %X, wait fc %d times, out of times Limited.", m_reqInfo.reqEcu, m_reqInfo.reqCanid, m_waitFcRetry);
            return N_WFT_OVRN;
        }
        m_waitFcRetry++;

        DocanSTTaskWaitFC* task = new DocanSTTaskWaitFC(this,
                                                        CAST_TASK_CB(&DocanNTTaskSendCommand::onWaitFCResult),
                                                        m_reqInfo, m_resInfo);
        return post(task);
    }

    void DocanNTTaskSendCommand::onWaitFCResult(STTask *task, uint32_t result)
    {
        DOCAN_LOG_D("onWaitFCResult reqEcu: %d, reqCanid: %X, result: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
        if (N_OK == result) {
            DocanSTTaskWaitFC* fftask = static_cast<DocanSTTaskWaitFC*>(task);
            if (nullptr == fftask) {
                result = N_TIMEOUT_Bs;
                onCallbackResult(result);
                return;
            }

            m_resInfo.resFs = fftask->getResInfo().resFs;
            if (CTS == m_resInfo.resFs) {
                m_resInfo.resBs = fftask->getResInfo().resBs;
                m_resInfo.resSTmin = fftask->getResInfo().resSTmin;
                result = startToSendCF();
            } else if (WAIT == m_resInfo.resFs) {
                result = startToWaitFC();
            }
            else {
                result = N_BUFFER_OVFLW;
            }

            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
        else {
            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
    }

    uint32_t DocanNTTaskSendCommand::startToSendCF()
    {
        DOCAN_LOG_D("startToSendCF reqEcu: %d, reqCanid: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid);
        DocanSTTaskSendCF* task = new DocanSTTaskSendCF(this,
                                                        CAST_TASK_CB(&DocanNTTaskSendCommand::onSendCFResult),
                                                        m_reqInfo, m_resInfo);
        return post(task);
    }

    void DocanNTTaskSendCommand::onSendCFResult(STTask *task, uint32_t result)
    {
        DOCAN_LOG_D("onSendCFResult reqEcu: %d, reqCanid: %X, result: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
        if (N_OK == result) {
            DocanSTTaskSendCF* cftask = static_cast<DocanSTTaskSendCF*>(task);
            if (nullptr == cftask) {
                result = N_TIMEOUT_Cr;
                onCallbackResult(result);
                return;
            }

            m_reqInfo.reqCompletedSize = cftask->getReqInfo().reqCompletedSize;
            m_resInfo.resContent = cftask->getResInfo().resContent;
            m_resInfo.resLen = cftask->getResInfo().resLen;
            m_resInfo.resCompletedSize = cftask->getResInfo().resCompletedSize;

            if (m_reqInfo.reqCompletedSize >= m_reqInfo.reqContent.size()) {
                if (m_reqInfo.suppressPosRsp) {
                    result = N_OK;
                    // suppress positive response
                    DOCAN_LOG_D("suppress positive response, reqEcu: %d, reqCanid: %X, result N_OK: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
                    onCallbackResult(result);
                    return;
                }

                // positive response
                if (cftask->getResInfo().resLen <= 0x07) {
                    // SF frame received
                    if (cftask->getResInfo().resContent[0] == m_reqInfo.reqContent[0] + 0x40) {
                        // match req service normal response
                        result = N_OK;
                        DOCAN_LOG_D("positive SF response match, reqEcu: %d, reqCanid: %X, result N_OK: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
                        onCallbackResult(result);
                        return;
                    }

                    // negativ req service response, eg 7f xx 31/78/xx ...
                    if (cftask->getResInfo().resContent[0] != 0x7F ||
                        cftask->getResInfo().resContent[1] != m_reqInfo.reqContent[0]
                        || cftask->getResInfo().resContent.size() > 3) {
                        // invalid natative req service response format
                        result = N_UNEXP_PDU;
                        DOCAN_LOG_D("invalid natative response format, reqEcu: %d, reqCanid: %X, result N_UNEXP_PDU: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
                        onCallbackResult(result);
                        return;
                    }

                    if (cftask->getResInfo().resContent[2] != 0x78) {
                        // negative req service response
                        result = N_OK;
                        DOCAN_LOG_D("negative response, reqEcu: %d, reqCanid: %X, NRC: %X, result N_OK: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, cftask->getResInfo().resContent[2], result);
                        onCallbackResult(result);
                        return;
                    }

                    // 0x78 response need do pending
                    result = startToWaitPending();
                }
                else {
                    // FF frame received
                    result = startToSendFC();
                }
            }
            else {
                // received next FC frame, requst data not send completed, continue send CF frame
                result = startToSendCF();
            }

            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
        else {
            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
    }

    uint32_t DocanNTTaskSendCommand::startToWaitPending()
    {
        DOCAN_LOG_D("startToWaitPending reqEcu: %d, reqCanid: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid);
        DocanSTTaskWaitPending* task = new DocanSTTaskWaitPending(this,
                                                        CAST_TASK_CB(&DocanNTTaskSendCommand::onWaitPendingResult),
                                                        m_reqInfo, m_resInfo);
        return post(task);
    }

    void DocanNTTaskSendCommand::onWaitPendingResult(STTask *task, uint32_t result)
    {
        DOCAN_LOG_D("onWaitPendingResult reqEcu: %d, reqCanid: %X, result: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
        if (N_OK == result) {
            DocanSTTaskWaitPending* pendingtask = static_cast<DocanSTTaskWaitPending*>(task);
            m_resInfo.resContent = pendingtask->getResInfo().resContent;
            m_resInfo.resLen = pendingtask->getResInfo().resLen;
            m_resInfo.resCompletedSize = pendingtask->getResInfo().resCompletedSize;

            // positive response
            if (pendingtask->getResInfo().resLen <= 0x07) {
                // SF frame received
                if (pendingtask->getResInfo().resContent[0] == m_reqInfo.reqContent[0] + 0x40) {
                    // match req service normal response
                    result = N_OK;
                    DOCAN_LOG_D("positive SF response match request, reqEcu: %d, reqCanid: %X, result N_OK: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
                    onCallbackResult(result);
                    return;
                }

                // negativ req service response, eg 7f xx 31/78/xx ...
                if (pendingtask->getResInfo().resContent[0] != 0x7F ||
                    pendingtask->getResInfo().resContent[1] != m_reqInfo.reqContent[0]
                    || pendingtask->getResInfo().resContent.size() > 3) {
                    // invalid natative req service response format
                    result = N_UNEXP_PDU;
                    DOCAN_LOG_D("invalid natative response format, reqEcu: %d, reqCanid: %X, result N_UNEXP_PDU: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
                    onCallbackResult(result);
                    return;
                }

                if (pendingtask->getResInfo().resContent[2] != 0x78) {
                    // negative req service response
                    result = N_OK;
                    DOCAN_LOG_D("negative response, reqEcu: %d, reqCanid: %X, NRC: %X, result N_OK: %X",  m_reqInfo.reqEcu, m_reqInfo.reqCanid, pendingtask->getResInfo().resContent[2], result);
                    onCallbackResult(result);
                    return;
                }

                // negative 0x78 pending
                result = startToWaitPending();
            }
            else {
                // FF frame received need send FC frame to receive next CF frame
                result = startToSendFC();
            }

            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
        else {
            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
    }

    uint32_t DocanNTTaskSendCommand::startToSendFC()
    {
        DOCAN_LOG_D("startToSendFC reqEcu: %d, reqCanid: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid);
        DocanSTTaskSendFC* task = new DocanSTTaskSendFC(this,
                                                        CAST_TASK_CB(&DocanNTTaskSendCommand::onSendFCResult),
                                                        m_reqInfo, m_resInfo);
        return post(task);
    }

    void DocanNTTaskSendCommand::onSendFCResult(STTask *task, uint32_t result)
    {
        DOCAN_LOG_D("onSendFCResult reqEcu: %d, reqCanid: %X, result: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
        if (N_OK == result) {
            DocanSTTaskSendFC* fctask = static_cast<DocanSTTaskSendFC*>(task);
            if (nullptr == fctask) {
                result = N_TIMEOUT_Bs;
                onCallbackResult(result);
                return;
            }

            m_resInfo.resContent = fctask->getResInfo().resContent;
            m_resInfo.resCompletedSize = fctask->getResInfo().resCompletedSize;
            if (m_resInfo.resCompletedSize >= m_resInfo.resLen) {
                // all data have received completed.
                result = N_OK;
                DOCAN_LOG_D("all data have received completed, reqEcu: %d, reqCanid: %X, result N_OK: %X", m_reqInfo.reqEcu, m_reqInfo.reqCanid, result);
                onCallbackResult(result);
                return;
            }

            // need continue to send next FC frame to continue receive CF frame
            result = startToSendFC();
            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
        else {
            if (eContinue != result) {
                onCallbackResult(result);
            }
        }
    }

    TaskReqInfo& DocanNTTaskSendCommand::getReqInfo()
    {
        return m_reqInfo;
    }

    TaskResInfo& DocanNTTaskSendCommand::getResInfo()
    {
        return m_resInfo;
    }


} // end of diag
} // end of netaos
} // end of hozon
/* EOF */
