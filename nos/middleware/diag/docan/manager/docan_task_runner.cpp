/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanTaskRunner implement
 */

#include "docan_task_runner.h"
#include "diag/docan/normaltasks/docan_nttask_sendcommand.h"
#include "diag/docan/common/docan_internal_def.h"
#include "diag/docan/log/docan_log.h"
#include "diag/docan/config/docan_config.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanTaskRunner* DocanTaskRunner::s_instance = nullptr;
    std::mutex DocanTaskRunner::s_instance_mutex;

    DocanTaskRunner* DocanTaskRunner::instance()
    {
        std::lock_guard<std::mutex> sync(s_instance_mutex);
        if (nullptr == s_instance) {
            s_instance = new DocanTaskRunner();
        }
        return s_instance;
    }

    void DocanTaskRunner::destroy()
    {
        std::lock_guard<std::mutex> sync(s_instance_mutex);
        if (nullptr != s_instance) {
            delete s_instance;
            s_instance = nullptr;
        }
    }

    DocanTaskRunner::DocanTaskRunner()
            : STTaskRunner()
            , m_sync()
    {
    }

    DocanTaskRunner::~DocanTaskRunner()
    {
    }

    int32_t DocanTaskRunner::Init(void)
    {
        init();
        return 0;
    }

    int32_t DocanTaskRunner::Start(void)
    {
        start();
        return 0;
    }

    int32_t DocanTaskRunner::Stop(void)
    {
        stop();
        return 0;
    }

    int32_t DocanTaskRunner::Deinit(void)
    {
        deinit();
        return 0;
    }

    int32_t DocanTaskRunner::UdsRequest(const std::string& who, uint16_t reqSa, uint16_t reqTa, const std::vector<uint8_t>& uds)
    {
        TaskReqInfo cmdInfo;
        cmdInfo.who = who;
        cmdInfo.N_SA = reqSa;
        cmdInfo.N_TA = reqTa;
        cmdInfo.N_TAtype = (reqTa == DIAG_FUNCTIONAL_ADDR_DOCAN) ? 1 : 2;
        cmdInfo.reqEcu = reqTa;
        cmdInfo.reqContent = uds;
        DocanNTTaskSendCommand* task = new DocanNTTaskSendCommand(nullptr,
                                                                  nullptr,
                                                                  cmdInfo,
                                                                  true);
        uint32_t reqId = (nullptr == task) ? 0 : task->reqId();
        int32_t ret = post(task);
        if (eContinue != ret) {
            DOCAN_LOG_E("post task fail, ret=%d.", ret);
            return 0;
        }
        return reqId;
    }

    void DocanTaskRunner::onInit()
    {
        DOCAN_LOG_D("DocanTaskRunner onInit()");
        // register command task channels

        std::vector<N_EcuInfo_t> infoList = DocanConfig::instance()->getEcuInfoList();
        for (auto it : infoList) {
            // register task channel
            getConfiguration()->configCommandChannel(DOCAN_TASK_CHANNEL_ECU + it.address_logical);
            getConfiguration()->configCommandChannel(DOCAN_TASK_CHANNEL_MONITOR + it.address_logical);
            // register command task and set command task channel
            getConfiguration()->registerCommand(DOCAN_COMMAND_SEND_CF + it.address_logical, DOCAN_TASK_CHANNEL_ECU + it.address_logical);
            getConfiguration()->registerCommand(DOCAN_COMMAND_SEND_FC + it.address_logical, DOCAN_TASK_CHANNEL_ECU + it.address_logical);
            getConfiguration()->registerCommand(DOCAN_COMMAND_SEND_FF + it.address_logical, DOCAN_TASK_CHANNEL_ECU + it.address_logical);
            getConfiguration()->registerCommand(DOCAN_COMMAND_SEND_SF + it.address_logical, DOCAN_TASK_CHANNEL_ECU + it.address_logical);
            getConfiguration()->registerCommand(DOCAN_COMMAND_WAIT_FC + it.address_logical, DOCAN_TASK_CHANNEL_ECU + it.address_logical);
            getConfiguration()->registerCommand(DOCAN_COMMAND_WAIT_PENDING + it.address_logical, DOCAN_TASK_CHANNEL_ECU + it.address_logical);
            getConfiguration()->registerCommand(DOCAN_COMMAND_ECU_MONITOR + it.address_logical, DOCAN_TASK_CHANNEL_MONITOR + it.address_logical);

            // register timer tasks
            getConfiguration()->registerTimer(DOCAN_TIMER_TASK_DELAY + it.address_logical);

            // register normal tasks, these tasks configure as the settig config could be serval times in excute queue.
            STTaskConfig::OPERATION_CONFIG operaQueueConfig;
            operaQueueConfig.maxOperationCount = DOCAN_DUPLICATE_TASK_IN_QUEUE_MAX;
            operaQueueConfig.queueMethod = STTaskConfig::QUEUE_METHOD_EXCLUDE_EXECUTING | STTaskConfig::QUEUE_METHOD_DELETE_FRONT_IF_OVERFLOW;
            getConfiguration()->registerOperation(DOCAN_NTTASK_SEND_COMMAND + it.address_logical, operaQueueConfig);
            getConfiguration()->registerOperation(DOCAN_NTTASK_COMMU_MONITOR + it.address_logical, operaQueueConfig);
        }
    }

    uint32_t DocanTaskRunner::onOperationStart(uint32_t operationId, STNormalTask* topTask)
    {

        return eContinue;
    }

    void DocanTaskRunner::onOperationEnd(uint32_t operationId, uint32_t result, STNormalTask* task)
    {
        if (nullptr != task) {
            switch (task->getOperationId()) {
            case DOCAN_NTTASK_INIT:
                {
                    if (eOK == result) {
                        DOCAN_LOG_D("Task init completed success!");
                    }
                    else {
                        DOCAN_LOG_D("Task init failed!");
                    }
                }
                break;
            default:
                // DOCAN_LOG_E("unknown top task, task operation=%d", task->getOperationId());
                return;
            }
        }
    }

    uint32_t DocanTaskRunner::onStepStart(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
    {
        return eContinue;
    }

    void DocanTaskRunner::onStepEnd(uint32_t operationId, uint32_t stepId, uint32_t result, STStepTask* stepTask)
    {
    }

    void DocanTaskRunner::onUnexpectedEvent(uint32_t eventKind, uint32_t eventId, STEvent* event)
    {
        if (nullptr == event) {
            DOCAN_LOG_E("event is nullptr");
            return;
        }
    }

    bool DocanTaskRunner::isOperationExist(uint32_t operation)
    {
        return getContext()->isOperationExist(operation);
    }

    bool DocanTaskRunner::isOperationRunning(uint32_t operation)
    {
        return getContext()->isOperationExecuting(operation);
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */