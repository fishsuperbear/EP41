/**
 * Copyright @ 2021 - 2023 Hozon Auto Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * Hozon Auto Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

/**
 * @file  STTaskContext.h
 * @brief Class of STTaskContext
 */

#ifndef STTASKCONTEXT_H
#define STTASKCONTEXT_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <map>
#include <list>
#include <mutex>
#include "STObject.h"

namespace hozon {
namespace netaos {
namespace sttask {

    class STTask;
    class STStepTask;
    class STCommandTask;
    class STNormalTask;
    class STEvent;
    class STCall;
    class STTaskConfig;
    /**
     * @brief Class of STTaskContext
     *
     * TBD.
     */
    class STTaskContext : public STObject
    {
    public:
        STTaskContext();
        virtual ~STTaskContext();

        void     init(STTaskConfig* configuration);

        bool     isOperationExist(uint32_t operationId);
        bool     isOperationExecuting(uint32_t operationId);
        bool     isOperationExecutingTask(STNormalTask* opTask);
        uint32_t curOperationCount(uint32_t operationId);
        uint32_t curOperationStep(uint32_t operationId);
        bool     setCurOperationStepTask(uint32_t operationId, uint32_t result);
        bool     isOperationStepExecuting(uint32_t operationId, uint32_t stepId);
        bool     interruptOperation(uint32_t operationId, uint32_t interruptReason);
        bool     restoreOperation(uint32_t operationId);

        uint32_t curCommandCountInChannel(uint32_t commandChannelId);
        bool     isCommandExcutingInChannel(uint32_t commandChannelId);
        uint32_t curExecutingCommandInChannel(uint32_t commandChannelId);
        bool     interruptCommand(uint32_t commandId, uint32_t interruptReason);

        void     onStepPost(STStepTask* stepTask);
        void     onOperationPost(STNormalTask* opTask);
        void     onStepStart(STStepTask* stepTask);
        void     onOperationStart(STNormalTask* opTask);
        void     onStepEnd(STStepTask* stepTask, uint32_t result);
        void     onOperationEnd(STNormalTask* opTask, uint32_t result);
        void     onStepDelete(STStepTask* stepTask);
        void     onOperationDelete(STNormalTask* opTask);

    private:
        bool     addToCommandTrace(STCommandTask* task, uint32_t commandChannelId);
        void     removeFromCommandTrace(STCommandTask* task, uint32_t commandChannelId);
        bool     addToOperationTrace(STNormalTask* task, uint32_t operationId);
        void     removeFromOperationTrace(STNormalTask* task, uint32_t operationId);


    private:
        STTaskContext(const STTaskContext&);
        STTaskContext& operator=(const STTaskContext&);

        std::mutex               m_syncObj;

        typedef std::list<STNormalTask*> TASK_QUEUE;
        typedef std::list<STCommandTask*> COMMAND_QUEUE;

        struct OPERATION_TRACE
        {
            STNormalTask* curOperationTask;
            STStepTask* curStepTask;
            TASK_QUEUE allOperationTasks;
        };

        /**
         * key: operationId
         */
        typedef std::map<uint32_t, OPERATION_TRACE> OPERATION_TRACE_MAP;
        OPERATION_TRACE_MAP m_operationTraces;


        struct COMMAND_TRACE
        {
            STCommandTask* executing;
            COMMAND_QUEUE commandTasks;
        };

        /**
         * key: channelId
         */
        typedef std::map<uint32_t, COMMAND_TRACE> COMMAND_TRACE_MAP;
        COMMAND_TRACE_MAP m_commandTraces;

        uint32_t      getOperationCommandFromQueue(uint32_t operationId, const COMMAND_QUEUE& commandQueue);

    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STTASKCONTEXT_H */
/* EOF */