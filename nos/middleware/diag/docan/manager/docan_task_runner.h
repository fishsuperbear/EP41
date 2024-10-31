/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanTaskRunner Header
 */

#ifndef DOCAN_TASK_RUNNER_H_
#define DOCAN_TASK_RUNNER_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <vector>
#include <mutex>
#include "diag/libsttask/STTaskRunner.h"

using namespace hozon::netaos::sttask;

namespace hozon {
namespace netaos {
namespace diag {

    class DocanServiceImpl;
    /**
     * @brief Class of DocanTaskRunner
     *
     * tasks handler main thread.
     */
    class DocanTaskRunner: public STTaskRunner
    {
    public:
        static DocanTaskRunner *instance();
        static void destroy();

        int32_t         Init(void);
        int32_t         Start(void);
        int32_t         Stop(void);
        int32_t         Deinit(void);

        /* interface for outside begin */
        int32_t UdsRequest(const std::string& who, uint16_t reqSa, uint16_t reqTa, const std::vector<uint8_t>& uds);
        /* interface for outside end */

        bool isOperationExist(uint32_t operation);
        bool isOperationRunning(uint32_t operation);
        virtual const char* getTaskThreadName() { return "task_thread"; };

    protected:
        virtual void        onInit();
        virtual uint32_t    onOperationStart(uint32_t operationId, STNormalTask* topTask);
        virtual void        onOperationEnd(uint32_t operationId, uint32_t result, STNormalTask* topTask);
        virtual uint32_t    onStepStart(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual void        onStepEnd(uint32_t operationId, uint32_t stepId, uint32_t result, STStepTask* stepTask);
        virtual void        onUnexpectedEvent(uint32_t eventKind, uint32_t eventId, STEvent* event);

    private:
        DocanTaskRunner();
        virtual ~DocanTaskRunner();

        DocanTaskRunner(const DocanTaskRunner&);
        DocanTaskRunner& operator=(const DocanTaskRunner&);

    private:
        mutable std::mutex m_sync;
        static DocanTaskRunner* s_instance;
        static std::mutex s_instance_mutex;

    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_TASK_RUNNER_H_
/* EOF */
