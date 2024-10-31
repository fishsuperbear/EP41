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
 * @file  STModuleManager.h
 * @brief Class of STModuleManager
 */

#ifndef STMODULEMANAGER_H
#define STMODULEMANAGER_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <map>

#include "STObject.h"
#include "STModuleBase.h"

namespace hozon {
namespace netaos {
namespace sttask {

    class STCommandTask;
    class STNormalTask;
    class STEvent;
    class STCall;
    class STModuleBase;
    /**
     * @brief Class of STModuleManager
     *
     * This class represent a manager for modules.
     */
    class STModuleManager : public STObject
    {
    public:
        STModuleManager();
        virtual ~STModuleManager();
        bool                addModule(STModuleBase* pModule);
        STModuleBase*       getModule(uint32_t moduleID);

        virtual void        onOperationPost(uint32_t operationId, STNormalTask* topTask);
        virtual uint32_t    onOperationStart(uint32_t operationId, STNormalTask* topTask);
        virtual void        onOperationStarted(uint32_t operationId, STNormalTask* topTask);
        virtual void        onOperationEnd(uint32_t operationId, uint32_t result, STNormalTask* topTask);
        virtual void        onOperationInterrupt(uint32_t operationId, uint32_t interruptReason, STNormalTask* topTask);

        virtual void        onStepPost(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual uint32_t    onStepStart(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual void        onStepStarted(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual void        onStepEnd(uint32_t operationId, uint32_t stepId, uint32_t result, STStepTask* stepTask);
        virtual void        onStepInterrupt(uint32_t operationId, uint32_t stepId, uint32_t interruptReason, STStepTask* stepTask);

        virtual void        onEvent(uint32_t eventKind, uint32_t eventId, STEvent* event);
        virtual void        onUnexpectedEvent(uint32_t eventKind, uint32_t eventId, STEvent* event);
        virtual void        onCall(uint32_t callKind, uint32_t callId, STCall* callObj);

    private:
        STModuleManager(const STModuleManager&);
        STModuleManager& operator=(const STModuleManager&);
        typedef std::map<uint32_t, STModuleBase*> MODULE_MAP;
        MODULE_MAP                               m_registeredModules;
    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STMODULEMANAGER_H */
/* EOF */