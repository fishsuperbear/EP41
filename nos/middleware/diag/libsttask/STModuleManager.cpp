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
 * @file STModuleManager.cpp
 * @brief Implements of STModuleManager class
 */

#include "STModuleManager.h"
#include "STTask.h"
#include "STLogDef.h"
#include "STCommandTask.h"
#include "STNormalTask.h"
#include "STEvent.h"
#include "STCall.h"

namespace hozon {
namespace netaos {
namespace sttask {

    STModuleManager::STModuleManager()
        : STObject(ST_OBJECT_TYPE_MODULEMANAGER)
    {
    }

    STModuleManager::~STModuleManager()
    {
        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            // uint32_t moduleID = it->first;
            STModuleBase* module = it->second;
            if (nullptr != module) {
                delete module;
                module = nullptr;
            }
        }

        m_registeredModules.clear();
    }

    bool STModuleManager::addModule(STModuleBase* pModule)
    {
        if (nullptr == pModule) {
            return false;
        }

        uint32_t moduleID = pModule->getModuleID();
        if (m_registeredModules.count(moduleID) > 0) {
            delete pModule;
            pModule = nullptr;
            return false;
        }

        pModule->setTaskRunner(getTaskRunner());
        m_registeredModules[moduleID] = pModule;
        return true;
    }

    STModuleBase* STModuleManager::getModule(uint32_t moduleID)
    {
        if (m_registeredModules.count(moduleID) > 0) {
            return m_registeredModules[moduleID];
        }
        return nullptr;
    }

    void STModuleManager::onOperationPost(uint32_t operationId, STNormalTask* topTask)
    {
        if (nullptr == topTask) {
            return;
        }

        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onOperationPost(operationId, topTask);
            }
        }
    }

    uint32_t STModuleManager::onOperationStart(uint32_t operationId, STNormalTask* topTask)
    {
        if (nullptr == topTask) {
            return eError;
        }

        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                uint32_t result = module->onOperationStart(operationId, topTask);
                if (eContinue != result) {
                    return result;
                }
            }
        }

        return eContinue;
    }

    void STModuleManager::onOperationStarted(uint32_t operationId, STNormalTask* topTask)
    {
        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onOperationStarted(operationId, topTask);
            }
        }
    }

    void STModuleManager::onOperationEnd(uint32_t operationId, uint32_t result, STNormalTask* topTask)
    {
        if (nullptr == topTask) {
            return;
        }

        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onOperationEnd(operationId, result, topTask);
            }
        }
    }

    void STModuleManager::onOperationInterrupt(uint32_t operationId, uint32_t interruptReason, STNormalTask* topTask)
    {
        if (nullptr == topTask) {
            return;
        }

        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onOperationInterrupt(operationId, interruptReason, topTask);
            }
        }
    }

    void STModuleManager::onStepPost(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
    {
        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onStepPost(operationId, stepId, stepTask);
            }
        }
    }

    uint32_t STModuleManager::onStepStart(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
    {
        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                uint32_t result = module->onStepStart(operationId, stepId, stepTask);
                if (eContinue != result) {
                    return result;
                }
            }
        }

        return eContinue;
    }

    void STModuleManager::onStepStarted(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
    {
        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onStepStarted(operationId, stepId, stepTask);
            }
        }
    }

    void STModuleManager::onStepEnd(uint32_t operationId, uint32_t stepId, uint32_t result, STStepTask* stepTask)
    {
        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onStepEnd(operationId, stepId, result, stepTask);
            }
        }
    }

    void STModuleManager::onStepInterrupt(uint32_t operationId, uint32_t stepId, uint32_t interruptReason, STStepTask* stepTask)
    {
        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onStepInterrupt(operationId, stepId, interruptReason, stepTask);
            }
        }
    }

    void STModuleManager::onEvent(uint32_t eventKind, uint32_t eventId, STEvent* event)
    {
        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onEvent(eventKind, eventId, event);
            }
        }
    }

    void STModuleManager::onUnexpectedEvent(uint32_t eventKind, uint32_t eventId, STEvent* event)
    {
        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onUnexpectedEvent(eventKind, eventId, event);
            }
        }
    }

    void STModuleManager::onCall(uint32_t callKind, uint32_t callId, STCall* callObj)
    {
        if (nullptr == callObj) {
            return;
        }

        for (MODULE_MAP::iterator it = m_registeredModules.begin(); it != m_registeredModules.end(); ++it) {
            STModuleBase* module = it->second;
            if (module) {
                module->onCall(callKind, callId, callObj);
            }
        }
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */
