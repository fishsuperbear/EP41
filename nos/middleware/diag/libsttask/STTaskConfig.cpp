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
 * @file STTaskConfig.cpp
 * @brief implements of STTaskConfiguratio
 */

#include "STTaskConfig.h"

namespace hozon {
namespace netaos {
namespace sttask {

    STTaskConfig::STTaskConfig()
        : STObject(ST_OBJECT_TYPE_CONFIRUATION)
    {
    }

    STTaskConfig::~STTaskConfig()
    {
    }

    void STTaskConfig::configCommandChannel(uint32_t channelId, uint32_t maxSize)
    {
        if (ST_TASK_INVALID_CHANNEL == channelId) {
            return;
        }

        CHANNEL_CONFIG channelInfo;
        channelInfo.maxSize = maxSize;
        m_commandChannels[channelId] = channelInfo;
    }

    void STTaskConfig::registerCommand(uint32_t commandId, uint32_t channelId)
    {
        if (ST_TASK_INVALID_STEP == commandId
            || ST_TASK_INVALID_CHANNEL == channelId) {
            return;
        }

        if (!isRegisteredCommandChannel(channelId)) {
            return;
        }

        STEP_CONFIG commandInfo;
        commandInfo.isCommand = true;
        commandInfo.commandChannelId = channelId;
        m_registeredSteps[commandId] = commandInfo;
    }

    void STTaskConfig::registerAllCommands(uint32_t commandIdMin, uint32_t commandIdMax, uint32_t channelId)
    {
        for (uint32_t i = commandIdMin; i < commandIdMax; ++i) {
            registerCommand(i, channelId);
        }
    }

    void STTaskConfig::registerTimer(uint32_t timerId)
    {
        if (ST_TASK_INVALID_STEP == timerId) {
            return;
        }
        STEP_CONFIG commandInfo;
        commandInfo.isCommand = false;
        m_registeredSteps[timerId] = commandInfo;
    }

    void STTaskConfig::registerAllTimers(uint32_t timerIdMin, uint32_t timerIdMax)
    {
        for (uint32_t i = timerIdMin; i < timerIdMax; ++i) {
            registerTimer(i);
        }
    }

    void STTaskConfig::registerOperation(uint32_t operationId)
    {
        OPERATION_CONFIG config;
        config.maxOperationCount = 1;
        config.queueMethod = DEFAULT_QUEUE_METHOD;
        registerOperation(operationId, config);
    }

    void STTaskConfig::registerOperation(uint32_t operationId, const OPERATION_CONFIG& config)
    {
        if (ST_TASK_INVALID_OPERATION == operationId) {
            return;
        }

        OPERATION_CONFIG operationInfo = config;
        m_registeredOperations[operationId] = operationInfo;
    }

    void STTaskConfig::registerAllOperations(uint32_t operationIdMin, uint32_t operationIdMax)
    {
        OPERATION_CONFIG config;
        config.maxOperationCount = 1;
        config.queueMethod = DEFAULT_QUEUE_METHOD;
        registerAllOperations(operationIdMin, operationIdMax, config);
    }

    void STTaskConfig::registerAllOperations(uint32_t operationIdMin, uint32_t operationIdMax, const OPERATION_CONFIG& config)
    {
        for (uint32_t i = operationIdMin; i < operationIdMax; ++i) {
            registerOperation(i, config);
        }
    }



    bool STTaskConfig::isRegisteredOperation(uint32_t operationId)
    {
        return m_registeredOperations.count(operationId) > 0 ? true : false;
    }

    bool STTaskConfig::isRegisteredCommand(uint32_t commandId)
    {
        if (m_registeredSteps.count(commandId) > 0) {
            return m_registeredSteps[commandId].isCommand;
        }
        return false;
    }

    bool STTaskConfig::isRegisteredCommandChannel(uint32_t channelId)
    {
        return m_commandChannels.count(channelId) > 0 ? true : false;
    }

    bool STTaskConfig::isRegisteredTimer(uint32_t timerId)
    {
        if (m_registeredSteps.count(timerId) > 0) {
            if (false == m_registeredSteps[timerId].isCommand) {
                return true;
            }
        }
        return false;
    }

    bool STTaskConfig::isRegisteredStep(uint32_t stepId)
    {
        if (m_registeredSteps.count(stepId) > 0) {
            return true;
        }
        return false;
    }


    uint32_t STTaskConfig::getMaxOperationCount(uint32_t operationId)
    {
        if (isRegisteredOperation(operationId)) {
            const OPERATION_CONFIG& operationInfo = m_registeredOperations[operationId];
            return operationInfo.maxOperationCount;
        }
        return 0;
    }

    uint8_t STTaskConfig::getOperationQueueMethod(uint32_t operationId)
    {
        if (isRegisteredOperation(operationId)) {
            const OPERATION_CONFIG& operationInfo = m_registeredOperations[operationId];
            return operationInfo.queueMethod;
        }
        return (QUEUE_METHOD_INCLUDE_EXECUTING | QUEUE_METHOD_DELETE_BACK_IF_OVERFLOW);
    }

    bool STTaskConfig::checkOperationQueueMethod(uint32_t operationId, uint8_t queueMethod)
    {
        if (isRegisteredOperation(operationId)) {
            const OPERATION_CONFIG& operationInfo = m_registeredOperations[operationId];
            uint8_t configMethod = operationInfo.queueMethod;

            return testOperationQueueMethod(configMethod, queueMethod);
        }

        return false;
    }

    bool STTaskConfig::testOperationQueueMethod(uint8_t methodToTest, uint8_t methodExpected)
    {
        if ((methodToTest & methodExpected) == methodExpected) {
            return true;
        }

        return false;
    }

    bool STTaskConfig::getOperationConfig(uint32_t operationId, STTaskConfig::OPERATION_CONFIG& config)
    {
        if (isRegisteredOperation(operationId)) {
            const OPERATION_CONFIG& operationInfo = m_registeredOperations[operationId];
            memcpy(&config, &operationInfo, sizeof(STTaskConfig::OPERATION_CONFIG));
            return true;
        }

        return false;
    }


    uint32_t STTaskConfig::getCommandChannel(uint32_t commandId)
    {
        if (isRegisteredCommand(commandId)) {
            const STEP_CONFIG& commandInfo = m_registeredSteps[commandId];
            return commandInfo.commandChannelId;
        }

        return ST_TASK_INVALID_CHANNEL;
    }

    uint32_t STTaskConfig::getCommandChannelMaxSize(uint32_t channelId)
    {
        if (isRegisteredCommandChannel(channelId)) {
            const CHANNEL_CONFIG& channelInfo = m_commandChannels[channelId];
            return channelInfo.maxSize;
        }
        return 0;
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */
