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
 * @file  STTaskConfig.h
 * @brief Class of STTaskConfig
 */

#ifndef STTASKCONFIGURATION_H
#define STTASKCONFIGURATION_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <map>
#include "STObject.h"

namespace hozon {
namespace netaos {
namespace sttask {

    /**
     * @brief Class of STTaskConfig
     *
     * TBD.
     */
    class STTaskConfig : public STObject
    {
    public:

        struct OPERATION_CONFIG
        {
            uint32_t maxOperationCount;
            uint8_t  queueMethod;
        };

        static const uint8_t      QUEUE_METHOD_EXCLUDE_EXECUTING           = 0x01;
        static const uint8_t      QUEUE_METHOD_INCLUDE_EXECUTING           = 0x02;
        static const uint8_t      QUEUE_METHOD_DELETE_FRONT_IF_OVERFLOW    = 0x10;
        static const uint8_t      QUEUE_METHOD_DELETE_BACK_IF_OVERFLOW     = 0x20;
        static const uint8_t      QUEUE_METHOD_DELETE_SELF_IF_OVERFLOW     = 0x40;


        static const uint8_t      DEFAULT_QUEUE_METHOD = (QUEUE_METHOD_INCLUDE_EXECUTING | QUEUE_METHOD_DELETE_SELF_IF_OVERFLOW);

        STTaskConfig();
        virtual ~STTaskConfig();

        void        configCommandChannel(uint32_t channelId, uint32_t maxSize = ST_TASK_SIZE_UNLIMITED);
        void        registerCommand(uint32_t commandId, uint32_t channelId);
        void        registerAllCommands(uint32_t commandIdMin, uint32_t commandIdMax, uint32_t channelId);
        void        registerTimer(uint32_t timerId);
        void        registerAllTimers(uint32_t timerIdMin, uint32_t timerIdMax);

        void        registerOperation(uint32_t operationId);
        void        registerOperation(uint32_t operationId, const OPERATION_CONFIG& config);
        void        registerAllOperations(uint32_t operationIdMin, uint32_t operationIdMax);
        void        registerAllOperations(uint32_t operationIdMin, uint32_t operationIdMax, const OPERATION_CONFIG& config);

        bool        isRegisteredOperation(uint32_t operationId);
        bool        isRegisteredCommand(uint32_t commandId);
        bool        isRegisteredCommandChannel(uint32_t channelId);
        bool        isRegisteredTimer(uint32_t timerId);
        bool        isRegisteredStep(uint32_t stepId);

        uint32_t    getMaxOperationCount(uint32_t operationId);
        uint8_t     getOperationQueueMethod(uint32_t operationId);
        bool        checkOperationQueueMethod(uint32_t operationId, uint8_t queueMethod);
        bool        testOperationQueueMethod(uint8_t methodToTest, uint8_t methodExpected);
        bool        getOperationConfig(uint32_t operationId, STTaskConfig::OPERATION_CONFIG& config);
        uint32_t    getCommandChannel(uint32_t commandId);
        uint32_t    getCommandChannelMaxSize(uint32_t channelId);

    private:
        struct CHANNEL_CONFIG
        {
            uint32_t          maxSize;
        };

        /**
         * key: channelId
         * value: channel info, type of CHANNEL_CONFIG
         */
        typedef std::map<uint32_t, CHANNEL_CONFIG>    CHANNEL_MAP;
        CHANNEL_MAP                         m_commandChannels;


        struct STEP_CONFIG
        {
            bool isCommand;
            uint32_t  commandChannelId;
        };

        /**
         * key: stepId(commandId or timerId)
         * value: command info, type of STEP_CONFIG
         */
        typedef std::map<uint32_t, STEP_CONFIG>    STEP_CONFIG_MAP;
        STEP_CONFIG_MAP                         m_registeredSteps;

        /**
         * key: operationId
         * value: operation info, type of OPERATION_CONFIG
         */
        typedef std::map<uint32_t, OPERATION_CONFIG>    OPERATION_CONFIG_MAP;
        OPERATION_CONFIG_MAP                m_registeredOperations;

        friend class STTaskContext;

    private:
        STTaskConfig(const STTaskConfig&);
        STTaskConfig& operator=(const STTaskConfig&);

    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STTASKCONFIGURATION_H */
/* EOF */