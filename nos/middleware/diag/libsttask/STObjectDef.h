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
 * @file  STObjectDef.h
 * @brief Class of STObjectDef
 */
#ifndef STOBJECTDEF_H
#define STOBJECTDEF_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

namespace hozon {
namespace netaos {
namespace sttask {

const uint32_t ST_TASK_INVALID_OPERATION = static_cast<uint32_t>(-1);
const uint32_t ST_TASK_INVALID_CHANNEL = static_cast<uint32_t>(-1);
const uint32_t ST_TASK_INVALID_STEP = static_cast<uint32_t>(-1);
const uint32_t ST_TASK_INVALID_MODULE = static_cast<uint32_t>(-1);
const uint32_t ST_TASK_SIZE_UNLIMITED = static_cast<uint32_t>(-1);
const uint32_t ST_TASK_INTERRUPTREASON_INVALID = static_cast<uint32_t>(-1);

const uint32_t ST_TIME_INFINITE = static_cast<uint32_t>(-1);

const uint32_t INFINITE = static_cast<uint32_t>(-1);

enum ST_OBJECT_TYPE
{
    ST_OBJECT_TYPE_GEN = 0,
    ST_OBJECT_TYPE_TASK,
    ST_OBJECT_TYPE_TASKRUNNER,
    ST_OBJECT_TYPE_MODULEMANAGER,
    ST_OBJECT_TYPE_CONTEXT,
    ST_OBJECT_TYPE_MODULE,
    ST_OBJECT_TYPE_CONFIRUATION,
};

enum ST_TASK_TYPE
{
    ST_TASK_TYPE_COMMAND = 0,
    ST_TASK_TYPE_TIMER,
    ST_TASK_TYPE_NORMAL,
};

enum ST_TASK_RESULT
{
    eNone = 0,
    eOK,
    eContinue,
    eMemErr,
    eError,
    eTimeout,
    eInterrupt,
    eBusy,
    eEvtResult,
    eChannelErr,
    eStartErr,
    eNotRegisteredOperation,
    eNotRegisteredCommand,
    eNotRegisteredTimer,
    eOperationChannelOverflow,
    eCommandChannelOverflow,
    eDeleteFront,
    eDeleteBack,
    eDefaultMax = 0xFF,
};

enum ST_TASK_COMMAND
{
    eCommand_DefaultMax = 0xFF,
};

enum ST_TASK_COMMAND_CHANNEL
{
    eCommandChannel_DefaultMax = 0xFF,
};


enum ST_TASK_OPERATON
{
    eOperation_HandleEvent = 0,
    eOperation_DefaultMax = 0xFF,
};

enum ST_TASK_CALLKIND
{
    eCallKind_DefaultMax = 0xFF,
};

enum ST_TASK_EVENTKIND
{
    eEventKind_TimerEvent = 0xFE, // IMPORTANT: client code must use eEventKind_DefaultMax or behind
    eEventKind_DefaultMax = 0xFF,
};

} // end of sttask
} // end of netaos
} // end of hozon
#endif /* STOBJECTDEF_H */
/* EOF */