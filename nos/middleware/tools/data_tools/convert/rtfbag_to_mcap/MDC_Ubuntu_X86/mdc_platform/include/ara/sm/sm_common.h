/*
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
* Description: Common data structure for SM
* Create: 2021-12-13
* Notes: NA
*/

#ifndef ARA_SM_COMMON_H
#define ARA_SM_COMMON_H

#include <functional>

#include "ara/core/string.h"
#include "ara/core/vector.h"
#include "ara/core/map.h"

namespace ara {
namespace sm {
struct StateChange {
    ara::core::String functionGroupName;
    ara::core::String stateName;
};

struct FunctionGroupStates {
    ara::core::String functionGroupName;
    ara::core::Vector<ara::core::String> stateNames;
};

enum class SmResultCode : uint8_t {
    kSuccess = 0U,  /* 执行成功 */
    kInvalid,       /* 入参非法 */
    kFailed,        /* 执行失败 */
    kTimeout,       /* 请求未响应 */
    kCommError,     /* 通信异常 */
    kFileError,     /* 文件操作错误 */
    kRejected,      /* 请求仲裁失败 */
    kBusy,          /* 请求繁忙 */
    kVersionError,  /* 版本不配套 */
    kEnd            /* 枚举终止位 */
};

enum class SysActionType : uint8_t {
    kSoft = 0U, /* 软复位 */
    kHard       /* 硬复位 */
};

enum class SysResetCause : uint8_t {
    kNormal = 0U,
    kUpdate
};

struct SysResetCode {
    SysActionType actionType;
    uint32_t actionTime;
};

enum class PlatformState : uint8_t {
    kDefault = 0U,      /* 起始 */
    kWorking,           /* 工作 */
    kUpgrade,           /* 升级 */
    kStandalone,        /* MCU独立运行 */
    kReset,             /* 复位 */
    kStandby,           /* 低功耗 */
    kShutdown,          /* 下电 */
    kEnd                /* 枚举结束位 */
};

using TaskHandler = std::function<SmResultCode(ara::core::String topic)>;
}
}

#endif
