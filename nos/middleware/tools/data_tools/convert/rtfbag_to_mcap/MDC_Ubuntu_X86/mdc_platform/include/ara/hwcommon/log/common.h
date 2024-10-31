/*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
* Description: LogStreamBuffer class header
* Create: 2019-7-2
*/
#ifndef INC_ARA_GODEL_COMMON_LOG_COMMON_H
#define INC_ARA_GODEL_COMMON_LOG_COMMON_H
#include <cstdint>
namespace ara    {
namespace godel  {
namespace common {
namespace log    {
enum class LogType : uint8_t {
    SCREEN_LOG = 0U,
    ARA_LOG,
    SYS_LOG
};
enum class LogLevel : uint8_t {
    VRTF_COMMON_LOG_OFF      = 0x00U,
    VRTF_COMMON_LOG_FATAL    = 0x01U,
    VRTF_COMMON_LOG_ERROR    = 0x02U,
    VRTF_COMMON_LOG_WARN     = 0x03U,
    VRTF_COMMON_LOG_INFO     = 0x04U,
    VRTF_COMMON_LOG_DEBUG    = 0x05U,
    VRTF_COMMON_LOG_VERBOSE  = 0x06U
};
enum class LogMode : uint8_t {
    LOG_REMOTE = 0U,
    LOG_FILE,
    LOG_CONSOLE
};
} // end log
} // end common
} // end godel
} // end ara
#endif

