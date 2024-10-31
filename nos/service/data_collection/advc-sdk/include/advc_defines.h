#ifndef ADVC_DEFINE_H
#define ADVC_DEFINE_H

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "Poco/JSON/Parser.h"
#include "util/string_util.h"

using Poco::Dynamic::Var;
using Poco::JSON::Array;
using Poco::JSON::Object;
using Poco::JSON::Parser;

namespace advc {

typedef enum log_out_type { ADVC_LOG_NULL = 0, ADVC_LOG_STDOUT, ADVC_LOG_SYSLOG } LOG_OUT_TYPE;

typedef enum { HTTP_HEAD, HTTP_GET, HTTP_PUT, HTTP_POST, HTTP_DELETE, HTTP_OPTIONS } HTTP_METHOD;

typedef enum advc_log_level {
    ADVC_LOG_ERR = 1,   // LOG_ERR
    ADVC_LOG_WARN = 2,  // LOG_WARNING
    ADVC_LOG_INFO = 3,  // LOG_INFO
    ADVC_LOG_DBG = 4    // LOG_DEBUG
} LOG_LEVEL;

#define LOG_LEVEL_STRING(level)                        \
    (((level) == ADVC_LOG_DBG)    ? "[ADVC-SDK DBG] "  \
     : ((level) == ADVC_LOG_INFO) ? "[ADVC-SDK INFO] " \
     : ((level) == ADVC_LOG_WARN) ? "[ADVC-SDK WARN] " \
     : ((level) == ADVC_LOG_ERR)  ? "[ADVC-SDK ERR] "  \
                                  : "[ADVC-SDK CRIT]")

#define ADVC_LOW_LOGPRN(level, fmt, ...)                                                                               \
    if (level <= ADVC_LOG_ERR) {                                                                                       \
        if (AdvcSysConfig::GetLogOutType() == ADVC_LOG_STDOUT) {                                                       \
            fprintf(stdout, "%s:%s(%d) " fmt "\n", LOG_LEVEL_STRING(level), __func__, __LINE__, ##__VA_ARGS__);        \
        } else if (AdvcSysConfig::GetLogOutType() == ADVC_LOG_SYSLOG) {                                                \
            LogUtil::Syslog(level, "%s:%s(%d) " fmt "\n", LOG_LEVEL_STRING(level), __func__, __LINE__, ##__VA_ARGS__); \
        } else {                                                                                                       \
        }                                                                                                              \
    } else {                                                                                                           \
    }

#define SDK_LOG_DBG(fmt, ...) ADVC_LOW_LOGPRN(ADVC_LOG_DBG, fmt, ##__VA_ARGS__)
#define SDK_LOG_INFO(fmt, ...) ADVC_LOW_LOGPRN(ADVC_LOG_INFO, fmt, ##__VA_ARGS__)
#define SDK_LOG_WARN(fmt, ...) ADVC_LOW_LOGPRN(ADVC_LOG_WARN, fmt, ##__VA_ARGS__)
#define SDK_LOG_ERR(fmt, ...) ADVC_LOW_LOGPRN(ADVC_LOG_ERR, fmt, ##__VA_ARGS__)
#define SDK_LOG_ADVC(level, fmt, ...) ADVC_LOW_LOGPRN(level, fmt, ##__VA_ARGS__)

}  // namespace advc
#endif
