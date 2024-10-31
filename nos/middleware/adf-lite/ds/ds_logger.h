#pragma once

#include "adf-lite/include/adf_lite_logger.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class DsLogger {
public:
    static DsLogger& GetInstance() {
        static DsLogger instance;
        return instance;
    }

    CtxLogger _logger;
};

#define DS_LOG_FATAL          CTX_LOG_FATAL(DsLogger::GetInstance()._logger)
#define DS_LOG_ERROR          CTX_LOG_ERROR(DsLogger::GetInstance()._logger)
#define DS_LOG_WARN           CTX_LOG_WARN(DsLogger::GetInstance()._logger)
#define DS_LOG_INFO           CTX_LOG_INFO(DsLogger::GetInstance()._logger)
#define DS_LOG_DEBUG          CTX_LOG_DEBUG(DsLogger::GetInstance()._logger)
#define DS_LOG_VERBOSE        CTX_LOG_VERBOSE(DsLogger::GetInstance()._logger)

}
}
}