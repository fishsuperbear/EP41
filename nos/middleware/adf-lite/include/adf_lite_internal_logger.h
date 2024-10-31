#pragma once

#include <iostream>
#include "adf-lite/include/adf_lite_logger.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class EarlyLogger {
public:
    ~EarlyLogger() {
        std::cout << std::endl;
    }

    template<typename T>
    EarlyLogger& operator<<(const T& value) {
        std::cout << value;
        return *this;
    }
};

#define WITH_LINE(x)    "\033[4m" << x << "\033[0m"
#define ADF_EARLY_LOG          hozon::netaos::adf_lite::EarlyLogger()

class AdfInternalLogger {
public:
    static AdfInternalLogger& GetInstance() {
        static AdfInternalLogger instance;
        return instance;
    }

    CtxLogger _logger;
};

#define ADF_INTERNAL_LOG_FATAL          CTX_LOG_FATAL(AdfInternalLogger::GetInstance()._logger)
#define ADF_INTERNAL_LOG_ERROR          CTX_LOG_ERROR(AdfInternalLogger::GetInstance()._logger)
#define ADF_INTERNAL_LOG_WARN           CTX_LOG_WARN(AdfInternalLogger::GetInstance()._logger)
#define ADF_INTERNAL_LOG_INFO           CTX_LOG_INFO(AdfInternalLogger::GetInstance()._logger)
#define ADF_INTERNAL_LOG_DEBUG          CTX_LOG_DEBUG(AdfInternalLogger::GetInstance()._logger)
#define ADF_INTERNAL_LOG_VERBOSE        CTX_LOG_VERBOSE(AdfInternalLogger::GetInstance()._logger)

#define EXEC_HEAD                   "(" << _config.executor_name << ") "
#define ADF_EXEC_LOG_FATAL          CTX_LOG_FATAL(AdfInternalLogger::GetInstance()._logger) << EXEC_HEAD
#define ADF_EXEC_LOG_ERROR          CTX_LOG_ERROR(AdfInternalLogger::GetInstance()._logger) << EXEC_HEAD
#define ADF_EXEC_LOG_WARN           CTX_LOG_WARN(AdfInternalLogger::GetInstance()._logger) << EXEC_HEAD
#define ADF_EXEC_LOG_INFO           CTX_LOG_INFO(AdfInternalLogger::GetInstance()._logger) << EXEC_HEAD
#define ADF_EXEC_LOG_DEBUG          CTX_LOG_DEBUG(AdfInternalLogger::GetInstance()._logger) << EXEC_HEAD
#define ADF_EXEC_LOG_VERBOSE        CTX_LOG_VERBOSE(AdfInternalLogger::GetInstance()._logger) << EXEC_HEAD

}
}
}