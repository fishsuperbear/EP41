#include "data_tools_logger.hpp"

#if !defined(RETURN_VAL_IF)
#define RETURN_VAL_IF(condition, val)      \
    if (condition) {                       \
        AWARN << #condition << " is met."; \
        return val;                        \
    }
#endif

// #define MODULE_NAME apollo::cyber::binary::GetName().c_str()

// #define COMMON_LOG_CRITICAL hozon::netaos::data_tool_common::BAGLogger::GetInstance().GetLogger()->LogCritical()
#define AERROR hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->LogError()
#define AWARN hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->LogWarn()
#define AINFO hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->LogInfo()
#define ADEBUG hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->LogDebug()
// #define COMMON_LOG_TRACE hozon::netaos::data_tool_common::BAGLogger::GetInstance().GetLogger()->LogTrace()