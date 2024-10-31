#pragma once

#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <thread>
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace data_tool_common {
class DataToolsLogger {
   public:
    static DataToolsLogger& GetInstance() {
        static DataToolsLogger instance;
        return instance;
    }

    ~DataToolsLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetBagLogger() const { return bag_logger_; }

    std::shared_ptr<hozon::netaos::log::Logger> GetTopicLogger() const { return topic_logger_; }

    std::shared_ptr<hozon::netaos::log::Logger> GetCommonLogger() const { return common_logger_; }

    void InitLogg(hozon::netaos::log::LogLevel bag_tool_level) {
#ifdef TARGET_PLATFORM
        std::string platform = TARGET_PLATFORM;
        if ("orin" == platform) {
            hozon::netaos::log::InitLogging("DATA_TOOLS", "hozon log test Application", bag_tool_level, hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 20);
        } else {
            hozon::netaos::log::InitLogging("DATA_TOOLS", "hozon log test Application", bag_tool_level, hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
        }
#else
        hozon::netaos::log::InitLogging("DATA_TOOLS", "hozon log test Application", bag_tool_level, hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 20);
#endif
    }

   private:
    DataToolsLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> bag_logger_{hozon::netaos::log::CreateLogger("BAG", "NETAOS BAG", hozon::netaos::log::LogLevel::kInfo)};
    std::shared_ptr<hozon::netaos::log::Logger> topic_logger_{hozon::netaos::log::CreateLogger("TOPIC", "NETAOS TOPIC", hozon::netaos::log::LogLevel::kInfo)};
    std::shared_ptr<hozon::netaos::log::Logger> common_logger_{hozon::netaos::log::CreateLogger("COMMON", "NETAOS COMMON", hozon::netaos::log::LogLevel::kInfo)};
};

#define NODE_LOG_HEAD getpid() << " " << (long int)syscall(__NR_gettid) << " " << __FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 << "(" << __LINE__ << ") |"

#define BAG_LOG_CRITICAL hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetBagLogger()->LogCritical() << NODE_LOG_HEAD
#define BAG_LOG_ERROR hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetBagLogger()->LogError() << NODE_LOG_HEAD
#define BAG_LOG_WARN hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetBagLogger()->LogWarn() << NODE_LOG_HEAD
#define BAG_LOG_INFO hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetBagLogger()->LogInfo() << NODE_LOG_HEAD
#define BAG_LOG_DEBUG hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetBagLogger()->LogDebug() << NODE_LOG_HEAD
#define BAG_LOG_TRACE hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetBagLogger()->LogTrace() << NODE_LOG_HEAD

#define TOPIC_LOG_CRITICAL hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetTopicLogger()->LogCritical() << NODE_LOG_HEAD
#define TOPIC_LOG_ERROR hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetTopicLogger()->LogError() << NODE_LOG_HEAD
#define TOPIC_LOG_WARN hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetTopicLogger()->LogWarn() << NODE_LOG_HEAD
#define TOPIC_LOG_INFO hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetTopicLogger()->LogInfo() << NODE_LOG_HEAD
#define TOPIC_LOG_DEBUG hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetTopicLogger()->LogDebug() << NODE_LOG_HEAD
#define TOPIC_LOG_TRACE hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetTopicLogger()->LogTrace() << NODE_LOG_HEAD

#define COMMON_LOG_CRITICAL hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->LogCritical() << NODE_LOG_HEAD
#define COMMON_LOG_ERROR hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->LogError() << NODE_LOG_HEAD
#define COMMON_LOG_WARN hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->LogWarn() << NODE_LOG_HEAD
#define COMMON_LOG_INFO hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->LogInfo() << NODE_LOG_HEAD
#define COMMON_LOG_DEBUG hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->LogDebug() << NODE_LOG_HEAD
#define COMMON_LOG_TRACE hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->LogTrace() << NODE_LOG_HEAD

#define BAG_LOG_CRITICAL_WITH_HEAD BAG_LOG_CRITICAL
#define BAG_LOG_ERROR_WITH_HEAD BAG_LOG_ERROR
#define BAG_LOG_WARN_WITH_HEAD BAG_LOG_WARN
#define BAG_LOG_INFO_WITH_HEAD BAG_LOG_INFO
#define BAG_LOG_DEBUG_WITH_HEAD BAG_LOG_DEBUG
#define BAG_LOG_TRACE_WITH_HEAD BAG_LOG_TRACE

//这是为了使用命令行工具能在屏幕显示相关信息而写的log工具

enum class LogLevel : uint8_t { kInfo = 0x00U, kWarn = 0x01U, kError = 0x02U, kOff = 0x03U };

class BagLogStream final {
   public:
    BagLogStream() = delete;

    BagLogStream(LogLevel level) noexcept { m_logLevel = level; };

    BagLogStream(BagLogStream&& other) : m_logLevel(other.m_logLevel), m_osLog(std::move(other.m_osLog)) {}

    ~BagLogStream() { LogOut(); };

    BagLogStream(const BagLogStream&) = delete;

    BagLogStream& operator=(const BagLogStream&) = delete;

    BagLogStream& operator=(BagLogStream&&) = delete;

    BagLogStream& operator<<(const char* const value) noexcept {
        m_osLog << value;
        return *this;
    };

    BagLogStream& operator<<(const std::string& value) noexcept {
        m_osLog << value;
        return *this;
    };

   private:
    void LogOut() {
        if (LogLevel::kError == m_logLevel) {
            std::cout << "\033[31m" << m_osLog.str() << "\033[0m" << std::endl;
        } else if (LogLevel::kWarn == m_logLevel) {
            std::cout << "\033[33m" << m_osLog.str() << "\033[0m" << std::endl;
        } else if (LogLevel::kInfo == m_logLevel) {
            std::cout << m_osLog.str() << std::endl;
        }
    };

    LogLevel m_logLevel;
    std::ostringstream m_osLog;
};

class BAGCONSOLELogger {
   public:
    static BAGCONSOLELogger& GetInstance() {
        static BAGCONSOLELogger instance;
        return instance;
    }

    ~BAGCONSOLELogger(){};

    void setLogLevel(LogLevel level) { _level = level; }

    BagLogStream LogError() noexcept {
        if (LogLevel::kOff == _level) {
            return BagLogStream{LogLevel::kOff};
        } else {
            return BagLogStream{LogLevel::kError};
        }
    }

    BagLogStream LogWarn() noexcept {
        if (LogLevel::kOff == _level) {
            return BagLogStream{LogLevel::kOff};
        } else {
            return BagLogStream{LogLevel::kWarn};
        }
    }

    BagLogStream LogInfo() noexcept {
        if (LogLevel::kOff == _level) {
            return BagLogStream{LogLevel::kOff};
        } else {
            return BagLogStream{LogLevel::kInfo};
        }
    }

   private:
    BAGCONSOLELogger(){};
    LogLevel _level = LogLevel::kOff;
};

#define CONCLE_BAG_LOG_ERROR hozon::netaos::data_tool_common::BAGCONSOLELogger::GetInstance().LogError()
#define CONCLE_BAG_LOG_WARN hozon::netaos::data_tool_common::BAGCONSOLELogger::GetInstance().LogWarn()
#define CONCLE_BAG_LOG_INFO hozon::netaos::data_tool_common::BAGCONSOLELogger::GetInstance().LogInfo()

}  // namespace data_tool_common
}  // namespace netaos
}  // namespace hozon
