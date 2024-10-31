#pragma once

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include "log/include/logging.h"

using namespace hozon::netaos::log;

class DefaultLogger {
public:
    static DefaultLogger& GetInstance() {
        static DefaultLogger instance;
        return instance;
    }
    ~DefaultLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

    void InitLogger() {
        hozon::netaos::log::InitLogging(
            "default",
            "default logger instance for test",
            LogLevel::kInfo,
            HZ_LOG2FILE,
            ".",
            10,
            20
        );

        logger_ = hozon::netaos::log::CreateLogger("default", "default logger instance",
                                                    hozon::netaos::log::LogLevel::kInfo);
    }

private:
    DefaultLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
};

#define DF_LOG_CRITICAL             (DefaultLogger::GetInstance().GetLogger()->LogCritical())
#define DF_LOG_ERROR                (DefaultLogger::GetInstance().GetLogger()->LogError())
#define DF_LOG_WARN                 (DefaultLogger::GetInstance().GetLogger()->LogWarn())
#define DF_LOG_INFO                 (DefaultLogger::GetInstance().GetLogger()->LogInfo())
#define DF_LOG_DEBUG                (DefaultLogger::GetInstance().GetLogger()->LogDebug())
#define DF_LOG_TRACE                (DefaultLogger::GetInstance().GetLogger()->LogTrace())

#define DEBUG_LOG(format, ...) \
 { \
    char print_msg[1024]= { 0 };    \
    struct timeval tv;              \
    gettimeofday(&tv, nullptr);     \
    struct tm *timeinfo = localtime(&tv.tv_sec);        \
    uint32_t milliseconds = tv.tv_usec / 1000;          \
    char time_buf[64] = { 0 };                          \
    memset(time_buf, 0x00, sizeof(time_buf));           \
    memset(print_msg, 0x00, sizeof(print_msg));         \
    snprintf(time_buf, sizeof(time_buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",         \
        timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,             \
        timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, milliseconds);          \
    snprintf(print_msg, sizeof(print_msg), (format), ##__VA_ARGS__);                   \
    printf("[%s] [%d %ld %s@%s(%d) | %s]\n", time_buf, getpid(), syscall(__NR_gettid), \
        __FUNCTION__, (nullptr == strrchr(__FILE__, '/')) ? __FILE__: (strrchr(__FILE__, '/') + 1), __LINE__, (print_msg)); \
 }
