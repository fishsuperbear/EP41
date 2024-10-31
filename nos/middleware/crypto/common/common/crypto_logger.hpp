/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: crypto server loger
 */

#ifndef CRYPTO_SERVER_LOGGER_H_
#define CRYPTO_SERVER_LOGGER_H_

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "log/include/logging.h"


namespace hozon {
namespace netaos {
namespace crypto {

/* Fm log class */
class CryptoLogger
{
public:
    CryptoLogger() : logger_(nullptr) {};
    virtual ~CryptoLogger() {};

    enum class CryptoLogLevelType {
        CRYPTO_KOFF = 0,
        CRYPTO_CRITICAL = 1,
        CRYPTO_ERROR = 2,
        CRYPTO_WARN = 3,
        CRYPTO_INFO = 4,
        CRYPTO_DEBUG = 5,
        CRYPTO_TRACE = 6
    };


    hozon::netaos::log::LogLevel CryptoParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<CryptoLogLevelType>(logLevel);
        switch (type) {
            case CryptoLogLevelType::CRYPTO_KOFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            case CryptoLogLevelType::CRYPTO_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case CryptoLogLevelType::CRYPTO_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case CryptoLogLevelType::CRYPTO_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case CryptoLogLevelType::CRYPTO_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case CryptoLogLevelType::CRYPTO_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case CryptoLogLevelType::CRYPTO_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    // only process can use this function
    void InitLogging(std::string appId = "DEFAULT_APP",  // the log id of application
        std::string appDescription = "default application", // the log id of application
        CryptoLogLevelType appLogLevel = CryptoLogLevelType::CRYPTO_ERROR, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
    )
    {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = CryptoParseLogLevel(static_cast<int32_t> (appLogLevel));
        hozon::netaos::log::InitLogging(
            appId,
            appDescription,
            applevel,
            outputMode,
            directoryPath,
            maxLogFileNum,
            maxSizeOfLogFile
        );
    }

    // context regist diagserver
    void CreateLogger(const std::string ctxId)
    {
        const hozon::netaos::log::LogLevel level = CryptoParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static CryptoLogger& GetInstance()
    {
        static CryptoLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> CryptoGetLogger() const
    {
        return logger_;
    }

    int32_t getLogLevel()
    {
        return level_;
    }

    void setLogLevel(int32_t level)
    {
        level_ = level;
    }

    std::string ToHexString(const unsigned char* p, size_t size) {
        std::ostringstream oss;
        for (size_t i = 0; i < size; ++i) {
            oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(*(p + i));
            if ((i + 1) % 32 == 0) {
                oss << std::endl;
            }
            else if ((i + 1) % 16 == 0) {
                oss << "    ";
            }
            else {
                oss << " ";
            }
        }

        return oss.str();
    }

private:
    CryptoLogger(const CryptoLogger&);
    CryptoLogger& operator=(const CryptoLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(CryptoLogLevelType::CRYPTO_INFO);
};

#define CRYP_HEAD                             \
    "P" << (long int)syscall(__NR_getpid) \
    << " T" << (long int)syscall(__NR_gettid) << " " << __FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "
#define CRYP_CRITICAL (CryptoLogger::GetInstance().CryptoGetLogger()->LogCritical() << CRYP_HEAD)
#define CRYP_ERROR (CryptoLogger::GetInstance().CryptoGetLogger()->LogError() << CRYP_HEAD)
#define CRYP_WARN (CryptoLogger::GetInstance().CryptoGetLogger()->LogWarn() << CRYP_HEAD)
#define CRYP_INFO (CryptoLogger::GetInstance().CryptoGetLogger()->LogInfo() << CRYP_HEAD)
#define CRYP_DEBUG (CryptoLogger::GetInstance().CryptoGetLogger()->LogDebug() << CRYP_HEAD)
#define CRYP_TRACE (CryptoLogger::GetInstance().CryptoGetLogger()->LogTrace() << CRYP_HEAD)
#define CRYP_ERROR_MESSAGE(code) "code " << static_cast<uint32_t>(code) << ": " << hozon::netaos::crypto::GetCryptoErrorDomain().Message(code)

}  // namespace crypto
}  // namespace netaos
}  // namespace hozon

#endif
// end of file
